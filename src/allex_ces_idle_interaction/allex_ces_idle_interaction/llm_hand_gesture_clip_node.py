#!/usr/bin/env python3
"""
LLM Publisher Node - CLIP 앙상블 추론 + RUN/STOP 제어
v1.2.0 - GUI에서 RUN 명령을 받으면 추론 시작, STOP 명령을 받으면 대기

시스템 구조:
- SPARK 1 PC: Camera Publisher (카메라 + YOLO 추적) → /allex_camera/target_crop/compressed 발행
- SPARK 2 PC: LLM Publisher (이 노드) → /allex_camera/target_crop/compressed 구독, /llm/response 발행
- Laptop: GUI → /llm/control 발행하여 LLM Publisher 제어
"""

import torch
import time
import json
from PIL import Image
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2

from concurrent.futures import ThreadPoolExecutor

# --- 설정 ---
MODEL_IDS = [
    "openai/clip-vit-large-patch14-336",
    #"facebook/metaclip-h14-fullcc2.5b"
]
USE_FP16 = True
TARGET_INFER_HZ = 7  # 목표 추론 Hz

# 실시간 스트리밍용 QoS
REALTIME_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

# 제어 명령용 QoS (신뢰성 보장)
RELIABLE_QOS = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    deadline=Duration(seconds=0, nanoseconds=0),
)

# --- 분류 라벨 ---
# 각 제스처(handshake, highfive, fist)에 대해 서브라벨 3개씩 정의
LABELS = [
    # handshake (3개 서브라벨)
    "a person reaching their hand forward to shake hands, hand near waist height, fingers relaxed",
    "a person extending their right hand forward for a handshake, arm slightly bent",
    "a close-up of two hands about to shake, fingers open and relaxed",
    # highfive (3개 서브라벨)
    "a person raising one hand high above their head for a high five, palm open",
    "a person leaning forward with hand up for a high five, palm facing forward",
    "two people with one hand each in the air about to high five",
    # fist bump (3개 서브라벨)
    "a person extending a closed fist forward for a fist bump",
    "two fists meeting in a fist bump gesture",
    "a person holding a clenched fist out in front of them for a friendly bump",
    # idle (1개 라벨)
    "a person standing normally with hands down and no interaction"
]
SHORT_LABELS = ("handshake", "highfive", "fist", "idle")


def check_gpu_status():
    """GPU 상태 확인 및 출력"""
    print("=" * 60)
    print("LLM Publisher - GPU 상태 확인")
    print("=" * 60)
    
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    메모리: {props.total_memory / 1024**3:.1f} GB")
        
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"현재 GPU 이름: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA를 사용할 수 없습니다! CPU로 실행됩니다.")
    
    print("=" * 60)


class LLMPublisher(Node):
    """LLM Publisher Node - CLIP 앙상블 추론 + RUN/STOP 제어"""
    
    def __init__(self) -> None:
        super().__init__('llm_publisher')
        
        # GPU 상태 확인
        check_gpu_status()
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 토픽 이름 (topics.json에서 가져와도 되지만, 독립 실행을 위해 하드코딩)
        self.image_topic = "/allex_camera/target_crop/compressed"
        self.result_topic = "/llm/response"
        self.control_topic = "/llm/control"
        
        # 이미지 구독 (실시간 QoS)
        self.image_subscription = self.create_subscription(
            CompressedImage,
            self.image_topic,
            self._image_callback,
            REALTIME_QOS
        )
        
        # 결과 발행
        self.result_publisher = self.create_publisher(
            String,
            self.result_topic,
            REALTIME_QOS
        )
        
        # 제어 명령 구독 (GUI에서 RUN/STOP)
        self.control_subscription = self.create_subscription(
            String,
            self.control_topic,
            self._control_callback,
            RELIABLE_QOS
        )
        
        # 상태 발행 (GUI에 현재 상태 알림)
        self.status_publisher = self.create_publisher(
            String,
            "/llm/status",
            10
        )
        
        # CLIP 모델 로드
        self.get_logger().info("📦 CLIP 모델 로드 중...")
        self._load_models()
        
        # 병렬 실행용 ThreadPool
        self.thread_pool = ThreadPoolExecutor(max_workers=len(MODEL_IDS))
        
        # 추론 제어
        self.last_infer_time = 0
        self.min_infer_interval = 1.0 / TARGET_INFER_HZ
        self.infer_hz = 0.0
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        # 주기적 상태 발행 타이머 (1초마다)
        self.status_timer = self.create_timer(1.0, self._publish_status)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("✅ LLM Publisher 초기화 완료!")
        self.get_logger().info(f"📡 이미지 구독: {self.image_topic}")
        self.get_logger().info(f"📡 결과 발행: {self.result_topic}")
        self.get_logger().info(f"📡 제어 구독: {self.control_topic}")
        self.get_logger().info(f"⚙️  목표 Hz: {TARGET_INFER_HZ}")
        self.get_logger().info(f"🔀 병렬 처리: {len(MODEL_IDS)}개 모델")
        self.get_logger().info("=" * 60)
        self.get_logger().info("⏸️  대기 중: RUN 명령을 기다립니다...")
    
    def _load_models(self):
        """CLIP 모델 로드"""
        from transformers import CLIPProcessor, CLIPModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.get_logger().info(f"🖥️  Device: {self.device}")
        
        self.models = []
        self.processors = []
        self.streams = []  # CUDA Streams
        
        for model_id in MODEL_IDS:
            self.get_logger().info(f"")
            self.get_logger().info(f"{'='*50}")
            self.get_logger().info(f"🧠 Loading: {model_id}")
            self.get_logger().info(f"{'='*50}")
            
            try:
                # 먼저 로컬 캐시에서 로드 시도
                model = CLIPModel.from_pretrained(model_id, local_files_only=True).to(self.device)
                processor = CLIPProcessor.from_pretrained(model_id, local_files_only=True)
            except Exception:
                # 로컬에 없으면 다운로드
                self.get_logger().info(f"   로컬 캐시에 없음, 다운로드 중...")
                model = CLIPModel.from_pretrained(model_id).to(self.device)
                processor = CLIPProcessor.from_pretrained(model_id)
            
            if USE_FP16 and self.device == "cuda":
                model = model.half()
            
            model.eval()
            self.models.append(model)
            self.processors.append(processor)
            
            # 각 모델용 CUDA Stream 생성
            if self.device == "cuda":
                self.streams.append(torch.cuda.Stream())
            
            self.get_logger().info(f"✅ {model_id} 로드 완료!")
        
        self.get_logger().info(f"")
        self.get_logger().info(f"🎯 총 {len(self.models)}개 모델 병렬 앙상블 준비 완료!")
    
    def _control_callback(self, msg: String):
        """제어 명령 콜백 - GUI에서 RUN/STOP"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type', '')
            
            if cmd_type == 'run' or cmd_type == 'start':
                if not self.is_running:
                    self.is_running = True
                    self.get_logger().info("▶️  RUN 명령 수신 - 추론 시작!")
                    self._publish_status()
            
            elif cmd_type == 'stop':
                if self.is_running:
                    self.is_running = False
                    self.get_logger().info("⏹️  STOP 명령 수신 - 추론 중지")
                    self._publish_status()
            
            elif cmd_type == 'status':
                # 상태 요청
                self._publish_status()
            
            else:
                self.get_logger().warn(f"알 수 없는 명령: {cmd_type}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"제어 명령 처리 실패: {e}")
    
    def _publish_status(self):
        """현재 상태를 발행"""
        try:
            status = {
                'running': self.is_running,
                'hz': round(self.infer_hz, 1),
                'models_loaded': len(self.models),
                'device': self.device,
                'timestamp': time.monotonic()
            }
            
            msg = String()
            msg.data = json.dumps(status)
            self.status_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"상태 발행 실패: {e}")
    
    def _image_callback(self, msg: CompressedImage):
        """이미지 콜백 - RUN 상태일 때만 추론"""
        # RUN 상태가 아니면 무시
        if not self.is_running:
            return
        
        curr_time = time.time()
        time_since_last = curr_time - self.last_infer_time
        
        # Hz 제한
        if time_since_last < self.min_infer_interval:
            return
        
        try:
            # CompressedImage → OpenCV 이미지
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            # BGR → RGB 변환 후 PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 병렬 추론 실행
            futures = []
            for idx in range(len(self.models)):
                future = self.thread_pool.submit(self._infer_single_model, idx, pil_image)
                futures.append(future)
            
            # 결과 수집 및 앙상블
            num_sub_labels = len(LABELS)  # 10개 (3+3+3+1)
            sublabel_probs = [0.0] * num_sub_labels
            for future in futures:
                model_probs = future.result()
                for i in range(num_sub_labels):
                    sublabel_probs[i] += model_probs[i] * 100
            
            # 서브라벨 → 메인 라벨 4개로 집계
            ensemble_probs = [0.0] * 4
            ensemble_probs[0] = sublabel_probs[0] + sublabel_probs[1] + sublabel_probs[2]  # handshake
            ensemble_probs[1] = sublabel_probs[3] + sublabel_probs[4] + sublabel_probs[5]  # highfive
            ensemble_probs[2] = sublabel_probs[6] + sublabel_probs[7] + sublabel_probs[8]  # fist
            ensemble_probs[3] = sublabel_probs[9]  # idle
            
            # CUDA 동기화
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # 커스텀 best 결정 로직
            IDLE_IDX = 3
            IDLE_THRESHOLD = 70.0
            
            if ensemble_probs[IDLE_IDX] >= IDLE_THRESHOLD:
                best_idx = IDLE_IDX
            else:
                non_idle_probs = ensemble_probs[:IDLE_IDX]
                best_idx = non_idle_probs.index(max(non_idle_probs))
            
            # Hz 계산
            if self.last_infer_time > 0:
                self.infer_hz = 1.0 / (curr_time - self.last_infer_time)
            self.last_infer_time = curr_time
            
            # 결과 발행
            result = {
                "best": SHORT_LABELS[best_idx],
                "probs": {SHORT_LABELS[i]: round(ensemble_probs[i], 1) for i in range(4)},
                "hz": round(self.infer_hz, 1),
                "timestamp": curr_time
            }
            
            result_msg = String()
            result_msg.data = json.dumps(result)
            self.result_publisher.publish(result_msg)
            
            # 프레임 카운트 및 주기적 로그
            self.frame_count += 1
            current_time = time.monotonic()
            if current_time - self.last_log_time > 5.0:
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                gpu_mem_str = ""
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    gpu_mem_str = f" | GPU 메모리: {allocated:.0f}MB"
                
                self.get_logger().info(
                    f"🧠 추론 중: {SHORT_LABELS[best_idx]} | "
                    f"Hz: {self.infer_hz:.1f}{gpu_mem_str}"
                )
                self.frame_count = 0
                self.last_log_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"추론 오류: {e}")
    
    def _infer_single_model(self, idx, pil_image):
        """단일 모델 추론 (병렬 실행용)"""
        model = self.models[idx]
        processor = self.processors[idx]
        
        # CUDA Stream 사용
        if self.device == "cuda" and idx < len(self.streams):
            stream = self.streams[idx]
            with torch.cuda.stream(stream):
                return self._run_inference(model, processor, pil_image)
        else:
            return self._run_inference(model, processor, pil_image)
    
    def _run_inference(self, model, processor, pil_image):
        """실제 추론 수행"""
        inputs = processor(
            text=LABELS,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        if USE_FP16 and self.device == "cuda":
            inputs = {k: v.half() if v.dtype == torch.float32 else v 
                     for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = outputs.logits_per_image.softmax(dim=1)
        return probs[0].cpu().tolist()
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        self.thread_pool.shutdown(wait=False)
        super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    llm_publisher = LLMPublisher()
    
    try:
        rclpy.spin(llm_publisher)
    except KeyboardInterrupt:
        llm_publisher.get_logger().info("\n⏹️ LLM Publisher 종료")
    finally:
        llm_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


