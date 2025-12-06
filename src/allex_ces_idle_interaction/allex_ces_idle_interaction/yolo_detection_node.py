#!/usr/bin/env python3
"""
YOLO Detection 전용 노드
이미지를 받아서 YOLO로 감지하고 BB box와 track_id를 발행
"""
import time
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

PERSON_CLASS_ID = 0  # COCO 사람 클래스 ID

def check_gpu_status():
    """GPU 상태 확인 및 출력"""
    print("=" * 60)
    print("YOLO Detection Node - GPU 상태 확인")
    print("=" * 60)
    
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
    
    print("=" * 60)


class YOLODetectionNode(Node):
    """YOLO Detection 전용 노드"""
    
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # GPU 상태 확인
        check_gpu_status()
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽명 파라미터 (Launch 파일에서 설정 가능)
        self.declare_parameter('camera_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('detections_topic', '/allex_camera/detections')
        self.declare_parameter('tracker_control_topic', '/allex_camera/tracker_control')
        
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        tracker_control_topic = self.get_parameter('tracker_control_topic').get_parameter_value().string_value
        
        # 입력 이미지 구독
        self.image_subscription = self.create_subscription(
            CompressedImage,
            camera_image_topic,
            self.image_callback,
            qos_profile,
        )
        
        # Detection 결과 발행 (BB box + track_id)
        self.detection_publisher = self.create_publisher(
            String,
            detections_topic,
            10
        )
        
        # YOLO 모델 초기화
        self._init_yolo_model()
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("YOLO Detection Node 초기화 완료")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    def _init_yolo_model(self):
        """YOLO 모델 초기화"""
        model_path = "yolo11n.pt"
        face_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
        
        # GPU 디바이스 설정
        device = 'cpu'
        if torch.cuda.is_available():
            try:
                device_props = torch.cuda.get_device_properties(0)
                compute_cap = f"{device_props.major}{device_props.minor}"
                self.get_logger().info(f"GPU Compute Capability: {device_props.major}.{device_props.minor} (sm_{compute_cap})")
                
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                device = 0
                self.get_logger().info(f"✓ GPU 사용 설정: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                self.get_logger().error(f"⚠️  GPU 사용 불가: {e}")
                raise RuntimeError("GPU를 사용할 수 없습니다.")
        
        # YOLO 모델 초기화
        self.yolo_model = YOLO(model_path)
        self.face_model = YOLO(face_model_path)
        
        # GPU로 모델 이동
        if device != 'cpu':
            self.yolo_model.to(device)
            self.face_model.to(device)
        
        self.conf_threshold = 0.7
        
        # GPU 워밍업
        self.get_logger().info("GPU 워밍업 시작...")
        warmup_start = time.monotonic()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.yolo_model.track(dummy, conf=self.conf_threshold, classes=[PERSON_CLASS_ID], verbose=False, imgsz=640)
        warmup_time = (time.monotonic() - warmup_start) * 1000
        self.get_logger().info(f"워밍업 추론 시간: {warmup_time:.1f}ms")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            self.get_logger().info(f"GPU 동기화 완료!! (메모리 사용: {allocated:.1f}MB)")
    
    def image_callback(self, msg: CompressedImage) -> None:
        """이미지 콜백 - YOLO Detection 수행"""
        if not self.is_running:
            return
        
        self.frame_count += 1
        frame_start = time.monotonic()
        
        # 압축된 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # YOLO Detection 수행 (track 사용하여 track_id 생성)
        # BOTSORT 설정 파일 무조건 사용 - 실패 시 즉시 오류 발생
        import os
        from ament_index_python import get_package_share_directory
        
        # 패키지 공유 디렉토리에서 설정 파일 찾기
        package_dir = get_package_share_directory('allex_ces_idle_interaction')
        tracker_config_path = os.path.join(package_dir, 'config', 'botsort.yaml')
        
        if not os.path.exists(tracker_config_path):
            self.get_logger().error(f"❌ BOTSORT 설정 파일을 찾을 수 없음: {tracker_config_path}")
            raise FileNotFoundError(f"BOTSORT 설정 파일이 필요합니다: {tracker_config_path}")
        
        # BOTSORT 설정 파일을 반드시 사용
        results = self.yolo_model.track(
            frame,
            conf=self.conf_threshold,
            classes=[PERSON_CLASS_ID],   # 사람만
            persist=True,                # 내부 tracker 상태 유지
            tracker=tracker_config_path,  # BOTSORT 설정 파일 사용 (필수)
            verbose=False,
            imgsz=640,
        )
        
        # Detection 결과 파싱
        detections = []
        if results and len(results) > 0:
            r = results[0]
            boxes = r.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
                    detections.append({
                        'track_id': int(tid),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(conf),
                        'centroid': [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
                    })
        
        # 처리 시간 계산
        process_time = (time.monotonic() - frame_start) * 1000
        
        # Detection 결과 발행
        self._publish_detections(detections, process_time_ms=process_time)
        
        # 주기적 성능 로그 (5초마다)
        current_time = time.monotonic()
        if current_time - self.last_log_time > 5.0:
            elapsed = current_time - self.last_log_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            gpu_mem_str = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                gpu_mem_str = f" | GPU 메모리: {allocated:.0f}MB"
            
            self.get_logger().info(
                f"Detection 중: {len(detections)}개 객체 | "
                f"처리 시간: {process_time:.1f}ms | FPS: {fps:.1f}{gpu_mem_str}"
            )
            self.frame_count = 0
            self.last_log_time = current_time
    
    def _publish_detections(self, detections, process_time_ms=None):
        """Detection 결과를 Topic으로 발행"""
        try:
            data = {
                'detections': detections,
                'performance': {
                    'process_time_ms': float(process_time_ms) if process_time_ms else 0.0
                },
                'timestamp': time.monotonic()
            }
            
            # JSON 문자열로 변환하여 발행
            json_str = json.dumps(data, ensure_ascii=False)
            msg = String()
            msg.data = json_str
            self.detection_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"Detection 결과 발행 실패: {e}")
    
    def set_running(self, running: bool):
        """실행 상태 설정"""
        self.is_running = running
        if running:
            self.get_logger().info("YOLO Detection 시작")
        else:
            self.get_logger().info("YOLO Detection 중지")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = YOLODetectionNode()
    
    # 제어 명령 구독 (간단한 구현)
    def control_callback(msg: String):
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            if cmd_type == 'run' or cmd_type == 'start':
                node.set_running(True)
            elif cmd_type == 'stop':
                node.set_running(False)
        except:
            pass
    
    # 제어 명령 구독 (노드 내부에서 처리)
    tracker_control_topic = node.get_parameter('tracker_control_topic').get_parameter_value().string_value
    control_subscription = node.create_subscription(
        String,
        tracker_control_topic,
        control_callback,
        10
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

