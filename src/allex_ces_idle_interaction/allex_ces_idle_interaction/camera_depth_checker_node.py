#!/usr/bin/env python3
"""
Camera Depth Checker Node
Compressed Depth 이미지를 받아서 YOLO로 사람 감지하고, 타겟에 대한 3D 포인트를 출력
"""
import time
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import String
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import threading
from collections import deque

PERSON_CLASS_ID = 0  # COCO 사람 클래스 ID


class CameraDepthCheckerNode(Node):
    """Camera Depth Checker Node - Depth + YOLO + 3D Point 추출"""
    
    def __init__(self):
        super().__init__('camera_depth_checker_node')
        
        # GPU 상태 확인
        self._check_gpu_status()
        
        # CvBridge 초기화 (Image 메시지 처리용)
        self.bridge = CvBridge()
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽명 파라미터 (압축되지 않은 Image 사용 - CompressedPublisher가 빈 데이터 보내는 문제)
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('color_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('output_topic', '/allex_camera/depth_points')
        
        depth_image_topic = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        color_image_topic = self.get_parameter('color_image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # 카메라 정보 구독 (캘리브레이션 파라미터)
        self.camera_info = None
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Depth 이미지 구독 (일반 Image 형식 - CompressedPublisher 문제로 raw 사용)
        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self.depth_image_callback,
            qos_profile,
        )
        
        # Color 이미지 구독 (YOLO용)
        self.color_image_subscription = self.create_subscription(
            CompressedImage,
            color_image_topic,
            self.color_image_callback,
            qos_profile,
        )
        
        # 결과 발행 (3D 포인트 + Detection 정보)
        self.output_publisher = self.create_publisher(
            String,
            output_topic,
            10
        )
        
        # 시각화 이미지 발행 (Detection bbox + ID + 거리 표시)
        self.declare_parameter('visualization_topic', '/allex_camera/depth_checker_visualization/compressed')
        visualization_topic = self.get_parameter('visualization_topic').get_parameter_value().string_value
        
        self.visualization_publisher = self.create_publisher(
            CompressedImage,
            visualization_topic,
            10
        )
        
        # 동기화를 위한 큐 (One-Q 처리)
        self.depth_queue = deque(maxlen=1)  # 최신 depth만 유지
        self.color_queue = deque(maxlen=1)  # 최신 color만 유지
        self.lock = threading.Lock()
        
        # YOLO 모델 초기화
        self._init_yolo_model()
        
        # 실행 상태 플래그 (StandAlone: 자동 실행)
        self.is_running = True
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("Camera Depth Checker Node 초기화 완료")
        self.get_logger().info("StandAlone 모드: 자동 실행 중...")
    
    def _check_gpu_status(self):
        """GPU 상태 확인 및 출력"""
        self.get_logger().info("=" * 60)
        self.get_logger().info("Camera Depth Checker Node - GPU 상태 확인")
        self.get_logger().info("=" * 60)
        
        self.get_logger().info(f"PyTorch 버전: {torch.__version__}")
        self.get_logger().info(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.get_logger().info(f"CUDA 버전: {torch.version.cuda}")
            self.get_logger().info(f"GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                self.get_logger().info(f"  GPU {i}: {props.name}")
        
        self.get_logger().info("=" * 60)
    
    def camera_info_callback(self, msg: CameraInfo):
        """카메라 정보 콜백 (캘리브레이션 파라미터 저장)"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info("카메라 정보 수신 완료")
            self.get_logger().info(f"  해상도: {msg.width}x{msg.height}")
            self.get_logger().info(f"  Focal Length: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}")
            self.get_logger().info(f"  Principal Point: cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}")
    
    def depth_image_callback(self, msg: Image):
        """Depth 이미지 콜백 (일반 Image 메시지 처리)
        StandAlone 모드: is_running 체크 없이 바로 처리
        """
        try:
            # CvBridge를 사용하여 depth 이미지 변환 (16UC1 형식 처리)
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # Depth 이미지 형식 확인 및 변환
            if depth_image.dtype == np.uint16:
                # 이미 uint16이면 그대로 사용
                pass
            elif depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
                # float 형식인 경우 (미터 단위) -> mm 단위 uint16으로 변환
                depth_image = (depth_image * 1000.0).astype(np.uint16)
            else:
                # 다른 형식인 경우 uint16으로 변환 시도
                self.get_logger().warn(f"예상하지 못한 depth 이미지 형식: {depth_image.dtype}, uint16으로 변환 시도")
                depth_image = depth_image.astype(np.uint16)
            
            # 2D 배열인지 확인 (3채널인 경우 첫 번째 채널만 사용)
            if len(depth_image.shape) == 3:
                depth_image = depth_image[:, :, 0]
            
            # 큐에 저장 (최신만 유지)
            with self.lock:
                self.depth_queue.append((depth_image, msg.header.stamp))
            
            # One-Q 처리: depth와 color가 모두 있으면 처리
            self._process_if_ready()
            
        except Exception as e:
            self.get_logger().error(f"Depth 이미지 처리 오류: {e}")
    
    def color_image_callback(self, msg: CompressedImage):
        """Color 이미지 콜백 (YOLO Detection용)
        StandAlone 모드: is_running 체크 없이 바로 처리
        """
        try:
            # 압축된 color 이미지 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if color_image is None:
                return
            
            # 큐에 저장 (최신만 유지)
            with self.lock:
                self.color_queue.append((color_image, msg.header.stamp))
            
            # One-Q 처리: depth와 color가 모두 있으면 처리
            self._process_if_ready()
            
        except Exception as e:
            self.get_logger().error(f"Color 이미지 처리 오류: {e}")
    
    def _process_if_ready(self):
        """Depth와 Color 이미지가 모두 준비되면 처리 (One-Q)"""
        with self.lock:
            if len(self.depth_queue) == 0 or len(self.color_queue) == 0:
                return
            
            depth_image, depth_stamp = self.depth_queue[0]
            color_image, color_stamp = self.color_queue[0]
        
        # 타임스탬프 차이가 너무 크면 스킵 (동기화 문제)
        time_diff = abs((depth_stamp.sec - color_stamp.sec) + 
                       (depth_stamp.nanosec - color_stamp.nanosec) * 1e-9)
        if time_diff > 0.1:  # 100ms 이상 차이나면 스킵
            return
        
        # 처리 시작
        frame_start = time.monotonic()
        self.frame_count += 1
        
        # YOLO Detection 수행
        detections = self._yolo_detect(color_image)
        
        # Depth에서 3D 포인트 추출
        points_3d = self._extract_3d_points(detections, depth_image, color_image)
        
        # 처리 시간 계산
        process_time = (time.monotonic() - frame_start) * 1000
        
        # 결과 발행 (시각화 이미지 포함)
        self._publish_results(detections, points_3d, process_time, color_image)
        
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
                f"Depth Checker: {len(detections)}개 객체 | "
                f"처리 시간: {process_time:.1f}ms | FPS: {fps:.1f}{gpu_mem_str}"
            )
            self.frame_count = 0
            self.last_log_time = current_time
    
    def _init_yolo_model(self):
        """YOLO 모델 초기화 (기존 모델 재사용)"""
        model_path = "yolo11n.pt"
        
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
        
        # GPU로 모델 이동
        if device != 'cpu':
            self.yolo_model.to(device)
        
        self.conf_threshold = 0.7
        
        # GPU 워밍업
        self.get_logger().info("GPU 워밍업 시작...")
        warmup_start = time.monotonic()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.yolo_model.track(
            dummy, 
            conf=self.conf_threshold, 
            classes=[PERSON_CLASS_ID], 
            verbose=False, 
            imgsz=640,
            persist=True,
        )
        warmup_time = (time.monotonic() - warmup_start) * 1000
        self.get_logger().info(f"워밍업 추론 시간: {warmup_time:.1f}ms")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**2
            self.get_logger().info(f"GPU 동기화 완료!! (메모리 사용: {allocated:.1f}MB)")
    
    def _yolo_detect(self, color_image):
        """YOLO로 사람 감지"""
        try:
            # YOLO Detection 수행
            results = self.yolo_model.track(
                color_image,
                conf=self.conf_threshold,
                classes=[PERSON_CLASS_ID],   # 사람만
                persist=True,
                verbose=False,
                imgsz=640,
            )
            
            # Detection 결과 파싱
            detections = []
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                if boxes is not None:
                    # track_id가 있는 경우와 없는 경우 모두 처리
                    if boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        ids = np.arange(len(boxes))
                    
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    
                    for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
                        centroid_x = (x1 + x2) / 2.0
                        centroid_y = (y1 + y2) / 2.0
                        
                        detections.append({
                            'track_id': int(tid),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'centroid': [float(centroid_x), float(centroid_y)]
                        })
            
            return detections
            
        except Exception as e:
            self.get_logger().error(f"YOLO Detection 오류: {e}")
            return []
    
    def _extract_3d_points(self, detections, depth_image, color_image):
        """Depth 이미지에서 3D 포인트 추출"""
        points_3d = []
        
        if self.camera_info is None:
            self.get_logger().warn("카메라 정보가 없어 3D 포인트 계산 불가")
            return points_3d
        
        # 카메라 내부 파라미터
        fx = self.camera_info.k[0]  # focal length x
        fy = self.camera_info.k[4]  # focal length y
        cx = self.camera_info.k[2]  # principal point x
        cy = self.camera_info.k[5]  # principal point y
        
        # Depth 이미지 크기 확인
        if depth_image.shape[:2] != color_image.shape[:2]:
            # 크기가 다르면 depth 이미지를 color 이미지 크기로 리사이즈
            depth_image = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
        
        # Depth 이미지 형식 확인 및 변환
        if len(depth_image.shape) == 3:
            # BGR로 인코딩된 경우 첫 번째 채널만 사용
            depth_image = depth_image[:, :, 0]
        
        # 각 Detection에 대해 3D 포인트 계산
        for det in detections:
            centroid_x, centroid_y = det['centroid']
            bbox = det['bbox']
            
            # 중심점 주변의 depth 값들을 샘플링 (노이즈 제거)
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, depth_image.shape[1] - 1))
            y1 = max(0, min(y1, depth_image.shape[0] - 1))
            x2 = max(0, min(x2, depth_image.shape[1] - 1))
            y2 = max(0, min(y2, depth_image.shape[0] - 1))
            
            # 중심점 주변 영역에서 depth 값 추출
            center_x = int(centroid_x)
            center_y = int(centroid_y)
            
            # 중심점 주변 작은 영역 (예: 5x5 픽셀)에서 depth 값 평균
            half_size = 2
            x_min = max(0, center_x - half_size)
            x_max = min(depth_image.shape[1], center_x + half_size + 1)
            y_min = max(0, center_y - half_size)
            y_max = min(depth_image.shape[0], center_y + half_size + 1)
            
            depth_roi = depth_image[y_min:y_max, x_min:x_max]
            valid_depths = depth_roi[depth_roi > 0]  # 0은 무효한 depth
            
            if len(valid_depths) == 0:
                # 유효한 depth가 없으면 스킵
                points_3d.append(None)
                continue
            
            # 중앙값 사용 (노이즈에 강함)
            depth_value = np.median(valid_depths)
            
            # Depth 단위 확인 (일반적으로 mm 단위, m로 변환 필요할 수 있음)
            # RealSense의 경우 보통 mm 단위이므로 m로 변환
            depth_m = depth_value / 1000.0  # mm -> m
            
            # 디버깅: depth 값이 비정상적으로 큰 경우 경고
            if depth_m > 10.0:  # 10m 이상이면 비정상
                self.get_logger().warn(
                    f"[Depth 디버깅] 비정상적인 depth 값: track_id={det.get('track_id', 'unknown')}, "
                    f"depth_value={depth_value:.1f}mm, depth_m={depth_m:.3f}m, "
                    f"centroid=({centroid_x:.1f}, {centroid_y:.1f}), "
                    f"bbox=({x1}, {y1}, {x2}, {y2})"
                )
            
            # 픽셀 좌표를 3D 좌표로 변환
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Z = depth
            u = centroid_x
            v = centroid_y
            
            x_3d = (u - cx) * depth_m / fx
            y_3d = (v - cy) * depth_m / fy
            z_3d = depth_m
            
            points_3d.append({
                'x': float(x_3d),
                'y': float(y_3d),
                'z': float(z_3d),
                'depth_mm': float(depth_value),
                'pixel': [float(u), float(v)]
            })
        
        return points_3d
    
    def _publish_results(self, detections, points_3d, process_time_ms, color_image):
        """결과 발행 (Detection + 3D Points + 시각화 이미지)"""
        try:
            # Detection과 3D 포인트를 결합
            results = []
            for det, point_3d in zip(detections, points_3d):
                result = {
                    'track_id': det['track_id'],
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'centroid': det['centroid'],
                    'point_3d': point_3d  # None이거나 {'x', 'y', 'z', 'depth_mm', 'pixel'}
                }
                results.append(result)
            
            data = {
                'detections_3d': results,
                'performance': {
                    'process_time_ms': float(process_time_ms) if process_time_ms else 0.0
                },
                'timestamp': time.monotonic()
            }
            
            # JSON 문자열로 변환하여 발행
            json_str = json.dumps(data, ensure_ascii=False)
            msg = String()
            msg.data = json_str
            self.output_publisher.publish(msg)
            
            # 디버깅: 발행된 depth 정보 로그 (주기적으로만)
            if not hasattr(self, '_last_publish_log_time'):
                self._last_publish_log_time = 0
            current_time = time.monotonic()
            if current_time - self._last_publish_log_time > 3.0:  # 3초마다 로그
                depth_summary = []
                for result in results:
                    track_id = result.get('track_id')
                    point_3d = result.get('point_3d')
                    if point_3d:
                        depth_m = point_3d.get('z', 0)
                        depth_summary.append(f"ID{track_id}:{depth_m:.3f}m")
                if depth_summary:
                    self.get_logger().info(f"[Depth 발행] {', '.join(depth_summary)}")
                self._last_publish_log_time = current_time
            
            # 시각화 이미지 생성 및 발행
            vis_image = self._draw_detections(color_image.copy(), results)
            self._publish_visualization(vis_image)
            
        except Exception as e:
            self.get_logger().error(f"결과 발행 실패: {e}")
    
    def _draw_detections(self, image, results):
        """Detection 결과를 이미지에 그리기 (bbox + ID + 거리)"""
        # 이미지 복사 (원본 유지)
        vis_image = image.copy()
        
        # 각 Detection에 대해 bbox 그리기
        for result in results:
            bbox = result['bbox']
            track_id = result['track_id']
            confidence = result['confidence']
            point_3d = result.get('point_3d')
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # bbox 그리기 (초록색)
            color = (0, 255, 0)  # BGR 형식
            thickness = 2
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            
            # ID와 거리 정보 텍스트
            if point_3d is not None:
                distance_m = point_3d['z']  # 미터 단위
                distance_mm = point_3d.get('depth_mm', distance_m * 1000.0)
                
                # 거리 텍스트 (미터 단위, 소수점 2자리)
                distance_text = f"ID:{track_id} | {distance_m:.2f}m ({distance_mm:.0f}mm)"
            else:
                distance_text = f"ID:{track_id} | N/A"
            
            # 텍스트 배경 (검은색 반투명)
            (text_width, text_height), baseline = cv2.getTextSize(
                distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 0, 0),
                -1  # 채워진 사각형
            )
            
            # 텍스트 그리기 (bbox 위에)
            cv2.putText(
                vis_image,
                distance_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),  # 초록색
                1,
                cv2.LINE_AA
            )
        
        # 우측 상단에 요약 정보 표시
        summary_texts = []
        if len(results) > 0:
            summary_texts.append(f"Objects: {len(results)}")
            for result in results:
                track_id = result['track_id']
                point_3d = result.get('point_3d')
                if point_3d is not None:
                    distance_m = point_3d['z']
                    summary_texts.append(f"ID {track_id}: {distance_m:.2f}m")
                else:
                    summary_texts.append(f"ID {track_id}: N/A")
        else:
            summary_texts.append("No detections")
        
        # 우측 상단에 텍스트 배치
        y_offset = 20
        for i, text in enumerate(summary_texts):
            (text_width, text_height), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # 배경 사각형
            x_start = vis_image.shape[1] - text_width - 10
            y_start = y_offset - text_height - 5
            cv2.rectangle(
                vis_image,
                (x_start - 5, y_start - 5),
                (vis_image.shape[1] - 5, y_offset + 5),
                (0, 0, 0),
                -1
            )
            
            # 텍스트 그리기
            cv2.putText(
                vis_image,
                text,
                (x_start, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),  # 노란색
                2,
                cv2.LINE_AA
            )
            y_offset += text_height + 10
        
        return vis_image
    
    def _publish_visualization(self, image):
        """시각화 이미지를 CompressedImage로 발행"""
        try:
            # 이미지를 JPEG로 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode('.jpg', image, encode_param)
            
            if not result:
                self.get_logger().warn("이미지 인코딩 실패")
                return
            
            # CompressedImage 메시지 생성
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = encimg.tobytes()
            
            self.visualization_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"시각화 이미지 발행 실패: {e}")
    
    def set_running(self, running: bool):
        """실행 상태 설정 (StandAlone 모드에서는 사용되지 않음)"""
        self.is_running = running
        if running:
            self.get_logger().info("Camera Depth Checker 시작")
        else:
            self.get_logger().info("Camera Depth Checker 중지")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = CameraDepthCheckerNode()
    
    # 제어 명령 구독
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
    
    # 제어 명령 구독
    control_subscription = node.create_subscription(
        String,
        '/allex_camera/tracker_control',
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

