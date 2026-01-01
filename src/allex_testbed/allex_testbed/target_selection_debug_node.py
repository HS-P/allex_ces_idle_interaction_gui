#!/usr/bin/env python3
"""
타겟 선택 디버깅 노드
이미지를 받아서 YOLO Detection + Tracking 결과를 시각화
Joystick으로 선택한 타겟은 붉은색, 나머지는 초록색 박스로 표시
"""
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
import os
import time

PERSON_CLASS_ID = 0  # COCO 사람 클래스 ID


class TargetSelectionDebugNode(Node):
    """타겟 선택 디버깅 노드 - YOLO Detection + 시각화"""
    
    def __init__(self):
        super().__init__('target_selection_debug_node')
        
        # CvBridge 초기화
        self.bridge = CvBridge()
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽명 파라미터
        self.declare_parameter('camera_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('tracking_result_topic', '/allex_camera/tracking_result')
        self.declare_parameter('manual_control_topic', '/allex_camera/manual_control')
        self.declare_parameter('output_image_topic', '/allex_camera/debug_image/compressed')
        
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        manual_control_topic = self.get_parameter('manual_control_topic').get_parameter_value().string_value
        output_image_topic = self.get_parameter('output_image_topic').get_parameter_value().string_value
        
        # 입력 이미지 구독
        self.image_subscription = self.create_subscription(
            CompressedImage,
            camera_image_topic,
            self.image_callback,
            qos_profile,
        )
        
        # Manual 제어 명령 구독 (타겟 ID 변경용)
        self.manual_control_subscription = self.create_subscription(
            String,
            manual_control_topic,
            self.manual_control_callback,
            10
        )
        
        # 출력 이미지 발행
        self.output_image_publisher = self.create_publisher(
            CompressedImage,
            output_image_topic,
            10
        )
        
        # Tracking 결과 발행 (YOLO Detection 결과를 tracking_result로 발행)
        self.tracking_result_publisher = self.create_publisher(
            String,
            tracking_result_topic,
            10
        )
        
        # YOLO 모델 초기화
        self._init_yolo_model()
        
        # 현재 상태 저장
        self.current_image = None
        self.current_target_id = None  # Joystick으로 선택된 타겟 ID
        self.tracked_objects = []  # [{'track_id': int, 'bbox': [x1, y1, x2, y2], 'centroid': (x, y)}, ...]
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Target Selection Debug Node 시작")
        self.get_logger().info(f"입력 이미지: {camera_image_topic}")
        self.get_logger().info(f"Tracking 결과 발행: {tracking_result_topic}")
        self.get_logger().info(f"Manual 제어 구독: {manual_control_topic}")
        self.get_logger().info(f"출력 이미지: {output_image_topic}")
        self.get_logger().info("즉시 YOLO Detection 시작 (GAZE 명령 없음, 시각화만)")
        self.get_logger().info("=" * 60)
    
    def _init_yolo_model(self):
        """YOLO 모델 초기화"""
        model_path = "yolo11n.pt"
        
        # GPU 디바이스 설정
        device = 'cpu'
        if torch.cuda.is_available():
            try:
                device = 0
                self.get_logger().info(f"✓ GPU 사용: {torch.cuda.get_device_name(0)}")
            except RuntimeError as e:
                self.get_logger().warn(f"⚠️  GPU 사용 불가: {e}, CPU 사용")
        else:
            self.get_logger().info("CPU 사용")
        
        # YOLO 모델 초기화
        self.yolo_model = YOLO(model_path)
        if device != 'cpu':
            self.yolo_model.to(device)
        
        self.conf_threshold = 0.7
        
        # BotSort 설정 파일 경로
        from ament_index_python import get_package_share_directory
        package_dir = get_package_share_directory('allex_ces_idle_interaction')
        self.tracker_config_path = os.path.join(package_dir, 'config', 'botsort.yaml')
        if not os.path.exists(self.tracker_config_path):
            self.get_logger().warn(f"BotSort 설정 파일을 찾을 수 없음: {self.tracker_config_path}")
            self.tracker_config_path = None
        
        self.get_logger().info("YOLO 모델 초기화 완료")
    
    def image_callback(self, msg: CompressedImage):
        """이미지 수신 콜백 - YOLO Detection 수행"""
        try:
            # CompressedImage를 OpenCV 이미지로 변환
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return
            
            self.current_image = frame
            
            # YOLO Detection 수행
            self._perform_yolo_detection(frame)
            
            # Tracking 결과 발행 (Joystick Node가 subscribe)
            self._publish_tracking_result()
            
            # 시각화 업데이트
            self._update_visualization()
        except Exception as e:
            self.get_logger().error(f"이미지 처리 오류: {e}")
    
    def _perform_yolo_detection(self, frame):
        """YOLO Detection 수행"""
        try:
            # YOLO Detection 수행 (track 사용하여 track_id 생성)
            results = self.yolo_model.track(
                frame,
                conf=self.conf_threshold,
                classes=[PERSON_CLASS_ID],   # 사람만
                persist=True,
                tracker=self.tracker_config_path if self.tracker_config_path else 'botsort.yaml',
                verbose=False,
                imgsz=640,
            )
            
            # Detection 결과 파싱
            self.tracked_objects = []
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    ids = boxes.id.cpu().numpy().astype(int)
                    xyxy = boxes.xyxy.cpu().numpy()
                    
                    for tid, (x1, y1, x2, y2) in zip(ids, xyxy):
                        self.tracked_objects.append({
                            'track_id': int(tid),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'centroid': [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)]
                        })
        except Exception as e:
            self.get_logger().error(f"YOLO Detection 오류: {e}")
    
    def _publish_tracking_result(self):
        """Tracking 결과 발행 (Joystick Node가 subscribe)"""
        try:
            # tracking_result 메시지 생성
            # 정렬하지 않고 YOLO Detection 결과 순서 그대로 발행
            # Joystick Node에서 x 좌표 기준으로 정렬함
            tracking_data = {
                'state': 'tracking',  # 디버그 노드는 항상 tracking 상태
                'target_track_id': self.current_target_id,  # 현재 선택된 타겟 ID (None일 수 있음)
                'tracked_objects': []
            }
            
            # tracked_objects 형식으로 변환 (정렬하지 않음, Joystick Node에서 정렬)
            for obj in self.tracked_objects:
                tracking_data['tracked_objects'].append({
                    'track_id': obj['track_id'],
                    'bbox': obj['bbox'],
                    'centroid': list(obj['centroid'])
                })
            
            # 메시지 발행
            msg = String()
            msg.data = json.dumps(tracking_data, ensure_ascii=False)
            self.tracking_result_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Tracking 결과 발행 오류: {e}")
    
    def manual_control_callback(self, msg: String):
        """Manual 제어 명령 콜백 (타겟 ID 변경)"""
        try:
            data = json.loads(msg.data)
            cmd_type = data.get('type', '')
            
            if cmd_type == 'set_target':
                # 타겟 ID 변경
                new_target_id = data.get('target_id', None)
                if new_target_id is not None:
                    self.current_target_id = int(new_target_id)
                    self.get_logger().info(f"[DEBUG NODE] 타겟 ID 변경: {self.current_target_id}")
                    # 이미지가 있으면 시각화 업데이트
                    if self.current_image is not None:
                        self._update_visualization()
            elif cmd_type == 'set_state':
                # 상태 변경 (타겟 ID 포함)
                new_target_id = data.get('target_id', None)
                if new_target_id is not None:
                    self.current_target_id = int(new_target_id)
                    self.get_logger().info(f"[DEBUG NODE] 상태 변경 + 타겟 ID: {self.current_target_id}")
                    # 이미지가 있으면 시각화 업데이트
                    if self.current_image is not None:
                        self._update_visualization()
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"Manual 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().warn(f"Manual 제어 명령 처리 실패: {e}")
    
    def _update_visualization(self):
        """이미지에 박스와 번호 그리기"""
        if self.current_image is None:
            return
        
        # 이미지 복사 (원본 유지)
        img = self.current_image.copy()
        
        # x 좌표 기준으로 정렬 (왼쪽부터)
        sorted_objects = sorted(self.tracked_objects, key=lambda obj: obj['centroid'][0])
        
        # 각 객체에 대해 박스 그리기
        for i, obj in enumerate(sorted_objects):
            track_id = obj['track_id']
            bbox = obj['bbox']
            
            # 현재 타겟이면 붉은색, 아니면 초록색
            if track_id == self.current_target_id:
                color = (0, 0, 255)  # BGR: 빨간색
                thickness = 3
            else:
                color = (0, 255, 0)  # BGR: 초록색
                thickness = 2
            
            # 박스 그리기
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # 번호 표시 (x 좌표 순서대로 0, 1, 2, ...)
            label = f"#{i}: ID{track_id}"
            
            # 텍스트 배경
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                img,
                (int(x1), int(y1) - text_height - 10),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # 텍스트
            cv2.putText(
                img,
                label,
                (int(x1), int(y1) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),  # 흰색
                2,
                cv2.LINE_AA
            )
            
            # 중심점 표시
            centroid = obj['centroid']
            cv2.circle(img, (int(centroid[0]), int(centroid[1])), 5, color, -1)
        
        # 현재 타겟 정보 표시 (화면 상단)
        if self.current_target_id is not None:
            info_text = f"Current Target: ID {self.current_target_id}"
            cv2.putText(
                img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),  # 빨간색
                2,
                cv2.LINE_AA
            )
        else:
            info_text = "No Target Selected"
            cv2.putText(
                img,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (128, 128, 128),  # 회색
                2,
                cv2.LINE_AA
            )
        
        # 결과 이미지 발행
        self._publish_image(img)
    
    def _publish_image(self, img: np.ndarray):
        """OpenCV 이미지를 CompressedImage로 변환하여 발행"""
        try:
            # OpenCV 이미지를 JPEG로 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            
            if not result:
                self.get_logger().error("이미지 인코딩 실패")
                return
            
            # CompressedImage 메시지 생성
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = encimg.tobytes()
            
            # 발행
            self.output_image_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"이미지 발행 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = TargetSelectionDebugNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
