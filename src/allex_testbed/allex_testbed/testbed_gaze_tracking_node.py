#!/usr/bin/env python3
"""
Testbed Gaze Tracking Node - YOLO 인식, 얼굴 추정, TRACKING, PID 제어 통합
"""
import json
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Float64MultiArray
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict
from collections import namedtuple
from enum import Enum
import os
from ament_index_python import get_package_share_directory

# 타입 정의
TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'bbox', 'centroid', 'state', 'confidence', 'age'
])

TargetInfo = namedtuple('TargetInfo', [
    'point',      # 타겟 중심점 (x, y) 또는 None
    'state',      # 현재 추적 상태
    'track_id',   # 타겟 track_id 또는 None
])

class TrackingState(Enum):
    """추적 상태"""
    IDLE = "idle"
    TRACKING = "tracking"
    LOST = "lost"
    SEARCHING = "searching"


class TestbedGazeTrackingNode(Node):
    """Testbed Gaze Tracking Node - YOLO, 얼굴 추정, TRACKING, PID 제어 통합"""
    
    def __init__(self):
        super().__init__('testbed_gaze_tracking_node')
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽 파라미터
        self.declare_parameter('camera_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('tracking_result_topic', '/allex_testbed/tracking_result')
        self.declare_parameter('pid_tune_topic', '/allex_testbed/pid_tune')
        self.declare_parameter('tracking_image_topic', '/allex_testbed/tracking_image')
        
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        pid_tune_topic = self.get_parameter('pid_tune_topic').get_parameter_value().string_value
        tracking_image_topic = self.get_parameter('tracking_image_topic').get_parameter_value().string_value
        
        # 이미지 구독
        self.image_subscription = self.create_subscription(
            CompressedImage,
            camera_image_topic,
            self.image_callback,
            qos_profile
        )
        
        # PID 게인 튜닝 구독
        self.pid_tune_subscription = self.create_subscription(
            String,
            pid_tune_topic,
            self._pid_tune_callback,
            10
        )
        
        # 추적 결과 발행
        self.tracking_result_publisher = self.create_publisher(
            String,
            tracking_result_topic,
            10
        )

        # Annotated 이미지 발행
        self.tracking_image_publisher = self.create_publisher(
            CompressedImage,
            tracking_image_topic,
            10
        )
        
        # YOLO 모델 초기화
        self._init_yolo_model()
        
        # 카메라 파라미터
        self.frame_width = 1280.0
        self.frame_height = 720.0
        
        # 추적 상태
        self.state = TrackingState.IDLE
        self.target_track_id: Optional[int] = None
        self.lost_frames = 0
        self.max_lost_frames = 120  # 약 4초 @30FPS (BotSort track_buffer와 동일하게)
        
        # 가림 대응: 마지막으로 본 타겟 위치 저장
        self.last_target_position: Optional[Tuple[float, float]] = None
        self.last_target_bbox: Optional[Tuple[float, float, float, float]] = None
        
        # 실행 상태
        self.is_running = False
        
        # 프레임 처리 최적화: 최신 프레임만 처리
        self.last_frame_time = 0.0
        self.min_frame_interval = 1.0 / 30.0  # 최대 30 FPS로 제한
        # 주의: 프레임 스킵이 많으면 추적이 불안정해질 수 있음
        
        self.get_logger().info("Testbed Gaze Tracking Node 초기화 완료")
    
    def _init_yolo_model(self):
        """YOLO 모델 초기화"""
        model_path = "yolo11n.pt"
        
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            self.get_logger().info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            self.get_logger().info("CPU 사용")
        
        self.yolo_model = YOLO(model_path)
        self.yolo_model.to(device)
        
        # BotSort 설정 파일 경로
        try:
            package_dir = get_package_share_directory('allex_ces_idle_interaction')
            tracker_config_path = os.path.join(package_dir, 'config', 'botsort.yaml')
            if not os.path.exists(tracker_config_path):
                raise FileNotFoundError(tracker_config_path)
            self.tracker_config_path = tracker_config_path
            self.get_logger().info(f"BotSort 설정 사용: {tracker_config_path}")
        except Exception as e:
            self.tracker_config_path = None
            self.get_logger().warn(f"BotSort 설정 파일을 찾을 수 없음: {e}")
        
        self.get_logger().info("YOLO 모델 초기화 완료")
    
    def _pixel_to_angle(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """타겟 픽셀 좌표를 목 각도로 변환"""
        center_x = self.frame_width / 2.0
        center_y = self.frame_height / 2.0
        
        offset_x = target_x - center_x
        offset_y = target_y - center_y
        
        horizontal_fov_deg = 120.0
        vertical_fov_deg = 45.0
        
        yaw_deg = (offset_x / self.frame_width) * horizontal_fov_deg
        pitch_deg = (offset_y / self.frame_height) * vertical_fov_deg
        
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        # 방향 정의에 맞게 변환
        yaw_rad = -yaw_rad  # Neck Yaw: 좌측 방향이 양수
        
        return yaw_rad, pitch_rad
    
    def _clip_angles(self, yaw_rad: float, pitch_rad: float) -> Tuple[float, float]:
        """각도를 제한 범위 내로 클리핑"""
        yaw_rad = max(self.yaw_min, min(self.yaw_max, yaw_rad))
        pitch_rad = max(self.pitch_min, min(self.pitch_max, pitch_rad))
        return yaw_rad, pitch_rad

    def _find_closest_person(self, detections: List[Dict], frame_shape: Tuple[int, int], 
                            exclude_track_id: Optional[int] = None) -> Optional[int]:
        """프레임 중심에 가장 가까운 사람 track_id 반환
        exclude_track_id: 제외할 track_id (가림 상황에서 가리는 사람 제외)"""
        if not detections:
            return None
        h, w = frame_shape
        cx = w / 2.0
        cy = h / 2.0
        min_dist = float('inf')
        closest_id = None
        for det in detections:
            # 제외할 track_id는 건너뛰기
            if exclude_track_id is not None and det['track_id'] == exclude_track_id:
                continue
            px, py = det['centroid']
            dist = math.hypot(px - cx, py - cy)
            if dist < min_dist:
                min_dist = dist
                closest_id = det['track_id']
        return closest_id
    
    def _calculate_iou(self, bbox1: Tuple[float, float, float, float], 
                       bbox2: Tuple[float, float, float, float]) -> float:
        """두 bbox 간의 IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 각 bbox의 면적
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _find_nearest_to_last_position(self, detections: List[Dict]) -> Optional[int]:
        """마지막 타겟 위치에 가장 가까운 사람 track_id 반환 (가림 대응)
        매우 엄격한 IoU 조건을 사용하여 가리는 사람을 선택하지 않도록 함"""
        if not detections or self.last_target_position is None or self.last_target_bbox is None:
            return None
        
        last_x, last_y = self.last_target_position
        last_bbox = self.last_target_bbox
        best_match_id = None
        best_iou = 0.0
        
        for det in detections:
            px, py = det['centroid']
            dist = math.hypot(px - last_x, py - last_y)
            
            # 마지막 위치에서 150픽셀 이내에 있는 경우만 고려 (더 엄격)
            if dist < 150:
                # IoU 계산하여 같은 사람인지 확인
                det_bbox = det['bbox']
                iou = self._calculate_iou(last_bbox, det_bbox)
                
                # 매우 엄격한 IoU 조건: 0.6 이상이어야 함 (가리는 사람은 IoU가 낮음)
                # 가림 상황에서도 원래 타겟의 일부는 보이므로 IoU가 높아야 함
                if iou > 0.6 and iou > best_iou:
                    best_iou = iou
                    best_match_id = det['track_id']
        
        # IoU > 0.6 조건을 만족하는 경우만 반환 (가리는 사람 제외)
        return best_match_id if best_iou > 0.6 else None
    
    def _process_tracking(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> TargetInfo:
        """IDLE → TRACKING → LOST → SEARCHING 간 단순 FSM"""
        # IDLE: 타겟 없으면 가장 가까운 사람 선택
        if self.state == TrackingState.IDLE:
            if detections:
                closest_id = self._find_closest_person(detections, frame_shape)
                if closest_id is not None:
                    self.target_track_id = closest_id
                    self.state = TrackingState.TRACKING
                    self.lost_frames = 0
                    self.get_logger().info(f"타겟 선택: ID={self.target_track_id}")
                else:
                    return TargetInfo(point=None, state=self.state, track_id=None)
            else:
                return TargetInfo(point=None, state=self.state, track_id=None)
        
        # TRACKING: 타겟 존재 여부 확인
        target_det = next((det for det in detections if det['track_id'] == self.target_track_id), None)
        
        if self.state == TrackingState.TRACKING:
            if target_det is None:
                self.lost_frames += 1
                # BotSort가 자동으로 재매칭하는 것을 완전히 신뢰
                # 수동 재매칭 로직 제거 - BotSort가 최적화된 설정으로 처리
                
                if self.lost_frames >= self.max_lost_frames:
                    self.state = TrackingState.SEARCHING
                    self.target_track_id = None
                    self.last_target_position = None
                    self.last_target_bbox = None
                    self.get_logger().info("타겟 손실: SEARCHING 전환")
                    return TargetInfo(point=None, state=self.state, track_id=None)
                else:
                    self.state = TrackingState.LOST
                    return TargetInfo(point=None, state=self.state, track_id=self.target_track_id)
            else:
                self.lost_frames = 0
                x1, y1, x2, y2 = target_det['bbox']
                target_x = float((x1 + x2) / 2.0)
                target_y = float(y1 + (y2 - y1) * 0.2)  # 얼굴 위치 (상단 20%)
                # 마지막 위치 업데이트
                self.last_target_position = (target_x, target_y)
                self.last_target_bbox = (x1, y1, x2, y2)
                return TargetInfo(point=(target_x, target_y), state=self.state, track_id=self.target_track_id)
        
        # LOST: 일정 프레임 내에 타겟 재발견 시 TRACKING 복귀, 아니면 SEARCHING
        # 중요: 타겟 ID를 유지하고 BotSort가 재매칭할 때까지 기다림
        if self.state == TrackingState.LOST:
            if target_det is not None:
                # BotSort가 같은 ID로 재매칭 성공
                self.state = TrackingState.TRACKING
                self.lost_frames = 0
                x1, y1, x2, y2 = target_det['bbox']
                target_x = float((x1 + x2) / 2.0)
                target_y = float(y1 + (y2 - y1) * 0.2)
                self.last_target_position = (target_x, target_y)
                self.last_target_bbox = (x1, y1, x2, y2)
                self.get_logger().info(f"LOST → TRACKING: ID={self.target_track_id} 재매칭 성공")
                return TargetInfo(point=(target_x, target_y), state=self.state, track_id=self.target_track_id)
            else:
                # BotSort가 자동으로 재매칭하는 것을 완전히 신뢰
                # 타겟 ID는 유지하고 기다림 (다른 사람으로 전환하지 않음)
                
                self.lost_frames += 1
                if self.lost_frames >= self.max_lost_frames:
                    # 4초 동안 재매칭 실패 시에만 SEARCHING 전환
                    self.state = TrackingState.SEARCHING
                    # SEARCHING 전환 시에도 마지막 위치는 유지 (같은 사람을 찾기 위해)
                    # target_track_id는 None으로 설정하되, last_target_position은 유지
                    old_target_id = self.target_track_id
                    self.target_track_id = None
                    self.get_logger().info(f"LOST 지속: SEARCHING 전환 (이전 ID={old_target_id}, 마지막 위치 유지)")
                return TargetInfo(point=None, state=self.state, track_id=self.target_track_id)
        
        # SEARCHING: 가장 가까운 사람을 다시 선택
        # 중요: 마지막 타겟 위치를 강력하게 고려하여 같은 위치 근처의 사람을 우선 선택
        if self.state == TrackingState.SEARCHING:
            if detections:
                # 마지막 타겟 위치가 있으면 그 근처의 사람을 강력하게 우선 선택
                if self.last_target_position is not None:
                    h, w = frame_shape
                    last_x, last_y = self.last_target_position
                    
                    # 마지막 타겟 위치에 가장 가까운 사람 선택 (가중치 적용)
                    best_id = None
                    min_weighted_dist = float('inf')
                    
                    for det in detections:
                        px, py = det['centroid']
                        # 마지막 타겟 위치와의 거리 (가중치 1.0)
                        dist_to_last = math.hypot(px - last_x, py - last_y)
                        # 프레임 중심과의 거리 (가중치 0.3 - 덜 중요)
                        cx, cy = w / 2.0, h / 2.0
                        dist_to_center = math.hypot(px - cx, py - cy)
                        # 가중 평균: 마지막 위치를 더 중요하게
                        weighted_dist = dist_to_last * 1.0 + dist_to_center * 0.3
                        
                        if weighted_dist < min_weighted_dist:
                            min_weighted_dist = weighted_dist
                            best_id = det['track_id']
                    
                    if best_id is not None:
                        self.target_track_id = best_id
                        self.state = TrackingState.TRACKING
                        self.lost_frames = 0
                        self.get_logger().info(f"SEARCHING → TRACKING: ID={self.target_track_id} (마지막 위치 우선, 거리={min_weighted_dist:.1f})")
                        target_det = next((det for det in detections if det['track_id'] == self.target_track_id), None)
                        if target_det is not None:
                            x1, y1, x2, y2 = target_det['bbox']
                            target_x = float((x1 + x2) / 2.0)
                            target_y = float(y1 + (y2 - y1) * 0.2)
                            self.last_target_position = (target_x, target_y)
                            self.last_target_bbox = (x1, y1, x2, y2)
                            return TargetInfo(point=(target_x, target_y), state=self.state, track_id=self.target_track_id)
                
                # 마지막 위치 정보가 없으면 기존 로직 사용
                exclude_id = self.target_track_id
                closest_id = self._find_closest_person(detections, frame_shape, exclude_track_id=exclude_id)
                if closest_id is not None:
                    self.target_track_id = closest_id
                    self.state = TrackingState.TRACKING
                    self.lost_frames = 0
                    self.get_logger().info(f"SEARCHING → TRACKING: ID={self.target_track_id}")
                    target_det = next((det for det in detections if det['track_id'] == self.target_track_id), None)
                    if target_det is not None:
                        x1, y1, x2, y2 = target_det['bbox']
                        target_x = float((x1 + x2) / 2.0)
                        target_y = float(y1 + (y2 - y1) * 0.2)
                        self.last_target_position = (target_x, target_y)
                        self.last_target_bbox = (x1, y1, x2, y2)
                        return TargetInfo(point=(target_x, target_y), state=self.state, track_id=self.target_track_id)
            return TargetInfo(point=None, state=self.state, track_id=None)
        
        return TargetInfo(point=None, state=self.state, track_id=None)
    
    def _publish_tracking_result(self, detections: List[Dict], target_info: TargetInfo):
        """추적 결과 발행"""
        objects_data = []
        for det in detections:
            # 모든 값을 Python 기본 타입으로 변환
            bbox = det['bbox']
            centroid = det['centroid']
            objects_data.append({
                'track_id': int(det['track_id']),
                'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                'centroid': [float(centroid[0]), float(centroid[1])],
                'confidence': float(det['confidence'])
            })
        
        # target_info.point도 Python 기본 타입으로 변환
        point = None
        if target_info.point is not None:
            point = [float(target_info.point[0]), float(target_info.point[1])]
        
        data = {
            'state': target_info.state.value,
            'target_info': {
                'track_id': int(target_info.track_id) if target_info.track_id is not None else None,
                'point': point,
                'state': target_info.state.value
            },
            'tracked_objects': objects_data,
            'timestamp': float(time.monotonic())
        }
        
        json_str = json.dumps(data, ensure_ascii=False)
        msg = String()
        msg.data = json_str
        self.tracking_result_publisher.publish(msg)

    def _publish_tracking_image(self, frame: np.ndarray, detections: List[Dict], target_track_id: Optional[int]):
        """Annotated 이미지 발행 (TARGET=빨강, 일반=초록)"""
        draw = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            track_id = det['track_id']
            conf = det['confidence']
            is_target = target_track_id is not None and track_id == target_track_id
            color = (0, 0, 255) if is_target else (0, 255, 0)
            thickness = 3 if is_target else 2
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{track_id} ({conf:.2f})"
            if is_target:
                label = f"TARGET {label}"
            cv2.putText(draw, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # 압축 발행
        success, enc = cv2.imencode('.jpg', draw, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = enc.tobytes()
            self.tracking_image_publisher.publish(msg)
    
    def image_callback(self, msg: CompressedImage):
        """이미지 콜백 - YOLO 인식, 얼굴 추정, TRACKING, PID 제어"""
        if not self.is_running:
            return
        
        # 프레임 스킵: 너무 빠르게 들어오는 프레임은 스킵
        current_time = time.monotonic()
        if current_time - self.last_frame_time < self.min_frame_interval:
            return
        self.last_frame_time = current_time
        
        try:
            # 이미지 디코딩
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                return
            
            h, w = frame.shape[:2]
            self.frame_width = float(w)
            self.frame_height = float(h)
            
            # YOLO + BotSort 추적
            # conf threshold를 낮춰서 가림 상황에서도 검출 확률 증가
            results = self.yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                classes=[0],  # 사람만
                imgsz=640,
                half=False,
                conf=0.2,  # 기본값보다 낮춰서 약한 검출도 포함 (가림 상황 대응)
                tracker=self.tracker_config_path if self.tracker_config_path else None
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        track_id = int(box.id.item()) if box.id is not None else None
                        if track_id is None:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        conf = float(box.conf[0].cpu().numpy())
                        centroid = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
                        detections.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2),
                            'centroid': centroid,
                            'confidence': conf
                        })
            
            # TRACKING 처리 (IDLE/TRACKING/LOST/SEARCHING)
            target_info = self._process_tracking(detections, (h, w))
            
            # 결과 발행
            self._publish_tracking_result(detections, target_info)
            self._publish_tracking_image(frame, detections, target_info.track_id)
            
        except Exception as e:
            self.get_logger().error(f"이미지 처리 실패: {e}")
    
    def _pid_tune_callback(self, msg: String):
        """PID 게인 튜닝 콜백 (이 노드는 PID 제어를 하지 않으므로 로그만 남김)"""
        try:
            data = json.loads(msg.data)
            self.get_logger().debug(f"PID 튜닝 명령 수신 (이 노드는 PID 제어를 하지 않음): {data}")
        except Exception as e:
            self.get_logger().warn(f"PID 튜닝 명령 파싱 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = TestbedGazeTrackingNode()
    
    # RUN/STOP 명령 처리
    def control_callback(msg):
        try:
            data = json.loads(msg.data)
            cmd_type = data.get('type', '').lower()
            if cmd_type == 'run' or cmd_type == 'start':
                node.is_running = True
                node.get_logger().info("RUN 시작 - 이미지 처리 활성화")
            elif cmd_type == 'stop':
                node.is_running = False
                node.get_logger().info("STOP - 이미지 처리 비활성화")
        except Exception as e:
            node.get_logger().warn(f"제어 명령 파싱 실패: {e}")
    
    control_sub = node.create_subscription(String, '/allex_testbed/control', control_callback, 10)
    node.get_logger().info("제어 토픽 구독 시작: /allex_testbed/control")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

