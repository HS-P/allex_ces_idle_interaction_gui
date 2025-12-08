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
from huggingface_hub import hf_hub_download
from typing import Optional, Tuple, List, Dict
from collections import namedtuple
from enum import Enum

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
        
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        pid_tune_topic = self.get_parameter('pid_tune_topic').get_parameter_value().string_value
        
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
        
        # 목 명령 Publisher
        self.neck_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_neck/joint_command',
            10
        )
        
        # 목 위치 Subscriber
        self.neck_position_subscription = self.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_neck/joint_positions_deg',
            self._neck_position_callback,
            10
        )
        
        # 허리 명령 Publisher
        self.waist_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_waist/joint_command',
            10
        )
        
        # 허리 위치 Subscriber
        self.waist_position_subscription = self.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_waist/joint_positions_deg',
            self._waist_position_callback,
            10
        )
        
        # YOLO 모델 초기화
        self._init_yolo_model()
        
        # 카메라 파라미터
        self.frame_width = 1280.0
        self.frame_height = 720.0
        
        # 각도 제한 범위 (라디안)
        self.yaw_min = math.radians(-80.0)
        self.yaw_max = math.radians(80.0)
        self.pitch_min = -0.0872665  # -5°
        self.pitch_max = 3.75246   # 215°
        
        # 현재 목 각도 (라디안)
        self.current_yaw_rad = 0.0
        self.current_pitch_rad = 0.0
        
        # 현재 허리 각도 (라디안)
        self.current_waist_yaw_rad = 0.0
        
        # 추적 상태
        self.state = TrackingState.IDLE
        self.target_track_id: Optional[int] = None
        self.lost_frames = 0
        self.max_lost_frames = 45
        
        # PID 제어 파라미터 (v1.3.0과 동일)
        self.kp_yaw = 1.55
        self.kp_pitch = 1.2
        self.ki_yaw = 0.2
        self.ki_pitch = 0.1
        self.kd_yaw = 0.02
        self.kd_pitch = 0.1
        
        # PID 제어 상태 변수
        self.integral_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_yaw = 0.0
        self.last_error_pitch = 0.0
        self.last_update_time = time.monotonic()
        
        # 숨쉬는 모션 변수
        self.breathing_start_time = time.monotonic()
        self.breathing_amplitude_deg = 5.0
        self.breathing_period_sec = 6.0
        
        # 실행 상태
        self.is_running = False
        
        # 프레임 처리 최적화: 최신 프레임만 처리
        self.last_frame_time = 0.0
        self.min_frame_interval = 1.0 / 30.0  # 최대 30 FPS로 제한
        
        self.get_logger().info("Testbed Gaze Tracking Node 초기화 완료")
    
    def _init_yolo_model(self):
        """YOLO 모델 초기화"""
        model_path = "yolo11n.pt"
        face_model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection",
            filename="model.pt"
        )
        
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
            self.get_logger().info(f"GPU 사용: {torch.cuda.get_device_name(0)}")
        else:
            self.get_logger().info("CPU 사용")
        
        self.yolo_model = YOLO(model_path)
        self.yolo_model.to(device)
        
        self.face_model = YOLO(face_model_path)
        self.face_model.to(device)
        
        self.get_logger().info("YOLO 모델 초기화 완료")
    
    def _neck_position_callback(self, msg: Float64MultiArray):
        """목 위치 콜백"""
        if len(msg.data) >= 2:
            pitch_deg = msg.data[0]
            yaw_deg = msg.data[1]
            self.current_pitch_rad = math.radians(pitch_deg)
            self.current_yaw_rad = math.radians(yaw_deg)
    
    def _waist_position_callback(self, msg: Float64MultiArray):
        """허리 위치 콜백"""
        if len(msg.data) >= 1:
            yaw_deg = msg.data[0]
            self.current_waist_yaw_rad = math.radians(yaw_deg)
    
    def _pid_tune_callback(self, msg: String):
        """PID 게인 튜닝 콜백"""
        try:
            data = json.loads(msg.data)
            if 'kp_yaw' in data:
                self.kp_yaw = float(data['kp_yaw'])
            if 'kp_pitch' in data:
                self.kp_pitch = float(data['kp_pitch'])
            if 'ki_yaw' in data:
                self.ki_yaw = float(data['ki_yaw'])
            if 'ki_pitch' in data:
                self.ki_pitch = float(data['ki_pitch'])
            if 'kd_yaw' in data:
                self.kd_yaw = float(data['kd_yaw'])
            if 'kd_pitch' in data:
                self.kd_pitch = float(data['kd_pitch'])
            
            # Integral 리셋
            if 'reset_integral' in data and data['reset_integral']:
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
            
            self.get_logger().info(
                f"PID 게인 업데이트: Kp=({self.kp_yaw:.2f}, {self.kp_pitch:.2f}), "
                f"Ki=({self.ki_yaw:.2f}, {self.ki_pitch:.2f}), "
                f"Kd=({self.kd_yaw:.2f}, {self.kd_pitch:.2f})"
            )
        except Exception as e:
            self.get_logger().error(f"PID 튜닝 파싱 실패: {e}")
    
    def is_facing_me(self, frame: np.ndarray, bbox: tuple) -> bool:
        """타겟이 나를 보고 있는지 확인 (얼굴 검출)"""
        if self.face_model is None:
            return False
        
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # 얼굴 영역 추출
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            return False
        
        # 얼굴 검출
        results = self.face_model(face_roi, verbose=False)
        return len(results[0].boxes) > 0
    
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
    
    def _pid_control(self, target_yaw_rad: float, target_pitch_rad: float) -> Tuple[float, float]:
        """PID 제어를 사용하여 목 증분 명령 계산"""
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))
        
        # 오차 계산
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_pitch = target_pitch_rad - self.current_pitch_rad
        
        # P 항
        p_yaw = self.kp_yaw * error_yaw
        p_pitch = self.kp_pitch * error_pitch
        
        # Integral 누적
        self.integral_yaw += error_yaw * dt
        self.integral_pitch += error_pitch * dt
        
        # Integral 제한
        max_integral = math.radians(60.0)
        self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))
        self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
        
        i_yaw = self.ki_yaw * self.integral_yaw
        i_pitch = self.ki_pitch * self.integral_pitch
        
        # D 항
        d_error_yaw = (error_yaw - self.last_error_yaw) / dt if dt > 0 else 0.0
        d_error_pitch = (error_pitch - self.last_error_pitch) / dt if dt > 0 else 0.0
        
        d_yaw = self.kd_yaw * d_error_yaw
        d_pitch = self.kd_pitch * d_error_pitch
        
        delta_yaw_rad = p_yaw + i_yaw + d_yaw
        delta_pitch_rad = p_pitch + i_pitch + d_pitch
        
        self.last_error_yaw = error_yaw
        self.last_error_pitch = error_pitch
        self.last_update_time = current_time
        
        return delta_yaw_rad, delta_pitch_rad
    
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float):
        """목 명령 전송 - PID 제어 후 절대각도로 전송"""
        target_yaw_rad, target_pitch_rad = self._clip_angles(target_yaw_rad, target_pitch_rad)
        
        # PID 제어로 증분 계산
        delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad)
        
        # 증분을 현재 위치에 더해서 절대각도로 변환
        absolute_yaw_rad = self.current_yaw_rad + delta_yaw_rad
        absolute_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        
        # 절대각도 제한 확인
        absolute_yaw_rad, absolute_pitch_rad = self._clip_angles(absolute_yaw_rad, absolute_pitch_rad)
        
        # 절대각도 명령으로 전송
        msg = Float64MultiArray()
        msg.data = [float(absolute_pitch_rad), float(absolute_yaw_rad)]
        self.neck_publisher.publish(msg)
        
        return absolute_yaw_rad, absolute_pitch_rad
    
    def _get_breathing_pitch(self) -> float:
        """숨쉬는 모션 Pitch 계산"""
        current_time = time.monotonic()
        elapsed_time = current_time - self.breathing_start_time
        breathing_pitch_rad = math.sin(2.0 * math.pi * elapsed_time / self.breathing_period_sec) * math.radians(self.breathing_amplitude_deg)
        return breathing_pitch_rad
    
    def _send_waist_command(self):
        """허리 명령 전송 - 현재 위치 유지하고 Pitch만 숨쉬는 모션"""
        waist_pitch_rad = self._get_breathing_pitch()
        msg = Float64MultiArray()
        msg.data = [float(self.current_waist_yaw_rad), float(waist_pitch_rad)]
        self.waist_publisher.publish(msg)
    
    def _process_tracking(self, detections: List[Dict], frame: np.ndarray) -> TargetInfo:
        """TRACKING 처리"""
        if self.state == TrackingState.IDLE:
            # 첫 번째 사람 선택
            if detections:
                self.target_track_id = detections[0]['track_id']
                self.state = TrackingState.TRACKING
                self.lost_frames = 0
                self.get_logger().info(f"타겟 선택: ID={self.target_track_id}")
        
        elif self.state == TrackingState.TRACKING:
            # 타겟 찾기
            target_det = next((det for det in detections if det['track_id'] == self.target_track_id), None)
            
            if target_det is None:
                self.lost_frames += 1
                if self.lost_frames >= self.max_lost_frames:
                    self.state = TrackingState.IDLE
                    self.target_track_id = None
                    self.get_logger().info("타겟 손실: IDLE로 전환")
                return TargetInfo(point=None, state=self.state, track_id=self.target_track_id)
            else:
                self.lost_frames = 0
                # 얼굴 중심점 계산
                x1, y1, x2, y2 = target_det['bbox']
                target_x = float((x1 + x2) / 2.0)
                target_y = float(y1 + (y2 - y1) * 0.2)  # 얼굴 위치 (상단 20%)
                
                return TargetInfo(
                    point=(target_x, target_y),
                    state=self.state,
                    track_id=self.target_track_id
                )
        
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
            
            # YOLO 인식 (최적화: imgsz, half precision 등)
            results = self.yolo_model.track(
                frame,
                persist=True,
                verbose=False,
                classes=[0],  # 사람만
                imgsz=640,    # 고정 이미지 크기로 속도 향상
                half=False    # FP16 사용 안 함 (호환성)
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        track_id = int(box.id.item()) if box.id is not None else None
                        if track_id is None:
                            continue
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # numpy 타입을 Python 기본 타입으로 변환
                        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                        conf = float(box.conf[0].cpu().numpy())
                        centroid = (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))
                        
                        detections.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2),
                            'centroid': centroid,
                            'confidence': conf
                        })
            
            # TRACKING 처리
            target_info = self._process_tracking(detections, frame)
            
            # 추적 결과 발행
            self._publish_tracking_result(detections, target_info)
            
            # TRACKING 상태일 때 PID 제어
            if target_info.state == TrackingState.TRACKING and target_info.point is not None:
                target_x, target_y = target_info.point
                
                # 픽셀을 각도로 변환
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y)
                
                # 상대 각도 제한
                max_relative_yaw_rad = math.radians(90.0)
                relative_yaw_rad = max(-max_relative_yaw_rad, min(max_relative_yaw_rad, relative_yaw_rad))
                
                # 목표 각도 계산
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # PID 제어
                self._send_neck_command(target_yaw_rad, target_pitch_rad)
            
            # 허리 명령 (항상)
            self._send_waist_command()
            
        except Exception as e:
            self.get_logger().error(f"이미지 처리 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = TestbedGazeTrackingNode()
    
    # RUN 명령 대기
    def start_callback(msg):
        try:
            data = json.loads(msg.data)
            if data.get('type') == 'run' or data.get('type') == 'start':
                node.is_running = True
                node.get_logger().info("RUN 시작")
        except:
            pass
    
    start_sub = node.create_subscription(String, '/allex_testbed/control', start_callback, 10)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

