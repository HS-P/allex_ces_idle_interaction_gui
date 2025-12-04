#!/usr/bin/env python3
"""
Tracking FSM Node - 모든 로직과 통신을 한 파일에 통합
BB Box와 ID를 받아서 FSM 처리
"""
import time
import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
from typing import List, Optional, Dict
from collections import namedtuple
from enum import Enum
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

cv2.setNumThreads(0)  # OpenCV의 멀티스레딩 비활성화

# 다른 파일에서 사용하는 타입들 export
TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'bbox', 'centroid', 'state', 'confidence', 'age'
])

TargetInfo = namedtuple('TargetInfo', [
    'point',      # 타겟 중심점 (x, y) 또는 None
    'state',      # 현재 추적 상태 (TrackingState)
    'track_id',   # 타겟 track_id 또는 None
])

class TrackingState(Enum):
    """추적 상태"""
    IDLE = "idle"           # 초기 대상 선택
    TRACKING = "tracking"   # 추적 중
    LOST = "lost"          # 추적 대상 놓침 (잠시 대기)
    SEARCHING = "searching" # 주변 두리번대기 (대상 선택)
    WAIST_FOLLOWER = "waist_follower"   # 허리 따라가기 (0도 유지)
    HELLO = "hello"        # 인사 제스처 (손 흔들기)
    INTERACTION = "interaction" # 인터렉션


class TrackingFSMNode(Node):
    """Tracking FSM Node - 모든 로직과 통신 통합"""
    
    def __init__(self):
        super().__init__('tracking_fsm_node')
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽명 파라미터 (Launch 파일에서 설정 가능)
        self.declare_parameter('detections_topic', '/allex_camera/detections')
        self.declare_parameter('camera_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('tracking_result_topic', '/allex_camera/tracking_result')
        self.declare_parameter('tracker_control_topic', '/allex_camera/tracker_control')
        self.declare_parameter('tracker_state_request_topic', '/allex_camera/tracker_state_request')
        self.declare_parameter('neck_angle_topic', '/allex_camera/neck_angle')
        
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        tracker_control_topic = self.get_parameter('tracker_control_topic').get_parameter_value().string_value
        tracker_state_request_topic = self.get_parameter('tracker_state_request_topic').get_parameter_value().string_value
        neck_angle_topic = self.get_parameter('neck_angle_topic').get_parameter_value().string_value
        
        # Detection 결과 구독 (YOLO Detection Node에서 발행)
        self.detection_subscription = self.create_subscription(
            String,
            detections_topic,
            self.detection_callback,
            qos_profile,
        )
        
        # 원본 이미지 구독 (얼굴 검출용)
        self.image_subscription = self.create_subscription(
            CompressedImage,
            camera_image_topic,
            self.image_callback,
            qos_profile,
        )
        
        # 제어 명령 구독 (GUI에서 오는 명령)
        self.control_subscription = self.create_subscription(
            String,
            tracker_control_topic,
            self._control_callback,
            10
        )
        
        # 상태 변경 요청 구독 (Controller 노드에서 발행)
        self.state_request_subscription = self.create_subscription(
            String,
            tracker_state_request_topic,
            self._state_request_callback,
            10
        )
        
        # 추적 결과 발행
        self.tracking_result_publisher = self.create_publisher(
            String,
            tracking_result_topic,
            10
        )
        
        # 목 각도 구독 (Controller 노드에서 발행)
        self.neck_angle_subscription = self.create_subscription(
            String,
            neck_angle_topic,
            self.neck_angle_callback,
            10
        )
        
        # 추적 상태 관리
        self.state = TrackingState.IDLE
        self.target_track_id: Optional[int] = None  # 추적 대상 ID
        self.lost_frames = 0  # 놓친 프레임 수
        self.max_lost_frames = 45  # 최대 놓친 프레임 수 (약 1.5초, 30FPS 기준)
        
        # Manual 모드 지원
        self.manual_mode = False  # True면 상태 자동 전이 비활성화
        
        # Interaction 모드 지원 (True: 타겟 자동 선택 활성화, False: IDLE 모드)
        self.interaction_mode = False
        
        # 타겟이 명시적으로 설정되었는지 표시 (상태 머신이 덮어쓰지 않도록)
        self.target_explicitly_set = False
        
        # 타겟 후보가 되기 위한 최소 지속 시간 (초)
        self.min_target_duration = 1.4
        # 각 track_id의 첫 등장 시간 추적
        self.track_id_first_seen: Dict[int, float] = {}
        
        # WAIST_FOLLOWER 전이를 위한 변수들 (목 각도 기반)
        self.neck_stable_start_time: Optional[float] = None  # 목 각도가 안정되기 시작한 시간
        self.neck_stable_duration = 5.0  # 목 각도가 안정되어야 하는 최소 시간 (초)
        self.neck_stable_threshold_deg = 3.0  # 목 각도 안정성 임계값 (도)
        self.last_neck_yaw_rad: Optional[float] = None  # 이전 목 각도 (라디안)
        self.neck_stable_reference_yaw_rad: Optional[float] = None  # 안정성 기준 목 각도 (라디안)
        self.pending_face_check: bool = False  # 얼굴 검출 대기 플래그
        
        # 얼굴 검출용 모델 (필요시)
        try:
            face_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
            self.face_model = YOLO(face_model_path)
        except:
            self.face_model = None
        
        # 목 각도 저장 (안정성 추적용)
        self.current_neck_yaw_rad = None
        
        # 최신 프레임 저장 (얼굴 검출용)
        self.latest_frame = None
        self.latest_frame_shape = None
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("Tracking FSM Node 초기화 완료")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    def _find_closest_person(self, detections: List[Dict], frame_shape: tuple, current_time: float) -> Optional[int]:
        """프레임 중심에 가장 가까운 사람 찾기 (최소 지속 시간 이상인 객체만 후보)"""
        if not detections:
            return None
        
        # 현재 프레임에 나타난 track_id 업데이트
        current_frame_ids = set()
        for det in detections:
            track_id = det['track_id']
            current_frame_ids.add(track_id)
            # 처음 보는 track_id면 등장 시간 기록
            if track_id not in self.track_id_first_seen:
                self.track_id_first_seen[track_id] = current_time
        
        # 사라진 track_id 제거 (메모리 관리)
        disappeared_ids = set(self.track_id_first_seen.keys()) - current_frame_ids
        for track_id in disappeared_ids:
            del self.track_id_first_seen[track_id]
        
        # 최소 지속 시간 이상인 객체만 필터링
        valid_detections = []
        for det in detections:
            track_id = det['track_id']
            if track_id in self.track_id_first_seen:
                duration = current_time - self.track_id_first_seen[track_id]
                if duration >= self.min_target_duration:
                    valid_detections.append(det)
        
        if not valid_detections:
            return None
        
        # 유효한 객체 중에서 가장 가까운 사람 찾기
        frame_center_y, frame_center_x = frame_shape[0] / 2, frame_shape[1] / 2
        
        min_distance = float('inf')
        closest_id = None
        
        for det in valid_detections:
            cx, cy = det['centroid']
            # 중심점까지의 거리 계산
            distance = np.sqrt((cx - frame_center_x)**2 + (cy - frame_center_y)**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_id = det['track_id']
        
        return closest_id
    
    def set_manual_mode(self, enabled: bool) -> None:
        """Manual 모드 설정"""
        self.manual_mode = enabled
    
    def set_interaction_mode(self, enabled: bool) -> None:
        """Interaction 모드 설정"""
        self.interaction_mode = enabled
        self.reset_timers()
        
        if enabled:
            self.state = TrackingState.INTERACTION
            self.target_track_id = None
            self.target_explicitly_set = False
        else:
            self.state = TrackingState.IDLE
            self.target_track_id = None
            self.target_explicitly_set = False
    
    def reset_timers(self) -> None:
        """모든 타이머 및 안정성 관련 변수 초기화"""
        self.neck_stable_start_time = None
        self.last_neck_yaw_rad = None
        self.neck_stable_reference_yaw_rad = None
        self.lost_frames = 0
        self.pending_face_check = False
    
    def set_state(self, state: TrackingState, target_track_id: Optional[int] = None) -> None:
        """Manual 모드에서 상태를 수동으로 설정"""
        if state == TrackingState.IDLE:
            self.reset_timers()
        
        if not self.manual_mode:
            return
        
        self.state = state
        if target_track_id is not None:
            self.target_track_id = int(target_track_id)
            self.target_explicitly_set = True
        elif state != TrackingState.TRACKING:
            self.target_track_id = None
            self.target_explicitly_set = False
        self.lost_frames = 0
    
    def set_target(self, target_track_id: int) -> None:
        """타겟 변경"""
        self.target_track_id = int(target_track_id)
        self.state = TrackingState.TRACKING
        self.lost_frames = 0
        self.target_explicitly_set = True
    
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
        
        crop = frame[y1:y2, x1:x2]
        results = self.face_model.predict(crop, conf=0.5, verbose=False)
        
        if results and len(results[0].boxes) > 0:
            return True
        return False
    
    def update_neck_angle(self, current_neck_yaw_rad: float) -> None:
        """목 각도를 업데이트하고 안정성을 확인"""
        current_time = time.monotonic()
        
        if self.state != TrackingState.TRACKING:
            self.neck_stable_start_time = None
            self.last_neck_yaw_rad = None
            self.neck_stable_reference_yaw_rad = None
            return
        
        if self.last_neck_yaw_rad is None:
            self.last_neck_yaw_rad = current_neck_yaw_rad
            self.neck_stable_start_time = None
            self.neck_stable_reference_yaw_rad = None
            return
        
        angle_change_deg = abs(math.degrees(current_neck_yaw_rad - self.last_neck_yaw_rad))
        
        if self.neck_stable_reference_yaw_rad is None:
            self.neck_stable_reference_yaw_rad = current_neck_yaw_rad
        
        reference_change_deg = abs(math.degrees(current_neck_yaw_rad - self.neck_stable_reference_yaw_rad))
        
        if reference_change_deg <= self.neck_stable_threshold_deg and angle_change_deg <= self.neck_stable_threshold_deg:
            if self.neck_stable_start_time is None:
                self.neck_stable_start_time = current_time
                self.neck_stable_reference_yaw_rad = current_neck_yaw_rad
            
            elapsed_time = current_time - self.neck_stable_start_time
            if elapsed_time >= self.neck_stable_duration:
                self.pending_face_check = True
        else:
            self.neck_stable_start_time = None
            self.neck_stable_reference_yaw_rad = None
        
        self.last_neck_yaw_rad = current_neck_yaw_rad
    
    def _process_fsm(self, detections: List[Dict], frame_shape: tuple, frame: Optional[np.ndarray] = None) -> tuple[List[TrackedObject], TargetInfo]:
        """Detection 결과를 받아서 FSM 처리"""
        current_time = time.monotonic()
        
        # 타겟이 설정되어 있으면 현재 프레임에 존재하는지 확인
        target_exists = (
            self.target_track_id is not None and
            any(det['track_id'] == self.target_track_id for det in detections)
        )
        
        # 상태 머신 처리
        if not detections:
            # 감지된 객체가 없으면 상태 업데이트
            if not self.manual_mode:
                match self.state:
                    case TrackingState.TRACKING:
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                    case TrackingState.LOST:
                        self.lost_frames += 1
                        if self.lost_frames >= self.max_lost_frames:
                            self.state = TrackingState.SEARCHING
                            if not self.target_explicitly_set:
                                self.target_track_id = None
                    case _:
                        pass
            
            target_info = TargetInfo(
                point=None,
                state=self.state,
                track_id=self.target_track_id
            )
            return [], target_info
        
        # 타겟이 존재하면 상태 업데이트
        if target_exists and self.state not in (TrackingState.WAIST_FOLLOWER, TrackingState.INTERACTION):
            if self.state not in (TrackingState.TRACKING, TrackingState.WAIST_FOLLOWER, TrackingState.INTERACTION):
                self.state = TrackingState.TRACKING
            self.lost_frames = 0
        
        # 상태 머신 처리 (Manual 모드가 아닐 때만 자동 전이)
        if not self.manual_mode:
            match self.state:
                case TrackingState.IDLE:
                    if self.target_explicitly_set and target_exists:
                        self.state = TrackingState.TRACKING
                        self.lost_frames = 0
                    elif not self.manual_mode:
                        if self.target_track_id is None:
                            closest_id = self._find_closest_person(detections, frame_shape, current_time)
                            if closest_id is not None:
                                self.target_track_id = closest_id
                                self.state = TrackingState.TRACKING
                                self.lost_frames = 0
                                self.target_explicitly_set = False
                        elif target_exists:
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                
                case TrackingState.INTERACTION:
                    if self.target_track_id is None or not target_exists:
                        closest_id = self._find_closest_person(detections, frame_shape, current_time)
                        if closest_id is not None:
                            self.target_track_id = closest_id
                            self.target_explicitly_set = False
                
                case TrackingState.TRACKING:
                    if self.pending_face_check and frame is not None:
                        target_det = next((det for det in detections if det['track_id'] == self.target_track_id), None)
                        if target_det is not None:
                            if self.is_facing_me(frame, target_det['bbox']):
                                self.state = TrackingState.WAIST_FOLLOWER
                                self.neck_stable_start_time = None
                                self.last_neck_yaw_rad = None
                                self.neck_stable_reference_yaw_rad = None
                            else:
                                self.neck_stable_start_time = None
                                self.last_neck_yaw_rad = None
                                self.neck_stable_reference_yaw_rad = None
                        self.pending_face_check = False
                    
                    if not target_exists and self.target_track_id is not None and not self.target_explicitly_set:
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                        self.neck_stable_start_time = None
                        self.last_neck_yaw_rad = None
                        self.neck_stable_reference_yaw_rad = None
                        self.pending_face_check = False
                
                case TrackingState.LOST:
                    if target_exists:
                        self.state = TrackingState.TRACKING
                        self.lost_frames = 0
                    else:
                        self.lost_frames += 1
                        if self.lost_frames >= self.max_lost_frames:
                            if self.target_track_id is None or not self.target_explicitly_set:
                                if self.target_track_id is not None:
                                    self.target_track_id = None
                                    self.target_explicitly_set = False
                                self.state = TrackingState.SEARCHING
                
                case TrackingState.SEARCHING:
                    if detections:
                        closest_id = self._find_closest_person(detections, frame_shape, current_time)
                        if closest_id is not None:
                            self.target_track_id = closest_id
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                            self.target_explicitly_set = False
                
                case TrackingState.WAIST_FOLLOWER:
                    pass
        
        # 추적 객체 생성
        tracked_objects: List[TrackedObject] = []
        target_point = None
        target_track_id = None
        
        for det in detections:
            if det['track_id'] == self.target_track_id:
                tracked_objects.append(
                    TrackedObject(
                        track_id=det['track_id'],
                        bbox=tuple(det['bbox']),
                        centroid=tuple(det['centroid']),
                        state="target",
                        confidence=det['confidence'],
                        age=0,
                    )
                )
                # 타겟 정보 저장 - 바운딩 박스 높이의 0.2 지점 (머리 쪽)
                x1, y1, x2, y2 = det['bbox']
                target_point = ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.2)
                target_track_id = det['track_id']
            else:
                tracked_objects.append(
                    TrackedObject(
                        track_id=det['track_id'],
                        bbox=tuple(det['bbox']),
                        centroid=tuple(det['centroid']),
                        state=self.state.value,
                        confidence=det['confidence'],
                        age=0,
                    )
                )
        
        # 타겟 정보 생성
        target_info = TargetInfo(
            point=target_point,
            state=self.state,
            track_id=target_track_id if target_track_id is not None else self.target_track_id
        )
        
        return tracked_objects, target_info
    
    def image_callback(self, msg: CompressedImage) -> None:
        """이미지 콜백 - 얼굴 검출용으로 저장"""
        if not self.is_running:
            return
        
        # 압축된 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            self.latest_frame = frame
            self.latest_frame_shape = frame.shape
    
    def detection_callback(self, msg: String) -> None:
        """Detection 결과 콜백 - FSM 처리"""
        if not self.is_running:
            return
        
        self.frame_count += 1
        frame_start = time.monotonic()
        
        try:
            # Detection 결과 파싱
            data = json.loads(msg.data)
            detections = data.get('detections', [])
            
            # 프레임 크기 가져오기
            frame_shape = self.latest_frame_shape if self.latest_frame_shape else (720, 1280)
            
            # FSM 처리
            tracked_objects, target_info = self._process_fsm(
                detections,
                frame_shape,
                self.latest_frame
            )
            
            # 목 각도 안정성 추적 (TRACKING 상태일 때만)
            if target_info.state == TrackingState.TRACKING and self.current_neck_yaw_rad is not None:
                self.update_neck_angle(self.current_neck_yaw_rad)
            
            # 처리 시간 계산
            process_time = (time.monotonic() - frame_start) * 1000
            
            # 추적 결과 발행
            self._publish_tracking_result(tracked_objects, target_info, process_time_ms=process_time)
            
            # 주기적 성능 로그 (5초마다)
            current_time = time.monotonic()
            if current_time - self.last_log_time > 5.0:
                elapsed = current_time - self.last_log_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                
                self.get_logger().info(
                    f"FSM 처리 중: {len(tracked_objects)}개 객체 | "
                    f"처리 시간: {process_time:.1f}ms | FPS: {fps:.1f}"
                )
                self.frame_count = 0
                self.last_log_time = current_time
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Detection 결과 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"FSM 처리 실패: {e}")
    
    def _publish_tracking_result(self, tracked_objects, target_info, process_time_ms=None):
        """추적 결과를 Topic으로 발행"""
        try:
            # 상태 정보 추출
            state_str = target_info.state.value if isinstance(target_info.state, TrackingState) else str(target_info.state)
            
            # 추적 객체 정보
            objects_data = []
            for obj in tracked_objects:
                objects_data.append({
                    'track_id': obj.track_id,
                    'bbox': list(obj.bbox),
                    'centroid': list(obj.centroid),
                    'state': obj.state,
                    'confidence': obj.confidence,
                    'age': obj.age
                })
            
            # JSON 데이터 구성
            data = {
                'state': state_str,
                'target_info': {
                    'track_id': target_info.track_id,
                    'point': list(target_info.point) if target_info.point else None,
                    'state': state_str
                },
                'tracked_objects': objects_data,
                'performance': {
                    'process_time_ms': float(process_time_ms) if process_time_ms else 0.0
                },
                'timestamp': time.monotonic()
            }
            
            # JSON 문자열로 변환하여 발행
            json_str = json.dumps(data, ensure_ascii=False)
            msg = String()
            msg.data = json_str
            self.tracking_result_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"추적 결과 발행 실패: {e}")
    
    def _control_callback(self, msg: String):
        """제어 명령 콜백"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'run' or cmd_type == 'start':
                self.is_running = True
                manual_mode = command.get('manual', False)
                self.set_manual_mode(manual_mode)
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드")
            
            elif cmd_type == 'stop':
                self.is_running = False
                self.set_state(TrackingState.IDLE, None)
                self.target_track_id = None
                self.target_explicitly_set = False
                self.get_logger().info("RUN 중지: IDLE 상태로 전환")
            
            elif cmd_type == 'set_mode':
                if self.is_running:
                    manual_mode = command.get('manual', False)
                    self.set_manual_mode(manual_mode)
                    self.get_logger().info(f"Manual 모드 설정: {manual_mode}")
            
            elif cmd_type == 'set_state':
                state_str = command.get('state', 'idle')
                target_id = command.get('target_id', None)
                try:
                    state = TrackingState[state_str.upper()]
                    self.set_state(state, target_id)
                    self.get_logger().info(f"상태 설정: {state_str}, 타겟 ID: {target_id}")
                except (KeyError, AttributeError) as e:
                    self.get_logger().error(f"잘못된 상태: {state_str}")
            
            elif cmd_type == 'set_target':
                target_id = command.get('target_id')
                if target_id is not None:
                    self.set_target(int(target_id))
                    if self.interaction_mode:
                        self.state = TrackingState.INTERACTION
                    else:
                        self.state = TrackingState.TRACKING
                    self.lost_frames = 0
                    self.get_logger().info(f"타겟 변경: {self.target_track_id}")
            
            elif cmd_type == 'set_interaction_mode':
                enabled = command.get('enabled', False)
                self.set_interaction_mode(enabled)
                if enabled:
                    self.get_logger().info("Interaction Mode 활성화")
                else:
                    self.get_logger().info("IDLE Mode 활성화")
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f"제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"제어 명령 처리 실패: {e}")
    
    def _state_request_callback(self, msg: String):
        """상태 변경 요청 콜백 (Controller 노드에서 발행)"""
        try:
            request = json.loads(msg.data)
            state_str = request.get('state', 'idle')
            target_id = request.get('target_id', None)
            
            try:
                state = TrackingState[state_str.upper()]
                self.set_state(state, target_id)
                self.get_logger().info(f"상태 변경 요청 수신: {state_str}")
            except (KeyError, AttributeError) as e:
                self.get_logger().error(f"잘못된 상태: {state_str}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"상태 변경 요청 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"상태 변경 요청 처리 실패: {e}")
    
    def neck_angle_callback(self, msg: String):
        """목 각도 콜백 - Controller 노드에서 발행한 목 각도 저장"""
        try:
            data = json.loads(msg.data)
            self.current_neck_yaw_rad = data.get('current_yaw_rad', None)
        except Exception as e:
            self.get_logger().warn(f"목 각도 파싱 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = TrackingFSMNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
