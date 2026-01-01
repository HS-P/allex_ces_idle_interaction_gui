#!/usr/bin/env python3
"""
Tracking FSM Node - 모든 로직과 통신을 한 파일에 통합
BB Box와 ID를 받아서 FSM 처리
"""
import time
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import cv2
import numpy as np
from cv_bridge import CvBridge
from typing import List, Optional, Dict, Tuple
from collections import namedtuple
from enum import Enum

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
    IDLE = "idle"           # 영자세로 돌아가기 (5초 동안 아무것도 안 함, 이후 자동으로 WAITING 전이)
    WAITING = "waiting"     # 타겟 찾기 (기존 IDLE의 타겟 찾기 로직)
    TRACKING = "tracking"   # 추적 중
    LOST = "lost"          # 추적 대상 놓침 (잠시 대기)
    SEARCHING = "searching" # 주변 두리번대기 (대상 선택)
    HELLO = "hello"        # 인사 제스처 (손 흔들기)
    HANDSHAKE = "handshake" # 악수 제스처


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
        
        detections_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        tracker_control_topic = self.get_parameter('tracker_control_topic').get_parameter_value().string_value
        tracker_state_request_topic = self.get_parameter('tracker_state_request_topic').get_parameter_value().string_value
        
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
        # depth=1로 설정하여 최신 메시지만 유지 (지연 최소화)
        self.tracking_result_publisher = self.create_publisher(
            String,
            tracking_result_topic,
            1  # 최신 메시지만 유지하여 지연 최소화
        )
        
        # 루틴 상태 피드백 구독 (HELLO/HANDSHAKE 상태에서 사용)
        # /debug/routine 토픽을 구독하여 루틴 실행 상태 확인
        self.routine_status_subscription = self.create_subscription(
            String,
            "/debug/routine",
            self._routine_status_callback,
            10
        )
        
        # Depth 이미지 구독 (HANDSHAKE/HELLO 분기 판단용)
        self.declare_parameter('depth_image_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        depth_image_topic = self.get_parameter('depth_image_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        
        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,
            self._depth_image_callback,
            qos_profile,
        )
        
        # 카메라 정보 구독 (캘리브레이션 파라미터)
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._camera_info_callback,
            10
        )
        
        self.cv_bridge = CvBridge()
        self.camera_info = None  # CameraInfo 저장
        self.latest_depth_image = None  # 최신 depth 이미지 저장
        
        self.get_logger().info(f"Depth 이미지 구독 시작: {depth_image_topic}")
        
        # 타겟의 Depth 정보 저장 (track_id -> depth_m)
        self.target_depth_map: Dict[int, float] = {}  # track_id -> depth in meters
        
        # 추적 상태 관리
        self.state = TrackingState.IDLE
        self.target_track_id: Optional[int] = None  # 추적 대상 ID
        self.lost_frames = 0  # 놓친 프레임 수
        self.max_lost_frames = 120  # 최대 놓친 프레임 수 (약 4초, 30FPS 기준) - 1.5초에서 4초로 증가
        
        # Manual 모드 지원
        self.manual_mode = False  # True면 상태 자동 전이 비활성화
        
        
        # 타겟이 명시적으로 설정되었는지 표시 (상태 머신이 덮어쓰지 않도록)
        self.target_explicitly_set = False
        
        # 타겟 후보가 되기 위한 최소 지속 시간 (초)
        self.min_target_duration = 1.4
        # 각 track_id의 첫 등장 시간 추적
        self.track_id_first_seen: Dict[int, float] = {}
        
        # TRACKING ROI 영역 변수 (좌우 끝 영역 제한) - gaze_controller와 동일한 값 사용
        self.tracking_roi_left_margin = 0.12  # 좌측 마진 (화면 너비의 12%)
        self.tracking_roi_right_margin = 0.12  # 우측 마진 (화면 너비의 12%)
        
        # 타겟 유지 시간: 타겟이 한 번 선택되면 이 시간 동안은 타겟 변경 방지 (초)
        self.target_lock_duration = 3.0  # 3초 동안 타겟 고정
        self.target_selected_time: Optional[float] = None  # 타겟이 선택된 시간
        
        # 마지막 타겟 위치 저장 (가림 대응 및 같은 사람 재선택용)
        self.last_target_position: Optional[Tuple[float, float]] = None
        self.last_target_bbox: Optional[Tuple[float, float, float, float]] = None
        
        # 최신 프레임 저장
        self.latest_frame = None
        self.latest_frame_shape = None
        
        # HELLO/HANDSHAKE 상태를 위한 변수들
        self.hello_routine_sent_time: Optional[float] = None  # HELLO 루틴 발행 시간
        self.handshake_routine_sent_time: Optional[float] = None  # HANDSHAKE 루틴 발행 시간
        self.hello_feedback_delay = 3.0  # 루틴 발행 후 피드백 확인 최소 대기 시간 (초) - 루틴이 안정적으로 완료될 때까지 대기
        self.handshake_feedback_delay = 3.0  # HANDSHAKE도 동일
        self.routine_stopped_confirmation_time = 0.5  # 루틴 종료 확인 후 추가 대기 시간 (초) - Ready 상태 확실히 확인
        self.current_routine_running = False  # 루틴 실행 중 여부 (True: 실행 중, False: 종료/비어있음)
        self.last_routine_status_time: Optional[float] = None  # 마지막 루틴 상태 수신 시간
        self.routine_stopped_time: Optional[float] = None  # 루틴이 종료된 것으로 감지된 시간
        self.hello_routine_sent = False  # HELLO 루틴 발행 여부
        self.handshake_routine_sent = False  # HANDSHAKE 루틴 발행 여부
        # 이미 HELLO/HANDSHAKE를 한 track_id 저장 (타겟 선택 시 제외)
        self.hello_done_track_ids = set()
        
        # 실행 상태 플래그
        self.is_running = False
        
        # IDLE 상태 타이머 (영자세로 돌아가기만 하는 시간)
        self.idle_start_time: Optional[float] = None  # IDLE 상태 진입 시간 또는 타겟 발견 시간
        self.idle_duration = 7.5  # IDLE 상태 유지 시간 (초) - 타겟 발견 후 5초 후 WAITING 전이
        
        # SEARCHING 상태 진입 시간 (최초 진입 후 5초 동안은 사람 탐색 안 함)
        self.searching_start_time: Optional[float] = None  # SEARCHING 상태 최초 진입 시간
        self.searching_cooldown_duration = 8.0  # SEARCHING 최초 진입 후 대기 시간 (초)
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("Tracking FSM Node 초기화 완료")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    def _find_closest_person(self, detections: List[Dict], frame_shape: tuple, current_time: float, current_target_id: Optional[int] = None) -> Optional[int]:
        """프레임 중심에 가장 가까운 사람 찾기 (최소 지속 시간 이상인 객체만 후보)
        
        Args:
            detections: 검출된 객체 리스트
            frame_shape: 프레임 크기 (height, width)
            current_time: 현재 시간
            current_target_id: 현재 타겟 ID (존재하면 우선 유지)
        """
        if not detections:
            return None
        
        # 현재 타겟이 존재하는 경우, 그 타겟을 우선적으로 반환 (타겟 안정성 유지)
        # 단, HELLO 완료 ID는 제외, ROI 영역 내에 있어야 함
        frame_height, frame_width = frame_shape
        roi_left = frame_width * self.tracking_roi_left_margin
        roi_right = frame_width * (1.0 - self.tracking_roi_right_margin)
        
        if current_target_id is not None and current_target_id not in self.hello_done_track_ids:
            for det in detections:
                if det['track_id'] == current_target_id:
                    # ROI 영역 체크: centroid의 x 좌표가 ROI 영역 내에 있는지 확인
                    cx, cy = det['centroid']
                    if roi_left <= cx <= roi_right:
                        # 현재 타겟이 ROI 영역 내에 존재하면 그대로 반환 (타겟 변경 방지)
                        return current_target_id
        
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
            if track_id in self.track_id_first_seen:
                del self.track_id_first_seen[track_id]
        
        # 최소 지속 시간 이상이고 HELLO 완료 ID가 아닌 객체만 필터링
        valid_detections = []
        frame_height, frame_width = frame_shape
        
        # ROI 영역 계산
        roi_left = frame_width * self.tracking_roi_left_margin
        roi_right = frame_width * (1.0 - self.tracking_roi_right_margin)
        
        for det in detections:
            track_id = det['track_id']
            # HELLO 완료 ID는 제외
            if track_id in self.hello_done_track_ids:
                continue
            
            # ROI 영역 체크: centroid의 x 좌표가 ROI 영역 내에 있는지 확인
            cx, cy = det['centroid']
            if cx < roi_left or cx > roi_right:
                # ROI 영역 외의 사람은 제외
                continue
            
            if track_id in self.track_id_first_seen:
                duration = current_time - self.track_id_first_seen[track_id]
                if duration >= self.min_target_duration:
                    valid_detections.append(det)
        
        if not valid_detections:
            return None
        
        # 유효한 객체 중에서 가장 가까운 사람 찾기 (ROI 영역 내의 사람만 대상)
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
    
    
    def reset_timers(self) -> None:
        """모든 타이머 및 안정성 관련 변수 초기화"""
        self.lost_frames = 0
        self.target_selected_time = None
    
    def set_state(self, state: TrackingState, target_track_id: Optional[int] = None) -> None:
        """Manual 모드에서 상태를 수동으로 설정"""
        if state == TrackingState.IDLE:
            self.reset_timers()
            self.idle_start_time = time.monotonic()  # IDLE 상태 진입 시간 기록
        elif state == TrackingState.WAITING:
            self.idle_start_time = None  # WAITING 진입 시 IDLE 타이머 초기화

        # HELLO/HANDSHAKE 전환 요청은 Auto 모드에서도 허용 (Controller에서 수신)
        if state == TrackingState.HELLO or state == TrackingState.HANDSHAKE:
            self.state = state
            if target_track_id is not None:
                self.target_track_id = int(target_track_id)
                self.target_explicitly_set = True
            self.lost_frames = 0
            return

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
        self.target_selected_time = time.monotonic()  # 타겟 선택 시간 기록
    
    def _process_fsm(self, detections: List[Dict], frame_shape: tuple, frame: Optional[np.ndarray] = None) -> tuple[List[TrackedObject], TargetInfo]:
        """Detection 결과를 받아서 FSM 처리"""
        current_time = time.monotonic()
        
        # 타겟이 설정되어 있으면 현재 프레임에 존재하는지 확인
        # 단, HELLO를 한 번 완료한 타겟은 존재하지 않는 것으로 처리 (더 이상 추적하지 않음)
        # 또한 ROI 영역 내에 있는지 확인
        frame_height, frame_width = frame_shape
        roi_left = frame_width * self.tracking_roi_left_margin
        roi_right = frame_width * (1.0 - self.tracking_roi_right_margin)
        
        target_exists = False
        if self.target_track_id is not None and self.target_track_id not in self.hello_done_track_ids:
            for det in detections:
                if det['track_id'] == self.target_track_id:
                    # ROI 영역 체크: centroid의 x 좌표가 ROI 영역 내에 있는지 확인
                    cx, cy = det['centroid']
                    if roi_left <= cx <= roi_right:
                        target_exists = True
                    break
        
        # 상태 머신 처리
        if not detections:
            # 감지된 객체가 없으면 상태 업데이트
            # Manual Mode에서도 안전을 위해 TRACKING -> LOST 전환은 허용
            match self.state:
                case TrackingState.TRACKING:
                    self.state = TrackingState.LOST
                    self.lost_frames = 0
                case TrackingState.LOST:
                    self.lost_frames += 1
                    # LOST -> SEARCHING 자동 전이는 Manual Mode에서 비활성화
                    if self.lost_frames >= self.max_lost_frames and not self.manual_mode:
                        self.state = TrackingState.SEARCHING
                        if not self.target_explicitly_set:
                            self.target_track_id = None
                        # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                        if self.searching_start_time is None:
                            self.searching_start_time = current_time
                            self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                    # Manual Mode에서는 LOST 상태 유지 (lost_frames만 증가)
                case _:
                    pass
            
            target_info = TargetInfo(
                point=None,
                state=self.state,
                track_id=self.target_track_id
            )
            return [], target_info
        
        # 타겟이 존재하면 상태 업데이트 (HELLO/HANDSHAKE/IDLE 상태에서는 전환하지 않음)
        # IDLE 상태는 시간 기반으로 자동 전이하므로 타겟 인식으로 전환하지 않음
        if target_exists and self.state != TrackingState.HELLO and self.state != TrackingState.HANDSHAKE and self.state != TrackingState.IDLE:
            if self.state != TrackingState.TRACKING:
                self.state = TrackingState.TRACKING
            self.lost_frames = 0
        
        # TRACKING 상태: 타겟 손실 감지 (Manual Mode에서도 안전을 위해 허용)
        if self.state == TrackingState.TRACKING:
            # 현재 타겟이 HELLO 완료 ID이면 LOST로 전환
            if self.target_track_id is not None and self.target_track_id in self.hello_done_track_ids:
                self.get_logger().info(
                    f"TRACKING: 타겟 ID={self.target_track_id}는 HELLO 완료 ID이므로 LOST로 전환"
                )
                self.target_track_id = None
                self.target_explicitly_set = False
                self.state = TrackingState.LOST
                self.lost_frames = 0
                # target_exists는 이미 False가 되므로 target_info 반환
                target_info = TargetInfo(
                    point=None,
                    state=self.state,
                    track_id=self.target_track_id
                )
                return [], target_info
            
            # 타겟이 존재하면 타겟 선택 시간 업데이트 (타겟 유지 중)
            if target_exists and self.target_track_id is not None:
                if self.target_selected_time is None:
                    self.target_selected_time = current_time
            
            # Manual Mode에서도 타겟 손실 시 LOST로 전환 (안전을 위해)
            if not target_exists and self.target_track_id is not None:
                # 타겟 lock 시간이 지나지 않았으면 타겟 유지 (ID 스위치 방지)
                should_keep_target = False
                if self.target_selected_time is not None:
                    elapsed_since_selection = current_time - self.target_selected_time
                    if elapsed_since_selection < self.target_lock_duration:
                        # 타겟 lock 기간 중이면 LOST로 전환하지 않음 (타겟 유지)
                        should_keep_target = True
                        self.get_logger().debug(
                            f"타겟 lock 중: ID={self.target_track_id}, "
                            f"경과={elapsed_since_selection:.1f}초/{self.target_lock_duration}초"
                        )
                
                if not should_keep_target:
                    self.state = TrackingState.LOST
                    self.lost_frames = 0
        
        # LOST 상태: 동일 타겟 복귀 감지 (Manual Mode에서도 허용)
        if self.state == TrackingState.LOST:
            # target_exists는 이미 hello_done_track_ids를 체크하므로,
            # HELLO 완료 ID는 자동으로 제외됨
            if target_exists:
                # 동일 타겟이 다시 나타남 -> TRACKING으로 복귀
                self.state = TrackingState.TRACKING
                self.lost_frames = 0
                # Auto Mode에서는 target_explicitly_set을 False로 유지
                # Manual Mode에서는 target_explicitly_set을 True로 유지
                if not self.manual_mode and self.target_explicitly_set:
                    self.target_explicitly_set = False
            elif self.target_track_id is not None and self.target_track_id in self.hello_done_track_ids:
                # 현재 타겟이 HELLO 완료 ID이면 타겟을 None으로 설정하고 LOST 상태 유지
                self.get_logger().info(
                    f"LOST: 타겟 ID={self.target_track_id}는 HELLO 완료 ID이므로 타겟 해제"
                )
                self.target_track_id = None
                self.target_explicitly_set = False
            else:
                self.lost_frames += 1
                # LOST -> SEARCHING 자동 전이는 Manual Mode에서 비활성화
                if self.lost_frames >= self.max_lost_frames and not self.manual_mode:
                    # Auto Mode에서는 항상 SEARCHING으로 전환 (새 타겟 자동 선택)
                    if self.target_track_id is not None:
                        # SEARCHING 전환 시 마지막 위치는 유지 (같은 사람을 찾기 위해)
                        # target_track_id만 None으로 설정
                        self.target_track_id = None
                        self.target_explicitly_set = False
                    self.state = TrackingState.SEARCHING
        
        # 상태 머신 처리 (Manual 모드가 아닐 때만 자동 전이)
        if not self.manual_mode:
            match self.state:
                case TrackingState.IDLE:
                    # IDLE 상태: 타겟을 찾으면 그 시점부터 5초 후 WAITING으로 전이
                    # 타겟이 발견되면 타이머 시작
                    if detections and len(detections) > 0:
                        # 타겟 발견 (detection이 있으면)
                        if self.idle_start_time is None:
                            self.idle_start_time = current_time
                            self.get_logger().info(f"[IDLE] 타겟 발견: 시작 시간={current_time:.3f}, {self.idle_duration}초 후 WAITING으로 자동 전이 예정")
                    elif self.idle_start_time is not None:
                        # 타겟이 사라지면 타이머 리셋
                        self.idle_start_time = None
                        self.get_logger().debug("[IDLE] 타겟 사라짐: 타이머 리셋")
                    
                    # 타겟 발견 후 5초가 지나면 자동으로 WAITING으로 전이
                    if self.idle_start_time is not None:
                        elapsed_time = current_time - self.idle_start_time
                        
                        # 디버깅: 1초마다 경과 시간 로그
                        if not hasattr(self, '_last_idle_log_time') or current_time - self._last_idle_log_time >= 1.0:
                            remaining_time = self.idle_duration - elapsed_time
                            self.get_logger().info(
                                f"[IDLE] 시간 체크: 경과={elapsed_time:.2f}초 / 목표={self.idle_duration}초, "
                                f"남은 시간={remaining_time:.2f}초"
                            )
                            self._last_idle_log_time = current_time
                        
                        if elapsed_time >= self.idle_duration:
                            self.state = TrackingState.WAITING
                            actual_elapsed = current_time - self.idle_start_time
                            self.idle_start_time = None
                            if hasattr(self, '_last_idle_log_time'):
                                delattr(self, '_last_idle_log_time')
                            self.get_logger().info(
                                f"[IDLE -> WAITING] 자동 전이 완료: 타겟 발견 후 {actual_elapsed:.3f}초 경과"
                            )
                    # IDLE 상태에서는 타겟 찾기 하지 않음 (영자세로 돌아가기만, 타겟 발견 후 시간 기반 자동 전이)
                
                case TrackingState.WAITING:
                    # WAITING 상태: 타겟 찾기 (기존 IDLE의 타겟 찾기 로직)
                    if self.target_explicitly_set and target_exists:
                        # Manual 모드에서 명시적으로 타겟 설정된 경우
                        self.state = TrackingState.TRACKING
                        self.lost_frames = 0
                        if self.target_selected_time is None:
                            self.target_selected_time = current_time
                    else:
                        # Auto Mode: 타겟이 없으면 자동으로 가장 가까운 사람 선택 (HELLO 완료 ID 제외)
                        if self.target_track_id is None:
                            # HELLO 완료 ID 제외
                            valid_detections = [det for det in detections if det['track_id'] not in self.hello_done_track_ids]
                            closest_id = self._find_closest_person(valid_detections, frame_shape, current_time, current_target_id=self.target_track_id)
                            if closest_id is not None:
                                self.target_track_id = closest_id
                                self.state = TrackingState.TRACKING
                                self.lost_frames = 0
                                self.target_explicitly_set = False  # Auto Mode에서는 항상 False
                                self.target_selected_time = current_time  # 타겟 선택 시간 기록
                        elif target_exists:
                            self.state = TrackingState.TRACKING
                            self.lost_frames = 0
                            # Auto Mode에서는 target_explicitly_set을 False로 유지
                            if self.target_explicitly_set:
                                self.target_explicitly_set = False
                            if self.target_selected_time is None:
                                self.target_selected_time = current_time
                
                # TRACKING과 LOST 케이스는 블록 밖에서 처리 (Manual Mode에서도 안전 관련 전환 허용)
                # case TrackingState.TRACKING: (제거됨 - 블록 밖에서 처리)
                # case TrackingState.LOST: (제거됨 - 블록 밖에서 처리)
                
                case TrackingState.SEARCHING:
                    # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                    if self.searching_start_time is None:
                        self.searching_start_time = current_time
                        self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                    
                    # 최초 진입 후 5초 동안은 사람 탐색 안 함
                    can_search = True
                    if self.searching_start_time is not None:
                        elapsed_since_searching = current_time - self.searching_start_time
                        if elapsed_since_searching < self.searching_cooldown_duration:
                            # 5초가 지나지 않았으면 타겟 선택하지 않음
                            can_search = False
                            self.get_logger().debug(
                                f"SEARCHING: 쿨다운 중 ({elapsed_since_searching:.2f}초/{self.searching_cooldown_duration}초), "
                                f"사람 탐색 건너뜀"
                            )
                    
                    if detections and can_search:
                        # 타겟 lock 시간이 지나지 않았으면 기존 타겟 유지 시도
                        target_found = False
                        if (self.target_track_id is not None and 
                            self.target_selected_time is not None):
                            # 기존 타겟이 HELLO 완료 ID이면 무시
                            if self.target_track_id not in self.hello_done_track_ids:
                                elapsed_since_selection = current_time - self.target_selected_time
                                if elapsed_since_selection < self.target_lock_duration:
                                    # 기존 타겟이 다시 나타났는지 확인 (HELLO 완료 ID 제외)
                                    if any(det['track_id'] == self.target_track_id and 
                                           det['track_id'] not in self.hello_done_track_ids 
                                           for det in detections):
                                        # 기존 타겟이 다시 나타남 - TRACKING으로 복귀
                                        self.state = TrackingState.TRACKING
                                        self.lost_frames = 0
                                        target_found = True
                                        # SEARCHING에서 벗어나므로 초기화
                                        self.searching_start_time = None
                                        self.get_logger().info(
                                            f"타겟 lock 중 기존 타겟 재발견: ID={self.target_track_id}"
                                        )
                            else:
                                # 기존 타겟이 HELLO 완료 ID이면 타겟 해제
                                self.target_track_id = None
                                self.target_explicitly_set = False
                        
                        # 기존 타겟을 찾지 못했으면 새로운 타겟 선택
                        if not target_found:
                            # HELLO를 한 타겟은 제외하고 선택
                            valid_detections = [det for det in detections if det['track_id'] not in self.hello_done_track_ids]
                            
                            if not valid_detections:
                                # 모든 후보가 HELLO를 한 경우, 타겟을 선정하지 않음
                                self.get_logger().info(
                                    f"SEARCHING: 모든 후보가 HELLO 완료 ID임 (총 {len(detections)}개). "
                                    f"타겟 선정 건너뜀."
                                )
                                # 타겟을 선정하지 않고 SEARCHING 상태 유지
                                closest_id = None
                            
                            # 마지막 타겟 위치를 우선 고려하여 같은 위치 근처의 사람 선택
                            closest_id = None
                            if self.last_target_position is not None:
                                # 마지막 타겟 위치에 가장 가까운 사람 선택 (Testbed 방식)
                                h, w = frame_shape
                                last_x, last_y = self.last_target_position
                                
                                best_id = None
                                min_weighted_dist = float('inf')
                                
                                for det in valid_detections:
                                    px, py = det['centroid']
                                    # 마지막 타겟 위치와의 거리 (가중치 1.0)
                                    dist_to_last = np.sqrt((px - last_x)**2 + (py - last_y)**2)
                                    # 프레임 중심과의 거리 (가중치 0.3 - 덜 중요)
                                    cx, cy = w / 2.0, h / 2.0
                                    dist_to_center = np.sqrt((px - cx)**2 + (py - cy)**2)
                                    # 가중 평균: 마지막 위치를 더 중요하게
                                    weighted_dist = dist_to_last * 1.0 + dist_to_center * 0.3
                                    
                                    if weighted_dist < min_weighted_dist:
                                        min_weighted_dist = weighted_dist
                                        best_id = det['track_id']
                                
                                if best_id is not None:
                                    closest_id = best_id
                                    self.get_logger().info(
                                        f"SEARCHING: 마지막 위치 우선 선택 ID={closest_id}, "
                                        f"거리={min_weighted_dist:.1f} (HELLO 완료 ID 제외)"
                                    )
                            
                            # 마지막 위치 정보가 없으면 기존 로직 사용 (HELLO 완료 ID 제외)
                            if closest_id is None and valid_detections:
                                closest_id = self._find_closest_person(valid_detections, frame_shape, current_time, current_target_id=self.target_track_id)
                            
                            if closest_id is not None:
                                self.target_track_id = closest_id
                                self.state = TrackingState.TRACKING
                                self.lost_frames = 0
                                self.target_explicitly_set = False
                                self.target_selected_time = current_time  # 새 타겟 선택 시간 기록
                                # SEARCHING에서 벗어나므로 초기화
                                self.searching_start_time = None
                
                case TrackingState.HELLO:
                    # HELLO 상태: 루틴 발행 후 루틴 종료 모니터링
                    current_time_check = time.monotonic()
                    
                    # 디버깅: HELLO 상태 진입 시 상세 정보
                    if not self.hello_routine_sent:
                        self.get_logger().warn(
                            f"[HELLO 디버깅] 루틴이 아직 발행되지 않음: "
                            f"hello_routine_sent={self.hello_routine_sent}, "
                            f"hello_routine_sent_time={self.hello_routine_sent_time}"
                        )
                    
                    # 루틴이 아직 발행되지 않았으면 발행 (allex_idle_interaction_node에서 처리)
                    # 여기서는 루틴 상태만 모니터링
                    if self.hello_routine_sent and self.hello_routine_sent_time is not None:
                        elapsed_time = current_time_check - self.hello_routine_sent_time
                        
                        # 디버깅: 주기적으로 상태 로그 (0.5초마다)
                        if not hasattr(self, '_last_hello_monitor_log_time'):
                            self._last_hello_monitor_log_time = 0
                        if current_time_check - self._last_hello_monitor_log_time > 0.5:
                            last_status_str = f"{self.last_routine_status_time:.2f}초 전" if self.last_routine_status_time else "수신 없음"
                            self.get_logger().info(
                                f"[HELLO 모니터링] 경과={elapsed_time:.2f}초/{self.hello_feedback_delay}초, "
                                f"루틴 실행 중={self.current_routine_running}, "
                                f"마지막 상태 수신={last_status_str}, "
                                f"target_track_id={self.target_track_id}"
                            )
                            self._last_hello_monitor_log_time = current_time_check
                        
                        # 루틴이 종료되었으면 (실행 중이 아니면) SEARCHING으로 전이
                        # 최소 대기 시간 경과 후에만 전환 (명령이 전달될 시간 확보)
                        # Manual Mode에서는 allex_idle_interaction_node에서 루틴 완료를 감지하므로 여기서는 상태 전환하지 않음
                        if elapsed_time >= self.hello_feedback_delay:
                            if not self.current_routine_running:
                                # 루틴 종료 확인 후 추가 대기 시간 체크 (Ready 상태 확실히 확인)
                                if self.routine_stopped_time is not None:
                                    time_since_stopped = current_time_check - self.routine_stopped_time
                                    if time_since_stopped >= self.routine_stopped_confirmation_time:
                                        # Manual Mode에서는 allex_idle_interaction_node에서 루틴 완료를 감지하므로 상태 전환하지 않음
                                        if self.manual_mode:
                                            self.get_logger().debug(
                                                f"[HELLO 조건 체크] Manual Mode: 루틴 종료 확인 완료하지만 상태 전환은 allex_idle_interaction_node에서 처리 "
                                                f"(경과={elapsed_time:.2f}초, 종료 후 {time_since_stopped:.2f}초)"
                                            )
                                        else:
                                            # Auto Mode: SEARCHING으로 전환
                                            # 디버깅: 조건 확인 상세 로그
                                            self.get_logger().info(
                                                f"[HELLO 조건 체크] 루틴 종료 확인 완료: "
                                                f"경과={elapsed_time:.2f}초 >= {self.hello_feedback_delay}초, "
                                                f"루틴 종료 후 {time_since_stopped:.2f}초 경과 >= {self.routine_stopped_confirmation_time}초, "
                                                f"루틴 실행 중={self.current_routine_running}"
                                            )
                                            
                                            # HELLO를 한 track_id 저장 (더 이상 타겟으로 선택하지 않음)
                                            if self.target_track_id is not None:
                                                self.hello_done_track_ids.add(self.target_track_id)
                                                self.get_logger().info(
                                                    f"HELLO 완료 ID 저장: track_id={self.target_track_id} "
                                                    f"(총 {len(self.hello_done_track_ids)}개 ID, 이제 타겟으로 선택되지 않음)"
                                                )
                                            
                                            # 타겟이 없어도 SEARCHING으로 전환 (상대방이 사라진 경우 대응)
                                            self.state = TrackingState.SEARCHING
                                            self.target_track_id = None
                                            self.target_explicitly_set = False
                                            self.hello_routine_sent = False
                                            self.hello_routine_sent_time = None
                                            self.current_routine_running = False
                                            self.routine_stopped_time = None
                                            # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                                            if self.searching_start_time is None:
                                                self.searching_start_time = current_time_check
                                                self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                                            self.get_logger().info(
                                                f"HELLO 완료: 루틴 종료 → SEARCHING 상태로 전환 "
                                                f"(총 경과 시간: {elapsed_time:.2f}초, 종료 확인 후: {time_since_stopped:.2f}초)"
                                            )
                                    else:
                                        # 루틴 종료 후 추가 대기 중
                                        self.get_logger().debug(
                                            f"[HELLO 대기 중] 루틴 종료 확인 대기: "
                                            f"종료 후 {time_since_stopped:.2f}초 < {self.routine_stopped_confirmation_time}초"
                                        )
                                else:
                                    # 루틴 종료 시간이 아직 기록되지 않았지만, 루틴이 비어있고 충분한 시간이 지났으면 SEARCHING으로 전환
                                    # 루틴이 비어있으면 종료된 것으로 간주하고 전환 (사람이 없어도 전환)
                                    if elapsed_time >= self.hello_feedback_delay + self.routine_stopped_confirmation_time:
                                        self.get_logger().info(
                                            f"[HELLO 조건 체크] 루틴 종료 시간 미기록이지만 충분한 시간 경과: "
                                            f"경과={elapsed_time:.2f}초 >= {self.hello_feedback_delay + self.routine_stopped_confirmation_time}초, "
                                            f"루틴 실행 중={self.current_routine_running}"
                                        )
                                        
                                        # HELLO를 한 track_id 저장
                                        if self.target_track_id is not None:
                                            self.hello_done_track_ids.add(self.target_track_id)
                                            self.get_logger().info(
                                                f"HELLO 완료 ID 저장: track_id={self.target_track_id} "
                                                f"(총 {len(self.hello_done_track_ids)}개 ID)"
                                            )
                                        
                                        # 타겟이 없어도 SEARCHING으로 전환 (사람이 없어도 전환)
                                        self.state = TrackingState.SEARCHING
                                        self.target_track_id = None
                                        self.target_explicitly_set = False
                                        self.hello_routine_sent = False
                                        self.hello_routine_sent_time = None
                                        self.current_routine_running = False
                                        self.routine_stopped_time = None
                                        # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                                        if self.searching_start_time is None:
                                            self.searching_start_time = current_time_check
                                            self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                                        self.get_logger().info(
                                            f"HELLO 완료: 루틴 비어있음 → SEARCHING 상태로 전환 "
                                            f"(총 경과 시간: {elapsed_time:.2f}초)"
                                        )
                                    else:
                                        # 루틴 종료 시간이 아직 기록되지 않음 (최소 대기 시간은 지났지만)
                                        self.get_logger().debug(
                                            f"[HELLO 대기 중] 루틴 종료 시간 미기록, "
                                            f"current_routine_running={self.current_routine_running}"
                                        )
                            else:
                                # 루틴이 아직 실행 중인 경우
                                self.get_logger().info(
                                    f"[HELLO 대기 중] 루틴이 아직 실행 중: "
                                    f"current_routine_running={self.current_routine_running}"
                                )
                        else:
                            # 아직 대기 시간이 지나지 않음
                            self.get_logger().debug(
                                f"[HELLO 대기 중] 경과 시간 부족: {elapsed_time:.2f}초 < {self.hello_feedback_delay}초"
                            )
                    else:
                        # hello_routine_sent가 False이거나 hello_routine_sent_time이 None
                        self.get_logger().debug(
                            f"[HELLO 대기 중] 루틴 발행 정보 부족: "
                            f"hello_routine_sent={self.hello_routine_sent}, "
                            f"hello_routine_sent_time={self.hello_routine_sent_time}"
                        )
                
                case TrackingState.HANDSHAKE:
                    # HANDSHAKE 상태: 루틴 발행 후 루틴 종료 모니터링 (HELLO와 동일)
                    current_time_check = time.monotonic()
                    
                    # 루틴이 아직 발행되지 않았으면 경고 로그
                    if not self.handshake_routine_sent:
                        self.get_logger().warn(
                            f"[HANDSHAKE 디버깅] 루틴이 아직 발행되지 않음: "
                            f"handshake_routine_sent={self.handshake_routine_sent}, "
                            f"handshake_routine_sent_time={self.handshake_routine_sent_time}"
                        )
                    
                    if self.handshake_routine_sent and self.handshake_routine_sent_time is not None:
                        elapsed_time = current_time_check - self.handshake_routine_sent_time
                        
                        # 디버깅: 주기적으로 상태 로그 (0.5초마다)
                        if not hasattr(self, '_last_handshake_monitor_log_time'):
                            self._last_handshake_monitor_log_time = 0
                        if current_time_check - self._last_handshake_monitor_log_time > 0.5:
                            last_status_str = f"{self.last_routine_status_time:.2f}초 전" if self.last_routine_status_time else "수신 없음"
                            self.get_logger().info(
                                f"[HANDSHAKE 모니터링] 경과={elapsed_time:.2f}초/{self.handshake_feedback_delay}초, "
                                f"루틴 실행 중={self.current_routine_running}, "
                                f"마지막 상태 수신={last_status_str}, "
                                f"target_track_id={self.target_track_id}"
                            )
                            self._last_handshake_monitor_log_time = current_time_check
                        
                        # 루틴이 종료되었으면 (실행 중이 아니면) SEARCHING으로 전이
                        # 최소 대기 시간 경과 후에만 전환 (명령이 전달될 시간 확보)
                        # Manual Mode에서는 allex_idle_interaction_node에서 루틴 완료를 감지하므로 여기서는 상태 전환하지 않음
                        if elapsed_time >= self.handshake_feedback_delay:
                            if not self.current_routine_running:
                                # 루틴 종료 확인 후 추가 대기 시간 체크 (Ready 상태 확실히 확인)
                                if self.routine_stopped_time is not None:
                                    time_since_stopped = current_time_check - self.routine_stopped_time
                                    if time_since_stopped >= self.routine_stopped_confirmation_time:
                                        # Manual Mode에서는 allex_idle_interaction_node에서 루틴 완료를 감지하므로 상태 전환하지 않음
                                        if self.manual_mode:
                                            self.get_logger().debug(
                                                f"[HANDSHAKE 조건 체크] Manual Mode: 루틴 종료 확인 완료하지만 상태 전환은 allex_idle_interaction_node에서 처리 "
                                                f"(경과={elapsed_time:.2f}초, 종료 후 {time_since_stopped:.2f}초)"
                                            )
                                        else:
                                            # Auto Mode: SEARCHING으로 전환
                                            # 디버깅: 조건 확인 상세 로그
                                            self.get_logger().info(
                                                f"[HANDSHAKE 조건 체크] 루틴 종료 확인 완료: "
                                                f"경과={elapsed_time:.2f}초 >= {self.handshake_feedback_delay}초, "
                                                f"루틴 종료 후 {time_since_stopped:.2f}초 경과 >= {self.routine_stopped_confirmation_time}초, "
                                                f"루틴 실행 중={self.current_routine_running}"
                                            )
                                            
                                            # HANDSHAKE를 한 track_id 저장 (더 이상 타겟으로 선택하지 않음)
                                            if self.target_track_id is not None:
                                                self.hello_done_track_ids.add(self.target_track_id)
                                                self.get_logger().info(
                                                    f"HANDSHAKE 완료 ID 저장: track_id={self.target_track_id} "
                                                    f"(총 {len(self.hello_done_track_ids)}개 ID, 이제 타겟으로 선택되지 않음)"
                                                )
                                            
                                            # 타겟이 없어도 SEARCHING으로 전환 (상대방이 사라진 경우 대응)
                                            self.state = TrackingState.SEARCHING
                                            self.target_track_id = None
                                            self.target_explicitly_set = False
                                            self.handshake_routine_sent = False
                                            self.handshake_routine_sent_time = None
                                            self.current_routine_running = False
                                            self.routine_stopped_time = None
                                            # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                                            if self.searching_start_time is None:
                                                self.searching_start_time = current_time_check
                                                self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                                            self.get_logger().info(
                                                f"HANDSHAKE 완료: 루틴 종료 → SEARCHING 상태로 전환 "
                                                f"(총 경과 시간: {elapsed_time:.2f}초, 종료 확인 후: {time_since_stopped:.2f}초)"
                                            )
                                    else:
                                        # 루틴 종료 후 추가 대기 중
                                        self.get_logger().debug(
                                            f"[HANDSHAKE 대기 중] 루틴 종료 확인 대기: "
                                            f"종료 후 {time_since_stopped:.2f}초 < {self.routine_stopped_confirmation_time}초"
                                        )
                                else:
                                    # 루틴 종료 시간이 아직 기록되지 않았지만, 루틴이 비어있고 충분한 시간이 지났으면 SEARCHING으로 전환
                                    # 루틴이 비어있으면 종료된 것으로 간주하고 전환
                                    # Manual Mode에서는 allex_idle_interaction_node에서 루틴 완료를 감지하므로 상태 전환하지 않음
                                    if elapsed_time >= self.handshake_feedback_delay + self.routine_stopped_confirmation_time:
                                        if self.manual_mode:
                                            self.get_logger().debug(
                                                f"[HANDSHAKE 조건 체크] Manual Mode: 루틴 종료 시간 미기록이지만 충분한 시간 경과, "
                                                f"상태 전환은 allex_idle_interaction_node에서 처리 (경과={elapsed_time:.2f}초)"
                                            )
                                        else:
                                            # Auto Mode: SEARCHING으로 전환
                                            self.get_logger().info(
                                                f"[HANDSHAKE 조건 체크] 루틴 종료 시간 미기록이지만 충분한 시간 경과: "
                                                f"경과={elapsed_time:.2f}초 >= {self.handshake_feedback_delay + self.routine_stopped_confirmation_time}초, "
                                                f"루틴 실행 중={self.current_routine_running}"
                                            )
                                            
                                            # HANDSHAKE를 한 track_id 저장
                                            if self.target_track_id is not None:
                                                self.hello_done_track_ids.add(self.target_track_id)
                                                self.get_logger().info(
                                                    f"HANDSHAKE 완료 ID 저장: track_id={self.target_track_id} "
                                                    f"(총 {len(self.hello_done_track_ids)}개 ID)"
                                                )
                                            
                                            # SEARCHING으로 전환
                                            self.state = TrackingState.SEARCHING
                                            self.target_track_id = None
                                            self.target_explicitly_set = False
                                            self.handshake_routine_sent = False
                                            self.handshake_routine_sent_time = None
                                            self.current_routine_running = False
                                            self.routine_stopped_time = None
                                            # SEARCHING 진입 시간 기록 (최초 진입 시에만)
                                            if self.searching_start_time is None:
                                                self.searching_start_time = current_time_check
                                                self.get_logger().info(f"SEARCHING 상태 최초 진입: {self.searching_cooldown_duration}초 동안 사람 탐색 안 함")
                                            self.get_logger().info(
                                                f"HANDSHAKE 완료: 루틴 비어있음 → SEARCHING 상태로 전환 "
                                                f"(총 경과 시간: {elapsed_time:.2f}초)"
                                            )
                                    else:
                                        # 루틴 종료 시간이 아직 기록되지 않음 (최소 대기 시간은 지났지만)
                                        self.get_logger().debug(
                                            f"[HANDSHAKE 대기 중] 루틴 종료 시간 미기록, "
                                            f"current_routine_running={self.current_routine_running}"
                                        )
                            else:
                                # 루틴이 아직 실행 중인 경우
                                self.get_logger().info(
                                    f"[HANDSHAKE 대기 중] 루틴이 아직 실행 중: "
                                    f"current_routine_running={self.current_routine_running}"
                                )
                        else:
                            # 아직 대기 시간이 지나지 않음
                            self.get_logger().debug(
                                f"[HANDSHAKE 대기 중] 경과 시간 부족: {elapsed_time:.2f}초 < {self.handshake_feedback_delay}초"
                            )
                    else:
                        # handshake_routine_sent가 False이거나 handshake_routine_sent_time이 None
                        self.get_logger().debug(
                            f"[HANDSHAKE 대기 중] 루틴 발행 정보 부족: "
                            f"handshake_routine_sent={self.handshake_routine_sent}, "
                            f"handshake_routine_sent_time={self.handshake_routine_sent_time}"
                        )
        
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
                # 마지막 타겟 위치 저장 (가림 대응 및 같은 사람 재선택용)
                self.last_target_position = target_point
                self.last_target_bbox = (x1, y1, x2, y2)
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
            # frame_shape는 (height, width) 형태로 저장 (channels 제외)
            self.latest_frame_shape = (frame.shape[0], frame.shape[1])
    
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
            
            # Depth 값 추출 및 저장 (frame_shape 전달)
            self._extract_depth_from_detection(detections, frame_shape)
            
            # FSM 처리
            tracked_objects, target_info = self._process_fsm(
                detections,
                frame_shape,
                self.latest_frame
            )
            
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
                'timestamp': time.monotonic(),
                'manual_mode': self.manual_mode,  # GUI 업데이트를 위한 manual_mode 정보 추가
                'target_selected_time': self.target_selected_time  # 타겟 선택 시간 (HELLO 전환 조건 체크용)
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
                manual_mode = command.get('manual', False)
                # Manual <-> Auto 전환 시 IDLE 상태로 변경
                if self.manual_mode != manual_mode:
                    self.set_manual_mode(manual_mode)
                    if self.is_running:
                        self.set_state(TrackingState.IDLE, None)
                        self.target_track_id = None
                        self.target_explicitly_set = False
                        self.get_logger().info(f"Manual 모드 전환: {self.manual_mode} -> {manual_mode} (IDLE 상태로 전환)")
                    else:
                        self.set_manual_mode(manual_mode)
                        self.get_logger().info(f"Manual 모드 설정: {manual_mode} (RUN 중이 아니므로 상태 변경 없음)")
                else:
                    self.set_manual_mode(manual_mode)
                    self.get_logger().debug(f"Manual 모드는 이미 {manual_mode}입니다.")
            
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
                # Manual 모드에서만 타겟 변경 허용
                if self.manual_mode:
                    target_id = command.get('target_id')
                    if target_id is not None:
                        # HELLO/HANDSHAKE 상태에서는 상태를 유지, 그 외에는 TRACKING으로 변경
                        if self.state in (TrackingState.HELLO, TrackingState.HANDSHAKE):
                            # 상태 유지: 타겟만 변경
                            self.target_track_id = int(target_id)
                            self.target_explicitly_set = True
                            self.lost_frames = 0
                            self.target_selected_time = time.monotonic()
                            self.get_logger().info(f"타겟 변경: {self.target_track_id} (상태 유지: {self.state.value})")
                        else:
                            # 상태 변경: set_target 사용 (TRACKING으로 변경)
                            self.set_target(int(target_id))
                            self.get_logger().info(f"타겟 변경: {self.target_track_id} (상태: TRACKING)")
                else:
                    self.get_logger().warn("Auto Mode에서는 타겟 변경이 허용되지 않습니다.")
                    
        except json.JSONDecodeError as e:
            self.get_logger().error(f"제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"제어 명령 처리 실패: {e}")
    
    def _state_request_callback(self, msg: String):
        """상태 변경 요청 콜백 (Controller 노드에서 발행)"""
        try:
            request = json.loads(msg.data)
            request_type = request.get('type', 'set_state')
            target_id = request.get('target_id', None)
            
            # hello_transition_ready: 위치 안정성 체크 완료, depth 기반 분기 판단 필요
            if request_type == 'hello_transition_ready':
                if target_id is None:
                    self.get_logger().warn("hello_transition_ready: target_id가 없습니다.")
                    return
                
                target_track_id = int(target_id)
                
                # Depth 값 확인하여 HELLO 또는 HANDSHAKE 결정
                target_state_str = 'hello'  # 기본값
                target_depth = None
                
                # 1순위: 저장된 map에서 조회
                if target_track_id in self.target_depth_map:
                    target_depth = self.target_depth_map[target_track_id]
                    self.get_logger().info(
                        f"[Depth 분기 체크] 저장된 map에서 조회: track_id={target_track_id}, "
                        f"depth={target_depth:.3f}m, 기준=1.5m"
                    )
                
                if target_depth is not None:
                    if target_depth <= 1.5:  # 1.5m 이내면 HANDSHAKE
                        target_state_str = 'handshake'
                        self.get_logger().info(
                            f"Depth 기반 분기: track_id={target_track_id}, "
                            f"depth={target_depth:.3f}m ({target_depth*1000:.1f}mm, ≤1.5m) → HANDSHAKE"
                        )
                    else:
                        self.get_logger().info(
                            f"Depth 기반 분기: track_id={target_track_id}, "
                            f"depth={target_depth:.3f}m ({target_depth*1000:.1f}mm, >1.5m) → HELLO"
                        )
                else:
                    self.get_logger().warn(
                        f"[Depth 분기 체크] Depth 정보 없음: track_id={target_track_id}, "
                        f"저장된 track_ids={list(self.target_depth_map.keys())}, "
                        f"depth 이미지={'있음' if self.latest_depth_image is not None else '없음'}, 기본값 HELLO 사용"
                    )
                
                # 결정된 상태로 전환
                try:
                    state = TrackingState[target_state_str.upper()]
                    self.state = state
                    
                    # HELLO/HANDSHAKE 상태로 전환 시 변수 초기화 (매번 리셋)
                    if state == TrackingState.HELLO:
                        self.hello_routine_sent_time = time.monotonic()
                        self.hello_routine_sent = True
                        self.current_routine_running = True  # 루틴 시작 시 실행 중으로 가정
                        self.routine_stopped_time = None  # 루틴 종료 시간 초기화
                        self.get_logger().info(
                            f"HELLO 상태로 전환: 루틴 발행 시간 기록, "
                            f"{self.hello_feedback_delay}초 후 루틴 종료 확인 시작"
                        )
                    elif state == TrackingState.HANDSHAKE:
                        self.handshake_routine_sent_time = time.monotonic()
                        self.handshake_routine_sent = True
                        self.current_routine_running = True  # 루틴 시작 시 실행 중으로 가정
                        self.routine_stopped_time = None  # 루틴 종료 시간 초기화
                        self.get_logger().info(
                            f"HANDSHAKE 상태로 전환: 루틴 발행 시간 기록, "
                            f"{self.handshake_feedback_delay}초 후 루틴 종료 확인 시작"
                        )
                    
                    if target_id is not None:
                        self.target_track_id = int(target_id)
                        # Auto 모드에서는 target_explicitly_set을 False로 유지
                        # Manual 모드에서만 True로 설정
                        self.target_explicitly_set = self.manual_mode
                    
                    self.get_logger().info(
                        f"Depth 기반 상태 전환: {target_state_str} → {self.state.value} "
                        f"(모드: {'Manual' if self.manual_mode else 'Auto'}, "
                        f"target_explicitly_set={self.target_explicitly_set})"
                    )
                except (KeyError, AttributeError) as e:
                    self.get_logger().error(f"잘못된 상태: {target_state_str}")
                return
            
            # 기존 set_state 요청 처리
            state_str = request.get('state', 'idle')
            try:
                state = TrackingState[state_str.upper()]
                
                # 상태 변경 요청은 manual_mode와 관계없이 처리 (자동 전환)
                # Controller에서 요청한 상태 전환은 항상 허용
                # 상태 변경 (manual_mode 체크 없이)
                self.state = state
                
                # HELLO/HANDSHAKE 상태로 전환 시 변수 초기화 (매번 리셋)
                if state == TrackingState.HELLO:
                    self.hello_routine_sent_time = time.monotonic()
                    self.hello_routine_sent = True
                    self.current_routine_running = True  # 루틴 시작 시 실행 중으로 가정
                    self.routine_stopped_time = None  # 루틴 종료 시간 초기화
                    self.get_logger().info(
                        f"HELLO 상태로 전환: 루틴 발행 시간 기록, "
                        f"{self.hello_feedback_delay}초 후 루틴 종료 확인 시작"
                    )
                elif state == TrackingState.HANDSHAKE:
                    self.handshake_routine_sent_time = time.monotonic()
                    self.handshake_routine_sent = True
                    self.current_routine_running = True  # 루틴 시작 시 실행 중으로 가정
                    self.routine_stopped_time = None  # 루틴 종료 시간 초기화
                    self.get_logger().info(
                        f"HANDSHAKE 상태로 전환: 루틴 발행 시간 기록, "
                        f"{self.handshake_feedback_delay}초 후 루틴 종료 확인 시작"
                    )
                
                if target_id is not None:
                    self.target_track_id = int(target_id)
                    self.target_explicitly_set = True
                elif state != TrackingState.TRACKING:
                    self.target_track_id = None
                    self.target_explicitly_set = False
                
                self.get_logger().info(f"상태 변경 요청 수신: {state_str} → {self.state.value}")
            except (KeyError, AttributeError) as e:
                self.get_logger().error(f"잘못된 상태: {state_str}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"상태 변경 요청 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"상태 변경 요청 처리 실패: {e}")
    
    def _routine_status_callback(self, msg: String):
        """루틴 상태 피드백 콜백 - /debug/routine 토픽에서 루틴 실행 상태 확인"""
        try:
            data = json.loads(msg.data)
            self.last_routine_status_time = time.monotonic()
            
            # 루틴이 비어있는지 확인 (루틴 종료 상태)
            nodes = data.get("nodes", [])
            if not nodes or len(nodes) == 0:
                # 루틴이 비어있으면 종료된 것으로 간주
                # 이전에 실행 중이었는데 지금 종료된 경우 routine_stopped_time 기록
                if self.current_routine_running:
                    self.routine_stopped_time = time.monotonic()
                # current_routine_running이 이미 False인 경우에도 routine_stopped_time이 None이면 현재 시간으로 설정
                # (handshake 상태에서 루틴 종료 확인을 위해)
                elif self.routine_stopped_time is None and (self.state == TrackingState.HELLO or self.state == TrackingState.HANDSHAKE):
                    self.routine_stopped_time = time.monotonic()
                self.current_routine_running = False
                if self.state == TrackingState.HELLO or self.state == TrackingState.HANDSHAKE:
                    self.get_logger().info(
                        f"[루틴 상태 수신] 루틴이 비어있음 (종료됨), "
                        f"현재 State={self.state.value}"
                    )
                return
            
            # 루트 노드 찾기 (parent == -1)
            root_node = None
            for node in nodes:
                if node.get("parent") == -1:
                    root_node = node
                    break
            
            if root_node:
                status = root_node.get("status")
                # status: 0=IDLE, 1=RUNNING, 2=SUCCESS, 3=FAILURE
                is_running = (status == 1)  # RUNNING인 경우만 실행 중으로 판단
                
                # 루틴이 종료된 것으로 변경되었는지 확인 (RUNNING -> IDLE/SUCCESS/FAILURE)
                if self.current_routine_running and not is_running:
                    self.routine_stopped_time = time.monotonic()
                    self.get_logger().info(
                        f"[루틴 상태 변화] 루틴 종료 감지: 상태={{{0: 'IDLE', 1: 'RUNNING', 2: 'SUCCESS', 3: 'FAILURE'}.get(status, 'Unknown')}}, "
                        f"종료 시간 기록: {self.routine_stopped_time:.2f}"
                    )
                
                self.current_routine_running = is_running
                
                if self.state == TrackingState.HELLO or self.state == TrackingState.HANDSHAKE:
                    status_str = {0: "IDLE", 1: "RUNNING", 2: "SUCCESS", 3: "FAILURE"}.get(status, f"Unknown({status})")
                    self.get_logger().info(
                        f"[루틴 상태 수신] 루트 노드 상태={status_str} ({status}), "
                        f"루틴 실행 중={is_running}, "
                        f"현재 State={self.state.value}, "
                        f"노드 이름={root_node.get('name', 'Unknown')}"
                    )
            else:
                # 루트 노드를 찾을 수 없으면 루틴이 없는 것으로 간주
                self.current_routine_running = False
                if self.state == TrackingState.HELLO or self.state == TrackingState.HANDSHAKE:
                    self.get_logger().warn(
                        f"[루틴 상태 수신] 루트 노드를 찾을 수 없음 (루틴 종료로 간주), "
                        f"현재 State={self.state.value}"
                    )
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"루틴 상태 피드백 JSON 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().warn(f"루틴 상태 피드백 처리 실패: {e}")
    
    def _camera_info_callback(self, msg: CameraInfo):
        """카메라 정보 콜백 - 캘리브레이션 파라미터 저장"""
        if self.camera_info is None:  # 한 번만 저장
            self.camera_info = msg
            self.get_logger().info(f"카메라 정보 수신: 해상도={msg.width}x{msg.height}")
    
    def _depth_image_callback(self, msg: Image):
        """Depth 이미지 콜백 - 최신 depth 이미지 저장"""
        try:
            # 16UC1 형식의 depth 이미지를 numpy 배열로 변환
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_image = depth_image
            self.latest_depth_image_shape = (depth_image.shape[0], depth_image.shape[1])  # (height, width)
        except Exception as e:
            self.get_logger().warn(f"Depth 이미지 변환 실패: {e}")
    
    def _extract_depth_from_detection(self, detections: List[Dict], frame_shape: tuple) -> None:
        """Detection 결과에서 depth 값을 추출하여 저장
        
        Args:
            detections: Detection 결과 리스트
            frame_shape: Color 이미지 프레임 크기 (height, width)
        """
        if self.latest_depth_image is None:
            return  # depth 이미지가 없으면 스킵
        
        depth_image = self.latest_depth_image
        depth_shape = self.latest_depth_image_shape
        
        if depth_shape is None:
            return
        
        # Color 이미지와 Depth 이미지의 해상도 차이 계산
        color_height, color_width = frame_shape
        depth_height, depth_width = depth_shape
        
        scale_x = depth_width / color_width
        scale_y = depth_height / color_height
        
        for det in detections:
            track_id = det.get('track_id')
            centroid = det.get('centroid')
            bbox = det.get('bbox')
            
            if track_id is None or centroid is None or bbox is None:
                continue
            
            centroid_x, centroid_y = centroid
            
            # Color 이미지 좌표를 Depth 이미지 좌표로 변환
            depth_center_x = int(centroid_x * scale_x)
            depth_center_y = int(centroid_y * scale_y)
            
            # 중심점 주변 작은 영역 (5x5 픽셀)에서 depth 값 샘플링
            half_size = 2
            x_min = max(0, depth_center_x - half_size)
            x_max = min(depth_image.shape[1], depth_center_x + half_size + 1)
            y_min = max(0, depth_center_y - half_size)
            y_max = min(depth_image.shape[0], depth_center_y + half_size + 1)
            
            depth_roi = depth_image[y_min:y_max, x_min:x_max]
            valid_depths = depth_roi[depth_roi > 0]  # 0은 무효한 depth
            
            if len(valid_depths) == 0:
                continue  # 유효한 depth가 없으면 스킵
            
            # 중앙값 사용 (노이즈에 강함)
            depth_value = np.median(valid_depths)
            
            # Depth 단위: RealSense의 경우 보통 mm 단위이므로 m로 변환
            depth_m = depth_value / 1000.0  # mm -> m
            
            # 저장
            old_depth = self.target_depth_map.get(track_id)
            self.target_depth_map[track_id] = depth_m
            
            # 디버깅: depth 값 업데이트 로그 (값이 변경되었을 때만)
            if old_depth is None or abs(old_depth - depth_m) > 0.1:  # 새 값이거나 10cm 이상 변경 시
                self.get_logger().info(
                    f"[Depth 추출] track_id={track_id}, "
                    f"depth={depth_m:.3f}m ({depth_m*1000:.1f}mm), "
                    f"color_centroid=({centroid_x:.1f}, {centroid_y:.1f}), "
                    f"depth_centroid=({depth_center_x}, {depth_center_y}), "
                    f"scale=({scale_x:.3f}, {scale_y:.3f}), "
                    f"color_shape={frame_shape}, depth_shape={depth_shape}"
                )


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
