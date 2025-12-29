#!/usr/bin/env python3
"""
목/허리 제어 노드 - 모든 로직과 통신을 한 파일에 통합
추적 결과를 받아서 로봇에 명령을 전송
"""
import json
import math
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from std_msgs.msg import String, Float64MultiArray
import time
from typing import Optional, Tuple, Dict

from .tracking_fsm_node import TargetInfo, TrackingState


class GazeControllerNode(Node):
    """목/허리 제어 노드 - 모든 로직과 통신 통합"""
    
    def __init__(self):
        super().__init__('gaze_controller_neck_waist_node')
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 추적 결과 구독
        self.tracking_result_subscription = self.create_subscription(
            String,
            "/allex_camera/tracking_result",
            self.tracking_result_callback,
            qos_profile
        )
        
        # 제어 명령 구독
        self.control_subscription = self.create_subscription(
            String,
            "/allex_camera/controller_control",
            self._control_callback,
            10
        )
        
        # 목 명령 Publisher
        self.neck_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_neck/joint_command',
            10
        )
        
        # 목 위치 Subscriber (현재 위치 파악용) - BEST_EFFORT QoS 사용
        self.neck_position_subscription = self.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_neck/joint_positions_deg',
            self._neck_position_callback,
            qos_profile
        )
        
        # 허리 명령 Publisher
        self.waist_publisher = self.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_waist/joint_command',
            10
        )
        
        # 허리 위치 Subscriber (현재 위치 파악용) - BEST_EFFORT QoS 사용
        self.waist_position_subscription = self.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_waist/joint_positions_deg',
            self._waist_position_callback,
            qos_profile
        )
        
        # 목 각도 발행 (Tracker 노드에서 사용)
        self.neck_angle_publisher = self.create_publisher(
            String,
            "/allex_camera/neck_angle",
            10
        )
        
        # Tracker 상태 변경 요청 Publisher (HELLO/HANDSHAKE 전이용)
        self.tracker_state_request_publisher = self.create_publisher(
            String,
            "/allex_camera/tracker_state_request",
            10
        )
        
        # Controller 제어 구독 (파라미터 설정 등)
        self.controller_control_subscription = self.create_subscription(
            String,
            "/allex_camera/controller_control",
            self._controller_control_callback,
            10
        )
        
        # 목 각도 발행 타이머 (30Hz)
        self.neck_angle_timer = self.create_timer(1.0 / 30.0, self._publish_neck_angle)
        
        # 제어 루프 타이머 (60Hz) - 타겟은 30Hz로 들어오지만 제어는 60Hz로 실행하여 더 부드럽게
        self.control_timer = self.create_timer(1.0 / 60.0, self._control_loop)
        
        # 타겟 정보 저장 (제어 루프에서 사용)
        self.last_target_info = None
        self.last_frame_width = 1280.0
        self.last_frame_height = 720.0
        self.last_target_data = None  # HELLO 체크용
        
        # 카메라 파라미터 (프레임 크기 기준)
        self.frame_width = 1280.0  # 프레임 너비 (픽셀)
        self.frame_height = 720.0  # 프레임 높이 (픽셀)
        
        # 각도 제한 범위 (라디안)
        self.yaw_min = math.radians(-65.0)    # -65° (SEARCHING용)
        self.yaw_max = math.radians(65.0)     # 65° (SEARCHING용)
        self.pitch_min = -0.0872665  # -5°
        self.pitch_max = 3.75246   # 215°
        
        # 허리 각도 제한 범위 (라디안)
        self.waist_yaw_min = math.radians(-85.0)  # -85°
        self.waist_yaw_max = math.radians(85.0)   # 85°
        
        # 현재 목 각도 (라디안) - 하드웨어에서 받은 실제 위치
        self.current_yaw_rad = 0.0
        self.current_pitch_rad = 0.0
        self.last_position_update_time = 0.0
        
        # 목표 명령 각도 (라디안) - 마지막으로 전송한 목표 각도
        self.target_yaw_rad = 0.0
        self.target_pitch_rad = 0.0
        
        # 목 목표 각도 스무딩용 (LOST -> TRACKING 전환 시 급격한 변화 방지)
        self.last_neck_target_yaw = None  # 마지막 목 목표 각도 (None이면 초기화 필요)
        
        # PID 제어 결과 스무딩용 (부드러운 움직임을 위해)
        self.last_neck_delta_yaw = 0.0  # 마지막 PID 제어 증분
        self.last_neck_delta_pitch = 0.0
        self.smoothing_alpha = 0.3  # Exponential smoothing 계수 (0~1, 작을수록 더 부드럽지만 느림)
        self.last_sent_neck_delta_yaw = 0.0  # 마지막으로 실제 전송한 목 증분 (추가 스무딩용)
        self.last_sent_neck_delta_pitch = 0.0  # 마지막으로 실제 전송한 목 증분 (추가 스무딩용)
        
        # 전체 시선각 스무딩용 (부드러운 움직임을 위해)
        self.last_desired_total_yaw = None  # 마지막 목표 전체 시선각
        self.total_yaw_smoothing_alpha = 0.6  # 전체 시선각 스무딩 계수 (0.4 -> 0.6: 30% 더 빠르게)
        
        # 영자세 (중앙 위치) - 절대 좌표 기준점
        self.home_yaw_rad = 0.0
        self.home_pitch_rad = 0.0
        self.left_right_angle = 40.0
        
        # IDLE 상태용 영자세 복귀 변수
        self.idle_return_start_time = None  # IDLE 상태 진입 시각
        self.idle_return_start_yaw = None  # IDLE 상태 진입 시 목 Yaw 위치
        self.idle_return_start_pitch = None  # IDLE 상태 진입 시 목 Pitch 위치
        self.idle_return_start_waist_yaw = None  # IDLE 상태 진입 시 허리 Yaw 위치
        self.idle_return_duration = 7.0  # 영자세 복귀 소요 시간 (7초)
        
        # SEARCHING 상태용 스캔 변수
        self.searching_start_time = None
        self.search_phase = 0  # 0: 우측(+40도)로, 1: 좌측(-40도)로
        self.search_target_yaw = 0.0  # 최종 목표 각도 (절대 각도)
        self.search_current_command_yaw = 0.0  # 현재 명령 각도 (증분 방식용)
        self.search_increment_rad = math.radians(0.3)  # 매 프레임마다 증가할 각도 (약 0.3도)
        
        # SEARCHING 타겟 변경 조건용 변수 (새로운 방식: 목과 허리 독립 제어)
        self.search_neck_target_yaw = None  # 목 목표 각도 (고정값)
        self.search_waist_target_yaw = None  # 허리 목표 각도 (고정값)
        self.search_neck_last_cmd = None  # 목 exponential smoothing용
        self.search_waist_last_cmd = None  # 허리 exponential smoothing용
        self.search_neck_last_time = None  # 목 시간 추적용
        self.search_waist_last_time = None  # 허리 시간 추적용
        self.search_waist_phase_start_time = None  # 허리 Phase 시작 시간 (초반 속도 제어용)
        self.tau_searching_neck = 1.09  # 목 exponential smoothing 시간 상수 (초) - 10% 빠르게
        self.tau_searching_waist = 0.6  # 허리 exponential smoothing 시간 상수 (초) - 20% 느리게
        self.waist_smoothing_factor = 0.85  # 허리 추가 스무딩 계수 (0.85 = 15% 스무딩)
        self.waist_initial_slow_duration = 1.5  # 허리 초반 느린 속도 지속 시간 (초)
        self.waist_initial_smoothing_factor = 0.6  # 허리 초반 스무딩 계수 (더 느리게)
        self.search_phase_timeout = 15.0  # Phase 타임아웃 (초) - 이 시간이 지나면 강제로 Phase 변경
        
        # PID 제어 파라미터 (일반 추적용, 12% 속도 증가)
        self.kp_yaw = 0.95  # P 게인 (Yaw) - 1.1 * 1.12 = 1.232 (12% 증가)
        self.kp_pitch = 1.15 # P 게인 (Pitch) - 1.2 * 1.12 = 1.344 (12% 증가)
        self.ki_yaw = 0.01    # I 게인 (Yaw) - 0.02 * 1.12 = 0.0224 (12% 증가)
        self.ki_pitch = 0.1344  # I 게인 (Pitch) - 0.12 * 1.12 = 0.1344 (12% 증가)
        self.kd_yaw = 0.0   # D 게인 (Yaw) - 낮춰서 움직임 억제 감소
        self.kd_pitch = 0.01 # D 게인 (Pitch)
        
        
        # SEARCHING 상태용 게인 (목 속도 감소)
        self.kp_yaw_searching = 0.30   # P 게인 (Yaw) - 검색 시 (속도 감소: 0.45 -> 0.30)
        self.kp_pitch_searching = 0.30 # P 게인 (Pitch) - 검색 시 (속도 감소: 0.45 -> 0.30)
        self.ki_yaw_searching = 0.02  # I 게인 (Yaw) - 검색 시
        self.ki_pitch_searching = 0.02 # I 게인 (Pitch) - 검색 시
        self.kd_yaw_searching = 0.01  # D 게인 (Yaw) - 검색 시
        self.kd_pitch_searching = 0.01 # D 게인 (Pitch) - 검색 시
        
        # PID 제어 상태 변수
        self.integral_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_yaw = 0.0
        self.last_error_pitch = 0.0
        self.last_update_time = time.monotonic()
        
        # 허리 제어 변수
        self.current_waist_yaw_rad = 0.0  # 현재 허리 각도 (라디안, 절대 좌표)
        self.last_waist_position_update_time = 0.0
        
        # TRACKING 상태에서 허리 추종용 P 게인 (P 게인만 사용)
        self.kp_waist_tracking = 1.8 # P 게인 (Waist Yaw) - 진동 방지 (20% 증가: 1.5 -> 1.8)
        
        # 허리 지연 파라미터 (Exponential smoothing 시간 상수, 초)
        # 허리 20% 빠르게: tau를 20% 줄임
        self.tau_waist = 0.15  # TRACKING용 지연 (50% 속도: 0.067 * 2 = 0.134초)
        self.tau_waist_searching = 1.5  # SEARCHING 모드용 지연 (40% 빠르게: 3.5 -> 2.1)
        self.max_delta_waist_tracking = math.radians(1.65)  # TRACKING 모드 허리 최대 변화량 (50% 속도: 3.3 * 0.5 = 1.65도)
        
        # 허리 제어 상태 변수
        self.last_waist_update_time = time.monotonic()
        self.last_waist_command = None  # 마지막으로 보낸 허리 명령 각도 (절대각, None이면 초기화 필요)
        self.last_sent_waist_command = None  # 마지막으로 실제 전송한 허리 명령 각도 (추가 스무딩용)
        
        # 상시 허리 Pitch sin 파형 움직임 (모든 상태에서 적용, 숨쉬는 것처럼, Yaw는 사람 추종)
        self.waist_breathe_start_time = time.monotonic()  # sin 파형 시작 시간
        self.waist_breathe_amplitude = math.radians(12.0)  # 진폭 15도 (총 30도 범위)
        self.waist_breathe_offset = math.radians(-7.0)  # 오프셋 -8도 (정면 7도 ~ 뒤로 23도)
        self.waist_breathe_period = 7.0  # sin 파형 주기 (초) - 6초 주기로 완만하게 움직임
        
        # TRACKING ROI 영역 변수 (좌우 끝 영역 제한)
        self.tracking_roi_left_margin = 0.15  # 좌측 마진 (화면 너비의 15%)
        self.tracking_roi_right_margin = 0.15  # 우측 마진 (화면 너비의 15%)
        # ROI 영역: [left_margin * width, (1 - right_margin) * width]
        # 예: 1280x720 화면에서 [192, 1088] 픽셀 영역만 사용
        
        # 현재 추적 중인 track_id (Track ID 변경 감지용)
        self.current_track_id = None
        
        # Manual 모드 추적 (tracking_result에서 받아옴)
        self.manual_mode = False
        
        # 실행 상태 플래그
        self.is_running = False
        
        # HELLO 전환 조건 변수 (현재 위치에서 ±2도 이내로 2초 유지)
        self.hello_position_threshold_deg = 2.0  # 기준 위치에서 ±2도 이내
        self.hello_stable_duration = 2.0  # 조건 유지 시간 (초)
        self.hello_stable_start_time = None  # 조건 만족 시작 시간
        # 상태 추적 분리 (HELLO 체크와 LOST 진입 감지 분리)
        self.prev_state_for_hello = None  # HELLO 전환 체크용
        self.prev_state_for_lost = None  # LOST 진입 감지용
        
        # 이미 HELLO를 한 track_id 저장 (중복 HELLO 방지)
        self.hello_done_track_ids = set()
        
        # LOST 상태 지수 감쇠 변수 (Exponential Deceleration)
        self.lost_start_time = None  # LOST 상태 진입 시간
        self.lost_last_time = None  # 마지막 LOST 업데이트 시간
        
        # 마지막으로 전송한 명령 위치 (절대 라디안) - 매 명령 전송 시 업데이트
        self.last_cmd_neck_yaw: Optional[float] = None
        self.last_cmd_neck_pitch: Optional[float] = None
        self.last_cmd_waist_yaw: Optional[float] = None
        
        # 이전 명령 위치 (속도 추정용)
        self.prev_cmd_neck_yaw: Optional[float] = None
        self.prev_cmd_neck_pitch: Optional[float] = None
        self.prev_cmd_waist_yaw: Optional[float] = None
        
        # 명령 전송 시간 추적 (목/허리 분리)
        self.last_cmd_time_neck: Optional[float] = None
        self.prev_cmd_time_neck: Optional[float] = None
        self.last_cmd_time_waist: Optional[float] = None
        self.prev_cmd_time_waist: Optional[float] = None
        
        # LOST 진입 시 캡처된 초기 속도 (rad/s)
        self.lost_v0_neck_yaw: float = 0.0
        self.lost_v0_neck_pitch: float = 0.0
        self.lost_v0_waist_yaw: float = 0.0
        
        # LOST 감속을 위한 마지막 타겟 위치 (TRACKING 상태에서 저장)
        self.lost_last_target_yaw: Optional[float] = None
        self.lost_last_target_pitch: Optional[float] = None
        self.lost_last_waist_target_yaw: Optional[float] = None
        
        # LOST 지수 감쇠 파라미터 (목표 위치 기반 exponential smoothing)
        # 얼굴이 목표 위치로 초당 3~5도씩 이동하도록 설정
        # alpha = dt / (tau + dt), dt ≈ 0.033초 (30Hz)일 때
        # 초당 속도 = alpha * distance / dt
        # 초당 3-5도 = alpha * distance / 0.033
        # 평균 거리 30도 기준: alpha * 30 / 0.033 = 3-5 → alpha = 0.0033-0.0055
        # alpha = 0.004 (중간값)일 때: 0.004 = 0.033 / (tau + 0.033) → tau ≈ 8.2초
        # alpha = 0.005 (빠른 쪽)일 때: 0.005 = 0.033 / (tau + 0.033) → tau ≈ 6.6초
        # alpha = 0.0033 (느린 쪽)일 때: 0.0033 = 0.033 / (tau + 0.033) → tau ≈ 10초
        # 평균적으로 tau = 8초 정도면 초당 약 4도씩 이동
        # 사용자 요청: 3배 빠르게 → tau를 1/3로 줄임 → 3.7초
        # 추가 요청: 2배 더 빠르게 → 3.7 / 2 = 1.85초
        self.tau_neck_lost = 1.5  # 얼굴 exponential smoothing 시간 상수 (초) - 초당 약 15도씩 이동 (6배 빠름)
        self.tau_waist_lost = 1.8  # 허리 전용 exponential smoothing 시간 상수 (초) - 허리가 목보다 더 천천히 멈춤
        
        # LOST용 rate limit (프레임당 최대 변화량) - exponential smoothing이 제대로 작동하도록 충분히 크게 설정
        # exponential smoothing 자체가 속도를 제한하므로, rate limit은 안전장치로만 사용 (매우 크게 설정)
        self.max_delta_neck_lost = math.radians(10.0)   # 목 yaw - exponential smoothing이 제한하므로 충분히 크게 설정
        self.max_delta_pitch_lost = math.radians(10.0)  # 목 pitch - exponential smoothing이 제한하므로 충분히 크게 설정
        self.max_delta_waist_lost = math.radians(0.4)  # 허리 (더 느리게, 부드러운 감속을 위해 더 작게)
        
        # 속도 클램프 상한 (비정상 값 방지)
        self.vmax_neck = math.radians(40.0)   # 목 최대 속도 (도/초)
        self.vmax_waist = math.radians(10.0)  # 허리 최대 속도 (도/초)
        
        # ===== Neck 절대각 기반 제어 파라미터 =====
        # Neck 속도 비율 (허리 대비 1.5~2.0배, 기본 1.8배)
        self.neck_speed_ratio = 1.35  # 1.5~2.0 범위 조절 가능
        
        # Neck 명령 스무딩 시간 상수 (초)
        self.tau_neck_cmd = 0.30  # 기본값: 0.30초 (TRACKING 기준, overshooting 방지를 위해 더 증가)
        self.tau_neck_cmd_searching = 0.214  # SEARCHING 모드 (70% 속도: 0.15 / 0.7 = 0.214초)
        self.tau_neck_cmd_lost = 0.20  # LOST 모드 (기존 tau_neck_lost와 별도)
        self.tau_neck_cmd_idle = 0.25  # IDLE 모드
        
        # Neck 최대 속도 상한 (rad/s) - 안전장치
        self.neck_max_rate_yaw_cap = math.radians(25.0)  # 초당 12도
        self.neck_max_rate_pitch_cap = math.radians(5.0)  # 초당 5도
        
        # Neck 명령 추적 변수 (publish_neck_abs용)
        self.last_neck_cmd_yaw_abs: Optional[float] = None  # 마지막 발행한 목 Yaw 절대각
        self.last_neck_cmd_pitch_abs: Optional[float] = None  # 마지막 발행한 목 Pitch 절대각
        self.last_neck_cmd_time: Optional[float] = None  # 마지막 명령 전송 시간
        
        self.get_logger().info("Gaze Controller Node 초기화 완료")
    
    def _neck_position_callback(self, msg: Float64MultiArray):
        """목 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트"""
        if len(msg.data) >= 2:
            pitch_deg = msg.data[0]
            yaw_deg = msg.data[1]
            self.current_pitch_rad = math.radians(pitch_deg)
            self.current_yaw_rad = math.radians(yaw_deg)
            self.last_position_update_time = time.monotonic()
            # 디버깅: 처음 몇 번만 로그 (너무 많이 찍히지 않도록)
            if self.last_position_update_time < 2.0:  # 처음 2초간만
                self.get_logger().debug(
                    f"목 위치 수신: pitch={pitch_deg:.2f}°, yaw={yaw_deg:.2f}°"
                )
        else:
            self.get_logger().warn(
                f"Invalid neck position message: data length={len(msg.data)} (expected >= 2)"
            )
    
    def _waist_position_callback(self, msg: Float64MultiArray):
        """허리 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트"""
        if len(msg.data) >= 1:
            yaw_deg = msg.data[0]
            self.current_waist_yaw_rad = math.radians(yaw_deg)
            self.last_waist_position_update_time = time.monotonic()
        else:
            self.get_logger().warn(
                f"Invalid waist position message: data length={len(msg.data)} (expected >= 1)"
            )
    
    def _pixel_to_angle(self, target_x: float, target_y: float, frame_width: float, frame_height: float) -> Tuple[float, float]:
        """타겟 픽셀 좌표를 목 각도로 변환"""
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        
        offset_x = target_x - center_x  # 양수: 우측, 음수: 좌측
        offset_y = target_y - center_y  # 양수: 하단, 음수: 상단
        
        horizontal_fov_deg = 120.0
        vertical_fov_deg = 45.0
        
        yaw_deg = (offset_x / frame_width) * horizontal_fov_deg
        pitch_deg = (offset_y / frame_height) * vertical_fov_deg
        
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
    
    def _maintain_current_position(self) -> Tuple[float, float]:
        """현재 목 위치 유지"""
        return self.current_yaw_rad, self.current_pitch_rad
    
    def _searching_behavior(self) -> Tuple[float, float]:
        """SEARCHING 상태: 목과 허리를 독립적으로 제어하여 스캔 (새로운 방식)
        - 목: 10~50도 (우측), -10~-50도 (좌측)
        - 허리: 0~20도 (우측), 0~-20도 (좌측)
        - 둘은 같은 방향으로 이동
        - 각각 exponential smoothing으로 천천히 이동
        - 둘 다 1도 이내 오차범위 내로 들어오면 다음 Phase로 전환
        """
        # SEARCHING 상태 진입 시 초기화
        if self.searching_start_time is None:
            self.searching_start_time = time.monotonic()
            # Phase 0: 우측, Phase 1: 좌측
            self.search_phase = 0
            # 목표 각도 초기화
            self.search_neck_target_yaw = None
            self.search_waist_target_yaw = None
            self.search_neck_last_cmd = None
            self.search_waist_last_cmd = None
            self.search_neck_last_time = None
            self.search_waist_last_time = None
        
        # 현재 시간
        current_time = time.monotonic()
        
        # Phase 변경 감지
        phase_changed = False
        if not hasattr(self, '_prev_search_phase_for_target'):
            self._prev_search_phase_for_target = self.search_phase
            phase_changed = True
        elif self._prev_search_phase_for_target != self.search_phase:
            phase_changed = True
            self._prev_search_phase_for_target = self.search_phase
        
        # Phase 변경 시 목표 각도 설정 및 초기화
        if phase_changed or self.search_neck_target_yaw is None or self.search_waist_target_yaw is None:
            if self.search_phase == 0:
                # 우측 방향: 목 20~50도, 허리 0~35도
                self.search_neck_target_yaw = math.radians(random.uniform(20.0, 50.0))
                self.search_waist_target_yaw = math.radians(random.uniform(0.0, 35.0))
            else:
                # 좌측 방향: 목 -20~-50도, 허리 0~-35도
                self.search_neck_target_yaw = math.radians(random.uniform(-50.0, -20.0))
                self.search_waist_target_yaw = math.radians(random.uniform(-35.0, 0.0))
            
            # 각도 제한
            self.search_neck_target_yaw = max(self.yaw_min, min(self.yaw_max, self.search_neck_target_yaw))
            self.search_waist_target_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, self.search_waist_target_yaw))
            
            # Exponential smoothing 초기화 (현재 센서 각도로 동기화)
            self.search_neck_last_cmd = self.current_yaw_rad
            self.search_waist_last_cmd = self.current_waist_yaw_rad
            self.search_neck_last_time = current_time
            self.search_waist_last_time = current_time
            self.search_waist_phase_start_time = current_time  # Phase 시작 시간 기록
        
        # 목 exponential smoothing
        dt_neck = current_time - self.search_neck_last_time if self.search_neck_last_time is not None else 0.033
        alpha_neck = dt_neck / (self.tau_searching_neck + dt_neck)
        neck_cmd = self.search_neck_last_cmd + alpha_neck * (self.search_neck_target_yaw - self.search_neck_last_cmd)
        neck_cmd = max(self.yaw_min, min(self.yaw_max, neck_cmd))
        self.search_neck_last_cmd = neck_cmd
        self.search_neck_last_time = current_time
        
        # 허리 exponential smoothing (추가 스무딩 적용, 초반에는 더 느리게)
        dt_waist = current_time - self.search_waist_last_time if self.search_waist_last_time is not None else 0.033
        alpha_waist = dt_waist / (self.tau_searching_waist + dt_waist)
        waist_cmd_raw = self.search_waist_last_cmd + alpha_waist * (self.search_waist_target_yaw - self.search_waist_last_cmd)
        
        # 초반 느린 속도 적용 (Phase 시작 후 일정 시간 동안)
        phase_elapsed_time = current_time - self.search_waist_phase_start_time if self.search_waist_phase_start_time is not None else 0.0
        if phase_elapsed_time < self.waist_initial_slow_duration:
            # 초반: 더 강한 스무딩 적용
            smoothing_factor = self.waist_initial_smoothing_factor
        else:
            # 이후: 일반 스무딩 적용
            smoothing_factor = self.waist_smoothing_factor
        
        # 추가 스무딩 적용 (부드러운 움직임)
        waist_cmd = self.search_waist_last_cmd + smoothing_factor * (waist_cmd_raw - self.search_waist_last_cmd)
        waist_cmd = max(self.waist_yaw_min, min(self.waist_yaw_max, waist_cmd))
        self.search_waist_last_cmd = waist_cmd
        self.search_waist_last_time = current_time
        
        # 목표 도달 판정 (1도 이내)
        target_reached_threshold = math.radians(1.0)
        neck_error = abs(self.search_neck_target_yaw - self.current_yaw_rad)
        waist_error = abs(self.search_waist_target_yaw - self.current_waist_yaw_rad)
        
        # Phase 경과 시간 확인
        phase_elapsed_time = current_time - self.search_waist_phase_start_time if self.search_waist_phase_start_time is not None else 0.0
        timeout_reached = phase_elapsed_time >= self.search_phase_timeout
        
        # 둘 다 도착하거나 타임아웃이 되면 다음 Phase로 전환
        if (neck_error <= target_reached_threshold and waist_error <= target_reached_threshold) or timeout_reached:
            if timeout_reached:
                self.get_logger().warn(
                    f"[SEARCHING Phase 타임아웃] {phase_elapsed_time:.1f}초 경과, "
                    f"neck_error={math.degrees(neck_error):.2f}도, "
                    f"waist_error={math.degrees(waist_error):.2f}도 - 강제 Phase 변경"
                )
            
            if self.search_phase == 0:
                # 우측 -> 좌측
                self.search_phase = 1
            else:
                # 좌측 -> 우측
                self.search_phase = 0
            
            # 새로운 목표 각도 설정을 위해 초기화
            self.search_neck_target_yaw = None
            self.search_waist_target_yaw = None
        
        # SEARCHING 시 목 Pitch를 6.5도 기울임
        searching_pitch_rad = self.home_pitch_rad + math.radians(6.5)
        
        # 목과 허리 명령 각도 반환 (실제로는 SEARCHING 상태에서 직접 명령을 내리므로 사용되지 않을 수 있음)
        # 하지만 호환성을 위해 전체 시선각으로 변환하여 반환
        total_target_yaw = neck_cmd + waist_cmd
        
        return total_target_yaw, searching_pitch_rad
    
    def _pid_control(self, target_yaw_rad: float, target_pitch_rad: float, use_searching_gain: bool = False, use_initial_gain: bool = False) -> Tuple[float, float]:
        """PID 제어를 사용하여 목 증분 명령 계산"""
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))
        
        if use_initial_gain:
            kp_yaw = self.kp_yaw_initial
            kp_pitch = self.kp_pitch_initial
            ki_yaw = self.ki_yaw_initial
            ki_pitch = self.ki_pitch_initial
            kd_yaw = self.kd_yaw_initial
            kd_pitch = self.kd_pitch_initial
        elif use_searching_gain:
            kp_yaw = self.kp_yaw_searching
            kp_pitch = self.kp_pitch_searching
            ki_yaw = self.ki_yaw_searching
            ki_pitch = self.ki_pitch_searching
            kd_yaw = self.kd_yaw_searching
            kd_pitch = self.kd_pitch_searching
        else:
            kp_yaw = self.kp_yaw
            kp_pitch = self.kp_pitch
            ki_yaw = self.ki_yaw
            ki_pitch = self.ki_pitch
            kd_yaw = self.kd_yaw
            kd_pitch = self.kd_pitch
        
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_pitch = target_pitch_rad - self.current_pitch_rad
        
        p_yaw = kp_yaw * error_yaw
        p_pitch = kp_pitch * error_pitch
        
        self.integral_yaw += error_yaw * dt
        self.integral_pitch += error_pitch * dt
        
        max_integral = math.radians(60.0)  # 적분 제한 증가 (Steady State Error 제거)
        self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))
        self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
        
        # 초기 추적 시 PID 출력 디버깅
        if use_initial_gain:
            if not hasattr(self, '_last_pid_debug_time') or time.monotonic() - self._last_pid_debug_time > 0.1:
                self.get_logger().info(
                    f"[초기 추적 PID 계산] "
                    f"error_yaw={math.degrees(error_yaw):.3f}도, "
                    f"kp_yaw={kp_yaw:.3f}, "
                    f"p_yaw={math.degrees(p_yaw):.3f}도, "
                    f"i_yaw={math.degrees(self.integral_yaw * ki_yaw):.3f}도, "
                    f"dt={dt:.4f}초"
                )
                self._last_pid_debug_time = time.monotonic()
        
        i_yaw = ki_yaw * self.integral_yaw
        i_pitch = ki_pitch * self.integral_pitch
        
        d_error_yaw = (error_yaw - self.last_error_yaw) / dt
        d_error_pitch = (error_pitch - self.last_error_pitch) / dt
        
        d_yaw = kd_yaw * d_error_yaw
        d_pitch = kd_pitch * d_error_pitch
        
        delta_yaw_rad = p_yaw + i_yaw + d_yaw
        delta_pitch_rad = p_pitch + i_pitch + d_pitch
        
        self.last_error_yaw = error_yaw
        self.last_error_pitch = error_pitch
        self.last_update_time = current_time
        
        return delta_yaw_rad, delta_pitch_rad
    
    def _get_waist_breathe_pitch(self) -> float:
        """상시 허리 Pitch sin 파형 계산 (모든 상태에서 적용, 숨쉬는 것처럼)
        
        Returns:
            float: sin 파형 Pitch 각도 (라디안, 정면 7도 ~ 뒤로 23도 범위)
        """
        current_time = time.monotonic()
        elapsed_time = current_time - self.waist_breathe_start_time
        
        # sin 파형 계산: -1 ~ +1 범위를 15도 진폭으로 변환 후 -8도 오프셋 추가
        # 결과: -23도 ~ +7도 (정면 7도, 뒤로 23도)
        sin_value = math.sin(2.0 * math.pi * elapsed_time / self.waist_breathe_period)
        pitch = sin_value * self.waist_breathe_amplitude + self.waist_breathe_offset
        
        return pitch
    
    def _publish_neck_abs(self, target_pitch_abs_rad: float, target_yaw_abs_rad: float, mode: str = "TRACKING"):
        """목 절대각 명령 발행 (공통 출력 shaping 레이어)
        
        모든 상태에서 목 명령은 이 함수를 통해서만 발행하여 절대각 통일을 보장합니다.
        내부에서 (1) 하드 클립, (2) 스무딩, (3) rate limit(rad/s), (4) 발행을 수행합니다.
        
        Args:
            target_pitch_abs_rad: 목표 Pitch 절대각 (라디안)
            target_yaw_abs_rad: 목표 Yaw 절대각 (라디안)
            mode: 모드 ("TRACKING", "SEARCHING", "LOST", "IDLE", "WAITING", "HELLO", "HANDSHAKE")
        """
        current_time = time.monotonic()
        
        # 1) dt 계산
        if self.last_neck_cmd_time is not None:
            dt = current_time - self.last_neck_cmd_time
            dt = max(0.001, min(dt, 0.1))
        else:
            dt = 1.0 / 60.0  # 기본값: 30Hz
        
        # 2) 하드 클립 (각도 제한)
        target_pitch_clipped = max(self.pitch_min, min(self.pitch_max, target_pitch_abs_rad))
        target_yaw_clipped = max(self.yaw_min, min(self.yaw_max, target_yaw_abs_rad))
        
        # 3) 초기화: 첫 호출 시 last를 현재 센서 위치로 설정
        if self.last_neck_cmd_yaw_abs is None:
            self.last_neck_cmd_yaw_abs = self.current_yaw_rad
            self.last_neck_cmd_pitch_abs = self.current_pitch_rad
        
        # 4) 상태별 tau 선택
        if mode == "SEARCHING":
            tau = self.tau_neck_cmd_searching
        elif mode == "LOST":
            tau = self.tau_neck_cmd_lost
        elif mode == "IDLE":
            tau = self.tau_neck_cmd_idle
        else:  # TRACKING, HELLO, HANDSHAKE, WAITING
            tau = self.tau_neck_cmd
        
        # 5) Exponential smoothing
        alpha = dt / (tau + dt)
        smoothed_pitch = self.last_neck_cmd_pitch_abs + alpha * (target_pitch_clipped - self.last_neck_cmd_pitch_abs)
        smoothed_yaw = self.last_neck_cmd_yaw_abs + alpha * (target_yaw_clipped - self.last_neck_cmd_yaw_abs)
        
        # 6) Waist 기준 속도 산출 (TRACKING 모드의 허리 속도 기준)
        # 허리의 max_delta_waist_tracking를 dt로 나눠 rad/s로 환산
        # dt가 작을 때를 대비해 안전 상한 적용
        waist_ref_rate_yaw = self.max_delta_waist_tracking / dt
        waist_ref_rate_yaw = min(waist_ref_rate_yaw, math.radians(200.0))  # 초당 200도 상한
        
        # 7) Neck 최대 속도 = 허리 기준 속도 × 비율
        neck_max_rate_yaw = self.neck_speed_ratio * waist_ref_rate_yaw
        neck_max_rate_yaw = min(neck_max_rate_yaw, self.neck_max_rate_yaw_cap)
        
        # Pitch도 동일 비율 적용 (또는 별도 ratio 가능)
        neck_max_rate_pitch = self.neck_speed_ratio * waist_ref_rate_yaw
        neck_max_rate_pitch = min(neck_max_rate_pitch, self.neck_max_rate_pitch_cap)
        
        # 8) Rate limit 적용 (rad/s → 프레임당 변화량으로 변환)
        max_delta_yaw = neck_max_rate_yaw * dt  # overshooting 방지를 위해 절반으로 감소
        max_delta_pitch = neck_max_rate_pitch * dt
        
        delta_yaw = smoothed_yaw - self.last_neck_cmd_yaw_abs
        delta_pitch = smoothed_pitch - self.last_neck_cmd_pitch_abs
        
        delta_yaw = max(-max_delta_yaw, min(max_delta_yaw, delta_yaw))
        delta_pitch = max(-max_delta_pitch, min(max_delta_pitch, delta_pitch))
        
        # 최종 출력값
        out_yaw_abs = self.last_neck_cmd_yaw_abs + delta_yaw
        out_pitch_abs = self.last_neck_cmd_pitch_abs + delta_pitch
        
        # 하드 클립 재적용
        out_yaw_abs = max(self.yaw_min, min(self.yaw_max, out_yaw_abs))
        out_pitch_abs = max(self.pitch_min, min(self.pitch_max, out_pitch_abs))
        
        # 9) 발행
        msg = Float64MultiArray()
        msg.data = [float(out_pitch_abs), float(out_yaw_abs)]  # [pitch, yaw] 순서, 절대각
        self.neck_publisher.publish(msg)
        
        # 10) 상태 업데이트
        self.last_neck_cmd_yaw_abs = out_yaw_abs
        self.last_neck_cmd_pitch_abs = out_pitch_abs
        self.last_neck_cmd_time = current_time
        
        # 명령 위치 추적도 업데이트 (LOST 등 다른 로직과 호환성)
        self.last_cmd_neck_yaw = out_yaw_abs
        self.last_cmd_neck_pitch = out_pitch_abs
        self.last_cmd_time_neck = current_time
    
    def _send_waist_command(self, absolute_waist_yaw_rad: float, fixed_pitch: float = None):
        """허리 명령 전송 (작은 변화에 대한 추가 스무딩 적용 + sin 파형 Pitch 또는 고정 Pitch)
        
        Args:
            absolute_waist_yaw_rad: 허리 Yaw 각도 (라디안)
            fixed_pitch: 고정 Pitch 각도 (라디안). None이면 breathe_pitch 사용
        """
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        
        # 작은 각도 변화에 대한 추가 스무딩 (끊김 방지)
        if self.last_sent_waist_command is not None:
            delta = abs(absolute_waist_yaw_rad - self.last_sent_waist_command)
            # 작은 변화 (약 0.5도 이하)에 대해 더 부드럽게 처리
            small_change_threshold = math.radians(0.5)  # 약 0.5도
            if delta < small_change_threshold:
                # 작은 변화는 더 강한 스무딩 적용 (0.25 = 25%만 반영, 더 부드럽게)
                smoothing_alpha = 0.25
                absolute_waist_yaw_rad = self.last_sent_waist_command + smoothing_alpha * (absolute_waist_yaw_rad - self.last_sent_waist_command)
            else:
                # 큰 변화는 중간 스무딩 (0.55 = 55% 반영, 약간 더 부드럽게)
                smoothing_alpha = 0.55
                absolute_waist_yaw_rad = self.last_sent_waist_command + smoothing_alpha * (absolute_waist_yaw_rad - self.last_sent_waist_command)
        
        # 각도 제한 재적용
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        
        # Pitch 결정: fixed_pitch가 있으면 사용, 없으면 breathe_pitch 사용
        if fixed_pitch is not None:
            waist_pitch = fixed_pitch
        else:
            # sin 파형 Pitch 추가 (상시 움직임, Yaw는 사람 추종)
            waist_pitch = self._get_waist_breathe_pitch()
        
        msg = Float64MultiArray()
        msg.data = [float(absolute_waist_yaw_rad), float(waist_pitch)]  # [yaw, pitch] 순서
        self.waist_publisher.publish(msg)
        
        # 마지막 전송 명령 업데이트
        self.last_sent_waist_command = absolute_waist_yaw_rad
        
        # 명령 위치 추적 (LOST 지수 감쇠용)
        current_time_cmd = time.monotonic()
        # 이전 명령을 prev로 이동
        self.prev_cmd_waist_yaw = self.last_cmd_waist_yaw
        self.prev_cmd_time_waist = self.last_cmd_time_waist
        # 현재 명령을 last로 업데이트
        self.last_cmd_waist_yaw = absolute_waist_yaw_rad
        self.last_cmd_time_waist = current_time_cmd
    
    def _waist_follow_total(self, desired_total_yaw: float, searching_mode: bool = False) -> float:
        """허리가 전체 시선각(desired_total_yaw)을 느리게 추종
        
        Args:
            desired_total_yaw: 목표 전체 시선각 (절대각, 라디안) = waist_yaw + neck_yaw
            searching_mode: True면 SEARCHING 모드 (TRACKING보다 더 느리게 움직임, tau_waist_searching 사용)
        
        Returns:
            float: 실제로 전송한 허리 명령 각도 (절대각, 라디안)
        """
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        
        # SEARCHING 모드에서는 허리 각도를 ±65도로 제한 (탐색 범위 제한)
        if searching_mode:
            desired_total_yaw = max(math.radians(-65.0), min(math.radians(65.0), desired_total_yaw))
        
        # 초기화: 첫 호출 시 last_waist_command를 현재 허리 위치로 설정
        if self.last_waist_command is None:
            self.last_waist_command = self.current_waist_yaw_rad
        
        # 허리 오차 계산: 목표 전체 시선각과 현재 허리 각도 간의 차이
        error_waist_yaw = desired_total_yaw - self.current_waist_yaw_rad
        
        if searching_mode:
            # SEARCHING 모드: Exponential smoothing (느린 추종, TRACKING보다 더 느리게)
            # Exponential smoothing: alpha = dt / (tau + dt)
            alpha = dt / (self.tau_waist_searching + dt)
            
            # 목표 허리 각도를 exponential smoothing으로 계산
            # waist_cmd = last_waist_command + alpha * (desired_total - last_waist_command)
            smoothed_waist_yaw = self.last_waist_command + alpha * (desired_total_yaw - self.last_waist_command)
            
            # Rate limit: 증분 제한 (스무딩은 증가했지만 게인은 높여서 반응성 유지)
            max_delta = math.radians(2.5)  # SEARCHING 모드: 게인 증가 (1.9 -> 2.5도, 약 32% 증가, 더 빠른 반응)
            delta_waist_yaw = smoothed_waist_yaw - self.last_waist_command
            delta_waist_yaw = max(-max_delta, min(max_delta, delta_waist_yaw))
            new_waist_yaw = self.last_waist_command + delta_waist_yaw
        else:
            # TRACKING 모드: Exponential smoothing (느린 추종, 반 박자 지연)
            # Exponential smoothing: alpha = dt / (tau + dt)
            alpha = dt / (self.tau_waist + dt)
            
            # 목표 허리 각도를 exponential smoothing으로 계산
            # waist_cmd = last_waist_command + alpha * (desired_total - last_waist_command)
            smoothed_waist_yaw = self.last_waist_command + alpha * (desired_total_yaw - self.last_waist_command)
            
            # Rate limit: 증분 제한
            # TRACKING 모드: 일반 추적 속도 (초당 16.5도, 30Hz 기준 0.55도/프레임, 10% 증가)
            max_delta = self.max_delta_waist_tracking  # 프레임당 변화량 (GUI에서 제어 가능)
            
            delta_waist_yaw = smoothed_waist_yaw - self.last_waist_command
            delta_waist_yaw = max(-max_delta, min(max_delta, delta_waist_yaw))
            new_waist_yaw = self.last_waist_command + delta_waist_yaw
        
        # 각도 제한 적용
        new_waist_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, new_waist_yaw))
        
        # 허리 명령 전송
        self._send_waist_command(new_waist_yaw)
        
        # 상태 업데이트
        self.last_waist_command = new_waist_yaw
        self.last_waist_update_time = current_time
        
        return new_waist_yaw
    
    def _waist_follow_target(self, desired_waist_yaw: float, searching_mode: bool = False, fixed_pitch: float = None) -> float:
        """허리가 절대각 목표(desired_waist_yaw)를 느리게 추종
        
        Args:
            desired_waist_yaw: 목표 허리 각도 (절대각, 라디안)
            searching_mode: True면 SEARCHING 모드 (TRACKING보다 더 느리게 움직임, tau_waist_searching 사용)
            fixed_pitch: 고정 Pitch 각도 (라디안). None이면 breathe_pitch 사용
        
        Returns:
            float: 실제로 전송한 허리 명령 각도 (절대각, 라디안)
        """
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        
        # 각도 제한 적용
        desired_waist_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, desired_waist_yaw))
        
        # 초기화: 첫 호출 시 last_waist_command를 현재 허리 위치로 설정
        if self.last_waist_command is None:
            self.last_waist_command = self.current_waist_yaw_rad
        
        if searching_mode:
            # SEARCHING 모드: Exponential smoothing (느린 추종, TRACKING보다 더 느리게)
            alpha = dt / (self.tau_waist_searching + dt)
            smoothed_waist_yaw = self.last_waist_command + alpha * (desired_waist_yaw - self.last_waist_command)
            max_delta = math.radians(3.5)  # SEARCHING 모드 (40% 증가: 2.5 -> 3.5도)
        else:
            # TRACKING 모드: Exponential smoothing (느린 추종)
            alpha = dt / (self.tau_waist + dt)
            smoothed_waist_yaw = self.last_waist_command + alpha * (desired_waist_yaw - self.last_waist_command)
            # Rate limit: 일반 추적 (초당 16.5도, 10% 증가)
            max_delta = self.max_delta_waist_tracking  # 일반: 초당 16.5도 (0.55도/프레임)
        
        delta_waist_yaw = smoothed_waist_yaw - self.last_waist_command
        delta_waist_yaw = max(-max_delta, min(max_delta, delta_waist_yaw))
        new_waist_yaw = self.last_waist_command + delta_waist_yaw
        
        # 각도 제한 재적용
        new_waist_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, new_waist_yaw))
        
        # 허리 명령 전송
        self._send_waist_command(new_waist_yaw, fixed_pitch=fixed_pitch)
        
        # 상태 업데이트
        self.last_waist_command = new_waist_yaw
        self.last_waist_update_time = current_time
        
        return new_waist_yaw
    
    def _waist_follow_neck(self, searching_mode: bool = False):
        """허리가 목 각도를 Exponential하게 추종 (TRACKING 상태에서 사용)
        
        Args:
            searching_mode: True면 SEARCHING 모드 (목보다 더 천천히 움직임)
        """
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        # Exponential 추종: 목의 목표 각도 + 현재 각도
        target_waist_yaw = self.target_yaw_rad + self.current_yaw_rad
        
        # SEARCHING 모드에서는 허리 각도를 ±40도로 제한 (탐색 범위 제한)
        if searching_mode:
            target_waist_yaw = max(math.radians(-65.0), min(math.radians(65.0), target_waist_yaw))
        
        # 허리 오차 계산
        error_waist_yaw = target_waist_yaw - self.current_waist_yaw_rad
        
        # kp_waist_tracking을 사용하여 오차에 비례한 증분 계산
        if searching_mode:
            # SEARCHING 모드: 3배 빠르게
            kp = self.kp_waist_tracking * 1.8  # 기존 0.6의 3배 (180% 속도)
        else:
            kp = self.kp_waist_tracking
        
        # kp를 곱한 증분
        delta_waist_yaw = kp * error_waist_yaw
        
        # 증분 제한
        if searching_mode:
            max_delta = math.radians(6.0)  # SEARCHING 모드: 3배 빠른 속도 (초당 약 6도, 30Hz 기준)
        else:
            max_delta = math.radians(15.0)
        delta_waist_yaw = max(-max_delta, min(max_delta, delta_waist_yaw))
        
        # 새 목표 허리 각도
        new_waist_yaw = self.current_waist_yaw_rad + delta_waist_yaw
        
        # 허리 명령 전송 (각도 제한 적용)
        self._send_waist_command(new_waist_yaw)
        
        # 상태 업데이트
        self.last_waist_update_time = current_time
    
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float, use_pid: bool = True, use_searching_gain: bool = False, is_initial_tracking: bool = False) -> Tuple[float, float]:
        """목 명령 전송 (스무딩 적용)
        
        Args:
            target_yaw_rad: 목표 Yaw 각도
            target_pitch_rad: 목표 Pitch 각도
            use_pid: PID 제어 사용 여부
            use_searching_gain: SEARCHING 게인 사용 여부
            is_initial_tracking: 초기 TRACKING 단계 여부 (1.5초 이내)
        """
        self.target_yaw_rad = target_yaw_rad
        self.target_pitch_rad = target_pitch_rad
        
        # 초기 추적 상세 디버깅 (_send_neck_command 진입)
        if is_initial_tracking:
            if not hasattr(self, '_last_send_cmd_debug_time') or time.monotonic() - self._last_send_cmd_debug_time > 0.1:
                error_yaw = target_yaw_rad - self.current_yaw_rad
                self.get_logger().info(
                    f"[초기 추적 _send_neck_command 진입] "
                    f"target_yaw={math.degrees(target_yaw_rad):.3f}도, "
                    f"current_yaw={math.degrees(self.current_yaw_rad):.3f}도, "
                    f"error_yaw={math.degrees(error_yaw):.3f}도, "
                    f"use_pid={use_pid}"
                )
                self._last_send_cmd_debug_time = time.monotonic()
        
        # PID 제어 또는 직접 계산
        if use_pid:
            delta_yaw_rad, delta_pitch_rad = self._pid_control(
                target_yaw_rad, target_pitch_rad, 
                use_searching_gain=use_searching_gain,
                use_initial_gain=is_initial_tracking  # 초기 추적 시 초기 게인 사용
            )
        else:
            delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
            delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
        
        # Rate limit 적용: 초기 추적 중에는 더 큰 값 사용하여 움직임 보장
        if is_initial_tracking:
            # 초기 추적 시: rate limit을 50%로 줄여서 속도 감소
            max_delta_angle = math.radians(10.0)  # 초기: 10.0도/프레임 (20도에서 50% 감소)
        else:
            max_delta_angle = math.radians(66.0)  # 일반 추적: 72.9도/프레임 (81도의 90%, 10% 하향)
        
        # 기본 rate limit
        delta_yaw_before_limit = delta_yaw_rad
        delta_yaw_rad = max(-max_delta_angle, min(max_delta_angle, delta_yaw_rad))
        delta_pitch_rad = max(-max_delta_angle, min(max_delta_angle, delta_pitch_rad))
        
        # 초기 추적 rate limit 적용 후 디버깅
        if is_initial_tracking:
            if abs(delta_yaw_before_limit) > abs(delta_yaw_rad) + 0.001:  # rate limit이 적용되었는지 확인
                if not hasattr(self, '_last_rate_limit_applied_time') or time.monotonic() - self._last_rate_limit_applied_time > 0.1:
                    self.get_logger().warn(
                        f"[초기 추적 Rate Limit 적용됨!] "
                        f"delta_yaw_before={math.degrees(delta_yaw_before_limit):.3f}도 -> "
                        f"delta_yaw_after={math.degrees(delta_yaw_rad):.3f}도 (제한됨)"
                    )
                    self._last_rate_limit_applied_time = time.monotonic()
        
        # PID 제어 결과 스무딩 (초기 추적 시 스무딩 최소화)
        if is_initial_tracking:
            # 초기 추적 시: 스무딩을 더 강하게 하여 속도 감소 (50% 속도)
            smoothing_alpha = 0.85  # 초기 추적: 더 강한 스무딩 (15%)
        elif use_searching_gain:
            smoothing_alpha = 0.75  # SEARCHING에서는 5% 스무딩 추가
        else:
            smoothing_alpha = 0.9  # TRACKING에서는 5% 스무딩 추가
        
        smoothed_delta_yaw = self.last_neck_delta_yaw + smoothing_alpha * (delta_yaw_rad - self.last_neck_delta_yaw)
        smoothed_delta_pitch = self.last_neck_delta_pitch + smoothing_alpha * (delta_pitch_rad - self.last_neck_delta_pitch)
        
        # 스무딩된 값도 rate limit 적용
        smoothed_delta_yaw = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_yaw))
        smoothed_delta_pitch = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_pitch))
        
        # 마지막 증분 저장 (PID 출력용)
        self.last_neck_delta_yaw = smoothed_delta_yaw
        self.last_neck_delta_pitch = smoothed_delta_pitch
        
        # 추가 스무딩: 작은 변화에 대한 부드러운 처리 (끊김 방지)
        # 초기 추적 시에는 추가 스무딩 최소화
        delta_yaw_magnitude = abs(smoothed_delta_yaw - self.last_sent_neck_delta_yaw)
        delta_pitch_magnitude = abs(smoothed_delta_pitch - self.last_sent_neck_delta_pitch)
        
        small_change_threshold = math.radians(0.3)  # 약 0.3도 (작은 변화)
        
        if is_initial_tracking:
            # 초기 추적 시: 스무딩을 더 강하게 하여 속도 감소 (50% 속도)
            final_smoothing_alpha = 0.5  # 50% 스무딩 (속도 감소)
            final_pitch_smoothing_alpha = 0.5
        else:
            if delta_yaw_magnitude < small_change_threshold:
                # 작은 변화만 약한 스무딩 (0.7 = 70% 반영)
                final_smoothing_alpha = 0.7
            else:
                # 큰 변화에도 5% 스무딩 (1.0 -> 0.95)
                final_smoothing_alpha = 0.95
            
            if delta_pitch_magnitude < small_change_threshold:
                final_pitch_smoothing_alpha = 0.7
            else:
                # 큰 변화에도 5% 스무딩 (1.0 -> 0.95)
                final_pitch_smoothing_alpha = 0.95
        
        final_delta_yaw = self.last_sent_neck_delta_yaw + final_smoothing_alpha * (smoothed_delta_yaw - self.last_sent_neck_delta_yaw)
        final_delta_pitch = self.last_sent_neck_delta_pitch + final_pitch_smoothing_alpha * (smoothed_delta_pitch - self.last_sent_neck_delta_pitch)
        
        # Rate limit 재적용 (최종 값) - 초기 추적과 일반 추적 모두 적용
        final_delta_yaw = max(-max_delta_angle, min(max_delta_angle, final_delta_yaw))
        final_delta_pitch = max(-max_delta_angle, min(max_delta_angle, final_delta_pitch))
        
        # 마지막 전송 명령 업데이트
        self.last_sent_neck_delta_yaw = final_delta_yaw
        self.last_sent_neck_delta_pitch = final_delta_pitch
        
        # 초기 추적 디버깅 로그 (_send_neck_command 내부)
        if is_initial_tracking:
            if not hasattr(self, '_last_initial_neck_cmd_log_time') or time.monotonic() - self._last_initial_neck_cmd_log_time > 0.1:
                self.get_logger().info(
                    f"[초기 추적 _send_neck_command] "
                    f"target_yaw={math.degrees(target_yaw_rad):.2f}도, "
                    f"current_yaw={math.degrees(self.current_yaw_rad):.2f}도, "
                    f"delta_yaw={math.degrees(delta_yaw_rad):.2f}도, "
                    f"smoothed_delta_yaw={math.degrees(smoothed_delta_yaw):.2f}도, "
                    f"final_delta_yaw={math.degrees(final_delta_yaw):.2f}도, "
                    f"max_delta_angle={math.degrees(max_delta_angle):.2f}도/프레임"
                )
                self._last_initial_neck_cmd_log_time = time.monotonic()
        
        # 초기 추적 최종 명령 전송 디버깅
        if is_initial_tracking:
            if not hasattr(self, '_last_final_cmd_debug_time') or time.monotonic() - self._last_final_cmd_debug_time > 0.1:
                self.get_logger().info(
                    f"[초기 추적 최종 명령] "
                    f"final_delta_yaw={math.degrees(final_delta_yaw):.3f}도, "
                    f"final_delta_pitch={math.degrees(final_delta_pitch):.3f}도, "
                    f"명령 전송 예정"
                )
                self._last_final_cmd_debug_time = time.monotonic()
        
        # 발행 제거: 모든 neck publish는 _publish_neck_abs로 통일
        # msg = Float64MultiArray()
        # msg.data = [float(final_delta_pitch), float(final_delta_yaw)]  # [pitch, yaw] 순서, 증분 명령
        # self.neck_publisher.publish(msg)
        
        # 초기 추적 명령 전송 확인
        if is_initial_tracking:
            if not hasattr(self, '_last_cmd_sent_confirm_time') or time.monotonic() - self._last_cmd_sent_confirm_time > 0.1:
                self.get_logger().info(
                    f"[초기 추적 명령 전송 완료] "
                    f"pitch_delta={final_delta_pitch:.6f}rad ({math.degrees(final_delta_pitch):.3f}도), "
                    f"yaw_delta={final_delta_yaw:.6f}rad ({math.degrees(final_delta_yaw):.3f}도)"
                )
                self._last_cmd_sent_confirm_time = time.monotonic()
        
        # _send_neck_command는 더 이상 publish하지 않음 (모든 publish는 _publish_neck_abs로 통일)
        # 반환값: 계산된 절대각 (target 기반)
        # 실제 발행은 호출 측에서 _publish_neck_abs를 사용해야 함
        expected_yaw_rad = target_yaw_rad
        expected_pitch_rad = target_pitch_rad
        
        return expected_yaw_rad, expected_pitch_rad
    
    def _update_control(self, target_info: TargetInfo, frame_width: float = None, frame_height: float = None) -> Optional[Tuple[float, float]]:
        """타겟 정보를 받아서 목 각도 계산 및 명령 전송"""
        if frame_width is None:
            frame_width = self.frame_width
        if frame_height is None:
            frame_height = self.frame_height
        
        state = target_info.state
        prev_lost_state = self.prev_state_for_lost  # LOST 진입 감지용 이전 상태 저장
        
        # 상태 전환 로그 (LOST 관련)
        if prev_lost_state != state:
            self.get_logger().info(
                f"[상태 전환] {prev_lost_state} -> {state}"
            )
            # LOST -> IDLE 전환 감지 (문제 가능성)
            if prev_lost_state == TrackingState.LOST and state == TrackingState.IDLE:
                self.get_logger().warn(
                    f"[주의] LOST -> IDLE 전환 감지! 이는 정상적이지 않을 수 있습니다."
                )
        
        # LOST 상태에서 다른 상태로 전환될 때 LOST 타이머 초기화
        if prev_lost_state == TrackingState.LOST and state != TrackingState.LOST:
            self.lost_start_time = None
            self.lost_last_time = None
            self.get_logger().debug(
                f"LOST 상태 종료: 타이머 초기화 | "
                f"이전 상태: LOST -> 현재 상태: {state}"
            )
        
        # IDLE 상태 진입 시 IDLE 복귀 타이머 초기화
        if prev_lost_state != TrackingState.IDLE and state == TrackingState.IDLE:
            self.idle_return_start_time = None
            self.idle_return_start_yaw = None
            self.idle_return_start_pitch = None
            self.idle_return_start_waist_yaw = None
            self.get_logger().debug(f"IDLE 상태 진입: 영자세 복귀 타이머 초기화")
        
        # IDLE 상태에서 다른 상태로 전환될 때 IDLE 타이머 초기화
        if prev_lost_state == TrackingState.IDLE and state != TrackingState.IDLE:
            self.idle_return_start_time = None
            self.idle_return_start_yaw = None
            self.idle_return_start_pitch = None
            self.idle_return_start_waist_yaw = None
        
        # SEARCHING이 아닌 상태로 전환될 때 초기화 (필요시)
        
        try:
            match state:
                case TrackingState.TRACKING if target_info.point is not None:
                    self.searching_start_time = None
                    self.search_phase = 0
                
                    # Track ID 변경 감지
                    current_time_check = time.monotonic()
                    if self.current_track_id != target_info.track_id:
                        self.current_track_id = target_info.track_id
                        if target_info.track_id is not None:
                            self.get_logger().info(f"새 타겟 추적 시작: track_id={target_info.track_id}")
                
                    # 픽셀 좌표를 ROI 영역으로 제한
                    target_x, target_y = target_info.point
                    roi_left = frame_width * self.tracking_roi_left_margin
                    roi_right = frame_width * (1.0 - self.tracking_roi_right_margin)
                    target_x_clipped = max(roi_left, min(roi_right, target_x))
                    
                    # ROI 제한이 적용되었는지 로깅 (디버깅용)
                    if abs(target_x - target_x_clipped) > 1.0:
                        if not hasattr(self, '_last_roi_clip_log_time') or time.monotonic() - self._last_roi_clip_log_time > 1.0:
                            self.get_logger().debug(
                                f"[ROI 제한] target_x={target_x:.1f} -> {target_x_clipped:.1f} "
                                f"(ROI: [{roi_left:.1f}, {roi_right:.1f}])"
                            )
                            self._last_roi_clip_log_time = time.monotonic()
                    
                    # 픽셀 오차를 상대 각도로 변환 (ROI 제한된 좌표 사용)
                    relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x_clipped, target_y, frame_width, frame_height)
                    
                    # 일반 추적: 목 중심 제어 + 허리 추종
                    # 전체 시선각 계산: 현재 total gaze + 필요한 변화량
                    current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
                    raw_desired_total_yaw = current_total_yaw + relative_yaw_rad
                    
                    # ===== 목 중심 제어 (목이 먼저 빠르게 추종) =====
                    # 목 타겟: raw_total_target_yaw - current_waist_yaw
                    neck_target_yaw = raw_desired_total_yaw - self.current_waist_yaw_rad
                    neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))
                    
                    # 목 타겟 계산 (exponential smoothing만 사용, 추가 스무딩 제거)
                    # _publish_neck_abs에서 exponential smoothing이 적용되므로 여기서는 rate limit만 적용
                    if self.last_neck_target_yaw is not None:
                        max_neck_target_delta = math.radians(5.0)  # 일반 추적: rate limit만 적용 (exponential smoothing이 부드럽게 처리)
                        neck_target_delta = neck_target_yaw - self.last_neck_target_yaw
                        neck_target_delta = max(-max_neck_target_delta, min(max_neck_target_delta, neck_target_delta))
                        neck_target_yaw = self.last_neck_target_yaw + neck_target_delta  # smoothing_factor 제거
                    else:
                        # 초기화: 계산된 목 타겟으로 바로 설정
                        self.last_neck_target_yaw = neck_target_yaw
                    
                    # ===== 허리 제어 (목 각도를 천천히 회수) =====
                    # 허리 타겟: 타겟 위치 + 현재 목 위치
                    # 예: 타겟이 47.9도에 있고 목이 47.9도로 가면, 허리는 타겟 위치 + 목 위치로 계산
                    waist_target_yaw = raw_desired_total_yaw + self.current_yaw_rad
                    waist_target_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, waist_target_yaw))
                    
                    # 허리 제어: 목 각도를 천천히 회수하도록 추종
                    waist_cmd = self._waist_follow_target(waist_target_yaw, searching_mode=False)
                    
                    # Pitch는 허리가 관여하지 않으므로 기존 방식 유지
                    target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                    # 하드 클립
                    target_pitch_rad = max(self.pitch_min, min(self.pitch_max, target_pitch_rad))
                
                    # 목 명령 전송 (절대각 기반)
                    self._publish_neck_abs(target_pitch_rad, neck_target_yaw, mode="TRACKING")
                
                    # 목표 각도 업데이트
                    self.last_neck_target_yaw = neck_target_yaw

                    # LOST 감속을 위한 마지막 타겟 명령 저장
                    self.lost_last_target_yaw = neck_target_yaw
                    self.lost_last_target_pitch = target_pitch_rad
                    self.lost_last_waist_target_yaw = waist_target_yaw
                
                    # 반환값: 발행한 절대각 (last_neck_cmd_yaw_abs 사용)
                    return self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else neck_target_yaw, \
                           self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else target_pitch_rad
            
                case TrackingState.LOST:
                    # LOST 상태: 마지막 타겟 위치 방향으로 exponential smoothing 감속
                    current_time_lost = time.monotonic()
                
                    # LOST 상태 진입 감지 (이전 상태가 LOST가 아니었을 때)
                    if prev_lost_state != TrackingState.LOST:
                        # LOST 상태로 새로 진입했으므로 초기화
                        self.lost_start_time = current_time_lost
                        self.lost_last_time = current_time_lost
                        
                        # 핵심: 목은 None이든 아니든 "항상" 센서 각도로 재동기화
                        # 이렇게 하면 last_cmd가 센서와 불일치하더라도 현재 센서 위치에서 시작
                        self.prev_cmd_neck_yaw = self.current_yaw_rad
                        self.prev_cmd_neck_pitch = self.current_pitch_rad
                        self.prev_cmd_time_neck = current_time_lost
                        
                        self.last_cmd_neck_yaw = self.current_yaw_rad
                        self.last_cmd_neck_pitch = self.current_pitch_rad
                        self.last_cmd_time_neck = current_time_lost
                        
                        # 허리는 None일 때만 초기화 (허리는 절대각 기반이라 보통 정확함)
                        if self.last_cmd_waist_yaw is None:
                            self.last_cmd_waist_yaw = self.current_waist_yaw_rad
                        
                        # 진입 로그 (상세)
                        sensor_cmd_diff = abs(self.current_yaw_rad - self.last_cmd_neck_yaw) if self.last_cmd_neck_yaw is not None else 0.0
                        lost_target_yaw_str = f"{math.degrees(self.lost_last_target_yaw):.2f}도" if self.lost_last_target_yaw is not None else "None (영자세 0도로 복귀)"
                        lost_target_pitch_str = f"{math.degrees(self.lost_last_target_pitch):.2f}도" if self.lost_last_target_pitch is not None else "None (현재 각도 유지)"
                        lost_target_waist_str = f"{math.degrees(self.lost_last_waist_target_yaw):.2f}도" if self.lost_last_waist_target_yaw is not None else "None (영자세 0도로 복귀)"
                        self.get_logger().info(
                            f"LOST 상태 진입: 마지막 타겟 위치로 exponential smoothing 감속 시작 | "
                            f"prev_lost_state={prev_lost_state}, "
                            f"센서위치(current_yaw)={math.degrees(self.current_yaw_rad):.2f}도, "
                            f"명령위치(last_cmd_yaw)={math.degrees(self.last_cmd_neck_yaw):.2f}도, "
                            f"마지막 타겟 위치: neck_yaw={lost_target_yaw_str}, neck_pitch={lost_target_pitch_str}, waist_yaw={lost_target_waist_str}, "
                            f"tau_neck_lost={self.tau_neck_lost:.2f}초, tau_waist_lost={self.tau_waist_lost:.2f}초"
                        )
                
                    # dt 계산
                    dt = current_time_lost - self.lost_last_time if self.lost_last_time is not None else (1.0 / 30.0)
                    dt = max(0.001, min(dt, 0.1))
                
                    # 목표 위치: 마지막 타겟 위치 (없으면 영자세 0도로 복귀)
                    if self.lost_last_target_yaw is not None:
                        target_neck_yaw = self.lost_last_target_yaw
                    else:
                        target_neck_yaw = 0.0
                    
                    if self.lost_last_target_pitch is not None:
                        target_neck_pitch = self.lost_last_target_pitch
                    else:
                        target_neck_pitch = self.current_pitch_rad  # Pitch는 현재 각도 유지
                    
                    if self.lost_last_waist_target_yaw is not None:
                        target_waist_yaw = self.lost_last_waist_target_yaw
                    else:
                        target_waist_yaw = 0.0
                
                    # 목표 위치 계산 (target_neck_yaw, target_neck_pitch는 이미 위에서 계산됨)
                    # _publish_neck_abs에서 내부적으로 exponential smoothing과 rate limit을 처리하므로
                    # 여기서는 목표 위치만 전달
                    
                    # 목 명령 전송 (절대각 기반, _publish_neck_abs 사용)
                    self._publish_neck_abs(target_neck_pitch, target_neck_yaw, mode="LOST")
                    
                    # _publish_neck_abs 호출 후 업데이트된 값 사용
                    new_neck_yaw = self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else target_neck_yaw
                    new_neck_pitch = self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else target_neck_pitch
                    
                    # 허리: Exponential smoothing으로 목표 위치(마지막 타겟)로 천천히 이동
                    if not hasattr(self, '_last_lost_cmd_log_time') or current_time_lost - self._last_lost_cmd_log_time > 0.1:
                        delta_yaw_for_log = new_neck_yaw - self.last_cmd_neck_yaw if self.last_cmd_neck_yaw is not None else 0.0
                        target_yaw_str = f"{math.degrees(target_neck_yaw):.3f}도 (마지막 타겟)" if self.lost_last_target_yaw is not None else "0.000도 (영자세)"
                        # alpha_neck 계산 (로그용)
                        alpha_neck = dt / (self.tau_neck_lost + dt) if hasattr(self, 'tau_neck_lost') else 0.0
                        self.get_logger().info(
                            f"[LOST 명령 상세] "
                            f"last_cmd_yaw={math.degrees(self.last_cmd_neck_yaw):.3f}도, "
                            f"new_yaw={math.degrees(new_neck_yaw):.3f}도, "
                            f"target_yaw={target_yaw_str}, "
                            f"alpha={alpha_neck:.4f}, "
                            f"delta_yaw(변화량)={math.degrees(delta_yaw_for_log):.3f}도, "
                            f"발행값(msg.data[1])={new_neck_yaw:.6f}rad ({math.degrees(new_neck_yaw):.3f}도) [절대각도]"
                        )
                        self._last_lost_cmd_log_time = current_time_lost
                    
                    # 명령 위치 추적 업데이트 (목)
                    self.prev_cmd_neck_yaw = self.last_cmd_neck_yaw
                    self.prev_cmd_neck_pitch = self.last_cmd_neck_pitch
                    self.prev_cmd_time_neck = self.last_cmd_time_neck
                    self.last_cmd_neck_yaw = new_neck_yaw
                    # Pitch 업데이트 (마지막 타겟 위치로 감속)
                    self.last_cmd_neck_pitch = new_neck_pitch
                    self.last_cmd_time_neck = current_time_lost
                    
                    # 허리: Exponential smoothing으로 목표 위치(마지막 타겟)로 천천히 이동
                    # LOST 진입 시 last_cmd가 current로 초기화되므로 항상 last_cmd 사용 가능
                    alpha_waist = dt / (self.tau_waist_lost + dt)
                    new_waist_yaw = self.last_cmd_waist_yaw + alpha_waist * (target_waist_yaw - self.last_cmd_waist_yaw)
                    
                    # 하드 리밋 적용
                    new_waist_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, new_waist_yaw))
                    
                    # Rate limit 적용
                    delta_waist_yaw = new_waist_yaw - self.last_cmd_waist_yaw
                    delta_waist_yaw = max(-self.max_delta_waist_lost, min(self.max_delta_waist_lost, delta_waist_yaw))
                    new_waist_yaw = self.last_cmd_waist_yaw + delta_waist_yaw
                    
                    # 허리 명령 전송
                    self._send_waist_command(new_waist_yaw)
                    
                    # LOST 시간 업데이트
                    self.lost_last_time = current_time_lost
                    
                    # 디버깅 로그 (0.2초마다 출력)
                    elapsed = current_time_lost - self.lost_start_time if self.lost_start_time is not None else 0.0
                    if not hasattr(self, '_last_lost_log_time') or current_time_lost - self._last_lost_log_time > 0.2:
                        neck_yaw_current = self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else self.current_yaw_rad
                        neck_pitch_current = self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else self.current_pitch_rad
                        neck_yaw_remaining = abs(neck_yaw_current - target_neck_yaw)
                        neck_pitch_remaining = abs(neck_pitch_current - target_neck_pitch) if self.lost_last_target_pitch is not None else 0.0
                        waist_remaining = abs(new_waist_yaw - target_waist_yaw)
                        self.get_logger().info(
                            f"[LOST exponential smoothing {elapsed:.2f}초] "
                            f"neck_yaw={math.degrees(neck_yaw_current):.2f}도 (목표: {math.degrees(target_neck_yaw):.2f}도, 남은거리: {math.degrees(neck_yaw_remaining):.2f}도), "
                            f"neck_pitch={math.degrees(neck_pitch_current):.2f}도 (목표: {math.degrees(target_neck_pitch):.2f}도, 남은거리: {math.degrees(neck_pitch_remaining):.2f}도), "
                            f"waist_yaw={math.degrees(new_waist_yaw):.2f}도 (목표: {math.degrees(target_waist_yaw):.2f}도, 남은거리: {math.degrees(waist_remaining):.2f}도)"
                        )
                        self._last_lost_log_time = current_time_lost
                    
                    # 반환값: 발행한 절대각
                    return self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else target_neck_yaw, \
                           self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else target_neck_pitch
            
                case TrackingState.HELLO | TrackingState.HANDSHAKE:
                    # HELLO/HANDSHAKE 상태에서도 TRACKING과 동일하게 목/허리 제어 (neck-lead / waist-follow 구조)
                    self.searching_start_time = None
                    self.search_phase = 0
                    # HELLO/HANDSHAKE 상태에서 neck/waist PID 상태는 유지 (진동 방지 초기화 X)
                    if target_info.point is None:
                        # 타겟 포인트 없으면 현재 위치 유지
                        return self.current_yaw_rad, self.current_pitch_rad
                    
                    current_time_check = time.monotonic()
                    
                    # 픽셀 오차를 상대 각도로 변환
                    target_x, target_y = target_info.point
                    relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                    
                    # 전체 시선각 계산: 현재 total gaze + 필요한 변화량 (스무딩 없이)
                    current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
                    raw_desired_total_yaw = current_total_yaw + relative_yaw_rad
                    
                    # ===== 목 중심 제어 (목이 먼저 빠르게 추종) =====
                    # 목 타겟: raw_total_target_yaw - current_waist_yaw (스무딩 없이, rate limit만)
                    neck_target_yaw = raw_desired_total_yaw - self.current_waist_yaw_rad
                    neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))
                    
                    # LOST -> TRACKING 전환 시 급격한 변화 방지: rate limit + 스무딩
                    if self.last_neck_target_yaw is not None:
                        # HELLO/HANDSHAKE는 초기 추적이 아니므로 일반 추적 로직 사용
                        max_neck_target_delta = math.radians(13.44)  # 일반 추적: 12.0 * 1.12 = 13.44도/프레임 (12% 증가)
                        smoothing_factor = 0.95  # 일반 추적: 5% 스무딩 적용
                        neck_target_delta = neck_target_yaw - self.last_neck_target_yaw
                        neck_target_delta = max(-max_neck_target_delta, min(max_neck_target_delta, neck_target_delta))
                        neck_target_yaw = self.last_neck_target_yaw + smoothing_factor * neck_target_delta
                    else:
                        # 초기화: 계산된 목 타겟으로 바로 설정 (첫 프레임에서 즉시 반응)
                        self.last_neck_target_yaw = neck_target_yaw
                    
                    # ===== 허리 제어 (목 각도를 천천히 회수) =====
                    # 허리 타겟: raw_total_target_yaw - current_neck_yaw (천천히 추종)
                    waist_target_yaw = raw_desired_total_yaw - self.current_yaw_rad
                    waist_target_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, waist_target_yaw))
                    
                    # 허리 제어: 목 각도를 천천히 회수하도록 추종 (TRACKING과 동일)
                    # HELLO/HANDSHAKE 상태에서는 허리 Pitch를 고정 (앞으로 살짝 숙인 상태: 7도)
                    waist_cmd = self._waist_follow_target(waist_target_yaw, searching_mode=False, fixed_pitch=math.radians(7.0))
                    
                    # Pitch는 허리가 관여하지 않으므로 기존 방식 유지
                    target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                    # 하드 클립
                    target_pitch_rad = max(self.pitch_min, min(self.pitch_max, target_pitch_rad))
                    
                    # 목 명령 전송 (절대각 기반)
                    mode_str = "HELLO" if state == TrackingState.HELLO else "HANDSHAKE"
                    self._publish_neck_abs(target_pitch_rad, neck_target_yaw, mode=mode_str)
                    
                    # 목표 각도 업데이트
                    self.last_neck_target_yaw = neck_target_yaw
                    
                    # LOST 감속을 위한 마지막 타겟 명령 저장 (HELLO/HANDSHAKE 상태에서도 저장)
                    self.lost_last_target_yaw = neck_target_yaw
                    self.lost_last_target_pitch = target_pitch_rad
                    self.lost_last_waist_target_yaw = waist_target_yaw
                    
                    # 반환값: 발행한 절대각
                    return self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else neck_target_yaw, \
                           self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else target_pitch_rad
            
                case TrackingState.SEARCHING:
                    # SEARCHING 동작: 목과 허리 독립 제어
                    command_yaw_rad, command_pitch_rad = self._searching_behavior()
                    
                    # _searching_behavior()에서 계산된 exponential smoothing 명령 사용
                    neck_cmd = self.search_neck_last_cmd if self.search_neck_last_cmd is not None else self.current_yaw_rad
                    waist_cmd = self.search_waist_last_cmd if self.search_waist_last_cmd is not None else self.current_waist_yaw_rad
                    
                    # 허리 명령 전송 (절대각도) - [yaw, pitch] 형식으로 전송
                    breathe_pitch = self._get_waist_breathe_pitch()  # 기존 breathe pitch 사용
                    msg_waist = Float64MultiArray()
                    msg_waist.data = [float(waist_cmd), float(breathe_pitch)]  # [yaw, pitch] 순서
                    self.waist_publisher.publish(msg_waist)
                    
                    # 허리 명령 전송 확인 로그 (매 프레임마다 출력)
                    if not hasattr(self, '_last_waist_cmd_log_time') or time.monotonic() - self._last_waist_cmd_log_time > 0.1:
                        self.get_logger().info(
                            f"[SEARCHING 허리 명령] waist_cmd={math.degrees(waist_cmd):.2f}도, "
                            f"waist_current={math.degrees(self.current_waist_yaw_rad):.2f}도, "
                            f"waist_target={math.degrees(self.search_waist_target_yaw):.2f}도 (발행됨)"
                        )
                        self._last_waist_cmd_log_time = time.monotonic()
                    
                    # 목 명령 전송 (절대각 기반, _publish_neck_abs 사용)
                    # _searching_behavior()에서 계산된 neck_cmd와 command_pitch_rad를 목표로 사용
                    self._publish_neck_abs(command_pitch_rad, neck_cmd, mode="SEARCHING")
                    
                    # 목표 각도 업데이트
                    self.last_neck_target_yaw = neck_cmd
                    
                    # 반환값: 발행한 절대각
                    yaw_rad = self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else neck_cmd
                    pitch_rad = self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else command_pitch_rad
                    
                    # 디버깅 로그 (1초마다 출력)
                    if not hasattr(self, '_last_search_log_time') or time.monotonic() - self._last_search_log_time > 1.0:
                        neck_target = self.search_neck_target_yaw if self.search_neck_target_yaw is not None else 0.0
                        waist_target = self.search_waist_target_yaw if self.search_waist_target_yaw is not None else 0.0
                        neck_error = abs(neck_target - self.current_yaw_rad) if self.search_neck_target_yaw is not None else 0.0
                        waist_error = abs(waist_target - self.current_waist_yaw_rad) if self.search_waist_target_yaw is not None else 0.0
                        
                        self.get_logger().info(
                            f"SEARCHING: 독립 제어 | "
                            f"neck_current={math.degrees(self.current_yaw_rad):.2f}도, "
                            f"neck_target={math.degrees(neck_target):.2f}도, "
                            f"neck_cmd={math.degrees(neck_cmd):.2f}도, "
                            f"neck_error={math.degrees(neck_error):.2f}도 | "
                            f"waist_current={math.degrees(self.current_waist_yaw_rad):.2f}도, "
                            f"waist_target={math.degrees(waist_target):.2f}도, "
                            f"waist_cmd={math.degrees(waist_cmd):.2f}도, "
                            f"waist_error={math.degrees(waist_error):.2f}도 | "
                            f"phase={self.search_phase}"
                        )
                        self._last_search_log_time = time.monotonic()
                    
                    return yaw_rad, pitch_rad
            
                case TrackingState.WAITING:
                    # WAITING 상태: 타겟 찾기 (SEARCHING과 동일한 동작)
                    # SEARCHING 동작: 목표 각도 계산
                    command_yaw_rad, command_pitch_rad = self._searching_behavior()
                    self.target_yaw_rad = self.search_target_yaw
                    self.target_pitch_rad = command_pitch_rad
                    
                    # LOST -> WAITING 전환 시 부드러운 전환 보장
                    if prev_lost_state == TrackingState.LOST:
                        # LOST에서 WAITING으로 전환될 때 현재 위치에서 시작
                        self.last_neck_target_yaw = self.current_yaw_rad
                        self.get_logger().info(
                            f"[LOST -> WAITING 전환] 부드러운 전환을 위해 last_neck_target_yaw를 현재 위치로 초기화: "
                            f"{math.degrees(self.current_yaw_rad):.2f}도"
                        )
                    
                    # search_target_yaw는 전체 시선각 목표로 해석 (스무딩 없이)
                    raw_desired_total_yaw = self.search_target_yaw
                    raw_desired_total_yaw = max(math.radians(-65.0), min(math.radians(65.0), raw_desired_total_yaw))
                    
                    # ===== 목 중심 제어 (목이 먼저 천천히 스캔) =====
                    # 목 타겟: raw_total_target_yaw - current_waist_yaw (스무딩 없이, rate limit만)
                    neck_target_yaw = raw_desired_total_yaw - self.current_waist_yaw_rad
                    neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))
                    
                    # Rate limit 적용 (WAITING도 천천히 움직이도록)
                    if self.last_neck_target_yaw is not None:
                        max_neck_delta = math.radians(3.0)  # 12도 -> 3도로 감소 (4배 느리게)
                        neck_delta = neck_target_yaw - self.last_neck_target_yaw
                        neck_delta = max(-max_neck_delta, min(max_neck_delta, neck_delta))
                        neck_target_yaw = self.last_neck_target_yaw + 0.9 * neck_delta  # 10% 스무딩 적용 (더 부드럽게)
                    else:
                        self.last_neck_target_yaw = neck_target_yaw
                    
                    # ===== 허리 제어 (목 각도를 천천히 회수) =====
                    # 허리 타겟: raw_total_target_yaw - current_neck_yaw (천천히 추종)
                    waist_target_yaw = raw_desired_total_yaw - self.current_yaw_rad
                    waist_target_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, waist_target_yaw))
                    
                    # 허리 제어: 목 각도를 천천히 회수하도록 추종
                    waist_cmd = self._waist_follow_target(waist_target_yaw, searching_mode=True)
                    
                    # Pitch는 기존 방식 유지
                    target_pitch_rad = command_pitch_rad
                    # 하드 클립
                    target_pitch_rad = max(self.pitch_min, min(self.pitch_max, target_pitch_rad))
                    
                    # 목 명령 전송 (절대각 기반, _publish_neck_abs 사용)
                    self._publish_neck_abs(target_pitch_rad, neck_target_yaw, mode="SEARCHING")
                    
                    # 목표 각도 업데이트
                    self.last_neck_target_yaw = neck_target_yaw
                    
                    # 디버깅 로그 (1초마다 출력)
                    if not hasattr(self, '_last_waiting_log_time') or time.monotonic() - self._last_waiting_log_time > 1.0:
                        self.get_logger().info(
                            f"WAITING: 타겟 찾기 중 | "
                            f"raw_total_target={math.degrees(raw_desired_total_yaw):.2f}도, "
                            f"waist_current={math.degrees(self.current_waist_yaw_rad):.2f}도, "
                            f"waist_target={math.degrees(waist_target_yaw):.2f}도, "
                            f"waist_cmd={math.degrees(waist_cmd):.2f}도, "
                            f"neck_current={math.degrees(self.current_yaw_rad):.2f}도, "
                            f"neck_target={math.degrees(neck_target_yaw):.2f}도, "
                            f"phase={self.search_phase}"
                        )
                        self._last_waiting_log_time = time.monotonic()
                    
                    # 반환값: 발행한 절대각
                    return self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else neck_target_yaw, \
                           self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else target_pitch_rad
            
                case TrackingState.IDLE | _:
                # IDLE 상태: 영자세(0도)로 50초 동안 천천히 선형 이동 (10% 속도)
                    self.searching_start_time = None
                    self.search_phase = 0
                    self._reset_waist_pid()
                    self._reset_hello_check()
                    self.integral_yaw = 0.0
                    self.integral_pitch = 0.0
                
                    current_time_idle = time.monotonic()
                    
                    # IDLE 상태 진입 시 시작 위치 기록
                    if self.idle_return_start_time is None:
                        self.idle_return_start_time = current_time_idle
                        self.idle_return_start_yaw = self.current_yaw_rad
                        self.idle_return_start_pitch = self.current_pitch_rad
                        self.idle_return_start_waist_yaw = self.current_waist_yaw_rad
                    
                    # 목표 위치: 영자세 (0도)
                    target_neck_yaw = 0.0
                    target_neck_pitch = 0.0
                    target_waist_yaw = 0.0
                    
                    # 경과 시간 계산
                    elapsed = current_time_idle - self.idle_return_start_time
                    
                    # 50초 동안 선형 보간 (0~1 사이의 진행률)
                    if elapsed >= self.idle_return_duration:
                        # 50초 경과: 목표 위치로 완전히 도달 (0도)
                        progress = 1.0
                        new_neck_yaw = 0.0
                        new_neck_pitch = 0.0
                        new_waist_yaw = 0.0
                    else:
                        # 진행 중: 선형 보간
                        progress = elapsed / self.idle_return_duration
                        
                        # 선형 보간으로 목표 위치 계산
                        new_neck_yaw = self.idle_return_start_yaw + (target_neck_yaw - self.idle_return_start_yaw) * progress
                        new_neck_pitch = self.idle_return_start_pitch + (target_neck_pitch - self.idle_return_start_pitch) * progress
                        
                        # 허리 선형 보간
                        if self.idle_return_start_waist_yaw is not None:
                            new_waist_yaw = self.idle_return_start_waist_yaw + (target_waist_yaw - self.idle_return_start_waist_yaw) * progress
                        else:
                            new_waist_yaw = self.current_waist_yaw_rad
                    
                    # 하드 리밋 적용
                    new_neck_yaw = max(self.yaw_min, min(self.yaw_max, new_neck_yaw))
                    new_neck_pitch = max(self.pitch_min, min(self.pitch_max, new_neck_pitch))
                    new_waist_yaw = max(self.waist_yaw_min, min(self.waist_yaw_max, new_waist_yaw))
                    
                    # 목 명령 전송: Yaw 0도, Pitch 0도 (절대각도 방식)
                    msg_neck = Float64MultiArray()
                    msg_neck.data = [float(new_neck_pitch), float(new_neck_yaw)]  # [pitch, yaw] 순서, 절대각도 명령 (Pitch 0도, Yaw 0도)
                    self.neck_publisher.publish(msg_neck)
                    
                    # 명령 위치 추적 업데이트
                    self.prev_cmd_neck_yaw = self.last_cmd_neck_yaw
                    self.prev_cmd_neck_pitch = self.last_cmd_neck_pitch
                    self.prev_cmd_time_neck = self.last_cmd_time_neck
                    self.last_cmd_neck_yaw = new_neck_yaw
                    self.last_cmd_neck_pitch = new_neck_pitch
                    self.last_cmd_time_neck = current_time_idle
                    
                    # 허리 명령 전송: Yaw 0도
                    self._send_waist_command(new_waist_yaw)
                
                    # 반환값: 발행한 절대각
                    return self.last_neck_cmd_yaw_abs if self.last_neck_cmd_yaw_abs is not None else new_neck_yaw, \
                           self.last_neck_cmd_pitch_abs if self.last_neck_cmd_pitch_abs is not None else new_neck_pitch
        finally:
            # 모든 상태 처리 후 prev_state_for_lost 업데이트 (LOST 진입 감지용)
            # try/finally로 감싸서 어떤 경우에도 업데이트되도록 보장
            self.prev_state_for_lost = state
    
    def _reset_waist_pid(self):
        """허리 제어 상태 초기화"""
        self.last_waist_command = None  # None으로 설정하여 다음 호출 시 현재 위치로 자동 초기화
        self.last_sent_waist_command = None  # 추가 스무딩용 변수도 초기화
    
    def _reset_hello_check(self):
        """HELLO 전환 체크 상태 초기화"""
        self.hello_stable_start_time = None
    
    def _check_hello_transition(self, target_track_id, current_state, target_selected_time):
        """HELLO 상태 전환 조건 체크 (타겟 선택 후 정지 상태에서 ±2도 이내로 1.45초 유지)
        
        Args:
            target_track_id: 현재 타겟 track_id
            current_state: 현재 상태
            target_selected_time: 타겟 선택 시간 (None이면 아직 선택되지 않음)
        """
        # 이미 HELLO를 한 track_id면 전환하지 않음
        if target_track_id is not None and target_track_id in self.hello_done_track_ids:
            return
        
        current_time = time.monotonic()
        
        # 타겟이 선택되지 않았으면 타이머를 시작하지 않음
        if target_selected_time is None:
            # 타이머가 시작되었으면 리셋
            if self.hello_stable_start_time is not None:
                self._reset_hello_check()
            self.prev_state_for_hello = current_state
            return
        
        # 기준 위치에서 현재 위치까지의 차이 계산
        is_stable = abs(math.degrees(self.current_yaw_rad)) <= self.hello_position_threshold_deg
        
        # 조건: 2도 이내로 들어왔을 때부터 시간 측정 시작
        if is_stable:
            # 2도 이내로 들어온 시점부터 타이머 시작
            if self.hello_stable_start_time is None:
                self.hello_stable_start_time = current_time
                self.get_logger().debug(
                    f"HELLO 체크 시작: 2도 이내 진입, 시간 측정 시작 "
                    f"현재 위치={math.degrees(self.current_yaw_rad):.2f}도"
                )
            
            elapsed_time = current_time - self.hello_stable_start_time
            
            # 디버깅 로그 (1초마다)
            if int(elapsed_time) > int(elapsed_time - 0.1) and elapsed_time >= 1.0:
                current_yaw_deg = math.degrees(self.current_yaw_rad)
                self.get_logger().info(
                    f"HELLO 체크: 경과={elapsed_time:.1f}초/{self.hello_stable_duration}초, "
                    f"현재 위치={current_yaw_deg:.2f}도 (2도 이내)"
                )
            
            # 2초 유지되면 Depth 값에 따라 HELLO 또는 HANDSHAKE로 전환
            if elapsed_time >= self.hello_stable_duration:
                # 로그용 값 저장
                current_yaw_deg = math.degrees(self.current_yaw_rad)
                
                # 위치 안정성 조건 만족 시 tracking_fsm_node에 전환 준비 완료 신호 전송
                # tracking_fsm_node에서 depth 값을 확인하여 HANDSHAKE/HELLO 분기 판단
                request = {
                    'type': 'hello_transition_ready',
                    'target_id': target_track_id
                }
                msg = String()
                msg.data = json.dumps(request)
                self.tracker_state_request_publisher.publish(msg)
                self.get_logger().info(
                    f"HELLO 전환 준비 완료: track_id={target_track_id}, "
                    f"tracking_fsm_node에서 depth 기반 분기 판단 대기"
                )
                
                # track_id 저장은 tracking_fsm_node에서 처리 (HELLO/HANDSHAKE 완료 후)
                self.get_logger().info(
                    f"HELLO 전환 준비 완료: 현재 위치={current_yaw_deg:.1f}도에서 "
                    f"±{self.hello_position_threshold_deg}도 이내로 {elapsed_time:.1f}초 유지, "
                    f"track_id={target_track_id}에 대한 depth 기반 분기 판단 요청 전송"
                )
                self._reset_hello_check()
        else:
            # 2도 이내에서 벗어나면 타이머 리셋
            if self.hello_stable_start_time is not None:
                self.hello_stable_start_time = None
                self.get_logger().debug(
                    f"HELLO 체크 리셋: 2도 이내에서 벗어남 "
                    f"현재 위치={math.degrees(self.current_yaw_rad):.2f}도"
                )
        
        # 이전 상태 업데이트 (HELLO 체크용)
        self.prev_state_for_hello = current_state
    
    def get_current_angles(self) -> Tuple[float, float]:
        """현재 목 각도 반환"""
        return self.current_yaw_rad, self.current_pitch_rad
    
    def get_target_angles(self) -> Tuple[float, float]:
        """목표 명령 각도 반환"""
        return self.target_yaw_rad, self.target_pitch_rad
    
    def get_waist_angles(self) -> Tuple[float, float]:
        """허리 각도 반환 (현재 각도, 목표 각도)"""
        # 목표 허리 각도는 목의 목표 각도 + 현재 각도 (Exponential 추종)
        target_waist_yaw = self.target_yaw_rad + self.current_yaw_rad
        return self.current_waist_yaw_rad, target_waist_yaw
    
    def tracking_result_callback(self, msg: String):
        """추적 결과 콜백 - 타겟 정보 저장 (제어는 60Hz 제어 루프에서 처리)"""
        if not self.is_running:
            return
        
        try:
            data = json.loads(msg.data)
            
            # Manual 모드 정보 업데이트
            self.manual_mode = data.get('manual_mode', False)
            
            target_info_data = data.get('target_info', {})
            state_str = data.get('state', 'idle')
            
            try:
                state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                state = TrackingState.IDLE
            
            target_info = TargetInfo(
                point=tuple(target_info_data.get('point')) if target_info_data.get('point') else None,
                state=state,
                track_id=target_info_data.get('track_id')
            )
            
            frame_width = 1280.0
            frame_height = 720.0
            
            # 타겟 정보 저장 (60Hz 제어 루프에서 사용)
            self.last_target_info = target_info
            self.last_frame_width = frame_width
            self.last_frame_height = frame_height
            self.last_target_data = data  # HELLO 체크용
            
            # TRACKING 상태에서 HELLO 전환 조건 체크 (Manual 모드가 아닐 때만)
            if state == TrackingState.TRACKING and not self.manual_mode:
                # target_selected_time을 tracking_result에서 받아옴
                target_selected_time = data.get('target_selected_time')
                self._check_hello_transition(target_info.track_id, state, target_selected_time)
            else:
                # TRACKING 상태가 아니면 타이머 리셋 및 이전 상태 업데이트 (HELLO 체크용)
                if self.prev_state_for_hello == TrackingState.TRACKING:
                    # TRACKING에서 다른 상태로 전환되었으므로 타이머 리셋
                    self._reset_hello_check()
                self.prev_state_for_hello = state
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"추적 결과 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"타겟 정보 저장 실패: {e}")
    
    def _control_loop(self):
        """60Hz 제어 루프 - 저장된 타겟 정보로 제어 명령 계산 및 발행"""
        if not self.is_running:
            return
        
        if self.last_target_info is None:
            return
        
        try:
            # 저장된 타겟 정보로 제어 명령 계산 및 발행
            # _update_control 내부에서 prev_state_for_lost가 업데이트됨 (try/finally로 보장)
            self._update_control(
                self.last_target_info, 
                frame_width=self.last_frame_width,
                frame_height=self.last_frame_height
            )
        except Exception as e:
            self.get_logger().error(f"제어 명령 생성 실패: {e}")
    
    def _publish_neck_angle(self):
        """목 각도 발행"""
        try:
            current_yaw, current_pitch = self.get_current_angles()
            target_yaw, target_pitch = self.get_target_angles()
            current_waist_yaw, target_waist_yaw = self.get_waist_angles()
            
            data = {
                'current_yaw_rad': float(current_yaw),
                'current_pitch_rad': float(current_pitch),
                'target_yaw_rad': float(target_yaw),
                'target_pitch_rad': float(target_pitch),
                'current_waist_yaw_rad': float(current_waist_yaw),
                'target_waist_yaw_rad': float(target_waist_yaw),
                'timestamp': time.monotonic()
            }
            
            msg = String()
            msg.data = json.dumps(data)
            self.neck_angle_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"목 각도 발행 실패: {e}")
    
    
    def _control_callback(self, msg: String):
        """제어 명령 콜백"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'run' or cmd_type == 'start':
                self.is_running = True
                self.get_logger().info("Controller RUN 시작")
            
            elif cmd_type == 'stop':
                self.is_running = False
                self.get_logger().info("Controller RUN 중지")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"제어 명령 처리 실패: {e}")
    
    def _controller_control_callback(self, msg: String):
        """Controller 제어 콜백 - 파라미터 설정 등"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'run' or cmd_type == 'start':
                self.is_running = True
                self.get_logger().info("Controller RUN 시작")
            
            elif cmd_type == 'stop':
                self.is_running = False
                self.get_logger().info("Controller RUN 중지")
            
            elif cmd_type == 'set_parameters':
                parameters = command.get('parameters', {})
                self._apply_parameters(parameters)
                self.get_logger().info(f"파라미터 적용 완료: {parameters}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Controller 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"Controller 제어 명령 처리 실패: {e}")
    
    def _apply_parameters(self, parameters: Dict):
        """파라미터 적용"""
        if 'kp_yaw' in parameters:
            self.kp_yaw = parameters['kp_yaw']
        if 'ki_yaw' in parameters:
            self.ki_yaw = parameters['ki_yaw']
        if 'kp_pitch' in parameters:
            self.kp_pitch = parameters['kp_pitch']
        if 'ki_pitch' in parameters:
            self.ki_pitch = parameters['ki_pitch']
        if 'total_yaw_smoothing_alpha' in parameters:
            self.total_yaw_smoothing_alpha = parameters['total_yaw_smoothing_alpha']
        if 'neck_target_alpha' in parameters:
            # neck_target_alpha는 _update_control에서 사용하는 변수
            if not hasattr(self, 'neck_target_alpha'):
                self.neck_target_alpha = parameters['neck_target_alpha']
            else:
                self.neck_target_alpha = parameters['neck_target_alpha']
        if 'pid_smoothing_alpha' in parameters:
            self.smoothing_alpha = parameters['pid_smoothing_alpha']
        if 'tau_waist' in parameters:
            self.tau_waist = parameters['tau_waist']
        if 'tau_waist_searching' in parameters:
            self.tau_waist_searching = parameters['tau_waist_searching']
        if 'max_delta_waist' in parameters:
            # max_delta_waist는 _waist_follow_total에서 사용하는 변수
            if not hasattr(self, 'max_delta_waist_tracking'):
                self.max_delta_waist_tracking = parameters['max_delta_waist']
            else:
                self.max_delta_waist_tracking = parameters['max_delta_waist']
    


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = GazeControllerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
