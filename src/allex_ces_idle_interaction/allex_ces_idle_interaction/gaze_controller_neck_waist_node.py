#!/usr/bin/env python3
"""
목/허리 제어 노드 - 모든 로직과 통신을 한 파일에 통합
추적 결과를 받아서 로봇에 명령을 전송
"""
import json
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from std_msgs.msg import String, Float64MultiArray
import time
from typing import Optional, Tuple

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
        
        # 목 위치 Subscriber (현재 위치 파악용)
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
        
        # 허리 위치 Subscriber (현재 위치 파악용)
        self.waist_position_subscription = self.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_waist/joint_positions_deg',
            self._waist_position_callback,
            10
        )
        
        # 목 각도 발행 (Tracker 노드에서 사용)
        self.neck_angle_publisher = self.create_publisher(
            String,
            "/allex_camera/neck_angle",
            10
        )
        
        # Tracker 상태 변경 요청 Publisher (HELLO 전이용)
        self.tracker_state_request_publisher = self.create_publisher(
            String,
            "/allex_camera/tracker_state_request",
            10
        )
        
        # 목 각도 발행 타이머 (30Hz)
        self.neck_angle_timer = self.create_timer(1.0 / 30.0, self._publish_neck_angle)
        
        # 카메라 파라미터 (프레임 크기 기준)
        self.frame_width = 1280.0  # 프레임 너비 (픽셀)
        self.frame_height = 720.0  # 프레임 높이 (픽셀)
        
        # 각도 제한 범위 (라디안)
        self.yaw_min = -3.31613    # -190°
        self.yaw_max = 1.74533     # 100°
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
        
        # 영자세 (중앙 위치) - 절대 좌표 기준점
        self.home_yaw_rad = 0.0
        self.home_pitch_rad = 0.0
        self.left_right_angle = 40.0
        
        # SEARCHING 상태용 스캔 변수
        self.searching_start_time = None
        self.search_phase = 0  # 0: 우측(+40도)로, 1: 좌측(-40도)로
        self.search_target_yaw = 0.0  # 최종 목표 각도 (절대 각도)
        self.search_current_command_yaw = 0.0  # 현재 명령 각도 (증분 방식용)
        self.search_increment_rad = math.radians(0.3)  # 매 프레임마다 증가할 각도 (약 0.3도)
        
        # PID 제어 파라미터 (일반 추적용)
        self.kp_yaw = 1.88   # P 게인 (Yaw)
        self.kp_pitch = 1.5 # P 게인 (Pitch)
        self.ki_yaw = 0.12    # I 게인 (Yaw) - Steady State Error 제거용
        self.ki_pitch = 0.3  # I 게인 (Pitch)
        self.kd_yaw = 0.01   # D 게인 (Yaw) - 낮춰서 움직임 억제 감소
        self.kd_pitch = 0.02 # D 게인 (Pitch)
        
        # SEARCHING 상태용 낮은 게인 (천천히 움직임)
        self.kp_yaw_searching = 0.7   # P 게인 (Yaw) - 검색 시 / IDLE 시
        self.kp_pitch_searching = 0.7 # P 게인 (Pitch) - 검색 시 / IDLE 시
        self.ki_yaw_searching = 0.05  # I 게인 (Yaw) - 검색 시
        self.ki_pitch_searching = 0.05 # I 게인 (Pitch) - 검색 시
        self.kd_yaw_searching = 0.02  # D 게인 (Yaw) - 검색 시
        self.kd_pitch_searching = 0.02 # D 게인 (Pitch) - 검색 시
        
        # PID 제어 상태 변수
        self.integral_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_yaw = 0.0
        self.last_error_pitch = 0.0
        self.last_update_time = time.monotonic()
        
        # 허리 제어 변수
        self.current_waist_yaw_rad = 0.0  # 현재 허리 각도 (라디안, 절대 좌표)
        self.last_waist_position_update_time = 0.0
        
        # TRACKING 상태에서 허리 추종용 PID 게인 (진동 방지를 위해 P 게인 낮춤)
        self.kp_waist_tracking = 1.2  # P 게인 (Waist Yaw) - 진동 방지
        self.ki_waist_tracking = 0.1  # I 게인 (Waist Yaw)
        self.kd_waist_tracking = 0.04  # D 게인 (Waist Yaw)
        
        # 허리-목 간 비선형 오프셋 파라미터 (steady state error 유지용)
        # 목 각도가 작을수록 오프셋을 더 크게 (목이 0도 근처일 때 허리가 앞서가도록)
        self.waist_base_scale = 1.6  # 기본 스케일 배수
        self.waist_max_offset_rad = math.radians(15.0)  # 최대 오프셋 (8도)
        self.waist_offset_decay_angle_rad = math.radians(20.0)  # 오프셋 감쇠 각도 (20도)
        
        # 허리 PID 제어 상태 변수
        self.integral_waist_yaw = 0.0
        self.last_error_waist_yaw = 0.0
        self.last_waist_update_time = time.monotonic()
        
        # 실행 상태 플래그
        self.is_running = False
        
        # HELLO 전환 조건 변수 (현재 위치에서 ±1도 이내로 1초 유지)
        self.hello_position_threshold_deg = 0.7  # 기준 위치에서 ±1도 이내
        self.hello_stable_duration = 1.85  # 조건 유지 시간 (초)
        self.hello_stable_start_time = None  # 조건 만족 시작 시간
        self.hello_reference_yaw_rad = None  # 기준 위치 (타이머 시작 시 저장)
        self.previous_state = None  # 이전 상태 추적 (TRACKING 진입 감지용)
        
        # 이미 HELLO를 한 track_id 저장 (중복 HELLO 방지)
        self.hello_done_track_ids = set()
        
        self.get_logger().info("Gaze Controller Node 초기화 완료")
    
    def _neck_position_callback(self, msg: Float64MultiArray):
        """목 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트"""
        if len(msg.data) >= 2:
            pitch_deg = msg.data[0]
            yaw_deg = msg.data[1]
            self.current_pitch_rad = math.radians(pitch_deg)
            self.current_yaw_rad = math.radians(yaw_deg)
            self.last_position_update_time = time.monotonic()
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
        """SEARCHING 상태 동작: 증분 방식으로 천천히 좌우 스캔"""
        if self.searching_start_time is None:
            self.searching_start_time = time.monotonic()
            self.search_phase = 0
            self.search_current_command_yaw = self.current_yaw_rad
            self.search_target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
        
        if self.search_phase == 0:
            target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
            target_yaw = min(target_yaw, self.yaw_max)
            self.search_target_yaw = target_yaw
            
            if self.search_current_command_yaw < target_yaw:
                self.search_current_command_yaw += self.search_increment_rad
                self.search_current_command_yaw = min(self.search_current_command_yaw, target_yaw)
            else:
                self.search_phase = 1
                self.search_target_yaw = self.home_yaw_rad - math.radians(self.left_right_angle)
                self.search_target_yaw = max(self.search_target_yaw, self.yaw_min)
        
        elif self.search_phase == 1:
            target_yaw = self.home_yaw_rad - math.radians(self.left_right_angle)
            target_yaw = max(target_yaw, self.yaw_min)
            self.search_target_yaw = target_yaw
            
            if self.search_current_command_yaw > target_yaw:
                self.search_current_command_yaw -= self.search_increment_rad
                self.search_current_command_yaw = max(self.search_current_command_yaw, target_yaw)
            else:
                self.search_phase = 0
                self.search_target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
                self.search_target_yaw = min(self.search_target_yaw, self.yaw_max)
        
        return self.search_current_command_yaw, self.home_pitch_rad
    
    def _pid_control(self, target_yaw_rad: float, target_pitch_rad: float, use_searching_gain: bool = False) -> Tuple[float, float]:
        """PID 제어를 사용하여 목 증분 명령 계산"""
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))
        
        if use_searching_gain:
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
    
    def _send_waist_command(self, absolute_waist_yaw_rad: float):
        """허리 명령 전송"""
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        msg = Float64MultiArray()
        msg.data = [float(absolute_waist_yaw_rad), 0.0]  # [yaw, pitch] 순서
        self.waist_publisher.publish(msg)
    
    def _waist_follow_neck(self):
        """허리가 목 각도를 천천히 추종 (TRACKING 상태에서 사용)"""
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        # 목표: 비선형 스케일링 (목 각도가 작을수록 오프셋을 더 크게)
        # 목이 0도 근처일 때 허리가 앞서가도록, 목이 크게 회전할 때는 오프셋 감소
        base_scale = self.waist_base_scale
        offset_scale = self.waist_max_offset_rad * math.exp(-abs(self.current_yaw_rad) / self.waist_offset_decay_angle_rad)
        offset_direction = 1.0 if self.current_yaw_rad >= 0 else -1.0
        target_waist_yaw = self.current_yaw_rad * base_scale + offset_scale * offset_direction
        
        # PID 제어 (3배 느린 게인)
        error_waist_yaw = target_waist_yaw - self.current_waist_yaw_rad
        
        # P
        p_waist = self.kp_waist_tracking * error_waist_yaw
        
        # I
        self.integral_waist_yaw += error_waist_yaw * dt
        max_integral = math.radians(30.0)
        self.integral_waist_yaw = max(-max_integral, min(max_integral, self.integral_waist_yaw))
        i_waist = self.ki_waist_tracking * self.integral_waist_yaw
        
        # D
        d_error_waist = (error_waist_yaw - self.last_error_waist_yaw) / dt
        d_waist = self.kd_waist_tracking * d_error_waist
        
        # 증분 명령 계산
        delta_waist_yaw = p_waist + i_waist + d_waist
        
        # 증분 제한 (1.45배 상향)
        max_delta = math.radians(8.5)
        delta_waist_yaw = max(-max_delta, min(max_delta, delta_waist_yaw))
        
        # 새 목표 허리 각도
        new_waist_yaw = self.current_waist_yaw_rad + delta_waist_yaw
        
        # 허리 명령 전송
        self._send_waist_command(new_waist_yaw)
        
        # 상태 업데이트
        self.last_error_waist_yaw = error_waist_yaw
        self.last_waist_update_time = current_time
    
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float, use_pid: bool = True, use_searching_gain: bool = False) -> Tuple[float, float]:
        """목 명령 전송"""
        self.target_yaw_rad = target_yaw_rad
        self.target_pitch_rad = target_pitch_rad
        
        if use_pid:
            delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad, use_searching_gain=use_searching_gain)
        else:
            delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
            delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
        
        max_delta_angle = math.radians(30.0)
        delta_yaw_rad = max(-max_delta_angle, min(max_delta_angle, delta_yaw_rad))
        delta_pitch_rad = max(-max_delta_angle, min(max_delta_angle, delta_pitch_rad))
        
        msg = Float64MultiArray()
        msg.data = [float(delta_pitch_rad), float(delta_yaw_rad)]  # [pitch, yaw] 순서, 증분 명령
        self.neck_publisher.publish(msg)
        
        expected_yaw_rad = self.current_yaw_rad + delta_yaw_rad
        expected_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        return expected_yaw_rad, expected_pitch_rad
    
    def _update_control(self, target_info: TargetInfo, frame_width: float = None, frame_height: float = None) -> Optional[Tuple[float, float]]:
        """타겟 정보를 받아서 목 각도 계산 및 명령 전송"""
        if frame_width is None:
            frame_width = self.frame_width
        if frame_height is None:
            frame_height = self.frame_height
        
        state = target_info.state
        
        match state:
            case TrackingState.TRACKING if target_info.point is not None:
                self.searching_start_time = None
                self.search_phase = 0
                
                # 목 제어: 빠르게 사람 추적
                target_x, target_y = target_info.point
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                yaw_rad, pitch_rad = self._send_neck_command(target_yaw_rad, target_pitch_rad, use_pid=True)
                
                # 허리 제어: 목 각도를 매우 천천히 추종 (3배 느린 PID)
                self._waist_follow_neck()
                
                return yaw_rad, pitch_rad
            
            case TrackingState.INTERACTION:
                # INTERACTION 모드: BB Box만 따고 목 명령 전송 안 함
                self.searching_start_time = None
                self.search_phase = 0
                self._reset_waist_pid()
                self._reset_hello_check()
                
                # 목 명령 전송 없이 현재 위치 유지
                return self.current_yaw_rad, self.current_pitch_rad
            
            case TrackingState.LOST:
                # LOST 상태: 현재 위치 유지, 명령 전송 안 함 (떨림 방지)
                self.searching_start_time = None
                self.search_phase = 0
                self._reset_waist_pid()
                self._reset_hello_check()
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                # 명령 전송 없이 현재 위치 유지
                return self.current_yaw_rad, self.current_pitch_rad
            
            case TrackingState.HELLO:
                # HELLO 상태에서도 TRACKING과 동일하게 목/허리 제어
                self.searching_start_time = None
                self.search_phase = 0
                # HELLO 상태에서 neck/waist PID 상태는 유지 (진동 방지 초기화 X)
                if target_info.point is None:
                    # 타겟 포인트 없으면 현재 위치 유지
                    return self.current_yaw_rad, self.current_pitch_rad
                
                target_x, target_y = target_info.point
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                yaw_rad, pitch_rad = self._send_neck_command(target_yaw_rad, target_pitch_rad, use_pid=True)
                
                # 허리 제어: 목 각도를 매우 천천히 추종 (3배 느린 PID) + 120%
                self._waist_follow_neck()
                
                return yaw_rad, pitch_rad
            
            case TrackingState.SEARCHING:
                command_yaw_rad, command_pitch_rad = self._searching_behavior()
                self.target_yaw_rad = self.search_target_yaw
                self.target_pitch_rad = command_pitch_rad
                command_yaw_rad, command_pitch_rad = self._clip_angles(command_yaw_rad, command_pitch_rad)
                
                relative_command_yaw_rad = command_yaw_rad - self.current_yaw_rad
                relative_command_pitch_rad = command_pitch_rad - self.current_pitch_rad
                
                msg = Float64MultiArray()
                msg.data = [float(relative_command_pitch_rad), float(relative_command_yaw_rad)]
                self.neck_publisher.publish(msg)
                return command_yaw_rad, command_pitch_rad
            
            case TrackingState.IDLE | _:
                self.searching_start_time = None
                self.search_phase = 0
                self._reset_waist_pid()
                self._reset_hello_check()
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                target_neck_yaw = 0.0
                target_neck_pitch = 0.0
                yaw_rad, pitch_rad = self._send_neck_command(
                    target_neck_yaw, 
                    target_neck_pitch, 
                    use_pid=True, 
                    use_searching_gain=True
                )
                
                target_waist_yaw = 0.0
                self._send_waist_command(target_waist_yaw)
                
                return yaw_rad, pitch_rad
    
    def _reset_waist_pid(self):
        """허리 PID 상태 초기화"""
        self.integral_waist_yaw = 0.0
        self.last_error_waist_yaw = 0.0
    
    def _reset_hello_check(self):
        """HELLO 전환 체크 상태 초기화"""
        self.hello_stable_start_time = None
        self.hello_reference_yaw_rad = None
    
    def _check_hello_transition(self, target_track_id, current_state):
        """HELLO 상태 전환 조건 체크 (현재 위치에서 ±1도 이내로 3초 유지)"""
        # 이미 HELLO를 한 track_id면 전환하지 않음
        if target_track_id is not None and target_track_id in self.hello_done_track_ids:
            return
        
        current_time = time.monotonic()
        
        # TRACKING 상태로 전환되었을 때만 타이머 시작 (이전 상태가 TRACKING이 아니었을 때)
        if self.previous_state != TrackingState.TRACKING:
            # TRACKING 상태로 새로 진입했으므로 타이머 시작
            self.hello_stable_start_time = current_time
            self.hello_reference_yaw_rad = self.current_yaw_rad
            self.get_logger().debug(
                f"HELLO 체크 시작: TRACKING 상태 진입, 기준 위치={math.degrees(self.hello_reference_yaw_rad):.1f}도"
            )
        
        # 타이머가 시작되지 않았으면 체크하지 않음 (안전장치)
        if self.hello_stable_start_time is None:
            self.previous_state = current_state
            return
        
        # 기준 위치에서 현재 위치까지의 차이 계산
        position_diff_deg = abs(math.degrees(self.current_yaw_rad - self.hello_reference_yaw_rad))
        is_stable = position_diff_deg <= self.hello_position_threshold_deg
        
        # 조건: 기준 위치에서 ±3도 이내 유지
        if is_stable:
            elapsed_time = current_time - self.hello_stable_start_time
            
            # 디버깅 로그 (1초마다)
            if int(elapsed_time) > int(elapsed_time - 0.1) and elapsed_time >= 1.0:
                self.get_logger().info(
                    f"HELLO 체크: 경과={elapsed_time:.1f}초/{self.hello_stable_duration}초, "
                    f"위치차이={position_diff_deg:.2f}도"
                )
            
            # 3초 유지되면 HELLO로 전환
            if elapsed_time >= self.hello_stable_duration:
                # 로그용 값 저장
                ref_yaw_deg = math.degrees(self.hello_reference_yaw_rad)
                
                # HELLO 상태로 전환 요청
                request = {
                    'type': 'set_state',
                    'state': 'hello',
                    'target_id': target_track_id
                }
                msg = String()
                msg.data = json.dumps(request)
                self.tracker_state_request_publisher.publish(msg)
                
                # HELLO를 한 track_id 저장
                if target_track_id is not None:
                    self.hello_done_track_ids.add(target_track_id)
                    self.get_logger().info(
                        f"HELLO 완료 ID 저장: track_id={target_track_id} "
                        f"(총 {len(self.hello_done_track_ids)}개 ID)"
                    )
                
                self.get_logger().info(
                    f"HELLO 전환: 기준 위치={ref_yaw_deg:.1f}도에서 "
                    f"±{self.hello_position_threshold_deg}도 이내로 {elapsed_time:.1f}초 유지"
                )
                self._reset_hello_check()
        else:
            # 기준 위치에서 ±5도 벗어나면 타이머 리셋 및 새 기준 위치 설정
            self.hello_stable_start_time = current_time
            self.hello_reference_yaw_rad = self.current_yaw_rad
        
        # 이전 상태 업데이트
        self.previous_state = current_state
    
    def get_current_angles(self) -> Tuple[float, float]:
        """현재 목 각도 반환"""
        return self.current_yaw_rad, self.current_pitch_rad
    
    def get_target_angles(self) -> Tuple[float, float]:
        """목표 명령 각도 반환"""
        return self.target_yaw_rad, self.target_pitch_rad
    
    def get_waist_angles(self) -> Tuple[float, float]:
        """허리 각도 반환 (현재 각도, 목표 각도)"""
        # 목표 허리 각도는 현재 목 각도의 142% (TRACKING에서 추종 중)
        target_waist_yaw = self.current_yaw_rad * 1.42
        return self.current_waist_yaw_rad, target_waist_yaw
    
    def tracking_result_callback(self, msg: String):
        """추적 결과 콜백 - 로봇 제어 명령 생성 및 전송"""
        if not self.is_running:
            return
        
        try:
            data = json.loads(msg.data)
            
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
            
            # TRACKING 상태에서 HELLO 전환 조건 체크
            if state == TrackingState.TRACKING:
                self._check_hello_transition(target_info.track_id, state)
            else:
                # TRACKING 상태가 아니면 타이머 리셋 및 이전 상태 업데이트
                if self.previous_state == TrackingState.TRACKING:
                    # TRACKING에서 다른 상태로 전환되었으므로 타이머 리셋
                    self._reset_hello_check()
                self.previous_state = state
            
            self._update_control(target_info, frame_width=frame_width, frame_height=frame_height)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"추적 결과 파싱 실패: {e}")
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
