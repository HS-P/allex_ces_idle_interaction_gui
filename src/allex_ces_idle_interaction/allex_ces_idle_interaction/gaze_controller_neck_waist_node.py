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
        
        # Tracker 상태 변경 요청 Publisher (WAIST_FOLLOWER 전이용)
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
        self.yaw_min = math.radians(-80.0)    # -60°
        self.yaw_max = math.radians(80.0)     # 60°
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
        self.search_increment_rad = math.radians(0.5)  # 매 프레임마다 증가할 각도 (약 0.5도)
        self.search_reached_threshold_rad = math.radians(1.0)  # 목표 도달 판정 임계값 (1도)
        
        # PID 제어 파라미터 (일반 추적용) - 스무딩으로 안정성 확보
        self.kp_yaw = 1.55   # P 게인 (Yaw)
        self.kp_pitch = 1.2 # P 게인 (Pitch)
        self.ki_yaw = 0.2   # I 게인 (Yaw)
        self.ki_pitch = 0.1 # I 게인 (Pitch)
        self.kd_yaw = 0.02   # D 게인 (Yaw) - 진동 억제
        self.kd_pitch = 0.1 # D 게인 (Pitch)
        
        # SEARCHING 상태용 게인 (매우 천천히 부드럽게 움직임)
        self.kp_yaw_searching = 0.1   # P 게인 (Yaw) - 천천히 움직임
        self.kp_pitch_searching = 0.3 # P 게인 (Pitch) - 검색 시
        self.ki_yaw_searching = 0.01  # I 게인 (Yaw) - 천천히 움직임
        self.ki_pitch_searching = 0.02 # I 게인 (Pitch) - 검색 시
        self.kd_yaw_searching = 0.0  # D 게인 (Yaw) - 안정성 향상
        self.kd_pitch_searching = 0.01 # D 게인 (Pitch) - 검색 시
        
        # PID 제어 상태 변수
        self.integral_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_yaw = 0.0
        self.last_error_pitch = 0.0
        self.last_update_time = time.monotonic()
        
        # WAIST_FOLLOWER 상태용 변수
        self.waist_follower_initial_neck_yaw = None  # WAIST_FOLLOWER 상태 진입 시점의 목 각도
        self.current_waist_yaw_rad = 0.0  # 현재 허리 각도 (라디안, 절대 좌표)
        self.last_waist_position_update_time = 0.0
        
        # WAIST_FOLLOWER 상태용 PID 게인 (허리 제어용)
        self.kp_waist_yaw = 0.34   # P 게인 (Waist Yaw)
        self.ki_waist_yaw = 0.015  # I 게인 (Waist Yaw)
        self.kd_waist_yaw = 0.001  # D 게인 (Waist Yaw)
        
        # WAIST_FOLLOWER 상태용 목 제어 PID 게인 (목을 0도로 이동) - 독립적으로 설정 가능
        self.kp_neck_yaw_waist_mode = 0.1   # P 게인 (Neck Yaw - Waist 모드)
        self.ki_neck_yaw_waist_mode = 0.01  # I 게인 (Neck Yaw - Waist 모드)
        self.kd_neck_yaw_waist_mode = 0.0  # D 게인 (Neck Yaw - Waist 모드)
        
        # 허리 PID 제어 상태 변수
        self.integral_waist_yaw = 0.0
        self.last_error_waist_yaw = 0.0
        self.last_waist_update_time = time.monotonic()
        self.integral_neck_yaw_waist_mode = 0.0
        self.last_error_neck_yaw_waist_mode = 0.0
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 목 각도 안정성 추적 변수 (WAIST_FOLLOWER 전이용)
        self.last_neck_yaw_rad = None
        self.neck_stable_start_time = None
        self.neck_stable_duration = 5.0
        self.neck_stable_threshold_deg = 3.0
        
        # 숨쉬는 모션 변수 (INTERACTION 모드 제외한 모든 모드에서 적용)
        self.breathing_start_time = time.monotonic()
        self.breathing_amplitude_deg = 5.0  # -5도에서 5도 사이
        self.breathing_period_sec = 6.0  # 6초 주기 (천천히 움직임)
        
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
        """SEARCHING 상태 동작: 절대각도 기반 좌우 스캔"""
        if self.searching_start_time is None:
            self.searching_start_time = time.monotonic()
            self.search_phase = 0  # 0: 우측으로 이동 시작
            # 현재 실제 위치에서 시작
            self.search_current_command_yaw = self.current_yaw_rad
        
        # 절대각도 목표 계산
        if self.search_phase == 0:
            # 우측으로 이동 중: home + 40도
            target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
            target_yaw = min(target_yaw, self.yaw_max)  # 60도 제한
            self.search_target_yaw = target_yaw
            
            # 현재 위치 피드백 기반으로 목표 도달 확인
            current_error_rad = abs(self.current_yaw_rad - self.search_target_yaw)
            
            if current_error_rad <= self.search_reached_threshold_rad:
                # 우측 목표 도달 → 좌측으로 전환
                self.search_phase = 1
                self.get_logger().info(
                    f"SEARCHING: 우측 도달 (목표={math.degrees(self.search_target_yaw):.1f}도, "
                    f"실제={math.degrees(self.current_yaw_rad):.1f}도, 오차={math.degrees(current_error_rad):.1f}도) → 좌측으로 전환"
                )
        
        elif self.search_phase == 1:
            # 좌측으로 이동 중: home - 40도
            target_yaw = self.home_yaw_rad - math.radians(self.left_right_angle)
            target_yaw = max(target_yaw, self.yaw_min)  # -60도 제한
            self.search_target_yaw = target_yaw
            
            # 현재 위치 피드백 기반으로 목표 도달 확인
            current_error_rad = abs(self.current_yaw_rad - self.search_target_yaw)
            
            if current_error_rad <= self.search_reached_threshold_rad:
                # 좌측 목표 도달 → 우측으로 전환
                self.search_phase = 0
                self.get_logger().info(
                    f"SEARCHING: 좌측 도달 (목표={math.degrees(self.search_target_yaw):.1f}도, "
                    f"실제={math.degrees(self.current_yaw_rad):.1f}도, 오차={math.degrees(current_error_rad):.1f}도) → 우측으로 전환"
                )
        
        # 절대각도 목표 반환 (PID 제어로 부드럽게 이동)
        return self.search_target_yaw, self.home_pitch_rad
    
    def _pid_control(self, target_yaw_rad: float, target_pitch_rad: float, use_searching_gain: bool = False) -> Tuple[float, float]:
        """PID 제어를 사용하여 목 증분 명령 계산
        스무딩을 통한 안정적인 제어:
        - 큰 움직임이 필요할 때는 큰 값을 전달 (스플라인이 빠르게 추종)
        - 현재 위치에서 오차만큼만 이동하도록 (상대각도 방식)
        - 스무딩으로 발산 및 진동 방지
        """
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))
        
        if use_searching_gain:
            # SEARCHING 상태는 고정 게인 사용
            kp_yaw = self.kp_yaw_searching
            kp_pitch = self.kp_pitch_searching
            ki_yaw = self.ki_yaw_searching
            ki_pitch = self.ki_pitch_searching
            kd_yaw = self.kd_yaw_searching
            kd_pitch = self.kd_pitch_searching
        else:
            # TRACKING 상태: 고정 게인 사용 (스무딩으로 안정성 확보)
            kp_yaw = self.kp_yaw
            kp_pitch = self.kp_pitch
            ki_yaw = self.ki_yaw
            ki_pitch = self.ki_pitch
            kd_yaw = self.kd_yaw
            kd_pitch = self.kd_pitch
        
        # 오차 계산: 목표 - 현재 (큰 값도 그대로 전달, 제한하지 않음)
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_pitch = target_pitch_rad - self.current_pitch_rad
        
        p_yaw = kp_yaw * error_yaw
        p_pitch = kp_pitch * error_pitch
        
        # Integral 누적 (Steady State Error 제거)
        self.integral_yaw += error_yaw * dt
        self.integral_pitch += error_pitch * dt
        
        # Integral 제한 (windup 방지, 하지만 충분히 크게 설정)
        max_integral = math.radians(60.0)  # 60도로 제한 (Steady State Error 제거를 위해 증가)
        self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))
        self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
        
        i_yaw = ki_yaw * self.integral_yaw
        i_pitch = ki_pitch * self.integral_pitch
        
        d_error_yaw = (error_yaw - self.last_error_yaw) / dt if dt > 0 else 0.0
        d_error_pitch = (error_pitch - self.last_error_pitch) / dt if dt > 0 else 0.0
        
        d_yaw = kd_yaw * d_error_yaw
        d_pitch = kd_pitch * d_error_pitch
        
        delta_yaw_rad = p_yaw + i_yaw + d_yaw
        delta_pitch_rad = p_pitch + i_pitch + d_pitch
        
        # 디버깅: PID 제어 확인
        if not use_searching_gain and abs(error_yaw) > math.radians(2.0):
            self.get_logger().debug(
                f"PID 제어: 목표={math.degrees(target_yaw_rad):.1f}도, "
                f"현재={math.degrees(self.current_yaw_rad):.1f}도, "
                f"오차={math.degrees(error_yaw):.1f}도, "
                f"Kp={kp_yaw:.2f}, Ki={ki_yaw:.2f}, "
                f"P={math.degrees(p_yaw):.2f}도, I={math.degrees(i_yaw):.2f}도, "
                f"D={math.degrees(d_yaw):.2f}도, "
                f"증분={math.degrees(delta_yaw_rad):.2f}도"
            )
        
        self.last_error_yaw = error_yaw
        self.last_error_pitch = error_pitch
        self.last_update_time = current_time
        
        return delta_yaw_rad, delta_pitch_rad
    
    def _pid_control_waist(self, target_waist_yaw_rad: float) -> float:
        """허리 PID 제어를 사용하여 증분 명령 계산"""
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        # 오차 계산: 목표 - 현재
        error_waist_yaw = target_waist_yaw_rad - self.current_waist_yaw_rad
        
        # P 항
        p_waist_yaw = self.kp_waist_yaw * error_waist_yaw
        
        # Integral 누적 (Steady State Error 제거)
        self.integral_waist_yaw += error_waist_yaw * dt
        
        # Integral 제한 (windup 방지)
        max_integral_waist = math.radians(60.0)  # 60도로 제한
        self.integral_waist_yaw = max(-max_integral_waist, min(max_integral_waist, self.integral_waist_yaw))
        
        i_waist_yaw = self.ki_waist_yaw * self.integral_waist_yaw
        
        # D 항 계산
        d_error_waist_yaw = (error_waist_yaw - self.last_error_waist_yaw) / dt if dt > 0 else 0.0
        d_waist_yaw = self.kd_waist_yaw * d_error_waist_yaw
        
        delta_waist_yaw_rad = p_waist_yaw + i_waist_yaw + d_waist_yaw
        
        # 디버깅: 허리 PID 제어 확인
        if abs(error_waist_yaw) > math.radians(2.0):
            self.get_logger().debug(
                f"허리 PID 제어: 목표={math.degrees(target_waist_yaw_rad):.1f}도, "
                f"현재={math.degrees(self.current_waist_yaw_rad):.1f}도, "
                f"오차={math.degrees(error_waist_yaw):.1f}도, "
                f"Kp={self.kp_waist_yaw:.2f}, Ki={self.ki_waist_yaw:.2f}, "
                f"P={math.degrees(p_waist_yaw):.2f}도, I={math.degrees(i_waist_yaw):.2f}도, "
                f"D={math.degrees(d_waist_yaw):.2f}도, "
                f"증분={math.degrees(delta_waist_yaw_rad):.2f}도"
            )
        
        self.last_error_waist_yaw = error_waist_yaw
        self.last_waist_update_time = current_time
        
        return delta_waist_yaw_rad
    
    def _get_breathing_pitch(self) -> float:
        """숨쉬는 모션 Pitch 계산 (-5도에서 5도 사이)"""
        current_time = time.monotonic()
        elapsed_time = current_time - self.breathing_start_time
        # sin 함수로 부드러운 숨쉬는 모션 생성
        breathing_pitch_rad = math.sin(2.0 * math.pi * elapsed_time / self.breathing_period_sec) * math.radians(self.breathing_amplitude_deg)
        return breathing_pitch_rad
    
    def _send_waist_command(self, target_waist_yaw_rad: float, use_pid: bool = True, enable_breathing: bool = False) -> float:
        """허리 명령 전송 - PID 제어 후 절대각도로 전송"""
        # 목표 각도를 제한 범위 내로 클리핑
        target_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, target_waist_yaw_rad))
        
        if use_pid:
            # PID 제어로 증분 계산
            delta_waist_yaw_rad = self._pid_control_waist(target_waist_yaw_rad)
        else:
            # PID 없이 직접 증분 계산
            delta_waist_yaw_rad = target_waist_yaw_rad - self.current_waist_yaw_rad
        
        # 증분을 현재 위치에 더해서 절대각도로 변환
        absolute_waist_yaw_rad = self.current_waist_yaw_rad + delta_waist_yaw_rad
        
        # 절대각도 제한 확인
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        
        # 숨쉬는 모션 추가 (허리 Pitch)
        waist_pitch_rad = 0.0
        if enable_breathing:
            waist_pitch_rad = self._get_breathing_pitch()
        
        # 절대각도 명령으로 전송
        msg = Float64MultiArray()
        msg.data = [float(absolute_waist_yaw_rad), float(waist_pitch_rad)]  # [yaw, pitch] 순서
        self.waist_publisher.publish(msg)
        
        return absolute_waist_yaw_rad
    
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float, use_pid: bool = True, use_searching_gain: bool = False) -> Tuple[float, float]:
        """목 명령 전송 - PID 제어 후 절대각도로 전송"""
        # 목표 각도를 60도 제한 범위 내로 클리핑
        target_yaw_rad, target_pitch_rad = self._clip_angles(target_yaw_rad, target_pitch_rad)
        
        self.target_yaw_rad = target_yaw_rad
        self.target_pitch_rad = target_pitch_rad
        
        if use_pid:
            # PID 제어로 증분 계산
            delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad, use_searching_gain=use_searching_gain)
        else:
            # PID 없이 직접 증분 계산
            delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
            delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
        
        # 증분을 현재 위치에 더해서 절대각도로 변환
        absolute_yaw_rad = self.current_yaw_rad + delta_yaw_rad
        absolute_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        
        # 절대각도 제한 확인
        absolute_yaw_rad, absolute_pitch_rad = self._clip_angles(absolute_yaw_rad, absolute_pitch_rad)
        
        # 절대각도 명령으로 전송
        msg = Float64MultiArray()
        msg.data = [float(absolute_pitch_rad), float(absolute_yaw_rad)]  # [pitch, yaw] 순서, 절대각도 명령
        self.neck_publisher.publish(msg)
        
        return absolute_yaw_rad, absolute_pitch_rad
    
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
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                target_x, target_y = target_info.point
                
                # 디버깅: 타겟 정보 확인
                self.get_logger().debug(
                    f"🎯 TRACKING: 타겟 ID={target_info.track_id}, "
                    f"포인트=({target_x:.1f}, {target_y:.1f}), "
                    f"프레임 크기=({frame_width:.0f}x{frame_height:.0f})"
                )
                
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                
                # 상대 각도를 먼저 제한 (한 번에 너무 큰 움직임 방지)
                # 하지만 절대 각도 제한을 고려하여 조정
                max_relative_yaw_rad = math.radians(90.0)  # 상대 각도 최대 ±90도
                relative_yaw_rad = max(-max_relative_yaw_rad, min(max_relative_yaw_rad, relative_yaw_rad))
                
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # 목표 각도를 60도 제한 범위 내로 클리핑 (절대 각도 기준)
                target_yaw_rad, target_pitch_rad = self._clip_angles(target_yaw_rad, target_pitch_rad)
                
                # 디버깅: 각도 정보 출력
                self.get_logger().debug(
                    f"각도 계산: 현재={math.degrees(self.current_yaw_rad):.1f}도, "
                    f"상대={math.degrees(relative_yaw_rad):.1f}도, "
                    f"목표={math.degrees(target_yaw_rad):.1f}도"
                )
                
                yaw_rad, pitch_rad = self._send_neck_command(target_yaw_rad, target_pitch_rad, use_pid=True)
                
                # 허리 Pitch 숨쉬는 모션 적용 (TRACKING 상태)
                # 허리는 현재 yaw 위치 유지하고 Pitch만 숨쉬는 모션
                self._send_waist_command(self.current_waist_yaw_rad, use_pid=False, enable_breathing=True)
                
                return yaw_rad, pitch_rad
            
            case TrackingState.INTERACTION:
                # INTERACTION 모드: BB Box만 따고 목 명령 전송 안 함
                self.searching_start_time = None
                self.search_phase = 0
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                # 목 명령 전송 없이 현재 위치 유지
                return self.current_yaw_rad, self.current_pitch_rad
            
            case TrackingState.LOST:
                self.searching_start_time = None
                self.search_phase = 0
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                # LOST 상태: 마지막 목 위치를 절대각도로 유지 (0도로 돌아가지 않음)
                # 현재 위치를 절대각도로 직접 전송하여 유지 (발산 방지)
                # PID를 사용하지 않음 (에러가 0이어도 Integral 누적으로 발산 가능)
                msg = Float64MultiArray()
                msg.data = [float(self.current_pitch_rad), float(self.current_yaw_rad)]  # [pitch, yaw] 순서, 절대각도 명령
                self.neck_publisher.publish(msg)
                
                # 허리도 숨쉬는 모션 추가
                waist_pitch_rad = self._get_breathing_pitch()
                msg_waist = Float64MultiArray()
                msg_waist.data = [float(self.current_waist_yaw_rad), float(waist_pitch_rad)]  # [yaw, pitch] 순서
                self.waist_publisher.publish(msg_waist)
                
                # Integral 초기화하여 누적 에러 제거
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                self.last_error_yaw = 0.0
                self.last_error_pitch = 0.0
                
                return self.current_yaw_rad, self.current_pitch_rad
            
            case TrackingState.SEARCHING:
                # 증분 방식으로 목표 위치 계산
                command_yaw_rad, command_pitch_rad = self._searching_behavior()
                self.target_yaw_rad = command_yaw_rad
                self.target_pitch_rad = command_pitch_rad
                
                # 각도 제한 (절대각도 기준)
                command_yaw_rad, command_pitch_rad = self._clip_angles(command_yaw_rad, command_pitch_rad)
                
                # 절대각도로 명령 전송 (PID 사용하여 부드럽게 이동)
                yaw_rad, pitch_rad = self._send_neck_command(command_yaw_rad, command_pitch_rad, use_pid=True, use_searching_gain=True)
                
                # 허리 Pitch 숨쉬는 모션 적용 (SEARCHING 상태)
                # 허리는 현재 yaw 위치 유지하고 Pitch만 숨쉬는 모션
                self._send_waist_command(self.current_waist_yaw_rad, use_pid=False, enable_breathing=True)
                
                # 디버깅: SEARCHING 각도 정보 출력
                self.get_logger().debug(
                    f"SEARCHING: 현재={math.degrees(self.current_yaw_rad):.1f}도, "
                    f"목표={math.degrees(command_yaw_rad):.1f}도, "
                    f"phase={self.search_phase}"
                )
                
                return yaw_rad, pitch_rad
            
            case TrackingState.WAIST_FOLLOWER:
                self.searching_start_time = None
                self.search_phase = 0
                
                if self.waist_follower_initial_neck_yaw is None:
                    self.waist_follower_initial_neck_yaw = self.current_yaw_rad + self.current_waist_yaw_rad
                
                target_neck_yaw = 0.0
                target_neck_pitch = self.current_pitch_rad
                
                current_time = time.monotonic()
                dt = current_time - self.last_update_time
                dt = max(0.001, min(dt, 0.1))
                
                # 오차 계산 (WAIST_FOLLOWER 상태는 고정 게인 사용, 스무딩으로 안정성 확보)
                error_neck_yaw = target_neck_yaw - self.current_yaw_rad
                error_neck_pitch = target_neck_pitch - self.current_pitch_rad
                
                # WAIST_FOLLOWER 상태는 고정 게인 사용 (스무딩으로 안정성 확보)
                p_neck_yaw = self.kp_neck_yaw_waist_mode * error_neck_yaw
                p_neck_pitch = self.kp_pitch_searching * error_neck_pitch
                
                self.integral_neck_yaw_waist_mode += error_neck_yaw * dt
                max_integral_neck = math.radians(30.0)
                self.integral_neck_yaw_waist_mode = max(-max_integral_neck, min(max_integral_neck, self.integral_neck_yaw_waist_mode))
                i_neck_yaw = self.ki_neck_yaw_waist_mode * self.integral_neck_yaw_waist_mode
                
                self.integral_pitch += error_neck_pitch * dt
                max_integral_pitch = math.radians(30.0)
                self.integral_pitch = max(-max_integral_pitch, min(max_integral_pitch, self.integral_pitch))
                i_neck_pitch = self.ki_pitch_searching * self.integral_pitch
                
                # D 항 계산
                d_error_neck_yaw = (error_neck_yaw - self.last_error_neck_yaw_waist_mode) / dt if dt > 0 else 0.0
                d_error_neck_pitch = (error_neck_pitch - self.last_error_pitch) / dt if dt > 0 else 0.0
                d_neck_yaw = self.kd_neck_yaw_waist_mode * d_error_neck_yaw
                d_neck_pitch = self.kd_pitch_searching * d_error_neck_pitch
                
                relative_neck_yaw = p_neck_yaw + i_neck_yaw + d_neck_yaw
                relative_neck_pitch = p_neck_pitch + i_neck_pitch + d_neck_pitch
                
                # 증분을 현재 위치에 더해서 절대각도로 변환 (다른 상태와 동일한 방식)
                absolute_neck_yaw_rad = self.current_yaw_rad + relative_neck_yaw
                absolute_neck_pitch_rad = self.current_pitch_rad + relative_neck_pitch
                
                # 절대각도 제한 확인
                absolute_neck_yaw_rad, absolute_neck_pitch_rad = self._clip_angles(absolute_neck_yaw_rad, absolute_neck_pitch_rad)
                
                # 절대각도 명령으로 전송 (다른 상태와 동일한 방식)
                msg_neck = Float64MultiArray()
                msg_neck.data = [float(absolute_neck_pitch_rad), float(absolute_neck_yaw_rad)]
                self.neck_publisher.publish(msg_neck)
                
                self.last_error_neck_yaw_waist_mode = error_neck_yaw
                self.last_error_pitch = error_neck_pitch
                self.last_update_time = current_time
                
                target_waist_yaw = self.waist_follower_initial_neck_yaw
                self._send_waist_command(target_waist_yaw, use_pid=True, enable_breathing=True)
                
                final_neck_target = 0.0
                final_neck_error = abs(math.degrees(final_neck_target - self.current_yaw_rad))
                waist_error_deg = abs(math.degrees(target_waist_yaw - self.current_waist_yaw_rad))
                target_reached_threshold_deg = 1.0
                
                if final_neck_error <= target_reached_threshold_deg and waist_error_deg <= target_reached_threshold_deg:
                    request = {
                        'type': 'set_state',
                        'state': 'tracking',
                        'target_id': target_info.track_id
                    }
                    msg = String()
                    msg.data = json.dumps(request)
                    self.tracker_state_request_publisher.publish(msg)
                    self.get_logger().info(
                        f"WAIST_FOLLOWER 목표 도착: 목={final_neck_error:.2f}도, 허리={waist_error_deg:.2f}도 → TRACKING 상태로 전환 요청"
                    )
                
                return self.current_yaw_rad + relative_neck_yaw, self.current_pitch_rad + relative_neck_pitch
            
            case TrackingState.IDLE | _:
                self.searching_start_time = None
                self.search_phase = 0
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                # IDLE 상태: 목표가 없으므로 현재 위치를 유지 (발산 방지)
                # Integral 초기화하여 누적 에러 제거
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                self.last_error_yaw = 0.0
                self.last_error_pitch = 0.0
                
                # 현재 위치를 목표로 설정하여 유지
                target_neck_yaw = self.current_yaw_rad
                target_neck_pitch = self.current_pitch_rad
                
                # PID 없이 현재 위치 유지 (발산 방지)
                yaw_rad, pitch_rad = self._send_neck_command(
                    target_neck_yaw, 
                    target_neck_pitch, 
                    use_pid=False  # PID 사용 안 함 (발산 방지)
                )
                
                # 허리는 0도로 유지 (PID 사용), 숨쉬는 모션 포함
                target_waist_yaw = 0.0
                self._send_waist_command(target_waist_yaw, use_pid=True, enable_breathing=True)
                
                return yaw_rad, pitch_rad
    
    def get_current_angles(self) -> Tuple[float, float]:
        """현재 목 각도 반환"""
        return self.current_yaw_rad, self.current_pitch_rad
    
    def get_target_angles(self) -> Tuple[float, float]:
        """목표 명령 각도 반환"""
        return self.target_yaw_rad, self.target_pitch_rad
    
    def get_waist_angles(self) -> Tuple[float, float]:
        """허리 각도 반환"""
        if self.waist_follower_initial_neck_yaw is not None:
            target_waist_yaw = self.waist_follower_initial_neck_yaw
        else:
            target_waist_yaw = 0.0
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
            
            if state == TrackingState.TRACKING:
                current_yaw, _ = self.get_current_angles()
                self._check_neck_stability(current_yaw)
            
            if state == TrackingState.WAIST_FOLLOWER:
                current_yaw, current_pitch = self.get_current_angles()
                current_waist_yaw, target_waist_yaw = self.get_waist_angles()
                
                final_neck_target = 0.0
                final_neck_error = abs(math.degrees(final_neck_target - current_yaw))
                waist_error_deg = abs(math.degrees(target_waist_yaw - current_waist_yaw))
                target_reached_threshold_deg = 1.0
                
                if final_neck_error <= target_reached_threshold_deg and waist_error_deg <= target_reached_threshold_deg:
                    request = {
                        'type': 'set_state',
                        'state': 'tracking',
                        'target_id': target_info.track_id
                    }
                    msg = String()
                    msg.data = json.dumps(request)
                    self.tracker_state_request_publisher.publish(msg)
                    self.get_logger().info(
                        f"WAIST_FOLLOWER 목표 도착: 목={final_neck_error:.2f}도, 허리={waist_error_deg:.2f}도 → TRACKING 상태로 전환 요청"
                    )
            
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
    
    def _check_neck_stability(self, current_neck_yaw_rad: float):
        """목 각도 안정성 확인 및 WAIST_FOLLOWER 전이 요청"""
        current_time = time.monotonic()
        
        if self.last_neck_yaw_rad is None:
            self.last_neck_yaw_rad = current_neck_yaw_rad
            self.neck_stable_start_time = None
            return
        
        angle_change_deg = abs(math.degrees(current_neck_yaw_rad - self.last_neck_yaw_rad))
        
        if angle_change_deg <= self.neck_stable_threshold_deg:
            if self.neck_stable_start_time is None:
                self.neck_stable_start_time = current_time
            
            elapsed_time = current_time - self.neck_stable_start_time
            if elapsed_time >= self.neck_stable_duration:
                request = {
                    'type': 'set_state',
                    'state': 'waist_follower',
                    'target_id': None
                }
                msg = String()
                msg.data = json.dumps(request)
                self.tracker_state_request_publisher.publish(msg)
                self.neck_stable_start_time = None
        else:
            self.neck_stable_start_time = None
        
        self.last_neck_yaw_rad = current_neck_yaw_rad


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
