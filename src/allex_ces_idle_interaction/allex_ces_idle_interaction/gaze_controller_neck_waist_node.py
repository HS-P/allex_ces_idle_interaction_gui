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
        
        # 전체 시선각 스무딩용 (부드러운 움직임을 위해)
        self.last_desired_total_yaw = None  # 마지막 목표 전체 시선각
        self.total_yaw_smoothing_alpha = 0.6  # 전체 시선각 스무딩 계수 (0.4 -> 0.6: 30% 더 빠르게)
        
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
        self.kp_yaw = 1.1   # P 게인 (Yaw)
        self.kp_pitch = 1.2 # P 게인 (Pitch)
        self.ki_yaw = 0.02    # I 게인 (Yaw) - Steady State Error 제거용
        self.ki_pitch = 0.12  # I 게인 (Pitch)
        self.kd_yaw = 0.0   # D 게인 (Yaw) - 낮춰서 움직임 억제 감소
        self.kd_pitch = 0.01 # D 게인 (Pitch)
        
        # SEARCHING 상태용 매우 낮은 게인 (매우 천천히 움직임)
        self.kp_yaw_searching = 0.3   # P 게인 (Yaw) - 검색 시 매우 느리게
        self.kp_pitch_searching = 0.3 # P 게인 (Pitch) - 검색 시 매우 느리게
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
        self.kp_waist_tracking = 1.5 # P 게인 (Waist Yaw) - 진동 방지
        
        # 허리 지연 파라미터 (Exponential smoothing 시간 상수, 초)
        # 목표지점 도달까지 30% 더 빠르게: tau를 30% 줄임 (1.5 -> 1.0)
        self.tau_waist = 1.0  # TRACKING용 느린 지연 (30% 더 빠른 도달)
        self.tau_waist_searching = 2.5  # SEARCHING 모드용 매우 느린 지연 (부드러운 움직임)
        self.max_delta_waist_tracking = math.radians(0.1)  # TRACKING 모드 허리 최대 변화량 (GUI에서 제어 가능)
        
        # 허리 제어 상태 변수
        self.last_waist_update_time = time.monotonic()
        self.last_waist_command = None  # 마지막으로 보낸 허리 명령 각도 (절대각, None이면 초기화 필요)
        
        # TRACKING 초기 빠른 추적용 변수
        self.tracking_start_time = None  # 현재 track_id로 추적 시작 시간
        self.current_track_id = None  # 현재 추적 중인 track_id
        self.initial_tracking_duration = 5.0  # 초기 빠른 추적 시간 (초)
        self.initial_tracking_max_delta = math.radians(3.5)  # 초기 추적 시 최대 변화량 (3.5도/프레임)
        
        # 실행 상태 플래그
        self.is_running = False
        
        # HELLO 전환 조건 변수 (현재 위치에서 ±1도 이내로 1초 유지)
        self.hello_position_threshold_deg = 1.5  # 기준 위치에서 ±1도 이내
        self.hello_stable_duration = 2.0  # 조건 유지 시간 (초)
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
        """SEARCHING 상태 동작: 랜덤 각도로 좌우 스캔
        - 좌측: -40도 ~ -10도 중 랜덤 선택
        - 우측: 10도 ~ 40도 중 랜덤 선택
        - 목표 도달 판정: Waist + Head = Target (전체 시선각 기준)
        - 반환: (목표 전체 시선각, pitch)
        """
        if self.searching_start_time is None:
            self.searching_start_time = time.monotonic()
            
            # 랜덤 각도 선택 (매번 스캔 시작 시 새로운 각도)
            self.search_left_angle = random.uniform(-40.0, -10.0)  # -40도 ~ -10도 중 랜덤
            self.search_right_angle = random.uniform(10.0, 40.0)   # 10도 ~ 40도 중 랜덤
            
            # 현재 전체 시선각 위치 계산
            current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
            left_target = self.home_yaw_rad + math.radians(self.search_left_angle)
            right_target = self.home_yaw_rad + math.radians(self.search_right_angle)
            
            # 현재 위치와 각 목표 사이의 거리 계산
            dist_to_left = abs(current_total_yaw - left_target)
            dist_to_right = abs(current_total_yaw - right_target)
            
            # 더 가까운 방향부터 시작
            if dist_to_left <= dist_to_right:
                self.search_phase = 1  # 좌측부터 시작
                self.search_target_yaw = left_target
            else:
                self.search_phase = 0  # 우측부터 시작
                self.search_target_yaw = right_target
        
        # 목표 도달 임계값 (약 2도 이내) - 전체 시선각 기준
        target_reached_threshold = math.radians(2.0)
        
        # 현재 전체 시선각 계산
        current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
        
        if self.search_phase == 0:
            # 우측으로 이동 (10도 ~ 40도 중 선택된 각도)
            target_total_yaw = self.home_yaw_rad + math.radians(self.search_right_angle)
            target_total_yaw = min(target_total_yaw, self.yaw_max)
            self.search_target_yaw = target_total_yaw
            
            # 전체 시선각 기준으로 목표 도달 판정
            total_error = target_total_yaw - current_total_yaw
            if abs(total_error) <= target_reached_threshold:
                # 목표에 도달 -> 좌측으로 전환
                self.search_phase = 1
                self.search_target_yaw = self.home_yaw_rad + math.radians(self.search_left_angle)
                self.search_target_yaw = max(self.search_target_yaw, self.yaw_min)
        
        elif self.search_phase == 1:
            # 좌측으로 이동 (-40도 ~ -10도 중 선택된 각도)
            target_total_yaw = self.home_yaw_rad + math.radians(self.search_left_angle)
            target_total_yaw = max(target_total_yaw, self.yaw_min)
            self.search_target_yaw = target_total_yaw
            
            # 전체 시선각 기준으로 목표 도달 판정
            total_error = target_total_yaw - current_total_yaw
            if abs(total_error) <= target_reached_threshold:
                # 목표에 도달 -> 우측으로 전환 (새로운 랜덤 각도 선택)
                self.search_phase = 0
                # 새로운 랜덤 각도 선택
                self.search_left_angle = random.uniform(-40.0, -10.0)
                self.search_right_angle = random.uniform(10.0, 40.0)
                self.search_target_yaw = self.home_yaw_rad + math.radians(self.search_right_angle)
                self.search_target_yaw = min(self.search_target_yaw, self.yaw_max)
        
        # SEARCHING 시 목 Pitch를 10도 기울임
        searching_pitch_rad = self.home_pitch_rad + math.radians(10.0)
        
        # 목표 전체 시선각과 pitch 반환
        return self.search_target_yaw, searching_pitch_rad
    
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
            
            # Rate limit: 증분 제한 (너무 급격한 변화 방지, SEARCHING은 매우 느리게)
            max_delta = math.radians(2.0)  # SEARCHING 모드: 매우 느린 속도 (초당 약 2도, 30Hz 기준)
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
            # 초기 5초간은 빠른 추적 (3.5도/프레임), 이후는 매우 느리게 (초당 약 2-3도, 프레임당 약 0.1도)
            if (self.tracking_start_time is not None and 
                current_time - self.tracking_start_time < self.initial_tracking_duration):
                max_delta = self.initial_tracking_max_delta  # 초기 빠른 추적 (3.5도/프레임)
            else:
                # TRACKING 모드: 매우 느린 속도 (초당 약 2-3도, 30Hz 기준 프레임당 약 0.08도)
                # 목이 빠르게 추종하고 허리는 천천히 따라가도록
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
            is_initial_tracking: 초기 TRACKING 단계 여부 (5초 이내)
        """
        self.target_yaw_rad = target_yaw_rad
        self.target_pitch_rad = target_pitch_rad
        
        if use_pid:
            delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad, use_searching_gain=use_searching_gain)
        else:
            delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
            delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
        
        # TRACKING 모드에서는 목이 항상 빠르게 추종 (허리가 느리게 따라가므로)
        # 초기 추적 단계에서는 더 빠르게 (3.5도/프레임), 일반 추적에서는 30도/프레임 유지
        if is_initial_tracking:
            max_delta_angle = self.initial_tracking_max_delta  # 초기: 3.5도/프레임
        else:
            max_delta_angle = math.radians(30.0)  # 일반 추적: 30도/프레임 (빠른 추종)
        
        # 기본 rate limit
        delta_yaw_rad = max(-max_delta_angle, min(max_delta_angle, delta_yaw_rad))
        delta_pitch_rad = max(-max_delta_angle, min(max_delta_angle, delta_pitch_rad))
        
        # PID 제어 결과 스무딩 (더 부드러운 움직임을 위해)
        # TRACKING 모드에서는 목이 빠르게 추종해야 하므로 스무딩을 적절히 적용
        # (허리가 느리게 따라가므로 목이 먼저 빠르게 움직여야 하지만, 끊김 방지를 위해 스무딩 필요)
        # 목표지점 도달까지 30% 더 빠르게: alpha를 높여서 더 빠르게 반응 (0.5 -> 0.65, 0.2 -> 0.35)
        if use_searching_gain:
            # SEARCHING 모드: 더 부드럽지만 30% 빠르게
            smoothing_alpha = 0.35  # SEARCHING에서는 부드럽지만 더 빠르게 (기존 0.2에서 75% 증가)
        else:
            # TRACKING 모드: 빠른 반응 (30% 더 빠르게)
            smoothing_alpha = 0.65  # TRACKING에서는 더 빠른 반응 (기존 0.5에서 30% 증가)
        
        smoothed_delta_yaw = self.last_neck_delta_yaw + smoothing_alpha * (delta_yaw_rad - self.last_neck_delta_yaw)
        smoothed_delta_pitch = self.last_neck_delta_pitch + smoothing_alpha * (delta_pitch_rad - self.last_neck_delta_pitch)
        
        # 스무딩된 값도 rate limit 적용
        smoothed_delta_yaw = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_yaw))
        smoothed_delta_pitch = max(-max_delta_angle, min(max_delta_angle, smoothed_delta_pitch))
        
        # 마지막 증분 저장
        self.last_neck_delta_yaw = smoothed_delta_yaw
        self.last_neck_delta_pitch = smoothed_delta_pitch
        
        msg = Float64MultiArray()
        msg.data = [float(smoothed_delta_pitch), float(smoothed_delta_yaw)]  # [pitch, yaw] 순서, 증분 명령
        self.neck_publisher.publish(msg)
        
        expected_yaw_rad = self.current_yaw_rad + smoothed_delta_yaw
        expected_pitch_rad = self.current_pitch_rad + smoothed_delta_pitch
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
                
                # Track ID 변경 감지 (새로운 사람 인식 시 초기 추적 시간 설정)
                current_time_check = time.monotonic()
                if self.current_track_id != target_info.track_id:
                    self.current_track_id = target_info.track_id
                    self.tracking_start_time = current_time_check
                    self.get_logger().info(
                        f"새 타겟 추적 시작: track_id={target_info.track_id}, "
                        f"초기 {self.initial_tracking_duration}초간 빠른 추적 모드"
                    )
                
                # 초기 추적 단계 여부 확인 (5초 이내)
                is_initial_tracking = (
                    self.tracking_start_time is not None and
                    current_time_check - self.tracking_start_time < self.initial_tracking_duration
                )
                
                # 픽셀 오차를 상대 각도로 변환
                target_x, target_y = target_info.point
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                
                # 전체 시선각 계산: 현재 total gaze + 필요한 변화량
                current_total_yaw = self.current_waist_yaw_rad + self.current_yaw_rad
                raw_desired_total_yaw = current_total_yaw + relative_yaw_rad
                
                # 전체 시선각 스무딩 (부드러운 움직임을 위해)
                if self.last_desired_total_yaw is not None:
                    # Exponential smoothing: smoothed = last + alpha * (current - last)
                    desired_total_yaw = self.last_desired_total_yaw + self.total_yaw_smoothing_alpha * (raw_desired_total_yaw - self.last_desired_total_yaw)
                else:
                    desired_total_yaw = raw_desired_total_yaw
                    self.last_desired_total_yaw = desired_total_yaw
                
                # 허리 제어: 전체 시선각을 느리게 추종 (exponential smoothing, 반 박자 지연)
                waist_cmd = self._waist_follow_total(desired_total_yaw, searching_mode=False)
                
                # 전체 시선각 업데이트
                self.last_desired_total_yaw = desired_total_yaw
                
                # 목 제어: 잔여분만 담당 (전체 시선각 - 허리 각도)
                neck_target_yaw = desired_total_yaw - waist_cmd
                
                # 목 각도 안전 범위 클램프 (허리가 포화되어 과도하게 커지는 경우 방지)
                neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))
                
                # LOST -> TRACKING 전환 시 급격한 변화 방지: 목 목표 각도 스무딩
                # 목은 항상 빠르게 반응하도록 (허리가 느리게 따라가므로 목이 빠르게 추종해야 함)
                if self.last_neck_target_yaw is not None:
                    # 목은 항상 빠른 반응 (프레임당 최대 12도 변화) - 허리가 느리게 따라가므로
                    max_neck_target_delta = math.radians(12.0)
                    
                    # Rate limit 적용
                    neck_target_delta = neck_target_yaw - self.last_neck_target_yaw
                    neck_target_delta = max(-max_neck_target_delta, min(max_neck_target_delta, neck_target_delta))
                    
                    # Exponential smoothing으로 더 부드럽게 (끊김 방지)
                    # 목표 각도 변화에 대한 스무딩 (0.85 = 기본값, GUI에서 neck_target_alpha로 제어 가능)
                    smoothing_factor = getattr(self, 'neck_target_alpha', 0.85)  # 목표 각도 변화에 대한 스무딩 (더 빠른 도달)
                    neck_target_yaw = self.last_neck_target_yaw + smoothing_factor * neck_target_delta
                else:
                    # 초기화: 현재 목 각도로 설정
                    self.last_neck_target_yaw = self.current_yaw_rad
                
                # Pitch는 허리가 관여하지 않으므로 기존 방식 유지
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # 목 명령 전송 (PID 제어, TRACKING에서는 항상 빠른 추종)
                # is_initial_tracking은 False로 전달 (초기 추적 후에도 목은 빠르게 유지)
                yaw_rad, pitch_rad = self._send_neck_command(
                    neck_target_yaw, target_pitch_rad, 
                    use_pid=True, 
                    is_initial_tracking=False  # 목은 항상 빠르게 추종 (허리가 느리게 따라가므로)
                )
                
                # 목표 각도 업데이트
                self.last_neck_target_yaw = neck_target_yaw
                
                # 디버깅 로그 (1초마다 출력)
                if not hasattr(self, '_last_tracking_log_time') or time.monotonic() - self._last_tracking_log_time > 1.0:
                    self.get_logger().info(
                        f"TRACKING: 전체시선각 기반 제어 | "
                        f"relative_yaw={math.degrees(relative_yaw_rad):.2f}도, "
                        f"current_total={math.degrees(current_total_yaw):.2f}도, "
                        f"desired_total={math.degrees(desired_total_yaw):.2f}도, "
                        f"waist_cmd={math.degrees(waist_cmd):.2f}도, "
                        f"neck_target={math.degrees(neck_target_yaw):.2f}도, "
                        f"neck_current={math.degrees(self.current_yaw_rad):.2f}도"
                    )
                    self._last_tracking_log_time = time.monotonic()
                
                return yaw_rad, pitch_rad
            
            case TrackingState.LOST:
                # LOST 상태: 현재 위치 유지, 명령 전송 안 함 (떨림 방지)
                self.searching_start_time = None
                self.search_phase = 0
                self._reset_waist_pid()
                self._reset_hello_check()
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                # 목 목표 각도 초기화 (TRACKING 진입 시 부드러운 전환을 위해)
                self.last_neck_target_yaw = None
                
                # 추적 시간 초기화
                self.tracking_start_time = None
                self.current_track_id = None
                
                # 전체 시선각 스무딩 초기화
                self.last_desired_total_yaw = None
                
                # 명령 전송 없이 현재 위치 유지
                return self.current_yaw_rad, self.current_pitch_rad
            
            case TrackingState.HELLO | TrackingState.HANDSHAKE:
                # HELLO/HANDSHAKE 상태에서도 TRACKING과 동일하게 목/허리 제어
                self.searching_start_time = None
                self.search_phase = 0
                # HELLO/HANDSHAKE 상태에서 neck/waist PID 상태는 유지 (진동 방지 초기화 X)
                if target_info.point is None:
                    # 타겟 포인트 없으면 현재 위치 유지
                    return self.current_yaw_rad, self.current_pitch_rad
                
                target_x, target_y = target_info.point
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(target_x, target_y, frame_width, frame_height)
                
                # 목표 각도 계산
                raw_target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                raw_target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # 목표 각도 스무딩 (부드러운 움직임)
                if self.last_neck_target_yaw is not None:
                    # Yaw 스무딩
                    yaw_delta = raw_target_yaw_rad - self.last_neck_target_yaw
                    max_yaw_delta = math.radians(10.0)  # 프레임당 최대 변화량
                    yaw_delta = max(-max_yaw_delta, min(max_yaw_delta, yaw_delta))
                    target_yaw_rad = self.last_neck_target_yaw + 0.7 * yaw_delta  # 스무딩 적용
                else:
                    target_yaw_rad = raw_target_yaw_rad
                    self.last_neck_target_yaw = target_yaw_rad
                
                # Pitch는 스무딩 없이 사용 (허리가 관여하지 않으므로)
                target_pitch_rad = raw_target_pitch_rad
                
                yaw_rad, pitch_rad = self._send_neck_command(target_yaw_rad, target_pitch_rad, use_pid=True)
                
                # 목표 각도 업데이트
                self.last_neck_target_yaw = target_yaw_rad
                
                # 스무딩 값 업데이트 (LOST/SEARCHING 전환 시 사용)
                self.last_smoothed_yaw_command = yaw_rad
                self.last_smoothed_pitch_command = pitch_rad
                
                # 이전 타겟 ID 업데이트
                self.previous_track_id = target_info.track_id
                
                # 허리 제어: 목 각도를 매우 천천히 추종 (3배 느린 PID) + 120%
                self._waist_follow_neck()
                
                return yaw_rad, pitch_rad
            
            case TrackingState.SEARCHING:
                # SEARCHING 동작: 목표 각도 계산
                command_yaw_rad, command_pitch_rad = self._searching_behavior()
                self.target_yaw_rad = self.search_target_yaw
                self.target_pitch_rad = command_pitch_rad
                
                # TRACKING과 동일한 전체 시선각 기반 제어 알고리즘 적용
                # search_target_yaw는 전체 시선각 목표로 해석
                raw_desired_total_yaw = self.search_target_yaw
                
                # 전체 시선각 스무딩 (SEARCHING도 부드럽지만 30% 더 빠르게)
                if self.last_desired_total_yaw is not None:
                    # SEARCHING에서는 부드럽지만 더 빠르게 (0.3 -> 0.45: 50% 증가, 약 30% 더 빠른 도달)
                    desired_total_yaw = self.last_desired_total_yaw + 0.45 * (raw_desired_total_yaw - self.last_desired_total_yaw)
                else:
                    desired_total_yaw = raw_desired_total_yaw
                    self.last_desired_total_yaw = desired_total_yaw
                
                # 허리 각도 제한 (탐색 범위 제한)
                desired_total_yaw = max(math.radians(-65.0), min(math.radians(65.0), desired_total_yaw))
                
                # 허리 제어: 전체 시선각을 추종 (SEARCHING 모드: 느리게)
                waist_cmd = self._waist_follow_total(desired_total_yaw, searching_mode=True)
                
                # 전체 시선각 업데이트
                self.last_desired_total_yaw = desired_total_yaw
                
                # 목 제어: 잔여분만 담당 (전체 시선각 - 허리 각도)
                neck_target_yaw = desired_total_yaw - waist_cmd
                
                # 목 각도 안전 범위 클램프
                neck_target_yaw = max(self.yaw_min, min(self.yaw_max, neck_target_yaw))
                
                # 목 목표 각도 스무딩 (SEARCHING에서도 부드럽지만 30% 더 빠르게)
                if self.last_neck_target_yaw is not None:
                    neck_delta = neck_target_yaw - self.last_neck_target_yaw
                    max_neck_delta = math.radians(8.0)  # SEARCHING에서는 더 느리게
                    neck_delta = max(-max_neck_delta, min(max_neck_delta, neck_delta))
                    # 0.5 -> 0.65: 30% 더 빠른 도달
                    neck_target_yaw = self.last_neck_target_yaw + 0.65 * neck_delta  # 스무딩 적용
                else:
                    self.last_neck_target_yaw = neck_target_yaw
                
                # Pitch는 기존 방식 유지
                target_pitch_rad = command_pitch_rad
                
                # 목 명령 전송 (PID 제어, SEARCHING 게인 사용)
                yaw_rad, pitch_rad = self._send_neck_command(
                    neck_target_yaw, 
                    target_pitch_rad, 
                    use_pid=True,
                    use_searching_gain=True
                )
                
                # 목표 각도 업데이트
                self.last_neck_target_yaw = neck_target_yaw
                
                # 디버깅 로그 (1초마다 출력)
                if not hasattr(self, '_last_search_log_time') or time.monotonic() - self._last_search_log_time > 1.0:
                    self.get_logger().info(
                        f"SEARCHING: 전체시선각 기반 제어 | "
                        f"desired_total={math.degrees(desired_total_yaw):.2f}도, "
                        f"waist_cmd={math.degrees(waist_cmd):.2f}도, "
                        f"neck_target={math.degrees(neck_target_yaw):.2f}도, "
                        f"neck_current={math.degrees(self.current_yaw_rad):.2f}도, "
                        f"waist_current={math.degrees(self.current_waist_yaw_rad):.2f}도, "
                        f"current_total={math.degrees(self.current_waist_yaw_rad + self.current_yaw_rad):.2f}도, "
                        f"phase={self.search_phase}"
                    )
                    self._last_search_log_time = time.monotonic()
                
                return yaw_rad, pitch_rad
            
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
        """허리 제어 상태 초기화"""
        self.last_waist_command = None  # None으로 설정하여 다음 호출 시 현재 위치로 자동 초기화
    
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
            
            # 3초 유지되면 Depth 값에 따라 HELLO 또는 HANDSHAKE로 전환
            if elapsed_time >= self.hello_stable_duration:
                # 로그용 값 저장
                ref_yaw_deg = math.degrees(self.hello_reference_yaw_rad)
                
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
                    f"HELLO 전환 준비 완료: 기준 위치={ref_yaw_deg:.1f}도에서 "
                    f"±{self.hello_position_threshold_deg}도 이내로 {elapsed_time:.1f}초 유지, "
                    f"track_id={target_track_id}에 대한 depth 기반 분기 판단 요청 전송"
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
        # 목표 허리 각도는 목의 목표 각도 + 현재 각도 (Exponential 추종)
        target_waist_yaw = self.target_yaw_rad + self.current_yaw_rad
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
