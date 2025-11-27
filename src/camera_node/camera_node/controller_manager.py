#!/usr/bin/env python3
"""
목 제어 관리자 - 타겟 추적 정보를 받아서 목 각도 계산 및 명령 전송
"""
from typing import Optional, Tuple
import math
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .tracker_manager import TargetInfo, TrackingState


class ControllerManager:
    """타겟 추적 정보를 받아서 목 제어 명령 생성 및 전송"""
    
    def __init__(self, node: Node):
        """
        ControllerManager 초기화
        
        Args:
            node: ROS2 Node 인스턴스 (Publisher/Subscriber 생성용)
        """
        self.node = node
        
        # 목 명령 Publisher
        self.neck_publisher = node.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_neck/joint_command',
            10
        )
        
        # 목 위치 Subscriber (현재 위치 파악용)
        self.neck_position_subscription = node.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_neck/joint_positions_deg',
            self._neck_position_callback,
            10
        )
        
        # 허리 명령 Publisher
        self.waist_publisher = node.create_publisher(
            Float64MultiArray,
            '/robot_inbound/theOne_waist/joint_command',
            10
        )
        
        # 허리 위치 Subscriber (현재 위치 파악용)
        self.waist_position_subscription = node.create_subscription(
            Float64MultiArray,
            '/robot_outbound_data/theOne_waist/joint_positions_deg',
            self._waist_position_callback,
            10
        )
        
        # 카메라 파라미터 (프레임 크기 기준)
        # 실제 카메라 FOV에 맞게 조정 필요
        self.frame_width = 1280.0  # 프레임 너비 (픽셀)
        self.frame_height = 720.0  # 프레임 높이 (픽셀)
        
        # 각도 제한 범위 (라디안)
        self.yaw_min = -3.31613    # -190°
        self.yaw_max = 1.74533     # 100°
        self.pitch_min = -0.0872665  # -5°
        self.pitch_max = 3.75246   # 215°
        
        # 허리 각도 제한 범위 (라디안)
        self.waist_yaw_min = math.radians(-85.0)  # -75°
        self.waist_yaw_max = math.radians(85.0)   # 75°
        
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
        
        # 목 명령 좌표계 관리
        # 목 명령은 상대좌표로 전송해야 함 (명령 시점의 현재 위치 기준)
        # subscribe로 받는 값들은 전부 절대좌표
        # 상대 각도 = 목표 절대 각도 - 현재 절대 각도 (명령 시점)
        
        # SEARCHING 상태용 스캔 변수
        self.searching_start_time = None
        self.search_phase = 0  # 0: 우측(+30도)로, 1: 좌측(-30도)로
        self.search_target_yaw = 0.0  # 최종 목표 각도 (절대 각도)
        self.search_current_command_yaw = 0.0  # 현재 명령 각도 (증분 방식용)
        self.search_increment_rad = math.radians(0.3)  # 매 프레임마다 증가할 각도 (약 0.3도)
        
        # PID 제어 파라미터 (일반 추적용)
        self.kp_yaw = 1.88   # P 게인 (Yaw)
        self.kp_pitch = 1.75 # P 게인 (Pitch)
        self.ki_yaw = 0.1    # I 게인 (Yaw) - 적분 게인
        self.ki_pitch = 0.1  # I 게인 (Pitch)
        self.kd_yaw = 0.05   # D 게인 (Yaw) - 미분 게인 (선택적)
        self.kd_pitch = 0.05 # D 게인 (Pitch)
        
        # SEARCHING 상태용 낮은 게인 (천천히 움직임)
        # IDLE 상태에서도 사용되므로 조금 더 올림
        self.kp_yaw_searching = 0.7   # P 게인 (Yaw) - 검색 시 / IDLE 시 (0.5 -> 0.7)
        self.kp_pitch_searching = 0.7 # P 게인 (Pitch) - 검색 시 / IDLE 시 (0.5 -> 0.7)
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
        
        # WAIST_FOLLOWER 상태용 변수
        self.waist_follower_initial_neck_yaw = None  # WAIST_FOLLOWER 상태 진입 시점의 목 각도 (허리 목표 각도로 사용)
        self.current_waist_yaw_rad = 0.0  # 현재 허리 각도 (라디안, 절대 좌표)
        self.last_waist_position_update_time = 0.0
        
        
        # WAIST_FOLLOWER 상태용 PID 게인 (허리 제어용)
        # 주의: 허리와 목의 물리적 특성(관성, 모터 토크 등)이 다르므로 
        # 이동 속도를 맞추기 위해 게인값을 조정해야 할 수 있습니다.
        # 허리가 느리면 kp_waist_yaw를 높이고, 빠르면 낮춥니다.
        self.kp_waist_yaw = 1.0   # P 게인 (Waist Yaw) - 사용자가 올린 값
        self.ki_waist_yaw = 0.1  # I 게인 (Waist Yaw) - steady-state error 제거를 위해 증가 (0.01 -> 0.05)
        self.kd_waist_yaw = 0.0  # D 게인 (Waist Yaw) - 안정성 조절
        
        # WAIST_FOLLOWER 상태용 목 제어 PID 게인 (목을 0도로 이동)
        # Searching Gain의 1/3만큼 작게 설정
        self.kp_neck_yaw_waist_mode = self.kp_yaw_searching / 3.0   # P 게인 (Neck Yaw - Waist 모드) - Searching의 1/3
        self.ki_neck_yaw_waist_mode = self.ki_yaw_searching / 3.0  # I 게인 (Neck Yaw - Waist 모드) - Searching의 1/3
        self.kd_neck_yaw_waist_mode = self.kd_yaw_searching / 3.0  # D 게인 (Neck Yaw - Waist 모드) - Searching의 1/3
        
        # 허리 PID 제어 상태 변수
        self.integral_waist_yaw = 0.0
        self.last_error_waist_yaw = 0.0
        self.integral_neck_yaw_waist_mode = 0.0
        self.last_error_neck_yaw_waist_mode = 0.0
        
        # ControllerManager 초기화 완료
    
    # Input : /robot_outbound_data/theOne_neck/joint_positions_deg
    # Output : self.current_yaw_rad, self.current_pitch_rad
    """ 목 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트 """
    def _neck_position_callback(self, msg: Float64MultiArray):
        """
        목 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트
        
        Args:
            msg: Float64MultiArray [pitch_deg, yaw_deg] 순서 (절대 좌표)
        """
        if len(msg.data) >= 2:
            # degree를 radian으로 변환 (절대 좌표)
            pitch_deg = msg.data[0]
            yaw_deg = msg.data[1]
            
            # 절대 좌표로 저장 (subscribe로 받는 값은 전부 절대좌표)
            self.current_pitch_rad = math.radians(pitch_deg)
            self.current_yaw_rad = math.radians(yaw_deg)
            self.last_position_update_time = time.monotonic()
        else:
            self.node.get_logger().warn(
                f"Invalid neck position message: data length={len(msg.data)} (expected >= 2)"
            )
    
    # Input : /robot_outbound_data/theOne_waist/joint_positions_deg
    # Output : self.current_waist_yaw_rad
    """ 허리 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트 """
    def _waist_position_callback(self, msg: Float64MultiArray):
        """
        허리 위치 콜백 - 하드웨어에서 현재 위치를 받아서 업데이트
        
        Args:
            msg: Float64MultiArray [yaw_deg] 순서 (허리는 yaw만)
        """
        if len(msg.data) >= 1:
            # degree를 radian으로 변환
            yaw_deg = msg.data[0]
            
            self.current_waist_yaw_rad = math.radians(yaw_deg)
            self.last_waist_position_update_time = time.monotonic()
        else:
            self.node.get_logger().warn(
                f"Invalid waist position message: data length={len(msg.data)} (expected >= 1)"
            )
    
    # Input : target_x, target_y, frame_width, frame_height
    # Output : relative_yaw_rad, relative_pitch_rad
    """ 타겟 픽셀 좌표를 목 각도로 변환 """
    def _pixel_to_angle(
        self, 
        target_x: float, 
        target_y: float,
        frame_width: float,
        frame_height: float
    ) -> Tuple[float, float]:
        """
        타겟 픽셀 좌표를 목 각도로 변환
        
        Args:
            target_x: 타겟 X 좌표 (픽셀)
            target_y: 타겟 Y 좌표 (픽셀)
            frame_width: 프레임 너비 (픽셀)
            frame_height: 프레임 높이 (픽셀)
            
        Returns:
            tuple: (yaw_rad, pitch_rad) 라디안 단위 각도
        """
        # 프레임 중심점 기준으로 오프셋 계산
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        
        offset_x = target_x - center_x  # 양수: 우측, 음수: 좌측
        offset_y = target_y - center_y  # 양수: 하단, 음수: 상단
        
        # FOV (Field of View) 가정 (실제 카메라 FOV에 맞게 조정 필요)
        # 예: 수평 FOV 60도, 수직 FOV 45도
        # 
        # horizontal_fov_deg 설명:
        # - 값이 크면: 같은 픽셀 오프셋에 대해 더 큰 각도로 변환됨 → 더 크게 움직임
        # - 값이 작으면: 같은 픽셀 오프셋에 대해 더 작은 각도로 변환됨 → 더 작게 움직임
        # 예시: 프레임 중심에서 100픽셀 오프셋일 때
        #   - horizontal_fov_deg = 60도 → 약 4.7도 움직임
        #   - horizontal_fov_deg = 90도 → 약 7.0도 움직임 (더 크게 움직임)
        horizontal_fov_deg = 120.0
        vertical_fov_deg = 45.0
        
        # 픽셀 오프셋을 각도로 변환
        # 각도 = (오프셋 / 프레임크기) * FOV
        # 예: 프레임 너비 1280픽셀, 오프셋 100픽셀, FOV 60도
        #     → 각도 = (100 / 1280) * 60 = 약 4.7도
        yaw_deg = (offset_x / frame_width) * horizontal_fov_deg
        pitch_deg = (offset_y / frame_height) * vertical_fov_deg
        
        # 라디안으로 변환
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        # 방향 정의에 맞게 변환
        # Neck Yaw: 좌측 방향이 양수 (CCW)
        # offset_x가 양수(우측)면 yaw는 음수, offset_x가 음수(좌측)면 yaw는 양수
        yaw_rad = -yaw_rad
        
        # Neck Pitch: 하단 방향이 양수 (CW)
        # offset_y가 양수(하단)면 pitch는 양수, offset_y가 음수(상단)면 pitch는 음수
        # 이미 맞게 계산됨
        
        return yaw_rad, pitch_rad
    
    # Input : yaw_rad, pitch_rad
    # Output : yaw_rad, pitch_rad
    """ 각도를 제한 범위 내로 클리핑 """
    def _clip_angles(self, yaw_rad: float, pitch_rad: float) -> Tuple[float, float]:
        """
        각도를 제한 범위 내로 클리핑
        
        Args:
            yaw_rad: 목 좌우 각도 (라디안)
            pitch_rad: 목 상하 각도 (라디안)
            
        Returns:
            tuple: 클리핑된 (yaw_rad, pitch_rad)
        """
        yaw_rad = max(self.yaw_min, min(self.yaw_max, yaw_rad))
        pitch_rad = max(self.pitch_min, min(self.pitch_max, pitch_rad))
        return yaw_rad, pitch_rad
    
    # Input : None
    # Output : self.current_yaw_rad, self.current_pitch_rad
    """ 현재 목 위치 유지 (마지막 위치 그대로), 타겟을 찾기 전까지 멈추기 위함"""
    def _maintain_current_position(self) -> Tuple[float, float]:
        """
        현재 목 위치 유지 (마지막 위치 그대로)
        
        Returns:
            tuple: (yaw_rad, pitch_rad) 현재 위치
        """
        return self.current_yaw_rad, self.current_pitch_rad
    
    # Input : None
    # Output : self.search_current_command_yaw, self.home_pitch_rad
    """ SEARCHING 상태 동작: 증분 방식으로 천천히 좌우 스캔 (0 -> 30 -> -30 반복), 타겟을 찾기 위함 """
    def _searching_behavior(self) -> Tuple[float, float]:
        """
        SEARCHING 상태 동작: 증분 방식으로 천천히 좌우 스캔 (0 -> 30 -> -30 반복)
        
        Returns:
            tuple: (yaw_rad, pitch_rad) 현재 명령 각도 (증분 방식)
        """
        # SEARCHING 시작 시 초기화
        if self.searching_start_time is None:
            self.searching_start_time = time.monotonic()
            self.search_phase = 0  # 0: 우측(+45도)로, 1: 좌측(-45도)로
            self.search_current_command_yaw = self.current_yaw_rad  # 현재 위치에서 시작
            self.search_target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)  # 첫 목표: +45도
        
        # Phase 0: 우측으로 이동 (+45도까지)
        if self.search_phase == 0:
            target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
            target_yaw = min(target_yaw, self.yaw_max)
            self.search_target_yaw = target_yaw
            
            # 증분 방식: 현재 명령 각도를 목표 방향으로 천천히 증가
            if self.search_current_command_yaw < target_yaw:
                # 목표 방향으로 증분
                self.search_current_command_yaw += self.search_increment_rad
                self.search_current_command_yaw = min(self.search_current_command_yaw, target_yaw)
            else:
                # 목표 도달, 좌측으로 이동 시작
                self.search_phase = 1
                self.search_target_yaw = self.home_yaw_rad - math.radians(self.left_right_angle)
                self.search_target_yaw = max(self.search_target_yaw, self.yaw_min)
        
        # Phase 1: 좌측으로 이동 (-45도까지)
        elif self.search_phase == 1:
            target_yaw = self.home_yaw_rad - math.radians(self.left_right_angle)
            target_yaw = max(target_yaw, self.yaw_min)
            self.search_target_yaw = target_yaw
            
            # 증분 방식: 현재 명령 각도를 목표 방향으로 천천히 감소
            if self.search_current_command_yaw > target_yaw:
                # 목표 방향으로 증분
                self.search_current_command_yaw -= self.search_increment_rad
                self.search_current_command_yaw = max(self.search_current_command_yaw, target_yaw)
            else:
                # 목표 도달, 다시 우측으로 이동 시작 (반복)
                self.search_phase = 0
                self.search_target_yaw = self.home_yaw_rad + math.radians(self.left_right_angle)
                self.search_target_yaw = min(self.search_target_yaw, self.yaw_max)
        
        # 증분된 명령 각도 반환 (PID 없이 직접 명령)
        return self.search_current_command_yaw, self.home_pitch_rad
    
    # Input : target_yaw_rad, target_pitch_rad, use_searching_gain
    # Output : output_yaw, output_pitch
    """ PID 제어를 사용하여 목 각도 계산 """
    def _pid_control(
        self, 
        target_yaw_rad: float, 
        target_pitch_rad: float,
        use_searching_gain: bool = False
    ) -> Tuple[float, float]:
        """
        PID 제어를 사용하여 목 증분 명령 계산 (절대 위치 제어)
        
        절대 위치 제어 방식:
        - 목표 위치(절대 좌표)와 현재 위치(피드백)의 오차를 계산
        - PID 제어를 통해 증분 명령(delta)을 생성
        - 하드웨어는 증분 명령을 받아 현재 위치에서 delta만큼 이동
        - 결과적으로 목표 절대 위치에 도달
        
        Args:
            target_yaw_rad: 목표 Yaw 각도 (라디안, 절대 좌표)
            target_pitch_rad: 목표 Pitch 각도 (라디안, 절대 좌표)
            use_searching_gain: True면 SEARCHING 상태용 낮은 게인 사용
            
        Returns:
            tuple: (delta_yaw_rad, delta_pitch_rad) - 증분 명령 (하드웨어가 이만큼 이동)
        """
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        
        # dt가 너무 작거나 큰 경우 제한 (안정성)
        dt = max(0.001, min(dt, 0.1))
        
        # 게인 선택 (SEARCHING 상태면 낮은 게인 사용)
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
        
        # 오차 계산
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_pitch = target_pitch_rad - self.current_pitch_rad
        
        # P 항 (비례)
        p_yaw = kp_yaw * error_yaw
        p_pitch = kp_pitch * error_pitch
        
        # I 항 (적분) - 누적 오차
        self.integral_yaw += error_yaw * dt
        self.integral_pitch += error_pitch * dt
        
        # 적분 항 제한 (windup 방지)
        max_integral = math.radians(30.0)  # 최대 30도까지 적분
        self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))
        self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
        
        i_yaw = ki_yaw * self.integral_yaw
        i_pitch = ki_pitch * self.integral_pitch
        
        # D 항 (미분) - 오차 변화율
        d_error_yaw = (error_yaw - self.last_error_yaw) / dt
        d_error_pitch = (error_pitch - self.last_error_pitch) / dt
        
        d_yaw = kd_yaw * d_error_yaw
        d_pitch = kd_pitch * d_error_pitch
        
        # PID 출력 계산 (증분 명령 = delta)
        # 절대 위치 제어: 목표-현재 오차를 기반으로 증분 명령 생성
        # 하드웨어는 증분 명령을 받아 현재 위치에서 delta만큼 이동
        delta_yaw_rad = p_yaw + i_yaw + d_yaw      # 증분 명령: Yaw 이동량
        delta_pitch_rad = p_pitch + i_pitch + d_pitch  # 증분 명령: Pitch 이동량
        
        # 상태 업데이트
        self.last_error_yaw = error_yaw
        self.last_error_pitch = error_pitch
        self.last_update_time = current_time
        
        return delta_yaw_rad, delta_pitch_rad
    
    # Input : waist_yaw_rad
    # Output : None
    """ 허리 명령 전송 """
    def _send_waist_command(self, absolute_waist_yaw_rad: float):
        """
        허리 명령 전송
        
        Args:
            absolute_waist_yaw_rad: 허리 좌우 각도 (라디안, 절대 각도)
        
        주의: 
        - Waist는 Yaw만 제어 (Index 0번이 Yaw)
        - Waist의 Pitch는 항상 0이어야 함
        - 메시지 포맷은 2차원 배열 [yaw, pitch]임
        - Waist 각도 제한: -75도 ~ 75도
        - 명령은 절대 각도로 전송
        """
        # 각도 제한 적용 (-75도 ~ 75도)
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        
        # 명령 메시지 생성 및 전송 (절대 각도로 전송)
        # Waist는 Yaw만 제어하지만, 메시지 포맷이 2차원 배열이면 Pitch를 0으로 포함
        # Index 0번이 Yaw라고 했으므로 [yaw, pitch] 형식으로 전송
        msg = Float64MultiArray()
        msg.data = [float(absolute_waist_yaw_rad), 0.0]  # [yaw, pitch] 순서, 절대 각도, Index 0번이 Yaw, Pitch는 항상 0
        self.waist_publisher.publish(msg)

    # Input : target_yaw_rad, target_pitch_rad, use_pid, use_searching_gain (목표 절대 좌표)
    # Output : expected_yaw_rad, expected_pitch_rad (예상 도달 절대 좌표)
    """ 목 명령 전송 (절대 위치 제어) """
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float, use_pid: bool = True, use_searching_gain: bool = False) -> Tuple[float, float]:
        """
        목 명령 전송 (절대 위치 제어 방식)
        
        절대 위치 제어 흐름:
        1. 목표 위치 설정 (절대 좌표)
        2. 현재 위치 피드백 (joint_positions_deg 토픽에서 수신)
        3. 오차 계산: 목표 - 현재
        4. PID 제어로 증분 명령(delta) 생성
        5. 하드웨어에 증분 명령 전송 → 목표 절대 위치에 도달
        
        Args:
            target_yaw_rad: 목표 Yaw 각도 (라디안, 절대 좌표)
            target_pitch_rad: 목표 Pitch 각도 (라디안, 절대 좌표)
            use_pid: PID 제어 사용 여부 (기본값: True)
            use_searching_gain: SEARCHING 상태용 낮은 게인 사용 여부 (기본값: False)
            
        Returns:
            tuple: (expected_yaw_rad, expected_pitch_rad) - 예상 도달 절대 좌표
        """
        # 목표 각도 저장 (절대 좌표, 내부 로직용)
        self.target_yaw_rad = target_yaw_rad
        self.target_pitch_rad = target_pitch_rad
        
        # PID 제어 적용 → 증분 명령(delta) 반환
        if use_pid:
            delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad, use_searching_gain=use_searching_gain)
        else:
            # PID 없이 직접 계산: 증분 = 목표 절대 좌표 - 현재 절대 좌표
            delta_yaw_rad = target_yaw_rad - self.current_yaw_rad
            delta_pitch_rad = target_pitch_rad - self.current_pitch_rad
        
        # 증분 명령 제한 (한 사이클에 과도한 움직임 방지)
        max_delta_angle = math.radians(30.0)  # 한 번에 최대 30도까지
        delta_yaw_rad = max(-max_delta_angle, min(max_delta_angle, delta_yaw_rad))
        delta_pitch_rad = max(-max_delta_angle, min(max_delta_angle, delta_pitch_rad))
        
        # 명령 메시지 생성 및 전송
        # 하드웨어는 증분 명령(delta)을 받아 현재 위치에서 delta만큼 이동
        msg = Float64MultiArray()
        msg.data = [float(delta_pitch_rad), float(delta_yaw_rad)]  # [pitch, yaw] 순서, 증분 명령
        
        self.neck_publisher.publish(msg)
        
        # 반환값: 예상 도달 절대 좌표 (현재 위치 + 증분 명령)
        expected_yaw_rad = self.current_yaw_rad + delta_yaw_rad
        expected_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        return expected_yaw_rad, expected_pitch_rad
    

    # Input : target_info, frame_width, frame_height
    # Output : yaw_rad, pitch_rad
    """ 타겟 정보를 받아서 목 각도 계산 및 명령 전송 """
    def update(self, target_info: TargetInfo, frame_width: float = None, frame_height: float = None, tracker_manager=None) -> Optional[Tuple[float, float]]:
        """
        타겟 정보를 받아서 목 각도 계산 및 명령 전송
        
        Args:
            target_info: 타겟 정보 (TargetInfo)
            frame_width: 프레임 너비 (기본값: self.frame_width)
            frame_height: 프레임 높이 (기본값: self.frame_height)
            tracker_manager: TrackerManager 인스턴스 (목 각도 안정성 추적용, 선택적)
            
        Returns:
            tuple: (yaw_rad, pitch_rad) 또는 None (명령 전송 실패 시)
        """
        if frame_width is None:
            frame_width = self.frame_width
        if frame_height is None:
            frame_height = self.frame_height
        
        state = target_info.state
        
        # TRACKING 상태에서 목 각도 안정성 추적 (WAIST_FOLLOWER 전이용)
        if state == TrackingState.TRACKING and tracker_manager is not None:
            tracker_manager.update_neck_angle(self.current_yaw_rad)
        
        match state:
            # TRACKING 또는 INTERACTION 상태: 타겟 추적
            case TrackingState.TRACKING | TrackingState.INTERACTION if target_info.point is not None:
                # SEARCHING 상태 리셋
                self.searching_start_time = None
                self.search_phase = 0
                
                # WAIST_FOLLOWER 상태 변수 리셋 (TRACKING으로 돌아오면 리셋)
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                # 타겟 좌표 추출
                target_x, target_y = target_info.point
                
                # 픽셀 좌표를 상대 각도로 변환 (프레임 중심 기준)
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(
                    target_x, target_y, frame_width, frame_height
                )
                
                # 목표 절대 각도 계산: 현재 목 위치 + 상대 각도
                # 카메라가 고정되어 있으므로, 타겟의 상대 위치를 현재 목 위치에 더함
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # PID 제어를 사용하여 명령 전송
                yaw_rad, pitch_rad = self._send_neck_command(target_yaw_rad, target_pitch_rad, use_pid=True)
                return yaw_rad, pitch_rad
            
            # LOST 상태: 마지막 위치 유지
            case TrackingState.LOST:
                # SEARCHING 상태 리셋
                self.searching_start_time = None
                self.search_phase = 0
                
                # WAIST_FOLLOWER 상태 변수 리셋
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                # 적분 항 리셋 (누적 오차 방지)
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                # 현재 위치 유지 (PID 없이)
                yaw_rad, pitch_rad = self._maintain_current_position()
                yaw_rad, pitch_rad = self._send_neck_command(yaw_rad, pitch_rad, use_pid=False)
                return yaw_rad, pitch_rad
            
            # SEARCHING 상태: 증분 방식으로 천천히 좌우 스캔 (PID 없이 직접 명령)
            case TrackingState.SEARCHING:
                command_yaw_rad, command_pitch_rad = self._searching_behavior()
                # 증분 방식이므로 PID 없이 직접 명령 전송
                # 목표 각도 저장 (화면 표시용)
                self.target_yaw_rad = self.search_target_yaw
                self.target_pitch_rad = command_pitch_rad
                # 각도 제한 적용 (절대 좌표)
                command_yaw_rad, command_pitch_rad = self._clip_angles(command_yaw_rad, command_pitch_rad)
                
                # 절대 좌표를 상대 좌표로 변환하여 전송 (명령 시점의 현재 위치 기준)
                relative_command_yaw_rad = command_yaw_rad - self.current_yaw_rad
                relative_command_pitch_rad = command_pitch_rad - self.current_pitch_rad
                
                # 명령 전송 (상대 좌표)
                msg = Float64MultiArray()
                msg.data = [float(relative_command_pitch_rad), float(relative_command_yaw_rad)]  # [pitch, yaw] 순서, 상대 좌표
                self.neck_publisher.publish(msg)
                return command_yaw_rad, command_pitch_rad
            
            # WAIST_FOLLOWER 상태: 목은 0도로, 허리는 WAIST_FOLLOWER 진입 시점의 목 각도로 이동 (TARGETING과 동일하게 계속 명령 전송)
            case TrackingState.WAIST_FOLLOWER:
                # SEARCHING 상태 리셋
                self.searching_start_time = None
                self.search_phase = 0
                
                # WAIST_FOLLOWER 진입 시점의 목 각도 저장 (한 번만)
                if self.waist_follower_initial_neck_yaw is None:
                    self.waist_follower_initial_neck_yaw = self.current_yaw_rad + self.current_waist_yaw_rad
                
                # 목: 허리가 이동한 만큼 목표 각도를 갱신 (목 + 허리 = 초기 목 각도 유지)
                # 초기: 목 = X도, 허리 = 0도
                # 목표: 목 = 0도, 허리 = X도
                # 이동 중: 목 목표 = 초기 목 각도 - 현재 허리 각도
                # 주의: 목 명령은 상대 각도로 전송, 허리 명령은 절대 각도로 전송
                target_neck_yaw = 0.0
                target_neck_pitch = self.current_pitch_rad
                
                # WAIST_FOLLOWER용 게인으로 PID 제어 (상대 각도 계산)
                current_time = time.monotonic()
                dt = current_time - self.last_update_time
                dt = max(0.001, min(dt, 0.1))
                
                error_neck_yaw = target_neck_yaw - self.current_yaw_rad
                error_neck_pitch = target_neck_pitch - self.current_pitch_rad
                
                # WAIST_FOLLOWER용 게인 사용
                p_neck_yaw = self.kp_neck_yaw_waist_mode * error_neck_yaw
                p_neck_pitch = self.kp_pitch_searching * error_neck_pitch  # Pitch는 Searching 게인 사용
                
                self.integral_neck_yaw_waist_mode += error_neck_yaw * dt
                max_integral_neck = math.radians(30.0)
                self.integral_neck_yaw_waist_mode = max(-max_integral_neck, min(max_integral_neck, self.integral_neck_yaw_waist_mode))
                i_neck_yaw = self.ki_neck_yaw_waist_mode * self.integral_neck_yaw_waist_mode
                
                self.integral_pitch += error_neck_pitch * dt
                max_integral_pitch = math.radians(30.0)
                self.integral_pitch = max(-max_integral_pitch, min(max_integral_pitch, self.integral_pitch))
                i_neck_pitch = self.ki_pitch_searching * self.integral_pitch
                
                d_error_neck_yaw = (error_neck_yaw - self.last_error_neck_yaw_waist_mode) / dt
                d_neck_yaw = self.kd_neck_yaw_waist_mode * d_error_neck_yaw
                
                d_error_neck_pitch = (error_neck_pitch - self.last_error_pitch) / dt
                d_neck_pitch = self.kd_pitch_searching * d_error_neck_pitch
                
                # 상대 각도 계산 (목 명령은 상대 각도로 전송)
                relative_neck_yaw = p_neck_yaw + i_neck_yaw + d_neck_yaw
                relative_neck_pitch = p_neck_pitch + i_neck_pitch + d_neck_pitch
                
                # 상대 각도 제한 적용 (과도한 움직임 방지)
                max_relative_angle = math.radians(30.0)  # 한 번에 최대 30도까지
                relative_neck_yaw = max(-max_relative_angle, min(max_relative_angle, relative_neck_yaw))
                relative_neck_pitch = max(-max_relative_angle, min(max_relative_angle, relative_neck_pitch))
                
                # 목 명령 전송 (상대 각도로 전송)
                msg_neck = Float64MultiArray()
                msg_neck.data = [float(relative_neck_pitch), float(relative_neck_yaw)]  # [pitch, yaw] 순서, 상대 각도
                self.neck_publisher.publish(msg_neck)
                
                self.last_error_neck_yaw_waist_mode = error_neck_yaw
                self.last_error_pitch = error_neck_pitch
                self.last_update_time = current_time
                
                # 허리: WAIST_FOLLOWER 진입 시점의 목 각도를 절대 각도로 직접 전송
                target_waist_yaw = self.waist_follower_initial_neck_yaw
                
                # 허리 명령 전송 (절대 각도로 전송, 예: -23.7도면 그냥 -23.7도 전송)
                self._send_waist_command(target_waist_yaw)
                
                # 목표 도달 확인 (1도 이내로 들어오면 TRACKING으로 전환)
                # 목의 최종 목표는 0도, 허리의 목표는 초기 목 각도
                final_neck_target = 0.0
                final_neck_error = abs(math.degrees(final_neck_target - self.current_yaw_rad))
                waist_error_deg = abs(math.degrees(target_waist_yaw - self.current_waist_yaw_rad))
                target_reached_threshold_deg = 1.0  # 1도 이내
                
                if final_neck_error <= target_reached_threshold_deg and waist_error_deg <= target_reached_threshold_deg:
                    # 목표 도착: TRACKING 상태로 전환
                    if tracker_manager is not None:
                        tracker_manager.state = TrackingState.TRACKING
                        # WAIST_FOLLOWER 상태 변수 리셋
                        self.waist_follower_initial_neck_yaw = None
                        self.integral_waist_yaw = 0.0
                        self.last_error_waist_yaw = 0.0
                        self.integral_neck_yaw_waist_mode = 0.0
                        self.last_error_neck_yaw_waist_mode = 0.0
                        self.node.get_logger().info(
                            f"WAIST_FOLLOWER 목표 도착: 목={final_neck_error:.2f}도, 허리={waist_error_deg:.2f}도 → TRACKING 상태로 전환"
                        )
                
                return self.current_yaw_rad + relative_neck_yaw, self.current_pitch_rad + relative_neck_pitch
            
            # IDLE 상태: 목과 허리를 0도로 천천히 이동 (Searching gain 사용)
            case TrackingState.IDLE | _:
                self.searching_start_time = None
                self.search_phase = 0
                
                # WAIST_FOLLOWER 상태 변수 리셋
                self.waist_follower_initial_neck_yaw = None
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.integral_neck_yaw_waist_mode = 0.0
                self.last_error_neck_yaw_waist_mode = 0.0
                
                # 적분 항 리셋
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                
                # 목과 허리를 0도로 천천히 이동 (Searching gain 사용)
                # 목: 0도로 이동
                target_neck_yaw = 0.0
                target_neck_pitch = 0.0
                yaw_rad, pitch_rad = self._send_neck_command(
                    target_neck_yaw, 
                    target_neck_pitch, 
                    use_pid=True, 
                    use_searching_gain=True  # 낮은 게인으로 천천히 이동
                )
                
                # 허리: 0도로 이동 (절대 각도로 전송)
                target_waist_yaw = 0.0
                self._send_waist_command(target_waist_yaw)
                
                return yaw_rad, pitch_rad
    
    # Input : None
    # Output : self.current_yaw_rad, self.current_pitch_rad
    """ 현재 목 각도 반환 (하드웨어에서 받은 실제 위치) """
    def get_current_angles(self) -> Tuple[float, float]:
        """
        현재 목 각도 반환 (하드웨어에서 받은 실제 위치)
        
        Returns:
            tuple: (yaw_rad, pitch_rad)
        """
        return self.current_yaw_rad, self.current_pitch_rad
    
    # Input : None
    # Output : self.target_yaw_rad, self.target_pitch_rad
    """ 목표 명령 각도 반환 (마지막으로 전송한 목표 각도) """
    def get_target_angles(self) -> Tuple[float, float]:
        """
        목표 명령 각도 반환 (마지막으로 전송한 목표 각도)
        
        Returns:
            tuple: (yaw_rad, pitch_rad)
        """
        return self.target_yaw_rad, self.target_pitch_rad
    
    # Input : None
    # Output : self.current_waist_yaw_rad, self.target_waist_yaw_rad
    """ 허리 각도 반환 """
    def get_waist_angles(self) -> Tuple[float, float]:
        """
        허리 각도 반환
        
        Returns:
            tuple: (current_waist_yaw_rad, target_waist_yaw_rad)
            target_waist_yaw_rad: WAIST_FOLLOWER 상태일 때는 WAIST_FOLLOWER 진입 시점의 목 각도, 아니면 0
        """
        if self.waist_follower_initial_neck_yaw is not None:
            # WAIST_FOLLOWER 상태: WAIST_FOLLOWER 진입 시점의 목 각도를 목표로 사용
            target_waist_yaw = self.waist_follower_initial_neck_yaw
        else:
            target_waist_yaw = 0.0
        return self.current_waist_yaw_rad, target_waist_yaw


