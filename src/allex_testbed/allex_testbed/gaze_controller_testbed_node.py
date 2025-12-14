#!/usr/bin/env python3
"""
얼굴/허리 제어 테스트베드 노드
- 가장 기본적인 PID 제어 알고리즘
- 목표 명령을 받아서 이동하는 방식 테스트
- PID 게인 튜닝 가능
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


class GazeControllerTestbedNode(Node):
    """얼굴/허리 제어 테스트베드 노드 - 기본 PID 제어"""
    
    def __init__(self):
        super().__init__('gaze_controller_testbed_node')
        
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
        
        # PID 게인 튜닝 명령 구독
        self.pid_tune_subscription = self.create_subscription(
            String,
            "/allex_testbed/pid_tune",
            self._pid_tune_callback,
            10
        )
        
        # 목표 명령 구독 (직접 각도 명령)
        self.target_command_subscription = self.create_subscription(
            String,
            "/allex_testbed/target_command",
            self._target_command_callback,
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
        
        # 목 각도 발행 (GUI용)
        self.neck_angle_publisher = self.create_publisher(
            String,
            "/allex_camera/neck_angle",
            10
        )
        
        # 상태 발행 (GUI용)
        self.status_publisher = self.create_publisher(
            String,
            "/allex_testbed/status",
            10
        )
        
        # 목 각도 발행 타이머 (30Hz)
        self.neck_angle_timer = self.create_timer(1.0 / 30.0, self._publish_status)
        
        # 직접 명령 모드용 제어 타이머 (30Hz)
        self.control_timer = self.create_timer(1.0 / 30.0, self._control_loop)
        
        # 카메라 파라미터
        self.frame_width = 1280.0
        self.frame_height = 720.0
        
        # 각도 제한 범위 (라디안)
        self.yaw_min = math.radians(-80.0)
        self.yaw_max = math.radians(80.0)
        self.pitch_min = -0.0872665  # -5°
        self.pitch_max = 3.75246   # 215°
        
        # 허리 각도 제한 범위 (라디안)
        self.waist_yaw_min = math.radians(-85.0)
        self.waist_yaw_max = math.radians(85.0)
        
        # 현재 목 각도 (라디안) - 하드웨어에서 받은 실제 위치
        self.current_yaw_rad = 0.0
        self.current_pitch_rad = 0.0
        
        # 현재 허리 각도 (라디안)
        self.current_waist_yaw_rad = 0.0
        
        # 목표 명령 각도 (라디안) - 직접 설정 가능
        self.target_yaw_rad = 0.0
        self.target_pitch_rad = 0.0
        self.target_waist_yaw_rad = 0.0
        
        # PID 제어 파라미터 (기본값 - 튜닝 가능)
        self.kp_yaw = 1.0
        self.kp_pitch = 1.0
        self.ki_yaw = 0.0
        self.ki_pitch = 0.0
        self.kd_yaw = 0.0
        self.kd_pitch = 0.0
        
        # 허리 PID 파라미터
        self.kp_waist_yaw = 0.5
        self.ki_waist_yaw = 0.0
        self.kd_waist_yaw = 0.0
        
        # PID 제어 상태 변수
        self.integral_yaw = 0.0
        self.integral_pitch = 0.0
        self.last_error_yaw = 0.0
        self.last_error_pitch = 0.0
        self.last_update_time = time.monotonic()
        
        # 허리 PID 상태 변수
        self.integral_waist_yaw = 0.0
        self.last_error_waist_yaw = 0.0
        self.last_waist_update_time = time.monotonic()
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 제어 모드 (0: 추적 모드, 1: 직접 명령 모드)
        self.control_mode = 0
        
        # 스무딩 파라미터 (0.0 ~ 1.0, 1.0이면 스무딩 없음)
        self.smoothing_factor = 1.0
        
        # 스무딩을 위한 이전 명령 저장
        self.smoothed_yaw_rad = 0.0
        self.smoothed_pitch_rad = 0.0
        
        # 테스트베드는 자동으로 실행 상태로 시작
        self.is_running = True
        
        self.get_logger().info("Gaze Controller Testbed Node 초기화 완료")
        self.get_logger().info(f"기본 PID 게인: Kp_yaw={self.kp_yaw}, Kp_pitch={self.kp_pitch}")
        self.get_logger().info("테스트베드 모드: 자동 실행 상태로 시작")
    
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
    
    def _pixel_to_angle(self, target_x: float, target_y: float, frame_width: float, frame_height: float) -> Tuple[float, float]:
        """타겟 픽셀 좌표를 목 각도로 변환"""
        center_x = frame_width / 2.0
        center_y = frame_height / 2.0
        
        offset_x = target_x - center_x
        offset_y = target_y - center_y
        
        horizontal_fov_deg = 120.0
        vertical_fov_deg = 45.0
        
        yaw_deg = (offset_x / frame_width) * horizontal_fov_deg
        pitch_deg = (offset_y / frame_height) * vertical_fov_deg
        
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        
        yaw_rad = -yaw_rad  # Neck Yaw: 좌측 방향이 양수
        
        return yaw_rad, pitch_rad
    
    def _clip_angles(self, yaw_rad: float, pitch_rad: float) -> Tuple[float, float]:
        """각도를 제한 범위 내로 클리핑"""
        yaw_rad = max(self.yaw_min, min(self.yaw_max, yaw_rad))
        pitch_rad = max(self.pitch_min, min(self.pitch_max, pitch_rad))
        return yaw_rad, pitch_rad
    
    def _pid_control(self, target_yaw_rad: float, target_pitch_rad: float) -> Tuple[float, float]:
        """기본 PID 제어 - 가장 간단한 형태"""
        current_time = time.monotonic()
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))
        
        # 오차 계산
        error_yaw = target_yaw_rad - self.current_yaw_rad
        error_pitch = target_pitch_rad - self.current_pitch_rad
        
        # P 항
        p_yaw = self.kp_yaw * error_yaw
        p_pitch = self.kp_pitch * error_pitch
        
        # I 항 (Integral 제한)
        self.integral_yaw += error_yaw * dt
        self.integral_pitch += error_pitch * dt
        
        max_integral = math.radians(30.0)
        self.integral_yaw = max(-max_integral, min(max_integral, self.integral_yaw))
        self.integral_pitch = max(-max_integral, min(max_integral, self.integral_pitch))
        
        i_yaw = self.ki_yaw * self.integral_yaw
        i_pitch = self.ki_pitch * self.integral_pitch
        
        # D 항
        d_error_yaw = (error_yaw - self.last_error_yaw) / dt if dt > 0 else 0.0
        d_error_pitch = (error_pitch - self.last_error_pitch) / dt if dt > 0 else 0.0
        
        d_yaw = self.kd_yaw * d_error_yaw
        d_pitch = self.kd_pitch * d_error_pitch
        
        # PID 출력
        delta_yaw_rad = p_yaw + i_yaw + d_yaw
        delta_pitch_rad = p_pitch + i_pitch + d_pitch
        
        # 스무딩 적용
        if self.smoothing_factor < 1.0:
            delta_yaw_rad = self.smoothing_factor * delta_yaw_rad + (1.0 - self.smoothing_factor) * (self.smoothed_yaw_rad - self.current_yaw_rad)
            delta_pitch_rad = self.smoothing_factor * delta_pitch_rad + (1.0 - self.smoothing_factor) * (self.smoothed_pitch_rad - self.current_pitch_rad)
            self.smoothed_yaw_rad = self.current_yaw_rad + delta_yaw_rad
            self.smoothed_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        
        self.last_error_yaw = error_yaw
        self.last_error_pitch = error_pitch
        self.last_update_time = current_time
        
        return delta_yaw_rad, delta_pitch_rad
    
    def _pid_control_waist(self, target_waist_yaw_rad: float) -> float:
        """허리 PID 제어"""
        current_time = time.monotonic()
        dt = current_time - self.last_waist_update_time
        dt = max(0.001, min(dt, 0.1))
        
        error_waist_yaw = target_waist_yaw_rad - self.current_waist_yaw_rad
        
        p_waist_yaw = self.kp_waist_yaw * error_waist_yaw
        
        self.integral_waist_yaw += error_waist_yaw * dt
        max_integral_waist = math.radians(30.0)
        self.integral_waist_yaw = max(-max_integral_waist, min(max_integral_waist, self.integral_waist_yaw))
        
        i_waist_yaw = self.ki_waist_yaw * self.integral_waist_yaw
        
        d_error_waist_yaw = (error_waist_yaw - self.last_error_waist_yaw) / dt if dt > 0 else 0.0
        d_waist_yaw = self.kd_waist_yaw * d_error_waist_yaw
        
        delta_waist_yaw_rad = p_waist_yaw + i_waist_yaw + d_waist_yaw
        
        self.last_error_waist_yaw = error_waist_yaw
        self.last_waist_update_time = current_time
        
        return delta_waist_yaw_rad
    
    def _send_neck_command(self, target_yaw_rad: float, target_pitch_rad: float) -> Tuple[float, float]:
        """목 명령 전송"""
        # 각도 제한
        target_yaw_rad, target_pitch_rad = self._clip_angles(target_yaw_rad, target_pitch_rad)
        
        # PID 제어
        delta_yaw_rad, delta_pitch_rad = self._pid_control(target_yaw_rad, target_pitch_rad)
        
        # 절대각도로 변환
        absolute_yaw_rad = self.current_yaw_rad + delta_yaw_rad
        absolute_pitch_rad = self.current_pitch_rad + delta_pitch_rad
        
        # 각도 제한 확인
        absolute_yaw_rad, absolute_pitch_rad = self._clip_angles(absolute_yaw_rad, absolute_pitch_rad)
        
        # 명령 전송
        msg = Float64MultiArray()
        msg.data = [float(absolute_pitch_rad), float(absolute_yaw_rad)]
        self.neck_publisher.publish(msg)
        
        return absolute_yaw_rad, absolute_pitch_rad
    
    def _send_waist_command(self, target_waist_yaw_rad: float) -> float:
        """허리 명령 전송"""
        target_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, target_waist_yaw_rad))
        
        delta_waist_yaw_rad = self._pid_control_waist(target_waist_yaw_rad)
        
        absolute_waist_yaw_rad = self.current_waist_yaw_rad + delta_waist_yaw_rad
        absolute_waist_yaw_rad = max(self.waist_yaw_min, min(self.waist_yaw_max, absolute_waist_yaw_rad))
        
        msg = Float64MultiArray()
        msg.data = [float(absolute_waist_yaw_rad), 0.0]  # [yaw, pitch]
        self.waist_publisher.publish(msg)
        
        return absolute_waist_yaw_rad
    
    def _target_command_callback(self, msg: String):
        """직접 목표 명령 콜백 (테스트용)"""
        try:
            data = json.loads(msg.data)
            cmd_type = data.get('type')
            
            if cmd_type == 'set_target':
                yaw_deg = data.get('yaw_deg', 0.0)
                pitch_deg = data.get('pitch_deg', 0.0)
                waist_yaw_deg = data.get('waist_yaw_deg', None)
                
                self.target_yaw_rad = math.radians(yaw_deg)
                self.target_pitch_rad = math.radians(pitch_deg)
                
                if waist_yaw_deg is not None:
                    self.target_waist_yaw_rad = math.radians(waist_yaw_deg)
                else:
                    # 허리 명령이 없으면 현재 위치 유지
                    self.target_waist_yaw_rad = self.current_waist_yaw_rad
                
                self.control_mode = 1  # 직접 명령 모드
                
                # PID 상태 초기화 (새로운 목표로 이동할 때)
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                self.last_error_yaw = 0.0
                self.last_error_pitch = 0.0
                
                self.get_logger().info(
                    f"목표 명령 설정: Yaw={yaw_deg:.1f}도, Pitch={pitch_deg:.1f}도, "
                    f"Waist={waist_yaw_deg:.1f}도 (설정됨)" if waist_yaw_deg is not None else f"Waist=현재 위치 유지"
                )
                
                # 즉시 명령 전송
                if self.is_running:
                    self._send_neck_command(self.target_yaw_rad, self.target_pitch_rad)
                    self._send_waist_command(self.target_waist_yaw_rad)
                    self.get_logger().info("목표 명령 즉시 전송 완료")
                else:
                    self.get_logger().warn("목표 명령 설정됨, 하지만 노드가 실행 중이 아님 (RUN 명령 필요)")
            
            elif cmd_type == 'reset':
                self.integral_yaw = 0.0
                self.integral_pitch = 0.0
                self.last_error_yaw = 0.0
                self.last_error_pitch = 0.0
                self.integral_waist_yaw = 0.0
                self.last_error_waist_yaw = 0.0
                self.get_logger().info("PID 상태 초기화")
                
        except Exception as e:
            self.get_logger().error(f"목표 명령 처리 실패: {e}")
    
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
            
            if 'kp_waist_yaw' in data:
                self.kp_waist_yaw = float(data['kp_waist_yaw'])
            if 'ki_waist_yaw' in data:
                self.ki_waist_yaw = float(data['ki_waist_yaw'])
            if 'kd_waist_yaw' in data:
                self.kd_waist_yaw = float(data['kd_waist_yaw'])
            
            if 'smoothing_factor' in data:
                self.smoothing_factor = max(0.0, min(1.0, float(data['smoothing_factor'])))
            
            self.get_logger().info(
                f"PID 게인 업데이트: Kp_yaw={self.kp_yaw:.3f}, Kp_pitch={self.kp_pitch:.3f}, "
                f"Ki_yaw={self.ki_yaw:.3f}, Ki_pitch={self.ki_pitch:.3f}, "
                f"Kd_yaw={self.kd_yaw:.3f}, Kd_pitch={self.kd_pitch:.3f}, "
                f"Smoothing={self.smoothing_factor:.2f}"
            )
            
        except Exception as e:
            self.get_logger().error(f"PID 튜닝 명령 처리 실패: {e}")
    
    def tracking_result_callback(self, msg: String):
        """추적 결과 콜백"""
        if not self.is_running:
            return
        
        # 직접 명령 모드에서는 추적 결과 무시 (제어 루프에서 처리)
        if self.control_mode == 1:
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
            
            if state == TrackingState.TRACKING and target_info.point is not None:
                target_x, target_y = target_info.point
                
                # 픽셀을 각도로 변환
                relative_yaw_rad, relative_pitch_rad = self._pixel_to_angle(
                    target_x, target_y, self.frame_width, self.frame_height
                )
                
                # 목표 각도 계산
                target_yaw_rad = self.current_yaw_rad + relative_yaw_rad
                target_pitch_rad = self.current_pitch_rad + relative_pitch_rad
                
                # 각도 제한
                target_yaw_rad, target_pitch_rad = self._clip_angles(target_yaw_rad, target_pitch_rad)
                
                # 명령 전송
                self._send_neck_command(target_yaw_rad, target_pitch_rad)
                
                # 허리는 현재 위치 유지
                self._send_waist_command(self.current_waist_yaw_rad)
            
        except Exception as e:
            self.get_logger().error(f"추적 결과 처리 실패: {e}")
    
    def _control_callback(self, msg: String):
        """제어 명령 콜백"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'run' or cmd_type == 'start':
                self.is_running = True
                self.control_mode = 0  # 추적 모드
                self.get_logger().info("Controller RUN 시작")
            
            elif cmd_type == 'stop':
                self.is_running = False
                self.get_logger().info("Controller RUN 중지")
                
        except Exception as e:
            self.get_logger().error(f"제어 명령 처리 실패: {e}")
    
    def _control_loop(self):
        """제어 루프 (직접 명령 모드용)"""
        if not self.is_running:
            return
        
        # 직접 명령 모드일 때 지속적으로 명령 전송
        if self.control_mode == 1:
            self._send_neck_command(self.target_yaw_rad, self.target_pitch_rad)
            if hasattr(self, 'target_waist_yaw_rad'):
                self._send_waist_command(self.target_waist_yaw_rad)
    
    def _publish_status(self):
        """상태 발행 (목 각도 및 PID 정보)"""
        try:
            data = {
                'current_yaw_rad': float(self.current_yaw_rad),
                'current_pitch_rad': float(self.current_pitch_rad),
                'target_yaw_rad': float(self.target_yaw_rad),
                'target_pitch_rad': float(self.target_pitch_rad),
                'current_waist_yaw_rad': float(self.current_waist_yaw_rad),
                'target_waist_yaw_rad': float(self.target_waist_yaw_rad),
                'pid_gains': {
                    'kp_yaw': float(self.kp_yaw),
                    'kp_pitch': float(self.kp_pitch),
                    'ki_yaw': float(self.ki_yaw),
                    'ki_pitch': float(self.ki_pitch),
                    'kd_yaw': float(self.kd_yaw),
                    'kd_pitch': float(self.kd_pitch),
                    'kp_waist_yaw': float(self.kp_waist_yaw),
                    'ki_waist_yaw': float(self.ki_waist_yaw),
                    'kd_waist_yaw': float(self.kd_waist_yaw),
                },
                'smoothing_factor': float(self.smoothing_factor),
                'control_mode': int(self.control_mode),
                'is_running': bool(self.is_running),
                'timestamp': time.monotonic()
            }
            
            msg = String()
            msg.data = json.dumps(data, ensure_ascii=False)
            self.status_publisher.publish(msg)
            
            # 기존 neck_angle 토픽도 발행 (호환성)
            neck_angle_msg = String()
            neck_angle_msg.data = json.dumps({
                'current_yaw_rad': float(self.current_yaw_rad),
                'current_pitch_rad': float(self.current_pitch_rad),
                'target_yaw_rad': float(self.target_yaw_rad),
                'target_pitch_rad': float(self.target_pitch_rad),
                'current_waist_yaw_rad': float(self.current_waist_yaw_rad),
                'target_waist_yaw_rad': float(self.target_waist_yaw_rad),
                'timestamp': time.monotonic()
            })
            self.neck_angle_publisher.publish(neck_angle_msg)
            
        except Exception as e:
            self.get_logger().error(f"상태 발행 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = GazeControllerTestbedNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

