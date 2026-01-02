#!/usr/bin/env python3
"""
ALLEX Idle Interaction 총괄 노드
- GUI 명령 처리
- 상태 관리 및 루틴 제어
- 추적 결과를 받아서 처리 및 발행
"""
import time
import json
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String, Int32MultiArray
import cv2
import numpy as np

from .tracking_fsm_node import TrackingState
from typing import Optional

cv2.setNumThreads(0)  # OpenCV의 멀티스레딩 비활성화


class RoutineController:
    """ROS2 Routine 시스템 제어 클래스"""
    
    def __init__(self, node: Node, robot_name: str = "X"):
        self.node = node
        self.robot_name = robot_name
        self.current_routine = None
        self.breathing_routine_running = False  # idle_breathing_rt 실행 상태 추적
        
        # RESET 대기 상태 추적
        self.waiting_for_reset = False  # RESET 대기 중인지
        self.resetting_routine_name = None  # 현재 RESET 중인 루틴 이름 (단일 루틴 RESET 시 사용)
        
        # START 명령 확인용 (플래그 기반)
        self.expected_routine_name = None  # START 명령 발행 시 예상하는 루틴 이름
        self.expected_routine_start_time = None  # START 명령 발행 시각
        
        # HMI 명령 Publisher
        self.command_pub = node.create_publisher(
            String,
            'hmi/robot_command',
            10
        )
        
        node.get_logger().info(f"RoutineController 초기화 완료 (로봇: {robot_name})")
    
    def publish_command(self, command: str):
        """명령을 토픽으로 발행"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.node.get_logger().info(f"Routine 명령 발행: {command}")
    
    def is_routine_idle(self) -> bool:
        """루틴이 Idle 상태인지 확인 (nodes.length == 0)"""
        return self.node.routine_nodes_count == 0
    
    def wait_for_idle(self, timeout=3.0, check_interval=0.1) -> bool:
        """
        Idle 상태가 될 때까지 대기 (블로킹)
        다른 콜백을 처리하면서 대기하기 위해 rclpy.spin_once() 사용
        
        Args:
            timeout: 최대 대기 시간 (초)
            check_interval: 확인 간격 (초)
        
        Returns:
            True: Idle 상태 도달 성공
            False: 타임아웃
        """
        import rclpy
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            # 다른 콜백(/debug/routine 등)이 실행될 수 있도록 spin_once 호출
            rclpy.spin_once(self.node, timeout_sec=check_interval)
            
            if self.is_routine_idle():
                elapsed = time.monotonic() - start_time
                self.node.get_logger().info(f"Idle 상태 확인 완료 (대기 시간: {elapsed:.2f}초)")
                return True
        
        self.node.get_logger().warn(f"Idle 상태 확인 타임아웃 ({timeout}초, 현재 nodes={self.node.routine_nodes_count})")
        return False
    
    def start_pause_reset_all_routines(self):
        """
        모든 루틴에 대해 PAUSE -> RESET 명령 발행 (STOP 명령 시에만 사용)
        (RESET 완료될 때까지 계속 호출됨)
        """
        # 모든 루틴 이름 목록 (현재 사용 중인 루틴들)
        all_routine_names = [
            "idle_breathing_rt",
            "idling_heart_rt",
            "idling_handshake_rt"
        ]
        
        # RESET 완료 플래그 초기화
        self.node.routine_reset_complete_flag = False
        # START 확인 플래그 초기화 (PAUSE/RESET 시작 시)
        self.expected_routine_name = None
        self.expected_routine_start_time = None
        
        # 모든 루틴에 대해 PAUSE 명령 발행 (0.2초 간격)
        for routine_name in all_routine_names:
            pause_command = f"{self.robot_name}::ROUTINE::{routine_name}::PAUSE"
            self.publish_command(pause_command)
            time.sleep(0.2)
        
        # 모든 루틴에 대해 RESET 명령 발행 (0.2초 간격)
        for routine_name in all_routine_names:
            reset_command = f"{self.robot_name}::ROUTINE::{routine_name}::RESET"
            self.publish_command(reset_command)
            time.sleep(0.2)
        
        self.node.get_logger().info(f"[PAUSE/RESET ALL] PAUSE/RESET 명령 발행 (현재 nodes={self.node.routine_nodes_count})")
    
    def start_pause_reset_single_routine(self, routine_name: str):
        """
        특정 루틴에 대해 PAUSE -> RESET 명령 발행 (0.2초 간격)
        """
        # RESET 완료 플래그 초기화
        self.node.routine_reset_complete_flag = False
        # START 확인 플래그 초기화 (PAUSE/RESET 시작 시)
        self.expected_routine_name = None
        self.expected_routine_start_time = None
        # RESET 중인 루틴 이름 저장
        self.resetting_routine_name = routine_name
        
        # 특정 루틴에 대해 PAUSE 명령 발행
        pause_command = f"{self.robot_name}::ROUTINE::{routine_name}::PAUSE"
        self.publish_command(pause_command)
        time.sleep(0.2)
        
        # 특정 루틴에 대해 RESET 명령 발행
        reset_command = f"{self.robot_name}::ROUTINE::{routine_name}::RESET"
        self.publish_command(reset_command)
        time.sleep(0.2)
        
        self.node.get_logger().info(f"[PAUSE/RESET] {routine_name} PAUSE/RESET 명령 발행 (현재 nodes={self.node.routine_nodes_count})")
    
    def is_reset_complete(self) -> bool:
        """
        RESET 완료 여부 확인 (콜백에서 설정된 플래그 확인)
        
        Returns:
            True: RESET 완료 (플래그가 True)
            False: RESET 미완료 (플래그가 False)
        """
        return self.node.routine_reset_complete_flag
    
    def transition_to_routine(self, from_routine: Optional[str], to_routine: str, old_state: TrackingState, new_state: TrackingState):
        """
        루틴 전환: 현재 루틴 PAUSE/RESET (0.2초 간격) → 피드백 확인 → 목표 루틴 START
        
        Args:
            from_routine: 현재 실행 중인 루틴 이름 (None이면 실행 중이 아님)
            to_routine: 목표 루틴 이름
            old_state: 이전 상태
            new_state: 새로운 상태
        """
        # RESET 완료 확인
        if self.is_reset_complete():
            # RESET 완료: 목표 루틴 START
            self.waiting_for_reset = False
            self.resetting_routine_name = None
            
            # RESET 후 시스템 안정화 대기 (1초)
            time.sleep(1.0)
            
            # 목표 루틴 START
            command = f"{self.robot_name}::ROUTINE::{to_routine}::START"
            self.current_routine = to_routine
            
            # breathing 루틴인지 확인
            if to_routine == "idle_breathing_rt":
                self.breathing_routine_running = True
            else:
                self.breathing_routine_running = False
            
            self.publish_command(command)
            
            # START 명령 확인용 플래그 설정
            self.expected_routine_name = to_routine
            self.expected_routine_start_time = time.monotonic()
            
            self.node.get_logger().info(
                f"{old_state.value} → {new_state.value}: {from_routine or 'None'} PAUSE → RESET → {to_routine} 시작"
            )
        else:
            # RESET 미완료: 현재 루틴 PAUSE/RESET 계속 발행 (0.2초 간격)
            self.waiting_for_reset = True
            if from_routine:
                self.start_pause_reset_single_routine(from_routine)
            else:
                # 현재 루틴이 없으면 바로 목표 루틴 START
                self.waiting_for_reset = False
                self.resetting_routine_name = None
                
                command = f"{self.robot_name}::ROUTINE::{to_routine}::START"
        self.current_routine = to_routine
        
        if to_routine == "idle_breathing_rt":
            self.breathing_routine_running = True
        else:
            self.breathing_routine_running = False
        
        self.publish_command(command)
        
        self.expected_routine_name = to_routine
        self.expected_routine_start_time = time.monotonic()
        
        self.node.get_logger().info(
                    f"{old_state.value} → {new_state.value}: 현재 루틴 없음, {to_routine} 시작"
        )
    
    def start_breathing(self):
        """숨쉬기 루틴 시작 (무한 반복) - 기존 루틴이 없을 경우 바로 실행"""
        # 이미 실행 중이면 중복 시작 방지
        if self.breathing_routine_running and self.current_routine == "idle_breathing_rt":
            self.node.get_logger().warn("idle_breathing_rt가 이미 실행 중입니다. 중복 시작 건너뜀.")
            return
        
        time.sleep(0.07)
        # RESET 이후 RUN 상태로 전환 (Waist와 Head가 READY 상태가 되는 것을 방지)
        status_run_command = "theOne_neck,theOne_waist::STATUS::RUN"
        self.publish_command(status_run_command)
        self.node.get_logger().info("STATUS::RUN 명령 발행 (Waist/Head RUN 상태로 전환)")
        
        # STATUS::RUN 명령 처리 대기
        time.sleep(0.07)
        
        # 루틴 시작
        routine_name = "idle_breathing_rt"
        command = f"{self.robot_name}::ROUTINE::{routine_name}::START"
        self.current_routine = routine_name
        self.breathing_routine_running = True
        self.publish_command(command)
        
        # START 명령 확인용 플래그 설정
        self.expected_routine_name = routine_name
        self.expected_routine_start_time = time.monotonic()
        
        self.node.get_logger().info(f"idle_breathing_rt 시작: {routine_name} (명령 발행 완료, 시작 확인 대기 중)")
    
    def stop_current_routine(self):
        """
        모든 루틴 중단: 모든 루틴 PAUSE → RESET → STOP (GUI STOP 명령 시 호출)
        (블로킹 없이, 콜백에서 플래그 확인하여 처리)
        """
        # 처음 호출: PAUSE/RESET 명령 발행 및 STOP 대기 플래그 설정
        self.node.get_logger().info(f"[STOP] 모든 루틴 중단 시작")
        self.node.waiting_for_stop = True
        # START 확인 플래그 초기화
        self.expected_routine_name = None
        self.expected_routine_start_time = None
        self.start_pause_reset_all_routines()
    
    def _complete_stop(self):
        """
        STOP 완료 처리 (콜백에서 호출)
        RESET 완료 후 모든 루틴 STOP 및 트래커/컨트롤러 STOP 명령 전송
        """
        if not self.node.waiting_for_stop:
            return
        
        self.node.get_logger().info("[STOP] 모든 루틴 RESET 완료 확인: Idle 상태 도달")
        self.node.waiting_for_stop = False
        self.waiting_for_reset = False
        
        # 모든 루틴에 대해 STOP 명령
        all_routine_names = [
            "idle_breathing_rt",
            "idling_heart_rt",
            "idling_handshake_rt"
        ]
        for routine_name in all_routine_names:
            stop_command = f"{self.robot_name}::ROUTINE::{routine_name}::STOP"
            self.publish_command(stop_command)
            self.node.get_logger().info(f"[STOP] 루틴 STOP: {routine_name}")
        
        # 트래커와 컨트롤러에 STOP 명령 전송
        tracker_command = {'type': 'stop'}
        controller_command = {'type': 'stop'}
        
        from std_msgs.msg import String
        tracker_msg = String()
        tracker_msg.data = json.dumps(tracker_command)
        self.node.tracker_control_publisher.publish(tracker_msg)
        
        controller_msg = String()
        controller_msg.data = json.dumps(controller_command)
        self.node.controller_control_publisher.publish(controller_msg)
        
        self.node.get_logger().info("[STOP] 트래커/컨트롤러 STOP 명령 전송 완료")
        
        # 상태 초기화
        self.current_routine = None
        self.breathing_routine_running = False
        self.node.actual_running_routine = None
        self.node.get_logger().info(f"[STOP] 모든 루틴 완전 중단 완료")


class AllexIdleInteractionNode(Node):
    """ALLEX Idle Interaction 총괄 노드"""
    
    def __init__(self):
        super().__init__("allex_idle_interaction_node")
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 입력 이미지 구독 (타겟 Crop 이미지 발행용)
        self.image_subscription = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.image_callback,
            qos_profile,
        )
        
        # 추적 결과 구독 (Tracker 노드에서 발행)
        self.tracking_result_subscription = self.create_subscription(
            String,
            "/allex_camera/tracking_result",
            self.tracking_result_callback,
            10
        )
        
        # 목 각도 구독 (Controller 노드에서 발행)
        self.neck_angle_subscription = self.create_subscription(
            String,
            "/allex_camera/neck_angle",
            self.neck_angle_callback,
            10
        )
        
        # 상태 및 추적 데이터 Publisher (GUI용)
        self.tracking_data_publisher = self.create_publisher(
            String,
            "/allex_camera/tracking_data",
            10
        )
        
        # 타겟 BB Box Crop 이미지 Publisher
        self.target_crop_publisher = self.create_publisher(
            CompressedImage,
            "/allex_camera/target_crop/compressed",
            10
        )
        
        # Tracker 제어 명령 Publisher
        self.tracker_control_publisher = self.create_publisher(
            String,
            "/allex_camera/tracker_control",
            10
        )
        
        # Controller 제어 명령 Publisher
        self.controller_control_publisher = self.create_publisher(
            String,
            "/allex_camera/controller_control",
            10
        )
        
        # Manual 제어 구독 (GUI에서 오는 명령)
        self.manual_control_subscription = self.create_subscription(
            String,
            "/allex_camera/manual_control",
            self._manual_control_callback,
            10
        )
        
        # /debug/routine 토픽 구독 (현재 실행 중인 루틴 추적용)
        self.routine_status_subscription = self.create_subscription(
            String,
            "/debug/routine",
            self._routine_status_callback,
            10
        )
        
        # Neck articulation 상태 구독 (READY 상태 체크용)
        self.neck_articulation_subscription = self.create_subscription(
            Int32MultiArray,
            "/robot_outbound_data/theOne_neck/articulation_now",
            self._neck_articulation_callback,
            10
        )
        
        # Traj status 구독 (TELEOP 및 READY/RUN/STOP 상태 확인용)
        self.traj_status_subscription = self.create_subscription(
            Int32MultiArray,
            "/robot_outbound_data/theOne_neck/traj_status",
            self._traj_status_callback,
            10
        )
        
        # HMI 명령 Publisher (TELEOP 제어용)
        self.hmi_command_publisher = self.create_publisher(
            String,
            '/hmi/robot_command',
            10
        )
        
        # 현재 실행 중인 루틴 이름 추적 (실제 실행 중인 루틴)
        self.actual_running_routine = None  # 실제 실행 중인 루틴 이름
        self.routine_nodes_count = 0  # 최신 nodes 개수 저장 (피드백 기반 제어용)
        self.routine_reset_complete_flag = False  # RESET 완료 플래그 (콜백에서 설정)
        self.waiting_for_stop = False  # STOP 대기 중인지
        
        # Neck articulation 상태 저장 (data[1]의 값: 4=READY, 5=RUN)
        self.neck_articulation_status = None  # None=알 수 없음, 4=READY, 5=RUN
        
        # Traj status 상태 저장
        self.teleop_status = None  # None=알 수 없음, 0=TELEOP 꺼짐, 1=TELEOP 켜짐
        self.robot_status = None  # None=알 수 없음, 0=Ready, 3=RUN, 5=STOP
        self.teleop_control_in_progress = False  # TELEOP 제어 진행 중 플래그
        self.waiting_for_teleop_off = False  # TELEOP OFF 대기 중인지
        self.waiting_for_ready_for_teleop_off = False  # TELEOP을 끄기 위해 READY 상태 대기 중인지
        
        # RoutineController 초기화
        self.routine_controller = RoutineController(self, robot_name="X")
        
        # 상태 관리
        self.previous_state = TrackingState.IDLE
        self.is_running = False
        
        # 최신 추적 결과 저장
        self.latest_tracking_result = None
        self.latest_frame = None
        self.latest_neck_yaw_rad = None
        self.latest_neck_angles = None
        self.latest_waist_angles = None
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        # Manual Mode 추적 (tracking_result에서 업데이트)
        self.current_manual_mode = False
        
        self.get_logger().info("ALLEX Idle Interaction Node 초기화 완료")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    def _neck_articulation_callback(self, msg: Int32MultiArray):
        """Neck articulation 상태 콜백 - READY 상태 체크용"""
        try:
            if len(msg.data) > 4:
                # data[1]이 상태 (값 4=READY, 값 5=RUN)
                # 사용자 설명에 따르면 인덱스 1의 값이 상태를 나타냄
                status_value = msg.data[1]
                self.neck_articulation_status = status_value
                # 디버깅: 상태 변경 시에만 로그 출력
                if status_value == 4:
                    self.get_logger().debug(f"Neck articulation 상태: READY (data[1]={status_value})")
                elif status_value == 5:
                    self.get_logger().debug(f"Neck articulation 상태: RUN (data[1]={status_value})")
            else:
                self.get_logger().warn(f"Neck articulation 데이터 길이 부족: {len(msg.data)}")
        except Exception as e:
            self.get_logger().warn(f"Neck articulation 상태 파싱 실패: {e}")
    
    def _traj_status_callback(self, msg: Int32MultiArray):
        """Traj status 콜백 - TELEOP 및 READY/RUN/STOP 상태 확인"""
        try:
            if len(msg.data) >= 2:
                # data[0]: TELEOP 상태 (0=꺼짐, 1=켜짐)
                # data[1]: READY/RUN/STOP 상태 (0=Ready, 3=RUN, 5=STOP)
                old_teleop = self.teleop_status
                old_robot = self.robot_status
                
                self.teleop_status = msg.data[0]
                self.robot_status = msg.data[1]
                
                # 상태 변경 시 로그 출력
                if old_teleop != self.teleop_status:
                    self.get_logger().info(f"[TELEOP] 상태 변경: {old_teleop} -> {self.teleop_status} (0=OFF, 1=ON)")
                    # TELEOP OFF 대기 중이면 확인
                    if self.waiting_for_teleop_off and self.teleop_status == 0:
                        self.get_logger().info("[TELEOP] TELEOP OFF 확인됨 (피드백)")
                        self.waiting_for_teleop_off = False
                        self.teleop_control_in_progress = False
                
                if old_robot != self.robot_status:
                    status_str = {0: "Ready", 3: "RUN", 5: "STOP"}.get(self.robot_status, f"Unknown({self.robot_status})")
                    self.get_logger().info(f"[ROBOT STATUS] 상태 변경: {old_robot} -> {self.robot_status} ({status_str})")
                    # READY 상태 대기 중이면 TELEOP OFF 수행
                    if self.waiting_for_ready_for_teleop_off and self._is_ready_state():
                        self.get_logger().info("[TELEOP STOP] READY 상태 도달: TELEOP OFF 명령 발행")
                        if self.teleop_status == 1:
                            self._publish_hmi_command("theOne_neck,theOne_waist::SCENARIO::TELEOP")
                            self.waiting_for_ready_for_teleop_off = False
                            self.waiting_for_teleop_off = True  # TELEOP OFF 피드백 대기
                        else:
                            self.get_logger().info("[TELEOP STOP] TELEOP이 이미 꺼져 있습니다.")
                            self.waiting_for_ready_for_teleop_off = False
                            self.waiting_for_teleop_off = False
                            self.teleop_control_in_progress = False
            else:
                self.get_logger().warn(f"Traj status 데이터 길이 부족: {len(msg.data)}")
        except Exception as e:
            self.get_logger().warn(f"Traj status 파싱 실패: {e}")
    
    def _publish_hmi_command(self, command: str):
        """HMI 명령 발행"""
        msg = String()
        msg.data = command
        self.hmi_command_publisher.publish(msg)
        self.get_logger().info(f"[HMI] 명령 발행: {command}")

    def _is_ready_state(self) -> bool:
        """
        현재 상태가 READY로 판단되는지 반환
        Interpretation fixed: READY if robot_status in {0,1,2} regardless of TELEOP.
        """
        try:
            return self.robot_status in (0, 1, 2)
        except Exception:
            return False

    def _is_running_state(self) -> bool:
        return self.robot_status == 3

    def _is_stop_state(self) -> bool:
        return self.robot_status == 5
    
    def _handle_teleop_for_run(self):
        """
        RUN 명령 시 TELEOP 제어
        
        예외 상황 처리:
        - TELEOP ON + RUN 상태: 이미 동작 중이므로 아무것도 하지 않음
        - TELEOP ON + Ready 상태: RUN 명령만 발행
        - TELEOP ON + STOP 상태: STOP -> READY -> RUN
        - TELEOP OFF + RUN 상태: STOP -> READY -> TELEOP ON -> RUN
        - TELEOP OFF + Ready 상태: TELEOP ON -> RUN
        - TELEOP OFF + STOP 상태: STOP -> READY -> TELEOP ON -> RUN
        """
        if self.teleop_control_in_progress:
            self.get_logger().debug("[TELEOP RUN] TELEOP 제어가 이미 진행 중입니다. 건너뜀")
            return
        
        self.get_logger().info("[TELEOP RUN] TELEOP 제어 시작")
        self.teleop_control_in_progress = True

        # 디버그: 현재 상태 요약 출력
        try:
            self.get_logger().debug(
                f"[TELEOP RUN DEBUG] teleop_status={self.teleop_status}, robot_status={self.robot_status}, "
                f"ready={self._is_ready_state()}, running={self._is_running_state()}, stop={self._is_stop_state()}"
            )
        except Exception:
            pass
        
        # TELEOP 상태 확인 및 분기
        # Case A: TELEOP ON
        if self.teleop_status == 1:
            # If robot_status unknown, try safe READY->RUN
            if self.robot_status is None:
                self.get_logger().warn("[TELEOP RUN] TELEOP ON but robot_status unknown: attempting READY->RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                time.sleep(0.2)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

            # RUNNING -> PASS
            if self._is_running_state():
                self.get_logger().info("[TELEOP RUN] TELEOP ON + RUNNING: already running, nothing to do")
                self.teleop_control_in_progress = False
                return

            # STOP -> READY -> RUN (per spec: send READY then RUN)
            if self._is_stop_state():
                self.get_logger().info("[TELEOP RUN] TELEOP ON + STOP: send READY then RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                time.sleep(0.5)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

            # READY -> RUN
            if self._is_ready_state():
                self.get_logger().info("[TELEOP RUN] TELEOP ON + READY: send RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

        # Case B: TELEOP OFF
        if self.teleop_status == 0:
            # RUNNING: STOP -> READY -> TELEOP ON -> RUN
            if self._is_running_state():
                self.get_logger().info("[TELEOP RUN] TELEOP OFF + RUNNING: STOP -> READY -> TELEOP ON -> RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::STOP")
                time.sleep(0.2)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                time.sleep(0.5)
                self._publish_hmi_command("theOne_neck,theOne_waist::SCENARIO::TELEOP")
                time.sleep(0.3)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

            # STOP: READY -> TELEOP ON -> RUN
            if self._is_stop_state():
                self.get_logger().info("[TELEOP RUN] TELEOP OFF + STOP: READY -> TELEOP ON -> RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                time.sleep(0.2)
                self._publish_hmi_command("theOne_neck,theOne_waist::SCENARIO::TELEOP")
                time.sleep(0.3)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

            # READY: TELEOP ON -> RUN
            if self._is_ready_state():
                self.get_logger().info("[TELEOP RUN] TELEOP OFF + READY: TELEOP ON -> RUN")
                self._publish_hmi_command("theOne_neck,theOne_waist::SCENARIO::TELEOP")
                time.sleep(0.3)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::RUN")
                self.teleop_control_in_progress = False
                return

        # Fallback: ensure flag cleared
        self.teleop_control_in_progress = False
    
    def _handle_teleop_for_stop(self):
        """
        STOP 명령 시 TELEOP 제어
        
        중요: TELEOP은 READY 상태에서만 꺼질 수 있음!
        
        처리 순서:
        1. 현재 상태 확인
        2. RUN/STOP 상태이면 -> STOP -> READY 전환 (피드백 대기)
        3. READY 상태가 되면 -> TELEOP OFF (피드백 대기)
        4. TELEOP OFF 확인 후 완료
        """
        if self.teleop_control_in_progress:
            self.get_logger().debug("[TELEOP STOP] TELEOP 제어가 이미 진행 중입니다. 건너뜀")
            return
        
        self.get_logger().info("[TELEOP STOP] TELEOP 제어 시작 (TELEOP은 READY에서만 꺼질 수 있음)")
        self.teleop_control_in_progress = True
        
        # TELEOP 상태 확인
        # Case A: already OFF -> nothing to do
        if self.teleop_status == 0:
            self.get_logger().info("[TELEOP STOP] TELEOP is already OFF")
            self.teleop_control_in_progress = False
            return

        # Case B: TELEOP ON -> must turn off but only in READY
        if self.teleop_status == 1:
            # If robot_status unknown: make READY then wait for READY feedback
            if self.robot_status is None:
                self.get_logger().warn("[TELEOP STOP] robot_status unknown: issuing READY and waiting")
                # Ensure robot is READY: send STOP then READY to be safe
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::STOP")
                time.sleep(0.2)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                self.waiting_for_ready_for_teleop_off = True
                return

            # If already READY -> toggle TELEOP (SCENARIO::TELEOP) to turn off
            if self._is_ready_state():
                self.get_logger().info("[TELEOP STOP] READY: toggling TELEOP OFF")
                self._publish_hmi_command("theOne_neck,theOne_waist::SCENARIO::TELEOP")
                self.waiting_for_teleop_off = True
                return

            # If RUNNING or STOP -> make READY first, then wait to toggle TELEOP
            if self._is_running_state() or self._is_stop_state():
                status_str = {3: "RUN", 5: "STOP"}.get(self.robot_status)
                self.get_logger().info(f"[TELEOP STOP] {status_str}: converting to READY then perform TELEOP OFF")
                # bring to READY
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::STOP")
                time.sleep(0.2)
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
                # wait for READY feedback to actually toggle TELEOP
                self.waiting_for_ready_for_teleop_off = True
                return

        # If TELEOP unknown: make READY then wait
        if self.teleop_status is None:
            self.get_logger().warn("[TELEOP STOP] TELEOP unknown: ensure READY then toggle")
            if self._is_running_state() or self._is_stop_state():
                self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::STOP")
                time.sleep(0.2)
            self._publish_hmi_command("theOne_neck,theOne_waist::STATUS::READY")
            self.waiting_for_ready_for_teleop_off = True
            return

        # Fallback: clear flag
        self.teleop_control_in_progress = False
    
    def _routine_status_callback(self, msg: String):
        """루틴 상태 피드백 콜백 - /debug/routine 토픽에서 실제 실행 중인 루틴 추적"""
        try:
            # Neck articulation 상태 체크: READY이면 RUN으로 변경 (RUN 중일 때만)
            # STOP 중일 때는 READY 상태를 유지해야 하므로 자동 전환하지 않음
            # data[1]의 값이 4면 READY, 5면 RUN
            if self.neck_articulation_status == 4 and self.is_running:  # READY (값 4)이고 RUN 중일 때만
                # ROBOT STATUS도 확인: READY 상태일 때만 RUN으로 전환
                if self._is_ready_state():
                    self.get_logger().info("Neck articulation이 READY 상태입니다. RUN으로 전환합니다. (RUN 중이므로)")
                    status_run_command = "theOne_neck,theOne_waist::STATUS::RUN"
                    self.routine_controller.publish_command(status_run_command)
                else:
                    self.get_logger().debug(f"Neck articulation이 READY이지만 robot_status={self.robot_status}이므로 RUN 전환하지 않음")
            elif self.neck_articulation_status == 4 and not self.is_running:
                self.get_logger().debug("Neck articulation이 READY 상태이지만 STOP 중이므로 RUN으로 전환하지 않음")
            
            data = json.loads(msg.data)
            
            # 루틴이 비어있는지 확인 (루틴 종료 상태)
            nodes = data.get("nodes", [])
            
            # nodes 개수 저장 (피드백 기반 제어용)
            # nodes가 리스트인지 확인하고, 빈 리스트도 처리
            if isinstance(nodes, list):
                old_nodes_count = self.routine_nodes_count
                self.routine_nodes_count = len(nodes)
                # nodes가 비어있으면 RESET 완료 플래그 설정
                if len(nodes) == 0:
                    # RESET 완료 플래그가 False에서 True로 변경되는 경우에만 로그 출력
                    if not self.routine_reset_complete_flag:
                        self.get_logger().info(f"[ROUTINE STATUS] RESET 완료 확인: nodes={old_nodes_count} -> 0 (플래그=True로 설정)")
                    self.routine_reset_complete_flag = True
                    # STOP 대기 중이면 STOP 완료 처리
                    if self.waiting_for_stop:
                        self.routine_controller._complete_stop()
                else:
                    self.routine_reset_complete_flag = False
            else:
                old_nodes_count = self.routine_nodes_count
                self.routine_nodes_count = 0
                if not self.routine_reset_complete_flag:
                    self.get_logger().info(f"[ROUTINE STATUS] RESET 완료 확인: nodes={old_nodes_count} -> 0 (플래그=True로 설정)")
                self.routine_reset_complete_flag = True
                # STOP 대기 중이면 STOP 완료 처리
                if self.waiting_for_stop:
                    self.routine_controller._complete_stop()
            
            if not nodes or len(nodes) == 0:
                # nodes가 비어있을 때: 이전에 실행 중이던 루틴이 완료되었을 수 있음
                # Manual Mode에서 Hello/Handshake 루틴이 완료되었는지 확인
                if self.actual_running_routine in ("idling_heart_rt", "idling_handshake_rt"):
                    if self.is_running:
                        # current_manual_mode는 tracking_result_callback에서 업데이트되므로
                        # latest_tracking_result에서도 확인
                        manual_mode = self.current_manual_mode
                        if 'manual_mode' in self.latest_tracking_result:
                            manual_mode = self.latest_tracking_result.get('manual_mode', False)
                        
                        if manual_mode:
                            # Manual Mode: 루틴이 완료되어 nodes가 비어있으면 Tracking으로 전환
                            self.get_logger().info(
                                f"[{self.actual_running_routine} 완료] Manual Mode: 루틴 완료 감지 (nodes 비어있음) → TRACKING 상태로 전환"
                            )
                            tracker_command = {
                                'type': 'set_state',
                                'state': 'tracking',
                                'target_id': None
                            }
                            tracker_msg = String()
                            tracker_msg.data = json.dumps(tracker_command)
                            self.tracker_control_publisher.publish(tracker_msg)
                            # 루틴 리셋
                            self.routine_controller.reset_current_routine_external()
                            # actual_running_routine 초기화
                            self.actual_running_routine = None
                
                self.actual_running_routine = None
                # START 명령 확인: nodes가 비어있어도 타임아웃 체크는 계속 수행
                # (START 명령 후 루틴이 시작되지 않은 경우 감지)
                if self.routine_controller.expected_routine_name and self.routine_controller.expected_routine_start_time:
                    elapsed = time.monotonic() - self.routine_controller.expected_routine_start_time
                    if elapsed > 0.5:  # 0.5초 후에도 시작되지 않으면 경고
                        self.get_logger().warn(
                            f"[ROUTINE START 경고] {self.routine_controller.expected_routine_name} 루틴이 "
                            f"START 명령 후 {elapsed:.3f}초 동안 시작되지 않음. (nodes가 비어있음)"
                        )
                        # 플래그 초기화 (경고 후에도 계속 확인하지 않음)
                        self.routine_controller.expected_routine_name = None
                        self.routine_controller.expected_routine_start_time = None
                return
            
            # 루트 노드 찾기 (parent == -1)
            root_node = None
            for node in nodes:
                if node.get("parent") == -1:
                        root_node = node
                        break
            
            # 실제 루틴 이름 찾기: nodes 배열에서 루틴 이름 패턴 찾기
            actual_routine_name = None
            for node in nodes:
                node_name = node.get("name", "")
                # 루틴 이름 패턴 확인 (idling_heart_rt, idling_handshake_rt, idle_breathing_rt)
                # 전체 이름이 루틴 이름과 일치하거나, 루틴 이름이 포함되어 있는 경우
                if node_name:
                    # 정확한 루틴 이름 매치
                    if node_name in ("idle_breathing_rt", "idling_heart_rt", "idling_handshake_rt"):
                        actual_routine_name = node_name
                        break
                    # 부분 매치 (루틴 이름이 노드 이름에 포함된 경우)
                    elif "_rt" in node_name and ("idling" in node_name or "breathing" in node_name):
                        # 더 구체적인 매치를 위해 정확한 이름 확인
                        if "idling_heart_rt" in node_name:
                            actual_routine_name = "idling_heart_rt"
                            break
                        elif "idling_handshake_rt" in node_name:
                            actual_routine_name = "idling_handshake_rt"
                            break
                        elif "idle_breathing_rt" in node_name:
                            actual_routine_name = "idle_breathing_rt"
                            break
                        # 정확한 매치가 없으면 첫 번째로 찾은 것을 사용
                        elif actual_routine_name is None and "_rt" in node_name:
                            actual_routine_name = node_name
            
            # 단일 루틴 RESET 완료 확인: RESET 중인 루틴이 실행 중이 아니면 RESET 완료
            if self.routine_controller.waiting_for_reset and self.routine_controller.resetting_routine_name:
                # 단일 루틴 RESET의 경우: 해당 루틴이 실행 중이 아니면 RESET 완료
                resetting_routine = self.routine_controller.resetting_routine_name
                is_resetting_routine_running = False
                for node in nodes:
                    node_name = node.get("name", "")
                    if resetting_routine in node_name:
                        is_resetting_routine_running = True
                        break
                
                if not is_resetting_routine_running and not self.routine_reset_complete_flag:
                    self.get_logger().info(f"[ROUTINE STATUS] 단일 루틴 RESET 완료 확인: {resetting_routine} 루틴이 실행 중이 아님 (플래그=True로 설정)")
                    self.routine_reset_complete_flag = True
                    self.routine_controller.resetting_routine_name = None
            
            if root_node:
                status = root_node.get("status")
                root_name = root_node.get("name", "")
                
                # status: 0=IDLE, 1=RUNNING, 2=SUCCESS, 3=FAILURE
                # 디버그: SUCCESS 상태 감지 로그
                if status == 2:  # SUCCESS: 루틴 완료
                    self.get_logger().info(f"[ROUTINE STATUS] 루틴 완료 감지: status=2 (SUCCESS), root_name={root_name}, actual_routine_name={actual_routine_name}")
                    # 실제 루틴 이름이 있으면 사용, 없으면 루트 노드 이름에서 추론
                    routine_name = actual_routine_name
                    if routine_name is None:
                        # 루트 노드 이름에서 루틴 이름 추론
                        if "Heart" in root_name or "heart" in root_name.lower():
                            routine_name = "idling_heart_rt"
                        elif "Handshake" in root_name or "handshake" in root_name.lower():
                            routine_name = "idling_handshake_rt"
                        elif "LoopWhile" in root_name or "breathing" in root_name.lower():
                            routine_name = "idle_breathing_rt"
                        else:
                            routine_name = root_name
                    
                    # Hello/Handshake 완료 후 상태 전환
                    if routine_name in ("idling_heart_rt", "idling_handshake_rt"):
                        if self.is_running:
                            # current_manual_mode는 tracking_result_callback에서 업데이트되므로
                            # latest_tracking_result에서도 확인
                            manual_mode = self.current_manual_mode
                            if 'manual_mode' in self.latest_tracking_result:
                                manual_mode = self.latest_tracking_result.get('manual_mode', False)
                            
                            if manual_mode:
                                # Manual Mode: TRACKING으로 전환
                                self.get_logger().info(f"[{routine_name} 완료] Manual Mode: TRACKING 상태로 전환 (current_manual_mode={self.current_manual_mode})")
                                tracker_command = {
                                    'type': 'set_state',
                                    'state': 'tracking',
                                    'target_id': None
                                }
                                tracker_msg = String()
                                tracker_msg.data = json.dumps(tracker_command)
                                self.tracker_control_publisher.publish(tracker_msg)
                                # 루틴 리셋
                                self.routine_controller.reset_current_routine_external()
                            else:
                                # Auto Mode: SEARCHING으로 전환 (기존 동작)
                                self.get_logger().info(f"[{routine_name} 완료] Auto Mode: SEARCHING 상태로 전환")
                                tracker_command = {
                                    'type': 'set_state',
                                    'state': 'searching',
                                    'target_id': None
                                }
                                tracker_msg = String()
                                tracker_msg.data = json.dumps(tracker_command)
                                self.tracker_control_publisher.publish(tracker_msg)
                                # 루틴 리셋
                                self.routine_controller.reset_current_routine_external()
                
                if status == 1:  # RUNNING인 경우만 실행 중으로 판단
                    # 실제 루틴 이름이 있으면 사용, 없으면 루트 노드 이름에서 추론
                    routine_name = actual_routine_name
                    if routine_name is None:
                        # 루트 노드 이름에서 루틴 이름 추론
                        if "Heart" in root_name or "heart" in root_name.lower():
                            routine_name = "idling_heart_rt"
                        elif "Handshake" in root_name or "handshake" in root_name.lower():
                            routine_name = "idling_handshake_rt"
                        elif "LoopWhile" in root_name or "breathing" in root_name.lower():
                            routine_name = "idle_breathing_rt"
                        else:
                            routine_name = root_name
                    if routine_name:
                        self.actual_running_routine = routine_name
                        
                        # START 명령 확인: 예상한 루틴이 시작되었는지 확인
                        if self.routine_controller.expected_routine_name and self.routine_controller.expected_routine_start_time:
                            expected_routine = self.routine_controller.expected_routine_name
                            elapsed = time.monotonic() - self.routine_controller.expected_routine_start_time
                            
                            # breathing 루틴인 경우: LoopWhile 노드 확인
                            if expected_routine == "idle_breathing_rt":
                                if root_name == "LoopWhile" or actual_routine_name == expected_routine:
                                    self.get_logger().info(
                                        f"[ROUTINE START 확인] {expected_routine} 루틴 시작 확인됨 "
                                        f"(루트 노드: {root_name}, 실제 루틴: {actual_routine_name}, 경과 시간: {elapsed:.3f}초)"
                                    )
                                    # 플래그 초기화
                                    self.routine_controller.expected_routine_name = None
                                    self.routine_controller.expected_routine_start_time = None
                            # handshake/hello 루틴인 경우: Sequence 노드 또는 HeartRoutine/HandshakeRoutine 노드 확인
                            elif expected_routine in ("idling_heart_rt", "idling_handshake_rt"):
                                # Sequence 노드이거나 HeartRoutine/HandshakeRoutine 노드이면 성공
                                # handshake/hello 루틴은 루트 노드가 Sequence 또는 HeartRoutine(X)/HandshakeRoutine(X)일 수 있음
                                is_routine_started = False
                                if root_name == "Sequence":
                                    is_routine_started = True
                                elif expected_routine == "idling_heart_rt" and ("Heart" in root_name or "heart" in root_name.lower()):
                                    is_routine_started = True
                                elif expected_routine == "idling_handshake_rt" and ("Handshake" in root_name or "handshake" in root_name.lower()):
                                    is_routine_started = True
                                elif actual_routine_name == expected_routine:
                                    # 실제 루틴 이름으로도 확인 가능
                                    is_routine_started = True
                                
                                if is_routine_started:
                                    self.get_logger().info(
                                        f"[ROUTINE START 확인] {expected_routine} 루틴 시작 확인됨 "
                                        f"(루트 노드: {root_name}, 실제 루틴: {actual_routine_name if actual_routine_name else 'N/A (루트 노드로 확인)'}, 경과 시간: {elapsed:.3f}초)"
                                    )
                                    # 플래그 초기화
                                    self.routine_controller.expected_routine_name = None
                                    self.routine_controller.expected_routine_start_time = None
                                elif elapsed > 0.5:
                                    # 0.5초 후에도 시작되지 않으면 재시도
                                    self.get_logger().warn(
                                        f"[ROUTINE START 재시도] {expected_routine} 루틴이 시작되지 않음 "
                                        f"(루트 노드: {root_name}, 실제 루틴: {actual_routine_name}), 재시도 중..."
                                    )
                                    # 재시도: START 명령 다시 발행
                                    command = f"{self.routine_controller.robot_name}::ROUTINE::{expected_routine}::START"
                                    self.routine_controller.publish_command(command)
                                    self.routine_controller.expected_routine_start_time = time.monotonic()
                else:
                    # RUNNING이 아니면 실행 중이 아님
                    self.actual_running_routine = None
            else:
                # 루트 노드를 찾을 수 없으면 루틴이 없는 것으로 간주
                self.actual_running_routine = None
            
            # START 명령 확인: 일정 시간(0.5초) 후에도 루틴이 시작되지 않으면 경고
            if self.routine_controller.expected_routine_name and self.routine_controller.expected_routine_start_time:
                elapsed = time.monotonic() - self.routine_controller.expected_routine_start_time
                if elapsed > 0.5:  # 0.5초 후에도 시작되지 않으면 경고
                    self.get_logger().warn(
                        f"[ROUTINE START 경고] {self.routine_controller.expected_routine_name} 루틴이 "
                        f"START 명령 후 {elapsed:.3f}초 동안 시작되지 않음. "
                        f"현재 실행 중인 루틴: {self.actual_running_routine}"
                    )
                    # 플래그 초기화 (경고 후에도 계속 확인하지 않음)
                    self.routine_controller.expected_routine_name = None
                    self.routine_controller.expected_routine_start_time = None
            
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"루틴 상태 피드백 JSON 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().warn(f"루틴 상태 피드백 처리 실패: {e}")
    
    def image_callback(self, msg: CompressedImage) -> None:
        """이미지 콜백 - 타겟 Crop 이미지 발행용으로 저장"""
        if not self.is_running:
            return
        
        # 압축된 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            self.latest_frame = frame
    
    def tracking_result_callback(self, msg: String):
        """추적 결과 콜백 - Tracker 노드에서 발행한 결과 처리"""
        try:
            data = json.loads(msg.data)
            self.latest_tracking_result = data
            
            # Manual Mode 추적 (tracking_result에서 manual_mode 가져오기)
            self.current_manual_mode = data.get('manual_mode', False)
            
            # 상태 변경 감지 및 루틴 전환 처리 (RUN 중일 때만)
            if self.is_running:
                state_str = data.get('state', 'idle')
                try:
                    current_state = TrackingState[state_str.upper()]
                except (KeyError, AttributeError):
                    current_state = TrackingState.IDLE
                
                if current_state != self.previous_state:
                    self._handle_state_change(self.previous_state, current_state)
                    self.previous_state = current_state
                else:
                    # 상태가 변경되지 않았어도 RESET 대기 중이면 계속 확인
                    if self.routine_controller.waiting_for_reset:
                        # 목표 루틴 결정
                        target_routine = self._get_target_routine_for_state(current_state)
                        if target_routine:
                            # transition_to_routine 호출 (이전 상태는 알 수 없으므로 None 전달)
                            self.routine_controller.transition_to_routine(
                                self.routine_controller.current_routine,
                                target_routine,
                                current_state,  # old_state 대신 current_state 사용
                                current_state   # new_state도 current_state 사용
                            )
            else:
                # RUN 중이 아니면 상태만 업데이트 (루틴 전환은 하지 않음)
                state_str = data.get('state', 'idle')
                try:
                    current_state = TrackingState[state_str.upper()]
                except (KeyError, AttributeError):
                    current_state = TrackingState.IDLE
                self.previous_state = current_state
            
            # GUI용 추적 데이터 발행 (목/허리 각도 정보 포함)
            self._publish_tracking_data(data)
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f"추적 결과 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"추적 결과 처리 실패: {e}")
    
    def neck_angle_callback(self, msg: String):
        """목 각도 콜백 - Controller 노드에서 발행한 목 각도 저장"""
        try:
            data = json.loads(msg.data)
            self.latest_neck_yaw_rad = data.get('current_yaw_rad', None)
            # 목/허리 각도 정보도 저장 (GUI 표시용)
            self.latest_neck_angles = {
                'current': {
                    'yaw_rad': data.get('current_yaw_rad', 0.0),
                    'pitch_rad': data.get('current_pitch_rad', 0.0)
                },
                'target': {
                    'yaw_rad': data.get('target_yaw_rad', 0.0),
                    'pitch_rad': data.get('target_pitch_rad', 0.0)
                }
            }
            self.latest_waist_angles = {
                'current': {
                    'yaw_rad': data.get('current_waist_yaw_rad', 0.0)
                },
                'target': {
                    'yaw_rad': data.get('target_waist_yaw_rad', 0.0)
                }
            }
        except Exception as e:
            self.get_logger().warn(f"목 각도 파싱 실패: {e}")
    
    def _publish_tracking_data(self, tracking_result_data):
        """GUI용 추적 데이터 발행"""
        try:
            # 목/허리 각도 정보는 Controller 노드에서 받은 최신 값 사용
            if self.latest_neck_angles is None:
                self.latest_neck_angles = {
                    'current': {'yaw_rad': 0.0, 'pitch_rad': 0.0},
                    'target': {'yaw_rad': 0.0, 'pitch_rad': 0.0}
                }
            if self.latest_waist_angles is None:
                self.latest_waist_angles = {
                    'current': {'yaw_rad': 0.0},
                    'target': {'yaw_rad': 0.0}
                }
            
            data = {
                **tracking_result_data,  # 추적 결과 데이터 복사
                'neck_angles': self.latest_neck_angles,
                'waist_angles': self.latest_waist_angles,
                'performance': {
                    'fps': 0.0,
                    'process_time_ms': tracking_result_data.get('performance', {}).get('process_time_ms', 0.0)
                },
                'center_zone': {
                    'elapsed_time': None,
                    'duration': 5.0
                },
                'timestamp': time.monotonic(),
                # GUI 업데이트를 위한 추가 정보
                'manual_mode': tracking_result_data.get('manual_mode', False),  # tracking_result에서 manual_mode 가져오기
                'is_running': self.is_running  # 현재 RUN/STOP 상태
            }
            
            json_str = json.dumps(data, ensure_ascii=False)
            msg = String()
            msg.data = json_str
            self.tracking_data_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"추적 데이터 발행 실패: {e}")
    
    def _publish_target_crop(self, frame: np.ndarray, tracked_objects_data, target_track_id):
        """타겟 BB Box Crop 이미지 발행"""
        try:
            # 타겟에 해당하는 객체 찾기
            target_obj_data = None
            for obj_data in tracked_objects_data:
                if obj_data.get('track_id') == target_track_id:
                    target_obj_data = obj_data
                    break
            
            if target_obj_data is None:
                return
            
            # BB Box 좌표 추출
            bbox = target_obj_data.get('bbox', [])
            if len(bbox) != 4:
                return
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 이미지 경계 확인 및 조정
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # 유효한 BB Box인지 확인
            if x2 <= x1 or y2 <= y1:
                return
            
            # Crop 수행
            crop_img = frame[y1:y2, x1:x2]
            
            if crop_img.size == 0:
                return
            
            # 정사각형으로 만들기 (비율 유지)
            crop_h, crop_w = crop_img.shape[:2]
            max_dim = max(crop_w, crop_h)
            
            # 정사각형 이미지 생성 (검은색 배경)
            square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            
            # 중앙에 crop 이미지 배치
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2
            square_img[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = crop_img
            
            # 336x336으로 resize
            resized_img = cv2.resize(square_img, (336, 336), interpolation=cv2.INTER_LINEAR)
            
            # JPEG 압축
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            result, encimg = cv2.imencode('.jpg', resized_img, encode_param)
            
            if not result:
                return
            
            # CompressedImage 메시지 생성 및 발행
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = "jpeg"
            msg.data = encimg.tobytes()
            self.target_crop_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"타겟 Crop 이미지 발행 실패: {e}")
    
    def _get_target_routine_for_state(self, state: TrackingState) -> str:
        """
        상태에 해당하는 목표 루틴 이름 반환
        
        Returns:
            루틴 이름 (idle_breathing_rt, idling_heart_rt, idling_handshake_rt) 또는 None
        """
        if state in (TrackingState.IDLE, TrackingState.WAITING, TrackingState.TRACKING, 
                     TrackingState.LOST, TrackingState.SEARCHING):
            return "idle_breathing_rt"
        elif state == TrackingState.HELLO:
            return "idling_heart_rt"
        elif state == TrackingState.HANDSHAKE:
            return "idling_handshake_rt"
        return None
    
    def _handle_state_change(self, old_state: TrackingState, new_state: TrackingState):
        """
        상태 변경 시 루틴 전환 처리
        
        로직:
        1. 상태에 따라 목표 루틴 결정
        2. 현재 실행 중인 루틴과 목표 루틴 비교
        3. 같으면 아무것도 하지 않음 (breathing -> breathing 등)
        4. 다르면 현재 루틴 PAUSE/RESET → 피드백 확인 → 목표 루틴 START
        """
        # 제외 케이스: TRACKING -> LOST, IDLE -> WAITING -> TRACKING
        if old_state == TrackingState.TRACKING and new_state == TrackingState.LOST:
            return  # 제외
        if old_state == TrackingState.IDLE and new_state == TrackingState.WAITING:
            return  # 제외
        if old_state == TrackingState.WAITING and new_state == TrackingState.TRACKING:
            return  # 제외
        
        # 목표 루틴 결정
        target_routine = self._get_target_routine_for_state(new_state)
        if target_routine is None:
            self.get_logger().warn(f"알 수 없는 상태에 대한 루틴: {new_state}")
            return
        
        # 현재 실행 중인 루틴 확인
        current_routine = self.routine_controller.current_routine
        
        # RESET 대기 중이면 루틴 전환을 하지 않음 (이미 전환 진행 중)
        if self.routine_controller.waiting_for_reset:
            self.get_logger().debug(f"{old_state.value} → {new_state.value}: RESET 대기 중이므로 루틴 전환 건너뜀")
            return
        
        # 같은 루틴이면 아무것도 하지 않음 (breathing -> breathing 등)
        if current_routine == target_routine:
            self.get_logger().debug(f"{old_state.value} → {new_state.value}: 동일 루틴 ({target_routine}) 유지")
            return
        
        # STOP 플래그 초기화 (interaction 상태 전환 시)
        if new_state in (TrackingState.HELLO, TrackingState.HANDSHAKE):
            self.waiting_for_stop = False
        
        # 루틴 전환 필요: 현재 루틴 PAUSE/RESET → 목표 루틴 START
        self.routine_controller.transition_to_routine(current_routine, target_routine, old_state, new_state)
    
    def _manual_control_callback(self, msg: String):
        """Manual 제어 콜백 - GUI에서 오는 명령 처리"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            # Tracker와 Controller에 명령 전달
            tracker_command = command.copy()
            controller_command = command.copy()
            
            if cmd_type == 'run' or cmd_type == 'start':
                # RESET 대기 중이면 플래그 리셋 (RUN 시작 시 초기화)
                if self.routine_controller.waiting_for_reset:
                    self.get_logger().warn("RUN 시작: 이전 RESET 대기 상태 리셋")
                    self.routine_controller.waiting_for_reset = False
                    self.routine_reset_complete_flag = False
                
                # TELEOP 제어 (RUN 명령 시)
                self._handle_teleop_for_run()
                
                self.is_running = True
                manual_mode = command.get('manual', False)
                tracker_command['manual'] = manual_mode
                # IDLE 상태로 시작하므로 idle_breathing_rt 시작
                if not self.routine_controller.breathing_routine_running:
                    self.routine_controller.start_breathing()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드")
            
            elif cmd_type == 'stop':
                # TELEOP 제어 (STOP 명령 시)
                self._handle_teleop_for_stop()
                
                self.is_running = False
                # STOP 처리: 루틴 PAUSE/RESET부터 시작 (RESET 완료 후 트래커/컨트롤러 STOP은 _complete_stop에서 처리)
                if not self.routine_controller.waiting_for_reset:
                    self.routine_controller.stop_current_routine()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info("RUN 중지: 루틴 RESET 대기 중...")
            
            elif cmd_type == 'set_mode':
                if self.is_running:
                    manual_mode = command.get('manual', False)
                    tracker_command['manual'] = manual_mode
                    # Manual <-> Auto 전환 시 IDLE 상태로 변경
                    tracker_command['type'] = 'set_state'
                    tracker_command['state'] = 'idle'
                    tracker_command['target_id'] = None
                    self.get_logger().info(f"Manual 모드 설정: {manual_mode} (IDLE 상태로 전환)")
            
            elif cmd_type == 'set_state':
                state_str = command.get('state', 'idle')
                target_id = command.get('target_id', None)
                tracker_command['state'] = state_str
                tracker_command['target_id'] = target_id
                self.get_logger().info(f"상태 설정: {state_str}, 타겟 ID: {target_id}")
                
                # set_state 명령은 tracker로 전송만 하고, 실제 상태 변경은 tracking_result_callback에서 처리
                # (중복 처리 방지를 위해 여기서는 루틴 전환을 하지 않음)
            
            elif cmd_type == 'set_target':
                target_id = command.get('target_id')
                if target_id is not None:
                    tracker_command['target_id'] = target_id
                    self.get_logger().info(f"타겟 변경: {target_id}")
            
            elif cmd_type == 'set_parameters':
                # 파라미터 설정 명령은 Controller로만 전달
                parameters = command.get('parameters', {})
                controller_command['parameters'] = parameters
                self.get_logger().info(f"파라미터 설정 요청: {parameters}")
            
            # Tracker에 명령 전송 (STOP은 RESET 완료 후 _complete_stop에서 전송)
            if cmd_type != 'stop':
                tracker_msg = String()
                tracker_msg.data = json.dumps(tracker_command)
                self.tracker_control_publisher.publish(tracker_msg)
            
            # Controller에 명령 전송 (run, stop, set_parameters) (STOP은 RESET 완료 후 _complete_stop에서 전송)
            if cmd_type in ['run', 'stop', 'set_parameters']:
                if cmd_type != 'stop':
                    controller_msg = String()
                    controller_msg.data = json.dumps(controller_command)
                    self.controller_control_publisher.publish(controller_msg)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Manual 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"Manual 제어 처리 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = AllexIdleInteractionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
