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
from std_msgs.msg import String, Int32MultiArray, Float64MultiArray
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
        
        # 루틴 제어 Publisher: hmi/robot_command (모든 루틴 명령 제어용)
        self.command_pub = node.create_publisher(
            String,
            'hmi/robot_command',
            10
        )
        
        # External Topic Publisher: /robot_inbound/routine/external_data (서브 루틴 제어용)
        qos_t1 = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )
        self.external_topic_pub = node.create_publisher(
            Float64MultiArray,
            '/robot_inbound/routine/external_data',
            qos_t1
        )
        
        # idle_interaction_selector_rt 실행 상태 추적
        self.selector_routine_running = False
        
        # Handshake/Hello 완료 감지용
        self.handshake_start_time = None  # Handshake 루틴 시작 시간
        self.hello_start_time = None  # Hello 루틴 시작 시간
        self.handshake_complete_check_started = False  # Handshake 완료 확인 시작 여부
        self.hello_complete_check_started = False  # Hello 완료 확인 시작 여부
        
        # 루틴 이름 -> External Topic 명령 코드 매핑
        self.routine_to_cmd_code = {
            "idle_breathing_rt": 21.0,      # Breathing RT
            "idling_heart_rt": 22.0,        # Heart RT
            "idling_handshake_rt": 23.0,    # Handshake RT
        }
        
        node.get_logger().info(f"RoutineController 초기화 완료 (로봇: {robot_name})")
    
    def publish_command(self, command: str):
        """명령을 /hmi/robot_command 토픽으로 발행 (모든 루틴 명령 제어용)"""
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)
        self.node.get_logger().info(f"Routine 명령 발행: {command}")
        time.sleep(0.01)  # 명령 처리 시간 대기
    
    def send_external_command(self, cmd_code: float):
        """External Topic으로 서브 루틴 명령 전송"""
        msg = Float64MultiArray()
        msg.data = [2.0, cmd_code]  # [2, 명령코드]
        self.external_topic_pub.publish(msg)
        self.node.get_logger().info(f"External Topic 명령 발행: {msg.data}")
        time.sleep(0.01)  # 명령 처리 시간 대기
    
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
    
    def wait_for_routine_running(self, routine_name: str, timeout=3.0, check_interval=0.1) -> bool:
        """
        특정 루틴이 실행 중인지 확인 (블로킹)
        
        Args:
            routine_name: 확인할 루틴 이름
            timeout: 최대 대기 시간 (초)
            check_interval: 확인 간격 (초)
        
        Returns:
            True: 루틴 실행 중 확인
            False: 타임아웃
        """
        import rclpy
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=check_interval)
            
            # 실제 실행 중인 루틴 확인
            if self.node.actual_running_routine == routine_name:
                elapsed = time.monotonic() - start_time
                self.node.get_logger().info(f"{routine_name} 루틴 실행 확인 완료 (대기 시간: {elapsed:.2f}초)")
                return True
        
        self.node.get_logger().warn(f"{routine_name} 루틴 실행 확인 타임아웃 ({timeout}초)")
        return False
    
    def wait_for_selector_running(self, timeout=3.0, check_interval=0.1) -> bool:
        """
        idle_interaction_selector_rt가 실행 중인지 확인 (블로킹)
        
        Args:
            timeout: 최대 대기 시간 (초)
            check_interval: 확인 간격 (초)
        
        Returns:
            True: selector 루틴 실행 중 확인
            False: 타임아웃
        """
        import rclpy
        start_time = time.monotonic()
        while (time.monotonic() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=check_interval)
            
            # /debug/routine에서 IdleInteractionSelector 노드 확인
            if self.node.routine_nodes_count > 0:
                # 루트 노드가 IdleInteractionSelector인지 확인
                # (실제로는 _routine_status_callback에서 확인하지만, 여기서는 nodes_count로 간접 확인)
                elapsed = time.monotonic() - start_time
                self.node.get_logger().info(f"idle_interaction_selector_rt 실행 확인 완료 (대기 시간: {elapsed:.2f}초)")
                return True
        
        self.node.get_logger().warn(f"idle_interaction_selector_rt 실행 확인 타임아웃 ({timeout}초)")
        return False
    
    def start_pause_reset_all_routines(self):
        """
        idle_interaction_selector_rt에 대해 PAUSE -> RESET 명령 발행 (STOP 명령 시에만 사용)
        (RESET 완료될 때까지 계속 호출됨)
        """
        # RESET 완료 플래그 초기화
        self.node.routine_reset_complete_flag = False
        # START 확인 플래그 초기화 (PAUSE/RESET 시작 시)
        self.expected_routine_name = None
        self.expected_routine_start_time = None
        
        # idle_interaction_selector_rt에 대해 PAUSE 명령 발행
        pause_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::PAUSE"
        self.publish_command(pause_command)
        time.sleep(0.02)  # PAUSE 처리 대기 (최소 20ms)
        
        # idle_interaction_selector_rt에 대해 RESET 명령 발행
        reset_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::RESET"
        self.publish_command(reset_command)
        time.sleep(0.1)  # RESET 처리 대기
        
        self.selector_routine_running = False
        self.node.get_logger().info(f"[PAUSE/RESET ALL] PAUSE/RESET 명령 발행 (현재 nodes={self.node.routine_nodes_count})")
    
    def reset_current_routine_external(self):
        """
        External Topic으로 현재 실행 중인 루틴 Reset (빠른 전환용)
        일반 루틴 전환 시 사용 (Handshake -> Hello, Hello -> Tracking 등)
        """
        # External Topic으로 Reset 명령 전송 (100 = Reset)
        self.send_external_command(100.0)
        self.node.get_logger().info(f"[EXTERNAL RESET] 현재 루틴 Reset 명령 발송 (External Topic: [2, 100])")
        # Reset 명령 발송 후 피드백 확인을 위한 로그
        self.node.get_logger().info(f"[EXTERNAL RESET] Reset 명령 발송 완료, 피드백 대기 중...")
    
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
        루틴 전환: External Topic으로 빠른 전환 (Reset 후 새 루틴 시작)
        Handshake -> Hello, Hello -> Tracking 등 일반 전환 시 사용
        
        Args:
            from_routine: 현재 실행 중인 루틴 이름 (None이면 실행 중이 아님)
            to_routine: 목표 루틴 이름
            old_state: 이전 상태
            new_state: 새로운 상태
        """
        # idle_interaction_selector_rt가 실행 중이 아니면 먼저 START
        if not self.selector_routine_running:
            selector_start_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::START"
            self.publish_command(selector_start_command)
            # 피드백 확인: selector 루틴이 실행 중인지 확인
            if self.wait_for_selector_running(timeout=2.0):
                self.selector_routine_running = True
            else:
                self.node.get_logger().warn("idle_interaction_selector_rt 시작 확인 실패, 계속 진행...")
                self.selector_routine_running = True  # 일단 진행
        
        # 현재 루틴이 있으면 External Topic으로 Reset
        if from_routine:
            self.reset_current_routine_external()
            # 피드백 확인: Idle 상태가 되었는지 확인 (최대 1초)
            self.wait_for_idle(timeout=1.0)
        
        # 목표 루틴을 External Topic으로 시작
        cmd_code = self.routine_to_cmd_code.get(to_routine)
        if cmd_code is None:
            self.node.get_logger().error(f"알 수 없는 루틴 이름: {to_routine}")
            return
        
        self.current_routine = to_routine
        
        # breathing 루틴인지 확인
        if to_routine == "idle_breathing_rt":
            self.breathing_routine_running = True
        else:
            self.breathing_routine_running = False
        
        # External Topic으로 서브 루틴 시작 (Reliable QoS이므로 확인 로직 제거)
        self.send_external_command(cmd_code)
        
        # Handshake/Hello 루틴 시작 시간 기록
        if to_routine == "idling_handshake_rt":
            self.handshake_start_time = time.monotonic()
            self.handshake_complete_check_started = False
            self.node.get_logger().info(f"Handshake 루틴 시작 시간 기록: {self.handshake_start_time}")
        elif to_routine == "idling_heart_rt" and new_state == TrackingState.HELLO:
            self.hello_start_time = time.monotonic()
            self.hello_complete_check_started = False
            self.node.get_logger().info(f"Hello 루틴 시작 시간 기록: {self.hello_start_time}")
        
        self.node.get_logger().info(
            f"{old_state.value} → {new_state.value}: {from_routine or 'None'} Reset([2,100]) → {to_routine}([2,{int(cmd_code)}]) 전환 완료"
        )
    
    def start_breathing(self):
        """숨쉬기 루틴 시작 (무한 반복) - 항상 Reset(100) 후 Breathing(21) 시작, 피드백 확인"""
        # 이미 실행 중이면 중복 시작 방지
        if self.breathing_routine_running and self.current_routine == "idle_breathing_rt":
            self.node.get_logger().warn("idle_breathing_rt가 이미 실행 중입니다. 중복 시작 건너뜀.")
            return
        
        # idle_interaction_selector_rt가 실행 중이 아니면 먼저 START
        if not self.selector_routine_running:
            selector_start_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::START"
            self.publish_command(selector_start_command)
            # 피드백 확인: selector 루틴이 실행 중인지 확인
            if self.wait_for_selector_running(timeout=2.0):
                self.selector_routine_running = True
            else:
                self.node.get_logger().warn("idle_interaction_selector_rt 시작 확인 실패, 계속 진행...")
                self.selector_routine_running = True  # 일단 진행
        
        # 항상 먼저 Reset (100) 전송
        self.node.get_logger().info("기존 루틴 Reset 후 Breathing 시작")
        self.send_external_command(100.0)  # Reset
        # 피드백 확인: Idle 상태가 되었는지 확인 (최대 1초)
        self.wait_for_idle(timeout=1.0)
        
        # 루틴 시작 (External Topic으로 Breathing RT 시작)
        routine_name = "idle_breathing_rt"
        cmd_code = self.routine_to_cmd_code[routine_name]  # 21.0
        self.current_routine = routine_name
        self.breathing_routine_running = True
        self.send_external_command(cmd_code)
        
        # Reliable QoS이므로 확인 로직 제거, 명령 발행 완료
        self.node.get_logger().info(f"idle_breathing_rt 시작: {routine_name} (Reset[2,100] -> Breathing[2,{int(cmd_code)}])")
    
    def stop_routine(self):
        """STOP 명령 시 현재 루틴 Reset(100)만 수행, 피드백 확인"""
        self.node.get_logger().info("STOP 명령: 현재 루틴 Reset")
        
        # idle_interaction_selector_rt가 실행 중이 아니면 Reset 불필요
        if not self.selector_routine_running:
            self.node.get_logger().info("STOP: selector 루틴이 실행 중이 아니므로 Reset 불필요")
            return
        
        # 현재 루틴이 있으면 Reset (100) 전송
        if self.current_routine is not None:
            self.node.get_logger().info(f"STOP: 기존 루틴({self.current_routine}) Reset")
            self.send_external_command(100.0)  # Reset
            # 피드백 확인: Idle 상태가 되었는지 확인 (최대 2초)
            self.wait_for_idle(timeout=2.0)
            # 상태 초기화
            self.current_routine = None
            self.breathing_routine_running = False
            self.node.get_logger().info("STOP: 루틴 Reset 완료")
        else:
            self.node.get_logger().info("STOP: 실행 중인 루틴이 없음")
    
    def cleanup_on_shutdown(self):
        """
        노드 종료 시(Ctrl+C) 루틴 정리
        오직 노드 종료 시에만 호출됨
        """
        self.node.get_logger().info("[SHUTDOWN] 노드 종료 중... 루틴 정리 시작")
        
        # START 확인 플래그 초기화
        self.expected_routine_name = None
        self.expected_routine_start_time = None
        
        # idle_interaction_selector_rt에 대해 PAUSE -> RESET 명령 발행
        # (블로킹 대기 없이 명령만 발행하고 즉시 반환)
        pause_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::PAUSE"
        self.publish_command(pause_command)
        time.sleep(0.02)  # PAUSE 처리 대기 (최소 20ms)
        
        reset_command = f"{self.robot_name}::ROUTINE::idle_interaction_selector_rt::RESET"
        self.publish_command(reset_command)
        
        self.node.get_logger().info("[SHUTDOWN] 루틴 정리 명령 발행 완료 (PAUSE -> RESET)")
        
        # 상태 초기화
        self.current_routine = None
        self.breathing_routine_running = False
        self.selector_routine_running = False
        self.node.actual_running_routine = None


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
        
        # 현재 실행 중인 루틴 이름 추적 (실제 실행 중인 루틴)
        self.actual_running_routine = None  # 실제 실행 중인 루틴 이름
        
        # 루틴 찾기 로그 최소화용 (상태가 바뀔 때만 출력)
        self.last_handshake_node_id = None
        self.last_handshake_node_status = None
        self.last_heart_node_id = None
        self.last_heart_node_status = None
        self.routine_nodes_count = 0  # 최신 nodes 개수 저장 (피드백 기반 제어용)
        self.routine_reset_complete_flag = False  # RESET 완료 플래그 (콜백에서 설정)
        
        # Neck articulation 상태 저장 (data[1]의 값: 4=READY, 5=RUN)
        self.neck_articulation_status = None  # None=알 수 없음, 4=READY, 5=RUN
        
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
    
    def _check_handshake_markers(self, sequence_id: int, all_nodes: list) -> bool:
        """
        Sequence 노드의 하위에 handshake 마커가 있는지 확인
        
        Args:
            sequence_id: 확인할 Sequence 노드의 ID
            all_nodes: 모든 노드 리스트
            
        Returns:
            True: handshake 마커가 있음 (handshake_routine_)
            False: handshake 마커가 없음 (heart_routine_)
        """
        # sequence_id의 직접 자식 찾기
        direct_children = [n for n in all_nodes if n.get("parent") == sequence_id]
        
        # GoToCartePoint나 hand_shaking 관련 trajectory 확인
        for child in direct_children:
            child_type = child.get("type", "")
            child_name = child.get("name", "").lower()
            
            # GoToCartePoint가 있으면 handshake
            if child_type == "GoToCartePoint":
                return True
            
            # hand_shaking 관련 trajectory가 있으면 handshake
            if "hand_shaking" in child_name or "handshake" in child_name:
                return True
            
            # 재귀적으로 확인 (Parallel 내부 등)
            if child_type in ["Parallel", "Sequence"]:
                if self._check_handshake_markers(child.get("id"), all_nodes):
                    return True
        
        return False
    
    def _check_all_children_completed(self, node_id: int, all_nodes: list) -> bool:
        """
        노드의 모든 직접 자식 노드가 완료되었는지 확인 (status=2 또는 status=0)
        
        Args:
            node_id: 확인할 노드의 ID
            all_nodes: 모든 노드 리스트
            
        Returns:
            True: 모든 직접 자식 노드가 완료됨 (status=2 또는 status=0)
            False: 아직 실행 중인 자식 노드가 있음
        """
        direct_children = [n for n in all_nodes if n.get("parent") == node_id]
        
        if len(direct_children) == 0:
            return False  # 자식이 없으면 완료로 간주하지 않음
        
        # 모든 직접 자식 노드가 완료되었는지 확인
        for child in direct_children:
            child_status = child.get("status", 0)
            # status=1 (RUNNING)이면 아직 실행 중
            if child_status == 1:
                return False
        
        # 모든 자식 노드가 status=0 (IDLE) 또는 status=2 (SUCCESS) 또는 status=3 (FAILURE)
        return True
    
    def _find_children_range(self, node_id: int, all_nodes: list) -> tuple[int, int]:
        """
        노드의 자식 노드 범위 찾기 (부모 노드가 0인 다음 노드 전까지가 자식 노드들)
        
        Args:
            node_id: 확인할 노드의 ID
            all_nodes: 모든 노드 리스트 (ID 순서대로 정렬되어 있다고 가정)
            
        Returns:
            (start_idx, end_idx): 자식 노드들의 시작 인덱스와 끝 인덱스 (end_idx는 포함하지 않음)
            자식 노드가 없으면 (None, None) 반환
        """
        # 노드를 ID 순서대로 정렬
        sorted_nodes = sorted(all_nodes, key=lambda n: n.get("id", 0))
        
        # node_id의 자식 노드 찾기
        children = []
        for node in sorted_nodes:
            if node.get("parent") == node_id:
                children.append(node)
        
        if len(children) == 0:
            return (None, None)
        
        # 자식 노드들의 ID 범위 찾기
        child_ids = [c.get("id") for c in children]
        min_child_id = min(child_ids)
        max_child_id = max(child_ids)
        
        # 부모 노드가 0인 다음 노드 찾기 (자식 노드 범위의 끝)
        # max_child_id 다음에 오는 노드 중 parent가 0인 노드 찾기
        end_id = None
        for node in sorted_nodes:
            node_id_val = node.get("id", 0)
            if node_id_val > max_child_id and node.get("parent") == 0:
                end_id = node_id_val
                break
        
        # 자식 노드 범위의 끝 인덱스 찾기
        if end_id is not None:
            # end_id 이전까지가 자식 노드 범위
            end_idx = None
            for idx, node in enumerate(sorted_nodes):
                if node.get("id") == end_id:
                    end_idx = idx
                    break
        else:
            # 부모 노드가 0인 다음 노드가 없으면, 모든 노드의 끝까지
            end_idx = len(sorted_nodes)
        
        # 시작 인덱스 찾기
        start_idx = None
        for idx, node in enumerate(sorted_nodes):
            if node.get("id") == min_child_id:
                start_idx = idx
                break
        
        return (start_idx, end_idx)
    
    def _check_all_descendants_completed(self, node_id: int, all_nodes: list, exclude_types: list = None) -> tuple[bool, Optional[int]]:
        """
        노드의 모든 자식 노드(직접 자식 + 하위 자식)가 완료되었는지 확인
        부모 노드가 0인 다음 노드 전까지가 자식 노드 범위로 간주
        
        Args:
            node_id: 확인할 노드의 ID
            all_nodes: 모든 노드 리스트
            exclude_types: 제외할 노드 타입 리스트 (예: ["WaitTrajectory"])
            
        Returns:
            (is_completed, current_index): 
            - is_completed: True면 모든 자식 노드가 완료됨, False면 아직 실행 중인 노드가 있음
            - current_index: 현재 실행 중인 노드의 인덱스 (완료되었으면 None)
        """
        if exclude_types is None:
            exclude_types = []
        
        # 노드를 ID 순서대로 정렬
        sorted_nodes = sorted(all_nodes, key=lambda n: n.get("id", 0))
        
        # 자식 노드 범위 찾기
        start_idx, end_idx = self._find_children_range(node_id, sorted_nodes)
        
        if start_idx is None or end_idx is None:
            return (False, None)  # 자식 노드가 없으면 완료로 간주하지 않음
        
        # 범위 내의 모든 노드가 완료되었는지 확인
        for idx in range(start_idx, end_idx):
            if idx >= len(sorted_nodes):
                break
            node = sorted_nodes[idx]
            node_type = node.get("type", "")
            node_name = node.get("name", "")
            
            # 제외할 타입인지 확인
            should_exclude = False
            for exclude_type in exclude_types:
                if exclude_type.lower() in node_type.lower() or exclude_type.lower() in node_name.lower():
                    should_exclude = True
                    break
            
            if should_exclude:
                continue  # 제외할 노드는 체크하지 않음
            
            node_status = node.get("status", 0)
            # status=1 (RUNNING)이면 아직 실행 중
            if node_status == 1:
                return (False, idx)  # 현재 실행 중인 인덱스 반환
        
        # 범위 내의 모든 노드가 status=0 (IDLE) 또는 status=2 (SUCCESS) 또는 status=3 (FAILURE)
        return (True, None)
    
    def _find_last_nominal_gain_group(self, node_id: int, all_nodes: list) -> Optional[dict]:
        """
        노드의 자식 노드 범위에서 마지막 nominal_gain_group 노드 찾기
        
        Args:
            node_id: 확인할 노드의 ID
            all_nodes: 모든 노드 리스트
            
        Returns:
            마지막 nominal_gain_group 노드 또는 None
        """
        # 노드를 ID 순서대로 정렬
        sorted_nodes = sorted(all_nodes, key=lambda n: n.get("id", 0))
        
        # 자식 노드 범위 찾기
        start_idx, end_idx = self._find_children_range(node_id, sorted_nodes)
        
        if start_idx is None or end_idx is None:
            return None
        
        # 범위 내에서 nominal_gain_group 찾기 (역순으로 검색하여 마지막 것 찾기)
        last_nominal_gain = None
        for idx in range(end_idx - 1, start_idx - 1, -1):
            if idx < 0 or idx >= len(sorted_nodes):
                continue
            node = sorted_nodes[idx]
            node_type = node.get("type", "")
            node_name = node.get("name", "")
            
            # nominal_gain_group 찾기
            if "nominal_gain_group" in node_type.lower() or "nominal_gain_group" in node_name.lower():
                last_nominal_gain = node
                break
        
        return last_nominal_gain
    
    def _routine_status_callback(self, msg: String):
        """루틴 상태 피드백 콜백 - /debug/routine 토픽에서 실제 실행 중인 루틴 추적"""
        try:
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
                else:
                    self.routine_reset_complete_flag = False
            else:
                old_nodes_count = self.routine_nodes_count
                self.routine_nodes_count = 0
                if not self.routine_reset_complete_flag:
                    self.get_logger().info(f"[ROUTINE STATUS] RESET 완료 확인: nodes={old_nodes_count} -> 0 (플래그=True로 설정)")
                self.routine_reset_complete_flag = True
            
            # RESET 상태 확인: nodes가 비어있으면 RESET 상태
            is_reset_state = (not nodes or len(nodes) == 0)
            
            if is_reset_state:
                self.actual_running_routine = None
                # Handshake/Hello 완료 확인 (RESET 상태: nodes가 비어있음)
                # 이 경우는 루틴이 완전히 종료된 상태이므로 완료로 간주
                current_time = time.monotonic()
                
                # Handshake 완료 확인
                if (self.routine_controller.handshake_start_time is not None and 
                    self.routine_controller.current_routine == "idling_handshake_rt"):
                    elapsed = current_time - self.routine_controller.handshake_start_time
                    if elapsed >= 2.0:
                        self.get_logger().info(f"[Handshake 완료 확인] RESET 상태 감지 (nodes=0), SEARCHING 상태로 전환")
                        # Tracker에 SEARCHING 상태 전환 요청
                        tracker_command = {
                            'type': 'set_state',
                            'state': 'searching',
                            'target_id': None
                        }
                        tracker_msg = String()
                        tracker_msg.data = json.dumps(tracker_command)
                        self.tracker_control_publisher.publish(tracker_msg)
                        # 초기화
                        self.routine_controller.handshake_start_time = None
                        self.routine_controller.handshake_complete_check_started = False
                        self.routine_controller.current_routine = None
                
                # Hello 완료 확인
                if (self.routine_controller.hello_start_time is not None and 
                    self.routine_controller.current_routine == "idling_heart_rt"):
                    elapsed = current_time - self.routine_controller.hello_start_time
                    if elapsed >= 2.0:
                        self.get_logger().info(f"[Hello 완료 확인] RESET 상태 감지 (nodes=0), SEARCHING 상태로 전환")
                        # Tracker에 SEARCHING 상태 전환 요청
                        tracker_command = {
                            'type': 'set_state',
                            'state': 'searching',
                            'target_id': None
                        }
                        tracker_msg = String()
                        tracker_msg.data = json.dumps(tracker_command)
                        self.tracker_control_publisher.publish(tracker_msg)
                        # 초기화
                        self.routine_controller.hello_start_time = None
                        self.routine_controller.hello_complete_check_started = False
                        self.routine_controller.current_routine = None
                
                return
            
            # 루트 노드 찾기 (parent == -1, type이 IdleInteractionSelector)
            root_node = None
            root_node_id = None
            for node in nodes:
                if node.get("parent") == -1:
                    root_node_type = node.get("type", "")
                    # type이 IdleInteractionSelector인지 확인
                    if "IdleInteractionSelector" in root_node_type or "idleinteractionselector" in root_node_type.lower():
                        root_node = node
                        root_node_id = node.get("id")
                        break
            
            # 실제 루틴 이름 찾기: IdleInteractionSelector의 자식 노드에서 status와 type으로 찾기
            actual_routine_name = None
            handshake_node = None  # Handshake 루틴 노드
            heart_node = None  # Heart 루틴 노드
            
            # 루트 노드의 자식 노드에서 실제 루틴 찾기 (status와 type으로 판단)
            if root_node_id is not None:
                # 먼저 모든 자식 노드를 수집
                child_nodes = []
                for node in nodes:
                    if node.get("parent") == root_node_id:
                        child_nodes.append(node)
                
                # 1단계: Breathing 루틴 찾기 (LoopWhile 타입 - 명확함)
                for node in child_nodes:
                    node_type = node.get("type", "")
                    node_status = node.get("status", 0)
                    if node_type == "LoopWhile":
                        if node_status == 1:  # RUNNING
                            actual_routine_name = "idle_breathing_rt"
                        break
                
                # 2단계: Handshake 루틴 찾기 (하위 구조 확인)
                # status와 관계없이 handshake_node를 찾아야 함 (완료 확인을 위해)
                for node in child_nodes:
                    node_type = node.get("type", "")
                    node_status = node.get("status", 0)
                    node_name = node.get("name", "").lower()
                    node_id = node.get("id")
                    
                    # 타입이나 이름으로 명확히 handshake인 경우
                    if ("handshake" in node_type.lower() or "torque" in node_type.lower() or 
                        "handshake" in node_name):
                        handshake_node = node
                        self.get_logger().debug(f"[루틴 찾기] handshake_node 찾음 (이름/타입): id={node_id}, status={node_status}")
                        if node_status == 1:  # RUNNING
                            actual_routine_name = "idling_handshake_rt"
                        break
                    
                    # Sequence 타입인 경우, 하위 구조 확인
                    if node_type == "Sequence":
                        has_handshake_marker = self._check_handshake_markers(node_id, nodes)
                        self.get_logger().debug(f"[루틴 찾기] Sequence(id={node_id}, status={node_status}) handshake 마커 확인: {has_handshake_marker}")
                        if has_handshake_marker:
                            handshake_node = node
                            # status와 관계없이 handshake_node로 설정 (완료 확인을 위해)
                            if node_status == 1:  # RUNNING
                                actual_routine_name = "idling_handshake_rt"
                            
                            # 상태가 바뀔 때만 로그 출력
                            if (self.last_handshake_node_id != node_id or 
                                self.last_handshake_node_status != node_status):
                                if node_status == 1:  # RUNNING
                                    self.get_logger().info(f"[루틴 찾기] handshake_node 찾음 (Sequence, RUNNING): id={node_id}, status={node_status}")
                                elif node_status == 2:  # SUCCESS
                                    self.get_logger().info(f"[루틴 찾기] handshake_node 찾음 (Sequence, SUCCESS): id={node_id}, status={node_status}")
                                else:
                                    self.get_logger().debug(f"[루틴 찾기] handshake_node 찾음 (Sequence, 기타): id={node_id}, status={node_status}")
                                self.last_handshake_node_id = node_id
                                self.last_handshake_node_status = node_status
                            break
                
                # 3단계: Heart 루틴 찾기 (나머지 Sequence 노드)
                # Handshake가 실행 중이면 heart_node 찾기 건너뛰기
                if actual_routine_name != "idling_handshake_rt":
                    # Handshake가 실행 중이 아니면 heart_node 찾기
                    for node in child_nodes:
                        node_type = node.get("type", "")
                        node_status = node.get("status", 0)
                        node_name = node.get("name", "").lower()
                        node_id = node.get("id")
                        
                        # 타입이나 이름으로 명확히 heart인 경우
                        if "heart" in node_type.lower() or "heart" in node_name:
                            heart_node = node
                            # RUNNING 상태일 때만 info 로그, 그 외는 debug
                            if node_status == 1:  # RUNNING
                                self.get_logger().info(f"[루틴 찾기] heart_node 찾음 (이름/타입): id={node_id}, status={node_status}")
                                actual_routine_name = "idling_heart_rt"
                            else:
                                self.get_logger().debug(f"[루틴 찾기] heart_node 찾음 (이름/타입, 비활성): id={node_id}, status={node_status}")
                            break
                        
                        # Sequence 타입이고 handshake 마커가 없는 경우 heart로 간주
                        if node_type == "Sequence":
                            # 이미 handshake_node로 확인된 경우 제외
                            if handshake_node is not None and handshake_node.get("id") == node_id:
                                self.get_logger().debug(f"[루틴 찾기] Sequence(id={node_id})는 이미 handshake_node로 확인됨")
                                continue
                            # handshake 마커가 없는 Sequence는 heart_routine_
                            has_handshake_marker = self._check_handshake_markers(node_id, nodes)
                            self.get_logger().debug(f"[루틴 찾기] Sequence(id={node_id}, status={node_status}) handshake 마커 확인: {has_handshake_marker}")
                            if not has_handshake_marker:
                                heart_node = node
                                if node_status == 1:  # RUNNING
                                    actual_routine_name = "idling_heart_rt"
                                
                                # 상태가 바뀔 때만 로그 출력
                                if (self.last_heart_node_id != node_id or 
                                    self.last_heart_node_status != node_status):
                                    if node_status == 1:  # RUNNING
                                        self.get_logger().info(f"[루틴 찾기] heart_node 찾음: id={node_id}, status={node_status}, type={node_type}")
                                    else:
                                        self.get_logger().debug(f"[루틴 찾기] heart_node 찾음 (비활성): id={node_id}, status={node_status}, type={node_type}")
                                    self.last_heart_node_id = node_id
                                    self.last_heart_node_status = node_status
                                break
                            else:
                                self.get_logger().debug(f"[루틴 찾기] Sequence(id={node_id})는 handshake 마커가 있어서 제외됨")
            
            # 단일 루틴 RESET 완료 확인: RESET 중인 루틴이 실행 중이 아니면 RESET 완료
            if self.routine_controller.waiting_for_reset and self.routine_controller.resetting_routine_name:
                # 단일 루틴 RESET의 경우: 해당 루틴이 실행 중이 아니면 RESET 완료
                resetting_routine = self.routine_controller.resetting_routine_name
                is_resetting_routine_running = False
                
                # 루트 노드의 자식 노드에서 status로 확인
                if root_node_id is not None:
                    for node in nodes:
                        if node.get("parent") == root_node_id:
                            node_status = node.get("status", 0)
                            node_type = node.get("type", "").lower()
                            
                            # status가 1(RUNNING)이고, type이 해당 루틴과 일치하면 실행 중
                            if node_status == 1:  # RUNNING
                                if resetting_routine == "idling_heart_rt" and "heart" in node_type:
                                    is_resetting_routine_running = True
                                    break
                                elif resetting_routine == "idling_handshake_rt" and "handshake" in node_type:
                                    is_resetting_routine_running = True
                                    break
                                elif resetting_routine == "idle_breathing_rt" and "breathing" in node_type:
                                    is_resetting_routine_running = True
                                    break
                
                if not is_resetting_routine_running and not self.routine_reset_complete_flag:
                    self.get_logger().info(f"[ROUTINE STATUS] 단일 루틴 RESET 완료 확인: {resetting_routine} 루틴이 실행 중이 아님 (플래그=True로 설정)")
                    self.routine_reset_complete_flag = True
                    self.routine_controller.resetting_routine_name = None
            
            if root_node:
                root_status = root_node.get("status", 0)
                root_name = root_node.get("name", "")
                
                # status: 0=IDLE, 1=RUNNING, 2=SUCCESS, 3=FAILURE
                # RESET 상태: 루트 노드가 IDLE 상태 (nodes는 이미 위에서 확인함)
                is_reset_state = (root_status == 0)
                
                if root_status == 1:  # RUNNING인 경우
                    # 실제 루틴 이름이 있으면 사용 (자식 노드의 status=1인 경우)
                    if actual_routine_name:
                        self.actual_running_routine = actual_routine_name
                    else:
                        # 자식 노드가 RUNNING 상태가 아니면 루틴이 실행 중이 아님
                        self.actual_running_routine = None
                    
                    # 현재 상태 확인 (Tracking State에서는 Handshake/Hello 완료 확인 안 함)
                    current_state_str = None
                    try:
                        state_msg = self.tracker_state_subscription.msg if hasattr(self.tracker_state_subscription, 'msg') else None
                        if state_msg is None:
                            # 직접 상태 요청 (간접 확인)
                            current_state_str = None
                    except:
                        current_state_str = None
                    
                    # Handshake/Hello 완료 확인 (2초 후부터 확인, Status 기반)
                    # Tracking State가 아니고, handshake_start_time이 설정되어 있을 때만 확인
                    # Tracking State에서는 handshake_start_time이 None이어야 함 (_handle_state_change에서 초기화)
                    current_time = time.monotonic()
                    
                    # Handshake 완료 확인 (하위 노드의 status 기반)
                    # handshake_start_time이 설정되어 있고, 실제로 Handshake 루틴이 실행 중일 때만 확인
                    if (self.routine_controller.handshake_start_time is not None and 
                        actual_routine_name == "idling_handshake_rt"):
                        elapsed = current_time - self.routine_controller.handshake_start_time
                        
                        # 2초 후부터 확인 시작 (타임아웃 완전 제거 - 무한 대기)
                        if elapsed >= 2.0:
                            # handshake_node가 있고 status=2 (SUCCESS)이면 완료로 간주
                            if handshake_node is not None:
                                handshake_status = handshake_node.get("status", 0)
                                handshake_node_id = handshake_node.get("id")
                                
                                if not self.routine_controller.handshake_complete_check_started:
                                    self.routine_controller.handshake_complete_check_started = True
                                    self.get_logger().info(f"[Handshake 완료 확인 시작] 경과 시간: {elapsed:.2f}초, handshake_node: id={handshake_node_id}, status={handshake_status}")
                                
                                # id나 status가 바뀔 때만 로그 출력
                                should_log = (self.last_handshake_node_id != handshake_node_id or 
                                            self.last_handshake_node_status != handshake_status)
                                
                                if should_log:
                                    self.get_logger().info(f"[Handshake 완료 확인] handshake_node 찾음: id={handshake_node_id}, status={handshake_status}, elapsed={elapsed:.2f}초")
                                    self.last_handshake_node_id = handshake_node_id
                                    self.last_handshake_node_status = handshake_status
                                
                                # reset()이 정상 작동하므로 부모 노드의 status만 확인
                                # handshake_node의 status가 2 (SUCCESS)이면 완료로 간주
                                is_completed = False
                                if handshake_status == 2:  # SUCCESS
                                    is_completed = True
                                    if should_log:
                                        self.get_logger().info(f"[Handshake 완료 확인] handshake_node status=2 (SUCCESS) 감지 - 부모 노드 완료 확인")
                                else:
                                    # RUNNING 상태 - 대기 중
                                    if should_log:
                                        self.get_logger().debug(f"[Handshake 완료 확인] handshake_node(id={handshake_node_id}) status={handshake_status} (RUNNING) - 대기 중")
                                
                                if is_completed:
                                    self.get_logger().info(f"[Handshake 완료 확인] 완료 감지, Reset 명령 발송 후 SEARCHING 상태로 전환")
                                    # Handshake 완료 시 Reset 명령 발송
                                    self.routine_controller.reset_current_routine_external()
                                    # Tracker에 SEARCHING 상태 전환 요청
                                    tracker_command = {
                                        'type': 'set_state',
                                        'state': 'searching',
                                        'target_id': None
                                    }
                                    tracker_msg = String()
                                    tracker_msg.data = json.dumps(tracker_command)
                                    self.tracker_control_publisher.publish(tracker_msg)
                                    # 초기화
                                    self.routine_controller.handshake_start_time = None
                                    self.routine_controller.handshake_complete_check_started = False
                                    self.routine_controller.current_routine = None
                            else:
                                # handshake_node를 찾지 못한 경우
                                # 모든 Sequence 노드를 다시 확인하여 완료된 handshake 노드 찾기
                                if should_log:
                                    self.get_logger().debug(f"[Handshake 완료 확인] handshake_node를 찾지 못함, elapsed={elapsed:.2f}초, current_routine={self.routine_controller.current_routine}")
                                if root_node_id is not None:
                                    for node in nodes:
                                        if node.get("parent") == root_node_id:
                                            node_type = node.get("type", "")
                                            node_status = node.get("status", 0)
                                            node_id = node.get("id")
                                            if node_type == "Sequence":
                                                # handshake 마커 확인
                                                if self._check_handshake_markers(node_id, nodes):
                                                    # reset()이 정상 작동하므로 부모 노드의 status만 확인
                                                    is_completed = False
                                                    if node_status == 2:  # SUCCESS
                                                        is_completed = True
                                                        self.get_logger().info(f"[Handshake 완료 확인] 재검색으로 handshake_node 찾음: id={node_id}, status={node_status} (SUCCESS) - 부모 노드 완료 확인")
                                                    
                                                    if is_completed:
                                                        # Handshake 완료 시 Reset 명령 발송
                                                        self.routine_controller.reset_current_routine_external()
                                                        # Tracker에 SEARCHING 상태 전환 요청
                                                        tracker_command = {
                                                            'type': 'set_state',
                                                            'state': 'searching',
                                                            'target_id': None
                                                        }
                                                        tracker_msg = String()
                                                        tracker_msg.data = json.dumps(tracker_command)
                                                        self.tracker_control_publisher.publish(tracker_msg)
                                                        # 초기화
                                                        self.routine_controller.handshake_start_time = None
                                                        self.routine_controller.handshake_complete_check_started = False
                                                        self.routine_controller.current_routine = None
                                                        break
                    
                    # Hello 완료 확인 (하위 노드의 status 기반)
                    # hello_start_time이 설정되어 있고, 실제로 Hello 루틴이 실행 중일 때만 확인
                    if (self.routine_controller.hello_start_time is not None and 
                        actual_routine_name == "idling_heart_rt"):
                        elapsed = current_time - self.routine_controller.hello_start_time
                        
                        # heart_node가 있고 status=2 (SUCCESS)이면 완료로 간주
                        if heart_node is not None:
                            heart_node_id = heart_node.get("id")
                            heart_node_status = heart_node.get("status", 0)
                            
                            # 2초 후부터 확인 시작
                            if elapsed >= 2.0:
                                if not self.routine_controller.hello_complete_check_started:
                                    self.routine_controller.hello_complete_check_started = True
                                    self.get_logger().info(f"[Hello 완료 확인 시작] 경과 시간: {elapsed:.2f}초, heart_node: id={heart_node_id}, status={heart_node_status}")
                                
                                # id나 status가 바뀔 때만 로그 출력
                                should_log_hello = (self.last_heart_node_id != heart_node_id or 
                                                   self.last_heart_node_status != heart_node_status)
                                
                                if should_log_hello:
                                    self.get_logger().info(f"[Hello 완료 확인] heart_node 찾음: id={heart_node_id}, status={heart_node_status}, elapsed={elapsed:.2f}초")
                                    self.last_heart_node_id = heart_node_id
                                    self.last_heart_node_status = heart_node_status
                                
                                # reset()이 정상 작동하므로 부모 노드의 status만 확인
                                # heart_node의 status가 2 (SUCCESS)이면 완료로 간주
                                is_completed = False
                                if heart_node_status == 2:  # SUCCESS
                                    is_completed = True
                                    if should_log_hello:
                                        self.get_logger().info(f"[Hello 완료 확인] heart_node status=2 (SUCCESS) 감지 - 부모 노드 완료 확인")
                                else:
                                    # RUNNING 상태 - 대기 중
                                    if should_log_hello:
                                        self.get_logger().debug(f"[Hello 완료 확인] heart_node(id={heart_node_id}) status={heart_node_status} (RUNNING) - 대기 중")
                                
                                if is_completed:
                                    self.get_logger().info(f"[Hello 완료 확인] 완료 감지, Reset 명령 발송 후 SEARCHING 상태로 전환")
                                    # Hello 완료 시 Reset 명령 발송
                                    self.routine_controller.reset_current_routine_external()
                                    # Tracker에 SEARCHING 상태 전환 요청
                                    tracker_command = {
                                        'type': 'set_state',
                                        'state': 'searching',
                                        'target_id': None
                                    }
                                    tracker_msg = String()
                                    tracker_msg.data = json.dumps(tracker_command)
                                    self.tracker_control_publisher.publish(tracker_msg)
                                    # 초기화
                                    self.routine_controller.hello_start_time = None
                                    self.routine_controller.hello_complete_check_started = False
                                    self.routine_controller.current_routine = None
                            else:
                                # heart_node를 찾지 못한 경우
                                # 모든 Sequence 노드를 다시 확인하여 완료된 heart 노드 찾기
                                if elapsed >= 2.0:
                                    should_log_hello = (self.last_heart_node_id is not None)
                                    if should_log_hello:
                                        self.get_logger().debug(f"[Hello 완료 확인] heart_node를 찾지 못함, elapsed={elapsed:.2f}초, current_routine={self.routine_controller.current_routine}")
                                    
                                    # 루트 노드의 자식 노드에서 heart 노드 재검색
                                    if root_node_id is not None:
                                        for node in nodes:
                                            if node.get("parent") == root_node_id:
                                                node_type = node.get("type", "")
                                                node_status = node.get("status", 0)
                                                node_id = node.get("id")
                                                node_name = node.get("name", "").lower()
                                                
                                                # heart 루틴 마커 확인 (heart, hello 등이 이름에 포함)
                                                if node_type == "Sequence" and ("heart" in node_name or "hello" in node_name):
                                                    # reset()이 정상 작동하므로 부모 노드의 status만 확인
                                                    is_completed = False
                                                    if node_status == 2:  # SUCCESS
                                                        is_completed = True
                                                        self.get_logger().info(f"[Hello 완료 확인] 재검색으로 heart_node 찾음: id={node_id}, status={node_status} (SUCCESS) - 부모 노드 완료 확인")
                                                    
                                                    if is_completed:
                                                        # Hello 완료 시 Reset 명령 발송
                                                        self.routine_controller.reset_current_routine_external()
                                                        # Tracker에 SEARCHING 상태 전환 요청
                                                        tracker_command = {
                                                            'type': 'set_state',
                                                            'state': 'searching',
                                                            'target_id': None
                                                        }
                                                        tracker_msg = String()
                                                        tracker_msg.data = json.dumps(tracker_command)
                                                        self.tracker_control_publisher.publish(tracker_msg)
                                                        # 초기화
                                                        self.routine_controller.hello_start_time = None
                                                        self.routine_controller.hello_complete_check_started = False
                                                        self.routine_controller.current_routine = None
                                                        break
                elif root_status == 2:  # SUCCESS: 루트 노드가 SUCCESS 상태
                    self.actual_running_routine = None
                    # Handshake/Hello 완료 확인 (루트 노드 SUCCESS 또는 하위 노드 status 확인)
                    current_time = time.monotonic()
                    
                    # Handshake 완료 확인 (하위 노드의 status 확인)
                    if self.routine_controller.handshake_start_time is not None:
                        elapsed = current_time - self.routine_controller.handshake_start_time
                        
                        if elapsed >= 2.0:
                            # reset()이 정상 작동하므로 부모 노드의 status만 확인
                            if handshake_node is not None:
                                handshake_status = handshake_node.get("status", 0)
                                handshake_node_id = handshake_node.get("id")
                                
                                is_completed = False
                                if handshake_status == 2:  # SUCCESS
                                    is_completed = True
                                    self.get_logger().info(f"[Handshake 완료 확인] handshake_node status={handshake_status} (SUCCESS) - 부모 노드 완료 확인")
                                
                                if is_completed:
                                    self.get_logger().info(f"[Handshake 완료 확인] 완료 감지, Reset 명령 발송 후 SEARCHING 상태로 전환")
                                    # Handshake 완료 시 Reset 명령 발송
                                    self.routine_controller.reset_current_routine_external()
                                    # Tracker에 SEARCHING 상태 전환 요청
                                    tracker_command = {
                                        'type': 'set_state',
                                        'state': 'searching',
                                        'target_id': None
                                    }
                                    tracker_msg = String()
                                    tracker_msg.data = json.dumps(tracker_command)
                                    self.tracker_control_publisher.publish(tracker_msg)
                                    # 초기화
                                    self.routine_controller.handshake_start_time = None
                                    self.routine_controller.handshake_complete_check_started = False
                                    self.routine_controller.current_routine = None
                            else:
                                # 루트 노드 SUCCESS로 완료 판단
                                self.get_logger().info(f"[Handshake 완료 확인] 루트 노드 SUCCESS 상태, SEARCHING 상태로 전환")
                                # Handshake 완료 시 Reset 명령 발송
                                self.routine_controller.reset_current_routine_external()
                                tracker_command = {
                                    'type': 'set_state',
                                    'state': 'searching',
                                    'target_id': None
                                }
                                tracker_msg = String()
                                tracker_msg.data = json.dumps(tracker_command)
                                self.tracker_control_publisher.publish(tracker_msg)
                                # 초기화
                                self.routine_controller.handshake_start_time = None
                                self.routine_controller.handshake_complete_check_started = False
                                self.routine_controller.current_routine = None
                    
                    # Hello 완료 확인 (하위 노드의 status 확인)
                    if self.routine_controller.hello_start_time is not None:
                        elapsed = current_time - self.routine_controller.hello_start_time
                        
                        if elapsed >= 2.0:
                            # reset()이 정상 작동하므로 부모 노드의 status만 확인
                            if heart_node is not None:
                                heart_status = heart_node.get("status", 0)
                                heart_node_id = heart_node.get("id")
                                
                                is_completed = False
                                if heart_status == 2:  # SUCCESS
                                    is_completed = True
                                    self.get_logger().info(f"[Hello 완료 확인] heart_node status={heart_status} (SUCCESS) - 부모 노드 완료 확인")
                                
                                if is_completed:
                                    self.get_logger().info(f"[Hello 완료 확인] 완료 감지, Reset 명령 발송 후 SEARCHING 상태로 전환")
                                    # Hello 완료 시 Reset 명령 발송
                                    self.routine_controller.reset_current_routine_external()
                                    # Tracker에 SEARCHING 상태 전환 요청
                                    tracker_command = {
                                        'type': 'set_state',
                                        'state': 'searching',
                                        'target_id': None
                                    }
                                    tracker_msg = String()
                                    tracker_msg.data = json.dumps(tracker_command)
                                    self.tracker_control_publisher.publish(tracker_msg)
                                    # 초기화
                                    self.routine_controller.hello_start_time = None
                                    self.routine_controller.hello_complete_check_started = False
                                    self.routine_controller.current_routine = None
                            else:
                                # 루트 노드 SUCCESS로 완료 판단
                                self.get_logger().info(f"[Hello 완료 확인] 루트 노드 SUCCESS 상태, SEARCHING 상태로 전환")
                                # Hello 완료 시 Reset 명령 발송
                                self.routine_controller.reset_current_routine_external()
                                tracker_command = {
                                    'type': 'set_state',
                                    'state': 'searching',
                                    'target_id': None
                                }
                                tracker_msg = String()
                                tracker_msg.data = json.dumps(tracker_command)
                                self.tracker_control_publisher.publish(tracker_msg)
                                # 초기화
                                self.routine_controller.hello_start_time = None
                                self.routine_controller.hello_complete_check_started = False
                                self.routine_controller.current_routine = None
                elif is_reset_state:  # RESET 상태: nodes가 비어있거나 루트 노드가 IDLE
                    self.actual_running_routine = None
                    # Handshake/Hello 완료 확인 (RESET 상태)
                    current_time = time.monotonic()
                    
                    # Handshake 완료 확인
                    if (self.routine_controller.handshake_start_time is not None and 
                        self.routine_controller.current_routine == "idling_handshake_rt"):
                        elapsed = current_time - self.routine_controller.handshake_start_time
                        if elapsed >= 2.0:
                            self.get_logger().info(f"[Handshake 완료 확인] RESET 상태 감지 (nodes={len(nodes)}, root_status={root_status}), SEARCHING 상태로 전환")
                            # Tracker에 SEARCHING 상태 전환 요청
                            tracker_command = {
                                'type': 'set_state',
                                'state': 'searching',
                                'target_id': None
                            }
                            tracker_msg = String()
                            tracker_msg.data = json.dumps(tracker_command)
                            self.tracker_control_publisher.publish(tracker_msg)
                            # 초기화
                            self.routine_controller.handshake_start_time = None
                            self.routine_controller.handshake_complete_check_started = False
                            self.routine_controller.current_routine = None
                    
                    # Hello 완료 확인
                    if (self.routine_controller.hello_start_time is not None and 
                        self.routine_controller.current_routine == "idling_heart_rt"):
                        elapsed = current_time - self.routine_controller.hello_start_time
                        if elapsed >= 2.0:
                            self.get_logger().info(f"[Hello 완료 확인] RESET 상태 감지 (nodes={len(nodes)}, root_status={root_status}), SEARCHING 상태로 전환")
                            # Tracker에 SEARCHING 상태 전환 요청
                            tracker_command = {
                                'type': 'set_state',
                                'state': 'searching',
                                'target_id': None
                            }
                            tracker_msg = String()
                            tracker_msg.data = json.dumps(tracker_command)
                            self.tracker_control_publisher.publish(tracker_msg)
                            # 초기화
                            self.routine_controller.hello_start_time = None
                            self.routine_controller.hello_complete_check_started = False
                            self.routine_controller.current_routine = None
                else:
                    # 루트 노드가 RUNNING/SUCCESS/RESET이 아니면 실행 중이 아님
                    self.actual_running_routine = None
            else:
                # 루트 노드를 찾을 수 없으면 루틴이 없는 것으로 간주
                self.actual_running_routine = None
            
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
        # Tracking State로 전환 시 handshake/hello_start_time 초기화
        if new_state == TrackingState.TRACKING:
            self.routine_controller.handshake_start_time = None
            self.routine_controller.hello_start_time = None
            self.routine_controller.handshake_complete_check_started = False
            self.routine_controller.hello_complete_check_started = False
        
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
                
                self.is_running = True
                manual_mode = command.get('manual', False)
                tracker_command['manual'] = manual_mode
                # IDLE 상태로 시작하므로 idle_breathing_rt 시작
                if not self.routine_controller.breathing_routine_running:
                    self.routine_controller.start_breathing()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드")
            
            elif cmd_type == 'stop':
                self.is_running = False
                # STOP 처리: 현재 루틴 Reset(100)만 수행
                self.routine_controller.stop_routine()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info("RUN 중지: 루틴 Reset 완료")
            
            elif cmd_type == 'set_mode':
                if self.is_running:
                    manual_mode = command.get('manual', False)
                    tracker_command['manual'] = manual_mode
                    self.get_logger().info(f"Manual 모드 설정: {manual_mode}")
            
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
            
            # Tracker에 명령 전송
            tracker_msg = String()
            tracker_msg.data = json.dumps(tracker_command)
            self.tracker_control_publisher.publish(tracker_msg)
            
            # Controller에 명령 전송
            if cmd_type in ['run', 'stop', 'set_parameters']:
                controller_msg = String()
                controller_msg.data = json.dumps(controller_command)
                self.controller_control_publisher.publish(controller_msg)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Manual 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"Manual 제어 처리 실패: {e}")
    
    def destroy_node(self):
        """노드 파괴 시 루틴 정리 (launch로 실행될 때도 작동)"""
        self.get_logger().info("[SHUTDOWN] 노드 파괴 시작, 루틴 정리 수행...")
        try:
            # 루틴 정리
            self.routine_controller.cleanup_on_shutdown()
        except Exception as e:
            self.get_logger().error(f"[SHUTDOWN] 루틴 정리 중 오류: {e}")
        finally:
            # 부모 클래스의 destroy_node 호출
            super().destroy_node()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = AllexIdleInteractionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # destroy_node에서 cleanup이 호출되므로 여기서는 destroy_node만 호출
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
