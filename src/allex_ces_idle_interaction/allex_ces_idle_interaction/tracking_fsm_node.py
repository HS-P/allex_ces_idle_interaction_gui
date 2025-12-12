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
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
import cv2
import numpy as np
from typing import List, Optional, Dict
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
    IDLE = "idle"           # 초기 대상 선택
    TRACKING = "tracking"   # 추적 중
    LOST = "lost"          # 추적 대상 놓침 (잠시 대기)
    SEARCHING = "searching" # 주변 두리번대기 (대상 선택)
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
        self.tracking_result_publisher = self.create_publisher(
            String,
            tracking_result_topic,
            10
        )
        
        # HAND 피드백 구독 (HELLO 상태에서 사용)
        self.hand_feedback_subscription = self.create_subscription(
            Int32MultiArray,
            '/robot_outbound_data/Hand_L_ring_wir/articulation_now',
            self._hand_feedback_callback,
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
        
        # 최신 프레임 저장
        self.latest_frame = None
        self.latest_frame_shape = None
        
        # HELLO 상태를 위한 변수들
        self.hello_routine_sent_time: Optional[float] = None  # HELLO 루틴 발행 시간
        self.hello_feedback_delay = 0.1  # 루틴 발행 후 피드백 확인 대기 시간 (초)
        self.current_hand_state: Optional[int] = None  # 현재 HAND 상태 (4: READY, 5: RUNNING)
        self.hello_routine_sent = False  # HELLO 루틴 발행 여부
        
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
        self.lost_frames = 0
    
    def set_state(self, state: TrackingState, target_track_id: Optional[int] = None) -> None:
        """Manual 모드에서 상태를 수동으로 설정"""
        if state == TrackingState.IDLE:
            self.reset_timers()

        # HELLO 전환 요청은 Auto 모드에서도 허용 (Controller에서 수신)
        if state == TrackingState.HELLO:
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
        if target_exists and self.state not in (TrackingState.INTERACTION, TrackingState.HELLO):
            if self.state not in (TrackingState.TRACKING, TrackingState.INTERACTION, TrackingState.HELLO):
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
                    # HELLO 상태 전환은 gaze_controller_neck_waist_node에서 처리
                    if not target_exists and self.target_track_id is not None and not self.target_explicitly_set:
                        self.state = TrackingState.LOST
                        self.lost_frames = 0
                
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
                
                case TrackingState.HELLO:
                    # HELLO 상태: 루틴 발행 후 HAND 피드백 모니터링
                    current_time_check = time.monotonic()
                    
                    # 루틴이 아직 발행되지 않았으면 발행 (allex_idle_interaction_node에서 처리)
                    # 여기서는 피드백만 모니터링
                    if self.hello_routine_sent and self.hello_routine_sent_time is not None:
                        elapsed_time = current_time_check - self.hello_routine_sent_time
                        
                        # 0.1초 후부터 피드백 확인
                        if elapsed_time >= self.hello_feedback_delay:
                            # HAND 상태가 READY(4)로 바뀌면 SEARCHING으로 전이
                            if self.current_hand_state == 4:  # READY
                                self.state = TrackingState.SEARCHING
                                self.target_track_id = None
                                self.target_explicitly_set = False
                                self.hello_routine_sent = False
                                self.hello_routine_sent_time = None
                                self.current_hand_state = None
                                self.get_logger().info(
                                    f"HELLO 완료: HAND 상태 READY → SEARCHING 상태로 전환"
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
                
                # 상태 변경 요청은 manual_mode와 관계없이 처리 (자동 전환)
                # Controller에서 요청한 상태 전환은 항상 허용
                # 상태 변경 (manual_mode 체크 없이)
                self.state = state
                
                # HELLO 상태로 전환 시 변수 초기화 (매번 리셋)
                if state == TrackingState.HELLO:
                    self.hello_routine_sent_time = time.monotonic()
                    self.hello_routine_sent = True
                    self.current_hand_state = None  # 초기화
                    self.get_logger().info(
                        f"HELLO 상태로 전환: 루틴 발행 시간 기록, "
                        f"{self.hello_feedback_delay}초 후 HAND 피드백 확인 시작"
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
    
    def _hand_feedback_callback(self, msg: Int32MultiArray):
        """HAND 피드백 콜백 - HELLO 상태에서 사용"""
        try:
            if len(msg.data) >= 2:
                # 두 번째 인자가 HAND 상태 (4: READY, 5: RUNNING)
                hand_state = int(msg.data[1])
                self.current_hand_state = hand_state
                
                if self.state == TrackingState.HELLO:
                    self.get_logger().debug(
                        f"HAND 피드백: 상태={hand_state} "
                        f"(4: READY, 5: RUNNING)"
                    )
        except Exception as e:
            self.get_logger().warn(f"HAND 피드백 파싱 실패: {e}")


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
