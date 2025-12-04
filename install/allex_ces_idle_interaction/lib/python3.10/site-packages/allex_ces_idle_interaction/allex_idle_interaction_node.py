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
from std_msgs.msg import String
import cv2
import numpy as np

from .tracking_fsm_node import TrackingState

cv2.setNumThreads(0)  # OpenCV의 멀티스레딩 비활성화


class RoutineController:
    """ROS2 Routine 시스템 제어 클래스"""
    
    def __init__(self, node: Node, robot_name: str = "the_bodyOne"):
        self.node = node
        self.robot_name = robot_name
        self.current_routine = None
        
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
    
    def start_idle_breathing(self):
        """IDLE 상태: 숨쉬기 루틴 시작 (무한 반복)"""
        command = f"{self.robot_name}::ROUTINE::idle_breathing_rt::START"
        self.current_routine = "idle_breathing_rt"
        self.publish_command(command)
    
    def start_hand_wave(self):
        """손 흔들기 루틴 시작 (1회 실행)"""
        command = f"{self.robot_name}::ROUTINE::hand_wave_rt::START"
        self.current_routine = "hand_wave_rt"
        self.publish_command(command)
    
    def switch_routine(self, new_routine: str):
        """루틴 전환: 기존 루틴 자동 중단 후 새 루틴 시작"""
        if new_routine == "idle_breathing_rt":
            self.start_idle_breathing()
        elif new_routine == "hand_wave_rt":
            self.start_hand_wave()
        else:
            self.node.get_logger().warn(f"알 수 없는 루틴: {new_routine}")
    
    def stop_current_routine(self):
        """현재 실행 중인 루틴 중단"""
        if self.current_routine:
            command = f"{self.robot_name}::ROUTINE::{self.current_routine}::RESET"
            self.publish_command(command)
            self.current_routine = None


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
        
        # RoutineController 초기화
        self.routine_controller = RoutineController(self, robot_name="the_bodyOne")
        
        # 상태 관리
        self.previous_state = TrackingState.IDLE
        self.is_running = False
        self.interaction_mode = False
        
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
            
            # 상태 변경 감지 및 루틴 전환 처리
            state_str = data.get('state', 'idle')
            try:
                current_state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                current_state = TrackingState.IDLE
            
            if current_state != self.previous_state:
                self._handle_state_change(self.previous_state, current_state)
                self.previous_state = current_state
            
            # 타겟 Crop 이미지 발행 (Interaction Mode에서만)
            if self.interaction_mode and self.latest_frame is not None:
                target_info_data = data.get('target_info', {})
                tracked_objects_data = data.get('tracked_objects', [])
                target_track_id = target_info_data.get('track_id')
                
                if target_track_id is not None:
                    self._publish_target_crop(self.latest_frame, tracked_objects_data, target_track_id)
            
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
                'timestamp': time.monotonic()
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
    
    def _handle_state_change(self, old_state: TrackingState, new_state: TrackingState):
        """상태 변경 시 루틴 전환 처리"""
        # INTERACTION 상태로 전환 시 루틴 중단
        if new_state == TrackingState.INTERACTION:
            if self.routine_controller.current_routine:
                self.routine_controller.stop_current_routine()
                self.get_logger().info("INTERACTION 상태: 루틴 중단")
            return
        
        # INTERACTION에서 다른 상태로 전환 시 루틴 재시작
        if old_state == TrackingState.INTERACTION:
            if new_state == TrackingState.HELLO:
                self.routine_controller.switch_routine("hand_wave_rt")
                self.get_logger().info("HELLO 상태: hand_wave_rt 시작")
            elif new_state in (TrackingState.IDLE, TrackingState.TRACKING, TrackingState.LOST, 
                              TrackingState.SEARCHING, TrackingState.WAIST_FOLLOWER):
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.get_logger().info(f"{new_state.value} 상태: idle_breathing_rt 시작")
            return
        
        # HELLO 상태로 전환 시 손 흔들기 루틴 시작
        if new_state == TrackingState.HELLO:
            self.routine_controller.switch_routine("hand_wave_rt")
            self.get_logger().info("HELLO 상태: hand_wave_rt 시작")
            return
        
        # HELLO에서 다른 상태로 전환 시 숨쉬기 루틴으로 전환
        if old_state == TrackingState.HELLO:
            if new_state in (TrackingState.IDLE, TrackingState.TRACKING, TrackingState.LOST, 
                            TrackingState.SEARCHING, TrackingState.WAIST_FOLLOWER):
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.get_logger().info(f"{new_state.value} 상태: idle_breathing_rt 시작")
            return
        
        # IDLE, TRACKING, LOST, SEARCHING, WAIST_FOLLOWER 상태들 간 전환 시
        if new_state in (TrackingState.IDLE, TrackingState.TRACKING, TrackingState.LOST, 
                        TrackingState.SEARCHING, TrackingState.WAIST_FOLLOWER):
            if self.routine_controller.current_routine != "idle_breathing_rt":
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.get_logger().info(f"{new_state.value} 상태: idle_breathing_rt 시작")
    
    def _manual_control_callback(self, msg: String):
        """Manual 제어 콜백 - GUI에서 오는 명령 처리"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            # Tracker와 Controller에 명령 전달
            tracker_command = command.copy()
            controller_command = command.copy()
            
            if cmd_type == 'run' or cmd_type == 'start':
                self.is_running = True
                manual_mode = command.get('manual', False)
                tracker_command['manual'] = manual_mode
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.previous_state = TrackingState.IDLE
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드")
            
            elif cmd_type == 'stop':
                self.is_running = False
                tracker_command['type'] = 'stop'
                controller_command['type'] = 'stop'
                self.routine_controller.stop_current_routine()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info("RUN 중지")
            
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
            
            elif cmd_type == 'set_target':
                target_id = command.get('target_id')
                if target_id is not None:
                    tracker_command['target_id'] = target_id
                    self.get_logger().info(f"타겟 변경: {target_id}")
            
            elif cmd_type == 'set_interaction_mode':
                enabled = command.get('enabled', False)
                self.interaction_mode = enabled
                tracker_command['enabled'] = enabled
                if enabled:
                    self.get_logger().info("Interaction Mode 활성화")
                else:
                    self.get_logger().info("IDLE Mode 활성화")
            
            # Tracker에 명령 전송
            tracker_msg = String()
            tracker_msg.data = json.dumps(tracker_command)
            self.tracker_control_publisher.publish(tracker_msg)
            
            # Controller에 명령 전송 (필요한 경우)
            if cmd_type in ['run', 'stop']:
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
