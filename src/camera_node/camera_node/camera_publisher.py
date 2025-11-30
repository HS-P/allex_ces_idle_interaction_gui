#!/usr/bin/env python3
"""
실시간 사람 추적 카메라 퍼블리셔 - 데이터 발행 전용
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
import torch

from .tracker_manager import TrackerManager, TrackingState, TargetInfo
from .controller_manager import ControllerManager

cv2.setNumThreads(0)  # OpenCV의 멀티스레딩 비활성화


class RoutineController:
    """ROS2 Routine 시스템 제어 클래스"""
    
    def __init__(self, node: Node, robot_name: str = "the_bodyOne"):
        """
        RoutineController 초기화
        
        Args:
            node: ROS2 Node 인스턴스
            robot_name: 로봇 이름 (기본값: "the_bodyOne")
        """
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
        """
        루틴 전환: 기존 루틴 자동 중단 후 새 루틴 시작
        ⭐ 가장 중요한 함수! 새 루틴 START만 보내면 기존 루틴이 자동으로 중단됨
        """
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


def check_gpu_status():
    """GPU 상태 확인 및 출력"""
    print("=" * 60)
    print("GPU 상태 확인")
    print("=" * 60)
    
    # PyTorch CUDA 확인
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"cuDNN 버전: {torch.backends.cudnn.version()}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    메모리: {props.total_memory / 1024**3:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        
        print(f"현재 GPU: {torch.cuda.current_device()}")
        print(f"현재 GPU 이름: {torch.cuda.get_device_name()}")
        
        # GPU 메모리 사용량
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU 메모리 할당됨: {allocated:.1f} MB")
        print(f"GPU 메모리 예약됨: {reserved:.1f} MB")
    else:
        print("⚠️  CUDA를 사용할 수 없습니다! CPU로 실행됩니다.")
        print("    - NVIDIA 드라이버 설치 확인")
        print("    - CUDA Toolkit 설치 확인")
        print("    - PyTorch GPU 버전 설치 확인: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    
    print("=" * 60)


class CameraPublisher(Node):
    """실시간 사람 추적 카메라 퍼블리셔 - 데이터 발행 전용"""
    
    # Input : None
    # Output : None
    """ 카메라 퍼블리셔 초기화 """
    def __init__(self) -> None:
        super().__init__("camera_publisher")
        
        # QoS 설정: BEST_EFFORT + 큐 깊이 30 (30FPS 대응)
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        # 입력 이미지 구독 (카메라 노드에서 발행하는 토픽 - 변경하지 않음)
        self.subscription = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.image_callback,
            qos_profile,
        )
        
        # 상태 및 추적 데이터 Publisher (STATE 포함)
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
        
        # Manual 제어 구독 (GUI에서 오는 명령)
        self.manual_control_subscription = self.create_subscription(
            String,
            "/allex_camera/manual_control",
            self._manual_control_callback,
            10
        )
        
        # GPU 상태 확인
        check_gpu_status()
        
        # TrackerManager 초기화
        self.tracker_manager = TrackerManager()
        self.get_logger().info(f"TrackerManager 초기화 완료: {self.tracker_manager.tracker_type}")
        
        # YOLO 모델 디바이스 확인
        yolo_device = self.tracker_manager.yolo_model.device
        self.get_logger().info(f"YOLO 모델 디바이스: {yolo_device}")
        if str(yolo_device) == 'cpu':
            self.get_logger().warn("⚠️  YOLO 모델이 CPU에서 실행 중입니다! 성능이 저하됩니다.")
        else:
            self.get_logger().info(f"✓ YOLO 모델이 GPU ({yolo_device})에서 실행 중입니다.")
        
        # ControllerManager 초기화 (현재 Node 전달)
        self.controller_manager = ControllerManager(self)
        
        # RoutineController 초기화
        self.routine_controller = RoutineController(self, robot_name="the_bodyOne")
        
        # 이전 상태 추적 (상태 변경 감지용)
        self.previous_state = TrackingState.IDLE
        
        # 실행 상태 플래그 (GUI의 RUN 명령을 받기 전까지는 False)
        self.is_running = False
        
        # 성능 모니터링 변수
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("카메라 이미지 구독 시작")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    # Input : msg: CompressedImage
    # Output : None
    """ 이미지 콜백 - 실시간 처리 및 데이터 발행 """
    def image_callback(self, msg: CompressedImage) -> None:
        # RUN 명령을 받기 전까지는 처리하지 않음
        if not self.is_running:
            return
        
        self.frame_count += 1
        frame_start = time.monotonic()
        
        # 압축된 이미지 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # 추적 처리 (YOLO 감지 + 매칭 알고리즘)
        tracked_objects, target_info = self.tracker_manager.process(frame)
        
        # 상태 변경 감지 및 루틴 전환 처리
        current_state = target_info.state
        if current_state != self.previous_state:
            self._handle_state_change(self.previous_state, current_state)
            self.previous_state = current_state
        
        # 목 제어 명령 생성 및 전송
        self.controller_manager.update(
            target_info, 
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            tracker_manager=self.tracker_manager  # 목 각도 안정성 추적용
        )
        
        # 처리 시간 계산
        current_time = time.monotonic()
        process_time = (current_time - frame_start) * 1000
        
        # 상태 및 추적 데이터 발행
        self._publish_tracking_data(tracked_objects, target_info, process_time_ms=process_time)
        
        # 타겟 BB Box Crop 이미지 발행 (Interaction Mode에서만, 타겟이 있는 경우)
        # IDLE Mode에서는 BB Box Publish 안 함
        if self.tracker_manager.interaction_mode and target_info.track_id is not None:
            self._publish_target_crop(frame, tracked_objects, target_info)
        
        # 주기적 성능 로그 (5초마다)
        
        if current_time - self.last_log_time > 5.0:
            elapsed = current_time - self.last_log_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
            # GPU 메모리 사용량 확인
            gpu_mem_str = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                gpu_mem_str = f" | GPU 메모리: {allocated:.0f}MB"
            
            self.get_logger().info(
                f"추적 중: {len(tracked_objects)}개 객체 | "
                f"처리 시간: {process_time:.1f}ms | FPS: {fps:.1f}{gpu_mem_str}"
            )
            self.frame_count = 0
            self.last_log_time = current_time
    
    # Input : tracked_objects, target_info
    # Output : None
    """ 추적 데이터를 Topic으로 발행 (STATE 포함, BB 처리 이미지 제외) """
    def _publish_tracking_data(self, tracked_objects, target_info, process_time_ms=None):
        try:
            # 현재 목 각도 정보 가져오기
            current_yaw, current_pitch = self.controller_manager.get_current_angles()
            target_yaw, target_pitch = self.controller_manager.get_target_angles()
            
            # 허리 각도 정보 가져오기
            current_waist_yaw, target_waist_yaw = self.controller_manager.get_waist_angles()
            
            # 상태 정보 추출
            state_str = target_info.state.value if isinstance(target_info.state, TrackingState) else str(target_info.state)
            
            # 추적 객체 정보 (BB 제외, 기본 정보만)
            objects_data = []
            for obj in tracked_objects:
                objects_data.append({
                    'track_id': obj.track_id,
                    'centroid': list(obj.centroid),  # 중심점만
                    'state': obj.state,
                    'confidence': obj.confidence,
                    'age': obj.age
                })
            
            # FPS 계산
            current_time = time.monotonic()
            elapsed = current_time - self.last_log_time if self.last_log_time > 0 else 0
            fps = self.frame_count / elapsed if elapsed > 0 else 0.0
            
            # Center Zone 경과 시간 가져오기
            center_zone_elapsed = self.tracker_manager.get_center_zone_elapsed_time()
            
            # JSON 데이터 구성
            data = {
                'state': state_str,
                'target_info': {
                    'track_id': target_info.track_id,
                    'point': list(target_info.point) if target_info.point else None,
                    'state': state_str
                },
                'tracked_objects': objects_data,
                'neck_angles': {
                    'current': {
                        'yaw_rad': float(current_yaw),
                        'pitch_rad': float(current_pitch)
                    },
                    'target': {
                        'yaw_rad': float(target_yaw),
                        'pitch_rad': float(target_pitch)
                    }
                },
                'waist_angles': {
                    'current': {
                        'yaw_rad': float(current_waist_yaw)
                    },
                    'target': {
                        'yaw_rad': float(target_waist_yaw)
                    }
                },
                'performance': {
                    'fps': float(fps),
                    'process_time_ms': float(process_time_ms) if process_time_ms else 0.0
                },
                'center_zone': {
                    'elapsed_time': float(center_zone_elapsed) if center_zone_elapsed is not None else None,
                    'duration': float(self.tracker_manager.center_zone_duration)
                },
                'timestamp': time.monotonic()
            }
            
            # JSON 문자열로 변환하여 발행
            json_str = json.dumps(data, ensure_ascii=False)
            msg = String()
            msg.data = json_str
            self.tracking_data_publisher.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f"추적 데이터 발행 실패: {e}")
    
    def _publish_target_crop(self, frame: np.ndarray, tracked_objects, target_info):
        """타겟 BB Box Crop 이미지 발행"""
        try:
            # 타겟에 해당하는 객체 찾기
            target_obj = None
            for obj in tracked_objects:
                if obj.track_id == target_info.track_id:
                    target_obj = obj
                    break
            
            if target_obj is None:
                return
            
            # BB Box 좌표 추출
            x1, y1, x2, y2 = target_obj.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
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
        """
        상태 변경 시 루틴 전환 처리
        
        규칙:
        - IDLE, TRACKING, LOST, SEARCHING, WAIST_FOLLOWER → idle_breathing_rt (무한 반복)
        - HELLO → hand_wave_rt (1회 실행)
        - INTERACTION → 루틴 없음 (중단)
        """
        # INTERACTION 상태로 전환 시 루틴 중단
        if new_state == TrackingState.INTERACTION:
            if self.routine_controller.current_routine:
                self.routine_controller.stop_current_routine()
                self.get_logger().info("INTERACTION 상태: 루틴 중단")
            return
        
        # INTERACTION에서 다른 상태로 전환 시 루틴 재시작
        if old_state == TrackingState.INTERACTION:
            # 새 상태에 맞는 루틴 시작
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
        # idle_breathing_rt 유지 (이미 실행 중이면 자동으로 유지됨)
        if new_state in (TrackingState.IDLE, TrackingState.TRACKING, TrackingState.LOST, 
                        TrackingState.SEARCHING, TrackingState.WAIST_FOLLOWER):
            # 루틴이 실행 중이 아니면 시작
            if self.routine_controller.current_routine != "idle_breathing_rt":
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.get_logger().info(f"{new_state.value} 상태: idle_breathing_rt 시작")
    
    def _manual_control_callback(self, msg: String):
        """Manual 제어 콜백 - GUI에서 오는 명령 처리"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            if cmd_type == 'run' or cmd_type == 'start':
                # RUN 명령 - 추적 시작
                self.is_running = True
                manual_mode = command.get('manual', False)
                self.tracker_manager.set_manual_mode(manual_mode)
                # IDLE 상태에서 시작하므로 숨쉬기 루틴 시작
                self.routine_controller.switch_routine("idle_breathing_rt")
                self.previous_state = TrackingState.IDLE
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드, idle_breathing_rt 시작")
            
            elif cmd_type == 'stop':
                # STOP 명령 - 추적 중지
                self.is_running = False
                # IDLE 상태로 전환
                self.tracker_manager.set_state(TrackingState.IDLE, None)
                self.tracker_manager.target_track_id = None
                self.tracker_manager.target_explicitly_set = False
                # 루틴 중단
                self.routine_controller.stop_current_routine()
                self.previous_state = TrackingState.IDLE
                self.get_logger().info("RUN 중지: IDLE 상태로 전환, 루틴 중단")
            
            elif cmd_type == 'set_mode':
                # 모드 설정 (manual: True/False) - RUN 중일 때만 적용
                if self.is_running:
                    manual_mode = command.get('manual', False)
                    self.tracker_manager.set_manual_mode(manual_mode)
                    self.get_logger().info(f"Manual 모드 설정: {manual_mode}")
                else:
                    self.get_logger().warn("RUN 상태가 아닙니다. 모드 변경을 무시합니다.")
            
            elif cmd_type == 'set_state':
                # 상태 설정
                state_str = command.get('state', 'idle')
                target_id = command.get('target_id', None)
                
                try:
                    state = TrackingState[state_str.upper()]
                    self.tracker_manager.set_state(state, target_id)
                    self.get_logger().info(f"상태 설정: {state_str}, 타겟 ID: {target_id}")
                except (KeyError, AttributeError) as e:
                    self.get_logger().error(f"잘못된 상태: {state_str}")
            
            elif cmd_type == 'set_target':
                # 타겟 변경 (Auto/Manual 모드 모두에서 작동)
                target_id = command.get('target_id')
                if target_id is not None:
                    old_target_id = self.tracker_manager.target_track_id
                    self.tracker_manager.set_target(int(target_id))
                    # Interaction Mode면 INTERACTION 상태 유지, 아니면 TRACKING
                    if self.tracker_manager.interaction_mode:
                        self.tracker_manager.state = TrackingState.INTERACTION
                    else:
                        self.tracker_manager.state = TrackingState.TRACKING
                    self.tracker_manager.lost_frames = 0
                    self.get_logger().info(
                        f"[타겟 변경] {old_target_id} → {self.tracker_manager.target_track_id} "
                        f"({'Manual' if self.tracker_manager.manual_mode else 'Auto'} 모드)"
                    )
                else:
                    self.get_logger().warn("타겟 ID가 제공되지 않았습니다.")
            
            elif cmd_type == 'set_interaction_mode':
                # Interaction Mode 설정 (IDLE Mode와 독립)
                enabled = command.get('enabled', False)
                self.tracker_manager.set_interaction_mode(enabled)
                if enabled:
                    self.get_logger().info("Interaction Mode 활성화: 타겟 자동 선택 및 BB Box 추적")
                else:
                    self.get_logger().info("IDLE Mode 활성화: 목/허리 0도 복귀")
            
            else:
                self.get_logger().warn(f"알 수 없는 명령 타입: {cmd_type}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Manual 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"Manual 제어 처리 실패: {e}")


def main(args=None):
    """메인 함수 - 순수 데이터 발행 모드"""
    # ROS2 초기화
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    
    try:
        # ROS2 spin 실행
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
