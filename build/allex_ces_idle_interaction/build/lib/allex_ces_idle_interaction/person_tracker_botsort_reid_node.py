#!/usr/bin/env python3
"""
YOLO + BoT-SORT(+ReID) 추적 노드 - 독립적인 ROS2 노드
이미지를 받아서 YOLO 추적을 수행하고 결과를 발행
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

from .person_tracker_botsort_reid import TrackerManager, TrackingState, TargetInfo

cv2.setNumThreads(0)  # OpenCV의 멀티스레딩 비활성화


def check_gpu_status():
    """GPU 상태 확인 및 출력"""
    print("=" * 60)
    print("Person Tracker Node - GPU 상태 확인")
    print("=" * 60)
    
    print(f"PyTorch 버전: {torch.__version__}")
    print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 버전: {torch.version.cuda}")
        print(f"GPU 개수: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
    
    print("=" * 60)


class PersonTrackerNode(Node):
    """YOLO + BoT-SORT(+ReID) 추적 노드"""
    
    def __init__(self):
        super().__init__('person_tracker_botsort_reid_node')
        
        # GPU 상태 확인
        check_gpu_status()
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 입력 이미지 구독
        self.image_subscription = self.create_subscription(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            self.image_callback,
            qos_profile,
        )
        
        # 제어 명령 구독 (GUI에서 오는 명령)
        self.control_subscription = self.create_subscription(
            String,
            "/allex_camera/tracker_control",
            self._control_callback,
            10
        )
        
        # 상태 변경 요청 구독 (Controller 노드에서 발행)
        self.state_request_subscription = self.create_subscription(
            String,
            "/allex_camera/tracker_state_request",
            self._state_request_callback,
            10
        )
        
        # 추적 결과 발행
        self.tracking_result_publisher = self.create_publisher(
            String,
            "/allex_camera/tracking_result",
            10
        )
        
        # 목 각도 구독 (Controller 노드에서 발행)
        self.neck_angle_subscription = self.create_subscription(
            String,
            "/allex_camera/neck_angle",
            self.neck_angle_callback,
            10
        )
        
        # TrackerManager 초기화
        self.tracker_manager = TrackerManager()
        self.get_logger().info(f"TrackerManager 초기화 완료: {self.tracker_manager.tracker_type}")
        
        # 목 각도 저장 (안정성 추적용)
        self.current_neck_yaw_rad = None
        
        # 실행 상태 플래그
        self.is_running = False
        
        # 성능 모니터링
        self.frame_count = 0
        self.last_log_time = time.monotonic()
        
        self.get_logger().info("Person Tracker Node 초기화 완료")
        self.get_logger().info("대기 중: RUN 명령을 기다립니다...")
    
    def image_callback(self, msg: CompressedImage) -> None:
        """이미지 콜백 - YOLO 추적 수행"""
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
        
        # 목 각도 안정성 추적 (TRACKING 상태일 때만)
        if target_info.state == TrackingState.TRACKING and self.current_neck_yaw_rad is not None:
            self.tracker_manager.update_neck_angle(self.current_neck_yaw_rad)
        
        # 처리 시간 계산
        process_time = (time.monotonic() - frame_start) * 1000
        
        # 추적 결과 발행
        self._publish_tracking_result(tracked_objects, target_info, process_time_ms=process_time)
        
        # 주기적 성능 로그 (5초마다)
        current_time = time.monotonic()
        if current_time - self.last_log_time > 5.0:
            elapsed = current_time - self.last_log_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            
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
                self.tracker_manager.set_manual_mode(manual_mode)
                self.get_logger().info(f"RUN 시작: {'Manual' if manual_mode else 'Auto'} 모드")
            
            elif cmd_type == 'stop':
                self.is_running = False
                self.tracker_manager.set_state(TrackingState.IDLE, None)
                self.tracker_manager.target_track_id = None
                self.tracker_manager.target_explicitly_set = False
                self.get_logger().info("RUN 중지: IDLE 상태로 전환")
            
            elif cmd_type == 'set_mode':
                if self.is_running:
                    manual_mode = command.get('manual', False)
                    self.tracker_manager.set_manual_mode(manual_mode)
                    self.get_logger().info(f"Manual 모드 설정: {manual_mode}")
            
            elif cmd_type == 'set_state':
                state_str = command.get('state', 'idle')
                target_id = command.get('target_id', None)
                try:
                    state = TrackingState[state_str.upper()]
                    self.tracker_manager.set_state(state, target_id)
                    self.get_logger().info(f"상태 설정: {state_str}, 타겟 ID: {target_id}")
                except (KeyError, AttributeError) as e:
                    self.get_logger().error(f"잘못된 상태: {state_str}")
            
            elif cmd_type == 'set_target':
                target_id = command.get('target_id')
                if target_id is not None:
                    self.tracker_manager.set_target(int(target_id))
                    if self.tracker_manager.interaction_mode:
                        self.tracker_manager.state = TrackingState.INTERACTION
                    else:
                        self.tracker_manager.state = TrackingState.TRACKING
                    self.tracker_manager.lost_frames = 0
                    self.get_logger().info(f"타겟 변경: {self.tracker_manager.target_track_id}")
            
            elif cmd_type == 'set_interaction_mode':
                enabled = command.get('enabled', False)
                self.tracker_manager.set_interaction_mode(enabled)
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
                self.tracker_manager.set_state(state, target_id)
                self.get_logger().info(f"상태 변경 요청 수신: {state_str}")
            except (KeyError, AttributeError) as e:
                self.get_logger().error(f"잘못된 상태: {state_str}")
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"상태 변경 요청 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"상태 변경 요청 처리 실패: {e}")
    
    def neck_angle_callback(self, msg: String):
        """목 각도 콜백 - Controller 노드에서 발행한 목 각도 저장"""
        try:
            data = json.loads(msg.data)
            self.current_neck_yaw_rad = data.get('current_yaw_rad', None)
        except Exception as e:
            self.get_logger().warn(f"목 각도 파싱 실패: {e}")


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = PersonTrackerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

