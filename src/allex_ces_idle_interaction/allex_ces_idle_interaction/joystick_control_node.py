#!/usr/bin/env python3
"""
Joystick Control Node - 8BitDo 컨트롤러 입력 처리 및 제어 명령 발행
j 키와 함께 눌러야 작동하는 안전장치 포함
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pynput import keyboard
import threading
import json
import time
import math
from typing import Set, Optional, List, Dict

from .tracking_fsm_node import TrackingState


class JoystickControlNode(Node):
    """8BitDo 컨트롤러 입력 처리 및 제어 명령 발행"""
    
    def __init__(self):
        super().__init__('joystick_control_node')
        
        # 현재 눌려진 키들
        self.pressed_keys: Set[keyboard.Key] = set()
        self.pressed_key_chars: Set[str] = set()
        
        # j 키가 눌려있는지 확인
        self.j_key_pressed = False
        
        # 키 입력 디바운싱 (중복 입력 방지)
        self.last_key_action_time = {}  # 키별 마지막 동작 시간
        self.key_debounce_time = 0.3  # 0.3초 디바운스
        
        # Manual 제어 명령 Publisher
        self.manual_control_publisher = self.create_publisher(
            String,
            "/allex_camera/manual_control",
            10
        )
        
        # 추적 결과 구독 (현재 상태 및 추적 객체 목록 확인용)
        self.tracking_result_subscription = self.create_subscription(
            String,
            "/allex_camera/tracking_result",
            self._tracking_result_callback,
            10
        )
        
        # 현재 상태 및 추적 객체 정보 저장
        self.current_state = TrackingState.IDLE
        self.current_target_id = None
        self.tracked_objects = []  # [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'centroid': (x,y)}, ...]
        self.tracked_objects_sorted = []  # x 좌표 기준 정렬된 리스트
        
        # 키보드 리스너 스레드 시작
        self.running = True
        self.keyboard_listener = None
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Joystick Control Node 시작")
        self.get_logger().info("모든 명령은 j 키와 함께 눌러야 작동합니다.")
        self.get_logger().info("j+c: Auto Run, j+d: Manual Run, j+f: Stop")
        self.get_logger().info("j+k: 타겟 선택 (왼쪽), j+m: 타겟 선택 (오른쪽)")
        self.get_logger().info("j+i: Handshake State, j+h: Idle State, j+g: Hello State")
        self.get_logger().info("ESC 키를 누르면 종료합니다.")
        self.get_logger().info("=" * 60)
        
        # 키보드 리스너 시작
        self._start_keyboard_listener()
    
    def _start_keyboard_listener(self):
        """키보드 리스너 스레드 시작"""
        def on_press(key):
            """키가 눌렸을 때 호출"""
            try:
                # j 키 확인
                if hasattr(key, 'char') and key.char == 'j':
                    if not self.j_key_pressed:
                        self.j_key_pressed = True
                        self._on_key_state_changed()
                    return
                
                # 일반 키 (char 속성이 있는 경우)
                if hasattr(key, 'char') and key.char is not None:
                    if key.char not in self.pressed_key_chars:
                        self.pressed_key_chars.add(key.char)
                        self._on_key_state_changed()
                else:
                    # 특수 키 (Key 객체)
                    if key not in self.pressed_keys:
                        self.pressed_keys.add(key)
                        self._on_key_state_changed()
            except AttributeError:
                pass
        
        def on_release(key):
            """키가 떼어졌을 때 호출"""
            try:
                # j 키 해제
                if hasattr(key, 'char') and key.char == 'j':
                    if self.j_key_pressed:
                        self.j_key_pressed = False
                    return
                
                # 일반 키
                if hasattr(key, 'char') and key.char is not None:
                    if key.char in self.pressed_key_chars:
                        self.pressed_key_chars.remove(key.char)
                else:
                    # 특수 키
                    if key in self.pressed_keys:
                        self.pressed_keys.remove(key)
                
                # ESC 키로 종료
                if key == keyboard.Key.esc:
                    self.get_logger().info("ESC 키 눌림 - 종료합니다.")
                    self.running = False
                    return False  # 리스너 종료
            except AttributeError:
                pass
        
        # 키보드 리스너 시작 (별도 스레드)
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        self.keyboard_listener.start()
    
    def _on_key_state_changed(self):
        """키 상태가 변경되었을 때 호출 (j 키와 조합 키 확인)"""
        # j 키가 눌려있지 않으면 무시
        if not self.j_key_pressed:
            return
        
        # j 키와 함께 눌린 다른 키 확인
        combined_keys = []
        for char in sorted(self.pressed_key_chars):
            if char != 'j':  # j 키 제외
                combined_keys.append(char)
        
        # 조합 키가 없으면 무시
        if len(combined_keys) == 0:
            return
        
        # 조합 키 처리 (첫 번째 키만 처리, 여러 키가 동시에 눌린 경우)
        key = combined_keys[0]
        self._handle_key_command(key)
    
    def _handle_key_command(self, key: str):
        """키 명령 처리 (디바운싱 포함)"""
        current_time = time.monotonic()
        
        # 디바운싱 체크
        if key in self.last_key_action_time:
            elapsed = current_time - self.last_key_action_time[key]
            if elapsed < self.key_debounce_time:
                return  # 디바운스 시간 내 재입력 무시
        
        self.last_key_action_time[key] = current_time
        
        # 키 명령 처리
        if key == 'c':
            self._handle_auto_run()
        elif key == 'd':
            self._handle_manual_run()
        elif key == 'f':
            self._handle_stop()
        elif key == 'k':
            self._handle_target_left()
        elif key == 'm':
            self._handle_target_right()
        elif key == 'i':
            self._handle_handshake_state()
        elif key == 'h':
            self._handle_idle_state()
        elif key == 'g':
            self._handle_hello_state()
        else:
            self.get_logger().debug(f"알 수 없는 키 명령: j+{key}")
    
    def _tracking_result_callback(self, msg: String):
        """추적 결과 콜백 - 현재 상태 및 추적 객체 목록 업데이트"""
        try:
            data = json.loads(msg.data)
            
            # 현재 상태 업데이트
            state_str = data.get('state', 'idle')
            try:
                self.current_state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                self.current_state = TrackingState.IDLE
            
            # 현재 타겟 ID 업데이트
            self.current_target_id = data.get('target_track_id', None)
            
            # 추적 객체 목록 업데이트
            tracked_objects_data = data.get('tracked_objects', [])
            self.tracked_objects = []
            for obj_data in tracked_objects_data:
                bbox = obj_data.get('bbox', [])
                centroid = obj_data.get('centroid', (0, 0))
                if len(bbox) == 4 and len(centroid) == 2:
                    self.tracked_objects.append({
                        'track_id': obj_data.get('track_id'),
                        'bbox': bbox,
                        'centroid': tuple(centroid)
                    })
            
            # 정렬 로직: 초기(타겟이 없을 때)는 중심 기준 좌우 분리, 타겟이 있으면 x 좌표 순서대로
            if self.current_target_id is None:
                # 초기: 화면 중심 기준으로 좌우 분리 후 거리 순 정렬
                if len(self.tracked_objects) > 0:
                    # 화면 중심 추정 (모든 객체의 centroid 범위를 사용하여 중심 계산)
                    all_x = [obj['centroid'][0] for obj in self.tracked_objects]
                    all_y = [obj['centroid'][1] for obj in self.tracked_objects]
                    center_x = (min(all_x) + max(all_x)) / 2.0 if all_x else 640.0
                    center_y = (min(all_y) + max(all_y)) / 2.0 if all_y else 360.0
                else:
                    center_x, center_y = 640.0, 360.0
                
                # 좌측/우측 분리
                left_objects = []  # 중심선 기준 좌측
                right_objects = []  # 중심선 기준 우측
                
                for obj in self.tracked_objects:
                    cx, cy = obj['centroid']
                    # 중심선 기준 좌측/우측 판단
                    if cx < center_x:
                        # 좌측: 중심선으로부터의 거리 계산 (음수로 표시하여 정렬 시 좌측 우선)
                        distance = -math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        left_objects.append((distance, obj))
                    else:
                        # 우측: 중심선으로부터의 거리 계산
                        distance = math.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        right_objects.append((distance, obj))
                
                # 좌측: 거리 순으로 정렬 (음수이므로 절댓값이 작은 것부터 = 가까운 것부터)
                left_objects.sort(key=lambda x: -x[0])  # 음수이므로 내림차순 정렬
                # 우측: 거리 순으로 정렬 (양수이므로 오름차순 정렬)
                right_objects.sort(key=lambda x: x[0])
                
                # 좌측 + 우측 순서로 합치기 (좌측 먼저, 그 다음 우측)
                self.tracked_objects_sorted = [obj for _, obj in left_objects] + [obj for _, obj in right_objects]
            else:
                # 타겟이 있으면: x 좌표(centroid[0]) 기준으로 정렬 (좌측부터)
                self.tracked_objects_sorted = sorted(
                    self.tracked_objects,
                    key=lambda obj: obj['centroid'][0]
                )
            
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"추적 결과 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().warn(f"추적 결과 처리 실패: {e}")
    
    def _publish_command(self, command: Dict):
        """명령을 토픽으로 발행"""
        msg = String()
        msg.data = json.dumps(command, ensure_ascii=False)
        self.manual_control_publisher.publish(msg)
        self.get_logger().info(f"[JOYSTICK] 명령 발행: {command}")
    
    def _handle_auto_run(self):
        """j+c: Auto Run (Manual Stop)"""
        self.get_logger().info("[JOYSTICK] Auto Run 명령")
        command = {
            'type': 'run',
            'manual': False  # Auto 모드
        }
        self._publish_command(command)
    
    def _handle_manual_run(self):
        """j+d: Manual Run (Auto Stop)"""
        self.get_logger().info("[JOYSTICK] Manual Run 명령")
        command = {
            'type': 'run',
            'manual': True  # Manual 모드
        }
        self._publish_command(command)
    
    def _handle_stop(self):
        """j+f: Stop"""
        self.get_logger().info("[JOYSTICK] Stop 명령")
        command = {
            'type': 'stop'
        }
        self._publish_command(command)
    
    def _handle_target_left(self):
        """j+k: 타겟 선택 (왼쪽으로 이동)"""
        if len(self.tracked_objects_sorted) == 0:
            self.get_logger().warn("[JOYSTICK] 추적 중인 객체가 없습니다.")
            return
        
        # 현재 타겟의 인덱스 찾기
        current_index = -1
        if self.current_target_id is not None:
            for i, obj in enumerate(self.tracked_objects_sorted):
                if obj['track_id'] == self.current_target_id:
                    current_index = i
                    break
        
        # 왼쪽으로 이동 (인덱스 감소)
        if current_index <= 0:
            # 가장 왼쪽이거나 타겟이 없으면 무시
            self.get_logger().info("[JOYSTICK] 이미 가장 왼쪽에 있거나 타겟이 없습니다.")
            return
        
        # 이전 타겟 선택 (왼쪽으로)
        new_target_id = self.tracked_objects_sorted[current_index - 1]['track_id']
        
        # IDLE 상태면 TRACKING으로 변경
        if self.current_state == TrackingState.IDLE:
            self.get_logger().info(f"[JOYSTICK] IDLE -> TRACKING, 타겟 ID: {new_target_id}")
            command = {
                'type': 'set_state',
                'state': 'tracking',
                'target_id': new_target_id
            }
        else:
            # 그 외 상태는 ID만 변경
            self.get_logger().info(f"[JOYSTICK] 타겟 ID 변경: {new_target_id}")
            command = {
                'type': 'set_target',
                'target_id': new_target_id
            }
        
        self._publish_command(command)
    
    def _handle_target_right(self):
        """j+m: 타겟 선택 (오른쪽으로 이동)"""
        if len(self.tracked_objects_sorted) == 0:
            self.get_logger().warn("[JOYSTICK] 추적 중인 객체가 없습니다.")
            return
        
        # 현재 타겟의 인덱스 찾기
        current_index = -1
        if self.current_target_id is not None:
            for i, obj in enumerate(self.tracked_objects_sorted):
                if obj['track_id'] == self.current_target_id:
                    current_index = i
                    break
        
        # 오른쪽으로 이동 (인덱스 증가)
        if current_index >= len(self.tracked_objects_sorted) - 1:
            # 가장 오른쪽이거나 타겟이 없으면 무시
            self.get_logger().info("[JOYSTICK] 이미 가장 오른쪽에 있거나 타겟이 없습니다.")
            return
        
        # 다음 타겟 선택 (오른쪽으로)
        new_target_id = self.tracked_objects_sorted[current_index + 1]['track_id']
        
        # IDLE 상태면 TRACKING으로 변경
        if self.current_state == TrackingState.IDLE:
            self.get_logger().info(f"[JOYSTICK] IDLE -> TRACKING, 타겟 ID: {new_target_id}")
            command = {
                'type': 'set_state',
                'state': 'tracking',
                'target_id': new_target_id
            }
        else:
            # 그 외 상태는 ID만 변경
            self.get_logger().info(f"[JOYSTICK] 타겟 ID 변경: {new_target_id}")
            command = {
                'type': 'set_target',
                'target_id': new_target_id
            }
        
        self._publish_command(command)
    
    def _handle_handshake_state(self):
        """j+i: Handshake State로 변경"""
        self.get_logger().info("[JOYSTICK] Handshake State로 변경")
        command = {
            'type': 'set_state',
            'state': 'handshake',
            'target_id': self.current_target_id  # 현재 타겟 ID 유지
        }
        self._publish_command(command)
    
    def _handle_idle_state(self):
        """j+h: Idle State로 변경"""
        self.get_logger().info("[JOYSTICK] Idle State로 변경")
        command = {
            'type': 'set_state',
            'state': 'idle'
        }
        self._publish_command(command)
    
    def _handle_hello_state(self):
        """j+g: Hello State로 변경"""
        self.get_logger().info("[JOYSTICK] Hello State로 변경")
        command = {
            'type': 'set_state',
            'state': 'hello',
            'target_id': self.current_target_id  # 현재 타겟 ID 유지
        }
        self._publish_command(command)
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        self.running = False
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = JoystickControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
        node.destroy_node()
        except:
            pass
        try:
        rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()
