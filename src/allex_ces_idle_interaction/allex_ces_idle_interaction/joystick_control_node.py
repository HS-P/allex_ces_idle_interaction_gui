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
import os
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
        self.current_manual_mode = False  # 현재 Manual Mode 여부
        
        # 키보드 리스너 스레드 시작
        self.running = True
        self.keyboard_listener = None
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Joystick Control Node 시작")
        self.get_logger().info("모든 명령은 j 키와 함께 눌러야 작동합니다.")
        self.get_logger().info("j+c: Auto Run, j+d: Manual Run")
        self.get_logger().info("j+k: 타겟 선택 (왼쪽), j+m: 타겟 선택 (오른쪽)")
        self.get_logger().info("j+i: Handshake State, j+h: Idle State, j+g: Hello State")
        self.get_logger().info("j+f: Stop")
        self.get_logger().info("ESC 키를 누르면 종료합니다.")
        self.get_logger().info("=" * 60)
        
        # 키보드 리스너 시작
        self._start_keyboard_listener()
    
    def _start_keyboard_listener(self):
        """키보드 리스너 스레드 시작"""
        def on_press(key):
            """키가 눌렸을 때 호출"""
            try:
                # Shift+Space (IME 전환) 감지: 모든 키 상태 초기화
                if key == keyboard.Key.space or key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    # IME 전환 중에는 키 상태를 초기화하지 않고 그냥 무시 (on_release에서 처리)
                    return
                
                # j 키 확인
                if hasattr(key, 'char') and key.char is not None:
                    char_lower = key.char.lower() if hasattr(key.char, 'lower') else key.char
                    if char_lower == 'j':
                        if not self.j_key_pressed:
                            self.j_key_pressed = True
                            self.get_logger().debug("[KEY DEBUG] j 키 누름")
                            self._on_key_state_changed()
                        return
                
                # 일반 키 (char 속성이 있는 경우)
                # 한글 입력 방지: ASCII 문자만 처리
                if hasattr(key, 'char') and key.char is not None:
                    # ASCII 문자이고 출력 가능한 문자만 처리 (한글 제외)
                    # 'f' 키의 경우 대소문자 모두 처리 (소문자로 통일)
                    char_lower = key.char.lower() if hasattr(key.char, 'lower') else key.char
                    if char_lower.isascii() and char_lower.isprintable() and len(str(char_lower)) == 1:
                        if char_lower not in self.pressed_key_chars:
                            self.pressed_key_chars.add(char_lower)
                            self.get_logger().debug(f"[KEY DEBUG] 키 누름 감지: '{char_lower}' (원본: '{key.char}')")
                            self._on_key_state_changed()
                else:
                    # 특수 키 (Key 객체) - Shift, Space는 제외 (IME 전환용)
                    if key not in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r, keyboard.Key.space):
                        if key not in self.pressed_keys:
                            self.pressed_keys.add(key)
                            self._on_key_state_changed()
            except (AttributeError, ValueError):
                pass
        
        def on_release(key):
            """키가 떼어졌을 때 호출"""
            try:
                # Shift+Space (IME 전환) 감지: 모든 키 상태 초기화
                if key == keyboard.Key.space or key == keyboard.Key.shift or key == keyboard.Key.shift_l or key == keyboard.Key.shift_r:
                    # IME 전환 후 키 상태 초기화 (먹통 방지)
                    self.pressed_key_chars.clear()
                    self.pressed_keys.clear()
                    self.j_key_pressed = False
                    self.get_logger().debug("[KEY DEBUG] IME 전환 감지: 키 상태 초기화")
                
                # j 키 해제
                if hasattr(key, 'char') and key.char is not None:
                    char_lower = key.char.lower() if hasattr(key.char, 'lower') else key.char
                    if char_lower == 'j':
                        if self.j_key_pressed:
                            self.j_key_pressed = False
                            self.get_logger().debug("[KEY DEBUG] j 키 해제")
                        return
                
                # 일반 키
                # 한글 입력 방지: ASCII 문자만 처리
                if hasattr(key, 'char') and key.char is not None:
                    # ASCII 문자이고 출력 가능한 문자만 처리 (한글 제외)
                    # 'f' 키의 경우 대소문자 모두 처리
                    char_lower = key.char.lower() if hasattr(key.char, 'lower') else key.char
                    if char_lower.isascii() and char_lower.isprintable() and len(str(char_lower)) == 1:
                        if char_lower in self.pressed_key_chars:
                            self.pressed_key_chars.remove(char_lower)
                            self.get_logger().debug(f"[KEY DEBUG] 키 해제 감지: '{char_lower}'")
                else:
                    # 특수 키
                    if key in self.pressed_keys:
                        self.pressed_keys.remove(key)
                
                # ESC 키로 종료
                if key == keyboard.Key.esc:
                    self.get_logger().info("ESC 키 눌림 - 종료합니다.")
                    self.running = False
                    return False  # 리스너 종료
            except (AttributeError, ValueError):
                pass
        
        # 키보드 리스너 시작 (별도 스레드)
        # suppress=False: 키 입력을 다른 애플리케이션으로도 전달 (기본값)
        # 한글 IME와의 호환성을 위해 suppress=False 유지
        self.keyboard_listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=False  # 키 입력을 다른 애플리케이션으로도 전달
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
        self.get_logger().debug(f"[KEY DEBUG] j+{key} 키 조합 감지, combined_keys={combined_keys}")
        self._handle_key_command(key)
    
    def _handle_key_command(self, key: str):
        """키 명령 처리 (디바운싱 포함)"""
        current_time = time.monotonic()
        
        # 디바운싱 체크
        if key in self.last_key_action_time:
            elapsed = current_time - self.last_key_action_time[key]
            if elapsed < self.key_debounce_time:
                self.get_logger().debug(f"[KEY DEBUG] j+{key} 디바운싱 무시 (elapsed={elapsed:.3f}s)")
                return  # 디바운스 시간 내 재입력 무시
        
        self.last_key_action_time[key] = current_time
        self.get_logger().info(f"[KEY] j+{key} 키 명령 처리 시작")
        
        # 키 명령 처리
        if key == 'c':
            self._handle_auto_run()
        elif key == 'd':
            self._handle_manual_run()
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
        elif key == 'f':
            self._handle_stop()
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
            
            # Manual Mode 업데이트
            self.current_manual_mode = data.get('manual_mode', False)
            
            # 현재 타겟 ID 업데이트
            # tracking_result에서 target_track_id 또는 target_info.track_id 가져오기
            target_track_id = data.get('target_track_id', None)
            if target_track_id is None:
                # target_info에서 가져오기 (tracking_fsm_node는 target_info.track_id로 발행)
                target_info = data.get('target_info', {})
                if isinstance(target_info, dict):
                    target_track_id = target_info.get('track_id', None)
            self.current_target_id = target_track_id
            if self.current_target_id is not None:
                self.get_logger().debug(f"[JOYSTICK] 타겟 ID 업데이트: {self.current_target_id}")
            elif self.current_target_id is None and len(self.tracked_objects_sorted) > 0:
                self.get_logger().debug(f"[JOYSTICK] 타겟 ID가 None입니다. tracking_result 데이터: {list(data.keys())}")
            
            # 추적 객체 목록 업데이트 및 정렬 (x 좌표 기준)
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
            
            # x 좌표(centroid[0]) 기준으로 정렬
            self.tracked_objects_sorted = sorted(
                self.tracked_objects,
                key=lambda obj: obj['centroid'][0]
            )
            
            # 상세 디버그 출력 (화면 clear 후 표 형식으로 표시)
            self._print_detailed_status()
            
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"추적 결과 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().warn(f"추적 결과 처리 실패: {e}")
    
    def _print_detailed_status(self):
        """상세 상태를 터미널에 표 형식으로 출력 (화면 clear)"""
        try:
            # 화면 clear (ANSI escape code)
            print('\033[2J\033[H', end='')  # Clear screen and move cursor to top
            
            # 헤더
            print("=" * 80)
            print("JOYSTICK CONTROL NODE - 타겟 선택 상태")
            print("=" * 80)
            print(f"현재 상태: {self.current_state.value.upper()}")
            print(f"현재 타겟 ID: {self.current_target_id if self.current_target_id is not None else 'None'}")
            print(f"추적 중인 객체 수: {len(self.tracked_objects_sorted)}")
            print("-" * 80)
            
            if len(self.tracked_objects_sorted) == 0:
                print("추적 중인 객체가 없습니다.")
            else:
                # 표 헤더
                print(f"{'인덱스':<8} {'Track ID':<12} {'X 좌표':<12} {'Y 좌표':<12} {'상태':<10}")
                print("-" * 80)
                
                # 각 객체 출력
                for i, obj in enumerate(self.tracked_objects_sorted):
                    track_id = obj['track_id']
                    x_coord = obj['centroid'][0]
                    y_coord = obj['centroid'][1]
                    
                    # 현재 타겟인지 확인
                    is_current = (track_id == self.current_target_id)
                    
                    # 현재 타겟은 강조 표시
                    if is_current:
                        marker = ">>> "
                        status = "CURRENT"
                    else:
                        marker = "    "
                        status = ""
                    
                    print(f"{marker}{i:<5} {track_id:<12} {x_coord:>10.1f} {y_coord:>10.1f} {status:<10}")
            
            print("=" * 80)
            print("명령: j+k (왼쪽), j+m (오른쪽), j+c (Auto Run), j+d (Manual Run), j+f (Stop)")
            print("=" * 80)
            
        except Exception as e:
            self.get_logger().error(f"상세 상태 출력 오류: {e}")
    
    def _publish_command(self, command: Dict):
        """명령을 토픽으로 발행"""
        msg = String()
        msg.data = json.dumps(command, ensure_ascii=False)
        self.manual_control_publisher.publish(msg)
        self.get_logger().info(f"[JOYSTICK] 명령 발행: {command}")
    
    def _handle_auto_run(self):
        """j+c: Auto Run (Manual Stop) - set_mode으로 IDLE 전환 후 run"""
        self.get_logger().info("[JOYSTICK] Auto Run 명령 (IDLE 전환 후 run)")
        # Manual Mode에서 Auto Mode로 전환하는 경우 IDLE로 전환
        if self.current_manual_mode:
            mode_command = {
                'type': 'set_mode',
                'manual': False  # Auto 모드
            }
            self._publish_command(mode_command)
            # 잠시 대기하여 set_mode가 처리되도록 함
            import time
            time.sleep(0.1)
        # run 명령
        run_command = {
            'type': 'run',
            'manual': False  # Auto 모드
        }
        self._publish_command(run_command)
    
    def _handle_manual_run(self):
        """j+d: Manual Run (Auto Stop) - set_mode으로 IDLE 전환 후 run"""
        self.get_logger().info("[JOYSTICK] Manual Run 명령 (IDLE 전환 후 run)")
        # Auto Mode에서 Manual Mode로 전환하는 경우 IDLE로 전환
        if not self.current_manual_mode:
            mode_command = {
                'type': 'set_mode',
                'manual': True  # Manual 모드
            }
            self._publish_command(mode_command)
            # 잠시 대기하여 set_mode가 처리되도록 함
            import time
            time.sleep(0.1)
        # run 명령
        run_command = {
            'type': 'run',
            'manual': True  # Manual 모드
        }
        self._publish_command(run_command)
    
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
        
        self.get_logger().info(f"[JOYSTICK LEFT] 현재 타겟 ID: {self.current_target_id}, 인덱스: {current_index}, 총 객체 수: {len(self.tracked_objects_sorted)}")
        
        # 상세 상태 출력
        self._print_detailed_status()
        
        # 타겟을 찾지 못한 경우 (current_index == -1): 화면 중앙(640)에서 가장 가까운 사람 선택
        if current_index == -1:
            # 화면 중앙 x 좌표 (1280 / 2 = 640)
            screen_center_x = 640.0
            
            # 중앙에서 가장 가까운 객체 찾기
            closest_obj = min(
                self.tracked_objects_sorted,
                key=lambda obj: abs(obj['centroid'][0] - screen_center_x)
            )
            closest_index = next(
                i for i, obj in enumerate(self.tracked_objects_sorted)
                if obj['track_id'] == closest_obj['track_id']
            )
            new_target_id = closest_obj['track_id']
            self.get_logger().info(
                f"[JOYSTICK LEFT] 타겟을 찾지 못함, 중앙({screen_center_x})에서 가장 가까운 타겟 선택: "
                f"ID{new_target_id} (인덱스 {closest_index}, x={closest_obj['centroid'][0]:.1f})"
            )
            if self.current_state == TrackingState.IDLE:
                command = {
                    'type': 'set_state',
                    'state': 'tracking',
                    'target_id': new_target_id
                }
            else:
                command = {
                    'type': 'set_target',
                    'target_id': new_target_id
                }
            self._publish_command(command)
            return
        
        # 왼쪽으로 이동 (인덱스 감소)
        if current_index <= 0:
            # 가장 왼쪽에 있으면 무시
            self.get_logger().info(f"[JOYSTICK LEFT] 이미 가장 왼쪽에 있음 (인덱스: {current_index}, 총 {len(self.tracked_objects_sorted)}개)")
            return
        
        # 이전 타겟 선택 (왼쪽으로)
        prev_index = current_index - 1
        new_target_id = self.tracked_objects_sorted[prev_index]['track_id']
        self.get_logger().info(f"[JOYSTICK LEFT] 인덱스 이동: {current_index} -> {prev_index}, 타겟 ID: {self.current_target_id} -> {new_target_id}")
        
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
            self.get_logger().info(f"[JOYSTICK] 타겟 ID 변경 (왼쪽으로): {self.current_target_id} -> {new_target_id}")
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
        
        self.get_logger().info(f"[JOYSTICK RIGHT] 현재 타겟 ID: {self.current_target_id}, 인덱스: {current_index}, 총 객체 수: {len(self.tracked_objects_sorted)}")
        
        # 상세 상태 출력
        self._print_detailed_status()
        
        # 타겟을 찾지 못한 경우 (current_index == -1): 화면 중앙(640)에서 가장 가까운 사람 선택
        if current_index == -1:
            # 화면 중앙 x 좌표 (1280 / 2 = 640)
            screen_center_x = 640.0
            
            # 중앙에서 가장 가까운 객체 찾기
            closest_obj = min(
                self.tracked_objects_sorted,
                key=lambda obj: abs(obj['centroid'][0] - screen_center_x)
            )
            closest_index = next(
                i for i, obj in enumerate(self.tracked_objects_sorted)
                if obj['track_id'] == closest_obj['track_id']
            )
            new_target_id = closest_obj['track_id']
            self.get_logger().info(
                f"[JOYSTICK RIGHT] 타겟을 찾지 못함, 중앙({screen_center_x})에서 가장 가까운 타겟 선택: "
                f"ID{new_target_id} (인덱스 {closest_index}, x={closest_obj['centroid'][0]:.1f})"
            )
            if self.current_state == TrackingState.IDLE:
                command = {
                    'type': 'set_state',
                    'state': 'tracking',
                    'target_id': new_target_id
                }
            else:
                command = {
                    'type': 'set_target',
                    'target_id': new_target_id
                }
            self._publish_command(command)
            return
        
        # 오른쪽으로 이동 (인덱스 증가)
        if current_index >= len(self.tracked_objects_sorted) - 1:
            # 가장 오른쪽에 있으면 무시
            self.get_logger().info(f"[JOYSTICK RIGHT] 이미 가장 오른쪽에 있음 (인덱스: {current_index}, 총 {len(self.tracked_objects_sorted)}개)")
            return
        
        # 다음 타겟 선택 (오른쪽으로)
        next_index = current_index + 1
        new_target_id = self.tracked_objects_sorted[next_index]['track_id']
        self.get_logger().info(f"[JOYSTICK RIGHT] 인덱스 이동: {current_index} -> {next_index}, 타겟 ID: {self.current_target_id} -> {new_target_id}")
        
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
            self.get_logger().info(f"[JOYSTICK] 타겟 ID 변경 (오른쪽으로): {self.current_target_id} -> {new_target_id}")
            command = {
                'type': 'set_target',
                'target_id': new_target_id
            }
        
        self._publish_command(command)
    
    def _handle_handshake_state(self):
        """j+i: Handshake State로 변경 (Manual Mode에서만 작동)"""
        if not self.current_manual_mode:
            self.get_logger().warn("[JOYSTICK] Auto Mode에서는 Handshake State로 변경할 수 없습니다.")
            return
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
        """j+g: Hello State로 변경 (Manual Mode에서만 작동)"""
        if not self.current_manual_mode:
            self.get_logger().warn("[JOYSTICK] Auto Mode에서는 Hello State로 변경할 수 없습니다.")
            return
        self.get_logger().info("[JOYSTICK] Hello State로 변경")
        command = {
            'type': 'set_state',
            'state': 'hello',
            'target_id': self.current_target_id  # 현재 타겟 ID 유지
        }
        self._publish_command(command)
    
    def _handle_stop(self):
        """j+f: Stop 명령"""
        self.get_logger().info("[JOYSTICK] Stop 명령")
        command = {
            'type': 'stop'
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
