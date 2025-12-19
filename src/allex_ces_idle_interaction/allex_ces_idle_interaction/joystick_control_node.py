#!/usr/bin/env python3
"""
Keyboard Control Node - 키보드로 GUI 기능 제어
GUI와 동일한 토픽(/allex_camera/manual_control)에 명령 발행

키보드 매핑:
- g: RUN/STOP 토글
- k: AUTO 모드 설정
- m: MANUAL 모드 설정
- e: 이전 타겟 선택
- f: 다음 타겟 선택
- c: State 다음으로 변경 (Manual 모드에서만)
- d: State 이전으로 변경 (Manual 모드에서만)

상태 순서: IDLE → TRACKING → LOST → SEARCHING → IDLE...
"""
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pynput import keyboard
import threading


class KeyboardControlNode(Node):
    """키보드 입력을 받아서 GUI 제어 명령 발행"""
    
    def __init__(self):
        super().__init__('keyboard_control_node')
        
        # Manual 제어 명령 Publisher (GUI와 동일한 토픽)
        self.manual_control_publisher = self.create_publisher(
            String,
            '/allex_camera/manual_control',
            10
        )
        
        # 상태 변수
        self.is_running = False  # RUN/STOP 상태
        self.is_manual_mode = False  # AUTO/MANUAL 모드
        
        # 타겟 관련
        self.current_target_id = None
        self.target_ids = []  # 감지된 타겟 ID 리스트 (tracking_data에서 업데이트)
        
        # Tracking 데이터 구독 (타겟 ID 목록 가져오기)
        self.tracking_data_subscription = self.create_subscription(
            String,
            '/allex_camera/tracking_data',
            self.tracking_data_callback,
            10
        )
        
        # 상태 순서 정의 (순환)
        self.state_order = ['idle', 'tracking', 'lost', 'searching']
        self.current_state_index = 0  # 현재 상태 인덱스
        
        # 키보드 입력 스레드 시작
        self.running = True
        self.keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self.keyboard_thread.start()
        
        self.get_logger().info("Keyboard Control Node 시작")
        self.get_logger().info("키보드 매핑:")
        self.get_logger().info("  g: RUN/STOP 토글")
        self.get_logger().info("  k: AUTO 모드")
        self.get_logger().info("  m: MANUAL 모드")
        self.get_logger().info("  e: 이전 타겟 선택")
        self.get_logger().info("  f: 다음 타겟 선택")
        self.get_logger().info("  c: State 다음 (Manual 모드에서만)")
        self.get_logger().info("  d: State 이전 (Manual 모드에서만)")
        self.get_logger().info("=" * 60)
        self.get_logger().info("키보드 입력 대기 중...")
    
    def tracking_data_callback(self, msg: String):
        """Tracking 데이터 콜백 - 타겟 ID 목록 업데이트"""
        try:
            data = json.loads(msg.data)
            tracked_objects = data.get('tracked_objects', [])
            self.target_ids = sorted([obj['track_id'] for obj in tracked_objects])
        except Exception as e:
            self.get_logger().error(f"Tracking 데이터 파싱 실패: {e}")
    
    def _keyboard_listener(self):
        """키보드 입력 리스너 (별도 스레드에서 실행)"""
        def on_press(key):
            """키가 눌렸을 때 호출"""
            try:
                # 일반 키 처리
                if hasattr(key, 'char') and key.char:
                    if key.char == 'g':
                        self._on_key_g()
                    elif key.char == 'k':
                        self._on_key_k()
                    elif key.char == 'm':
                        self._on_key_m()
                    elif key.char == 'e':
                        self._on_key_e()
                    elif key.char == 'f':
                        self._on_key_f()
                    elif key.char == 'c':
                        self._on_key_c()
                    elif key.char == 'd':
                        self._on_key_d()
            except AttributeError:
                # 특수 키 처리 (ESC 등)
                if key == keyboard.Key.esc:
                    self.running = False
                    return False  # 리스너 종료
        
        def on_release(key):
            """키가 떼어졌을 때 호출"""
            if key == keyboard.Key.esc:
                return False  # 리스너 종료
        
        # 키보드 리스너 시작
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    
    def _on_key_g(self):
        """g 키: RUN/STOP 토글"""
        self.get_logger().info(f"[KEY] g 키 눌림 (현재 상태: RUN={self.is_running})")
        self.is_running = not self.is_running
        if self.is_running:
            self._send_command({
                'type': 'run',
                'manual': self.is_manual_mode
            })
            self.get_logger().info(f"[KEY] → RUN 시작 ({'Manual' if self.is_manual_mode else 'Auto'})")
        else:
            self._send_command({'type': 'stop'})
            self.get_logger().info("[KEY] → RUN 중지")
    
    def _on_key_k(self):
        """k 키: AUTO 모드 설정"""
        self.get_logger().info(f"[KEY] k 키 눌림 (현재 모드: {'Manual' if self.is_manual_mode else 'Auto'})")
        self.is_manual_mode = False
        if self.is_running:
            self._send_command({
                'type': 'set_mode',
                'manual': False
            })
            self.get_logger().info("[KEY] → 모드 변경: AUTO")
    
    def _on_key_m(self):
        """m 키: MANUAL 모드 설정"""
        self.get_logger().info(f"[KEY] m 키 눌림 (현재 모드: {'Manual' if self.is_manual_mode else 'Auto'})")
        self.is_manual_mode = True
        if self.is_running:
            self._send_command({
                'type': 'set_mode',
                'manual': True
            })
            self.get_logger().info("[KEY] → 모드 변경: MANUAL")
    
    def _on_key_e(self):
        """e 키: 이전 타겟 선택"""
        self.get_logger().info(f"[KEY] e 키 눌림 (현재 타겟: {self.current_target_id}, 사용가능: {self.target_ids})")
        if self.is_running and self.target_ids:
            if self.current_target_id is None:
                self.current_target_id = self.target_ids[-1]
            else:
                try:
                    current_idx = self.target_ids.index(self.current_target_id)
                    prev_idx = (current_idx - 1) % len(self.target_ids)
                    self.current_target_id = self.target_ids[prev_idx]
                except ValueError:
                    self.current_target_id = self.target_ids[-1]
            
            self._send_command({
                'type': 'set_target',
                'target_id': self.current_target_id,
                'force': True
            })
            self.get_logger().info(f"[KEY] → 타겟 변경 (이전): {self.current_target_id}")
        else:
            self.get_logger().warn(f"[KEY] → 타겟 변경 실패 (RUN={self.is_running}, 타겟수={len(self.target_ids)})")
    
    def _on_key_f(self):
        """f 키: 다음 타겟 선택"""
        self.get_logger().info(f"[KEY] f 키 눌림 (현재 타겟: {self.current_target_id}, 사용가능: {self.target_ids})")
        if self.is_running and self.target_ids:
            if self.current_target_id is None:
                self.current_target_id = self.target_ids[0]
            else:
                try:
                    current_idx = self.target_ids.index(self.current_target_id)
                    next_idx = (current_idx + 1) % len(self.target_ids)
                    self.current_target_id = self.target_ids[next_idx]
                except ValueError:
                    self.current_target_id = self.target_ids[0]
            
            self._send_command({
                'type': 'set_target',
                'target_id': self.current_target_id,
                'force': True
            })
            self.get_logger().info(f"[KEY] → 타겟 변경 (다음): {self.current_target_id}")
        else:
            self.get_logger().warn(f"[KEY] → 타겟 변경 실패 (RUN={self.is_running}, 타겟수={len(self.target_ids)})")
    
    def _on_key_c(self):
        """c 키: State 다음으로 (Manual 모드에서만)"""
        if self.is_running and self.is_manual_mode:
            self.get_logger().info(f"[KEY] c 키 눌림 (현재 상태 인덱스: {self.current_state_index})")
            self.current_state_index = (self.current_state_index + 1) % len(self.state_order)
            next_state = self.state_order[self.current_state_index]
            self._send_command({
                'type': 'set_state',
                'state': next_state,
                'target_id': self.current_target_id if next_state == 'tracking' else None
            })
            self.get_logger().info(f"[KEY] → 상태 변경 (다음): {next_state.upper()}")
        else:
            self.get_logger().warn(f"[KEY] → State 변경 실패 (RUN={self.is_running}, Manual={self.is_manual_mode})")
    
    def _on_key_d(self):
        """d 키: State 이전으로 (Manual 모드에서만)"""
        if self.is_running and self.is_manual_mode:
            self.get_logger().info(f"[KEY] d 키 눌림 (현재 상태 인덱스: {self.current_state_index})")
            self.current_state_index = (self.current_state_index - 1) % len(self.state_order)
            prev_state = self.state_order[self.current_state_index]
            self._send_command({
                'type': 'set_state',
                'state': prev_state,
                'target_id': self.current_target_id if prev_state == 'tracking' else None
            })
            self.get_logger().info(f"[KEY] → 상태 변경 (이전): {prev_state.upper()}")
        else:
            self.get_logger().warn(f"[KEY] → State 변경 실패 (RUN={self.is_running}, Manual={self.is_manual_mode})")
    
    def _send_command(self, command: dict):
        """명령 전송"""
        try:
            msg = String()
            msg.data = json.dumps(command, ensure_ascii=False)
            self.manual_control_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"명령 전송 실패: {e}")
    
    def destroy_node(self):
        """노드 종료 시 키보드 리스너 정리"""
        self.running = False
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
