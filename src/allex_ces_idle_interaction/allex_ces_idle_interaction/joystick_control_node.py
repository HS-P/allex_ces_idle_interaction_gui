#!/usr/bin/env python3
"""
Joystick Control Node - 조이스틱 버튼 인식 및 출력
버튼 입력을 감지하고 로그로 출력 (단순 테스트용)

조합 버튼 인식 지원:
- 단일 버튼 입력 감지
- 두 개 이상의 버튼이 동시에 눌린 경우 조합으로 인식
"""
import rclpy
from rclpy.node import Node
import pygame
import time
from typing import Set


class JoystickControlNode(Node):
    """조이스틱 버튼 입력 감지 및 출력"""
    
    def __init__(self):
        super().__init__('joystick_control_node')
        
        # Pygame 초기화
        pygame.init()
        pygame.joystick.init()
        
        # 조이스틱 확인
        joystick_count = pygame.joystick.get_count()
        if joystick_count == 0:
            self.get_logger().error("조이스틱이 감지되지 않습니다!")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.get_logger().info(f"조이스틱 감지: {self.joystick.get_name()}")
            self.get_logger().info(f"버튼 개수: {self.joystick.get_numbuttons()}")
            self.get_logger().info(f"축 개수: {self.joystick.get_numaxes()}")
        
        # 버튼 상태 추적
        self.pressed_buttons: Set[int] = set()  # 현재 눌려진 버튼들
        
        # 타이머 설정 (조이스틱 이벤트 폴링용)
        self.timer = self.create_timer(0.01, self._update_joystick)  # 100Hz
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Joystick Control Node 시작")
        self.get_logger().info("버튼 입력을 감지하고 로그로 출력합니다.")
        self.get_logger().info("=" * 60)
    
    def _update_joystick(self):
        """조이스틱 상태 업데이트 (타이머 콜백)"""
        if self.joystick is None:
            return
        
        # Pygame 이벤트 처리
        pygame.event.pump()
        
        # 현재 프레임에서 눌려진 버튼들
        current_pressed: Set[int] = set()
        
        # 모든 버튼 상태 확인
        for button_id in range(self.joystick.get_numbuttons()):
            if self.joystick.get_button(button_id):
                current_pressed.add(button_id)
        
        # 버튼 상태 변화 감지
        newly_pressed = current_pressed - self.pressed_buttons  # 새로 눌린 버튼
        newly_released = self.pressed_buttons - current_pressed  # 새로 떼어진 버튼
        
        # 새로 눌린 버튼이 있는 경우
        if newly_pressed:
            if len(current_pressed) == 1:
                # 단일 버튼
                button_id = list(current_pressed)[0]
                self.get_logger().info(f"[BUTTON] 버튼 {button_id} 눌림 (단일)")
            else:
                # 조합 버튼
                buttons_str = ', '.join(sorted([str(b) for b in current_pressed]))
                self.get_logger().info(f"[BUTTON] 조합 버튼 눌림: [{buttons_str}] (총 {len(current_pressed)}개)")
        
        # 새로 떼어진 버튼이 있는 경우
        if newly_released:
            if len(current_pressed) == 0:
                # 모든 버튼 해제
                buttons_str = ', '.join(sorted([str(b) for b in self.pressed_buttons]))
                self.get_logger().info(f"[BUTTON] 모든 버튼 해제: [{buttons_str}]")
            elif len(current_pressed) == 1:
                # 단일 버튼만 남음
                button_id = list(current_pressed)[0]
                self.get_logger().info(f"[BUTTON] 버튼 {button_id}만 눌림 (단일, 다른 버튼 해제)")
            else:
                # 조합 버튼 상태 유지 (일부만 해제)
                buttons_str = ', '.join(sorted([str(b) for b in current_pressed]))
                self.get_logger().info(f"[BUTTON] 조합 버튼 상태: [{buttons_str}] (일부 해제)")
        
        # 상태 업데이트
        self.pressed_buttons = current_pressed
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        if self.joystick is not None:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()
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
