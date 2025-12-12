#!/usr/bin/env python3
import os
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from evdev import InputDevice, ecodes

class FootSwitchJoyNode(Node):
    def __init__(self):
        super().__init__('footswitch_joy_node')

        # by-id 심볼릭 링크 경로
        device_path = '/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd'

        # 권한 확인: 읽기/쓰기 가능 여부
        if not os.access(device_path, os.R_OK | os.W_OK):
            self.get_logger().error(
                f"Permission denied for '{device_path}'.\n"
                f"Please run the following command to grant access:\n"
                f"  sudo setfacl -m u:$USER:rw {device_path}\n"
                f"Then re-run this node."
            )
            sys.exit(1)

        # 독점 모드로 장치 열기
        self.dev = InputDevice(device_path)
        self.dev.grab()
        self.get_logger().info(f"Grabbed device: {self.dev.name} ({device_path})")

        # 퍼블리셔 생성
        self.pub = self.create_publisher(Joy, '/wontae/joy', 10)
        self.buttons = [0]
        self.axes = []

        # 100Hz 타이머
        self.timer = self.create_timer(0.05, self.timer_callback)

    def timer_callback(self):
        # 이벤트 처리: 한 번에 하나씩 읽기
        while True:
            event = self.dev.read_one()
            if event is None:
                break

            # KEY_B(코드 48)만 처리
            if event.type == ecodes.EV_KEY and event.code == ecodes.KEY_B:
                if event.value == 1:
                    self.buttons[0] = 1
                elif event.value == 0:
                    self.buttons[0] = 0

        # Joy 메시지 발행
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.axes = self.axes
        msg.buttons = self.buttons
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FootSwitchJoyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
