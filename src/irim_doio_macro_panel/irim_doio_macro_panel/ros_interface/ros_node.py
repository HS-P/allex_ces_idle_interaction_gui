from __future__ import annotations

from typing import Optional
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MacroPublisherNode(Node):
    def __init__(self, topic: str, qos_depth: int = 10):
        super().__init__("doio_macro_publisher")
        self._topic = topic
        self._pub = self.create_publisher(String, topic, qos_depth)
        self.get_logger().info(f"[INIT] Publishing String to {topic}")

    def publish_text(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._pub.publish(msg)
        self.get_logger().info(f"[PUB] {text}")
