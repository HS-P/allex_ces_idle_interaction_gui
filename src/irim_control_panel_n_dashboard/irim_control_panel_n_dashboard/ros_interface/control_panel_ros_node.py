# ros_interface/ros_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32MultiArray

import json

class ROSInterface(Node):
    def __init__(self, articulations):
        super().__init__("robot_command_gui_node")
        # 명령 전송 퍼블리셔
        self.pub = self.create_publisher(String, "/hmi/robot_command", 10)

        self.articulations = articulations

        # 각 articulation의 최신 데이터를 저장 (각 topic은 Int32MultiArray의 data 필드에 값이 들어있다고 가정)
        self.latest_data = {
            art: {"articulation_now": None,
                  "servo_status": None,
                  "control_mode": None,
                  "communication_code": None}
            for art in self.articulations
        }
        
        # subscriber 저장용 리스트
        self.subscribers = []
        for art in self.articulations:
            topic_articulation_now = f"/robot_outbound_data/{art}/articulation_now"
            topic_servo_status     = f"/robot_outbound_data/{art}/servo_status"
            topic_control_mode     = f"/robot_outbound_data/{art}/control_mode"
            topic_comm_code        = f"/robot_outbound_data/{art}/communication_code"
            
            self.subscribers.append(
                self.create_subscription(
                    Int32MultiArray, topic_articulation_now,
                    self.create_callback(art, "articulation_now"), 10)
            )
            self.subscribers.append(
                self.create_subscription(
                    Int32MultiArray, topic_servo_status,
                    self.create_callback(art, "servo_status"), 10)
            )
            self.subscribers.append(
                self.create_subscription(
                    Int32MultiArray, topic_control_mode,
                    self.create_callback(art, "control_mode"), 10)
            )
            self.subscribers.append(
                self.create_subscription(
                    Int32MultiArray, topic_comm_code,
                    self.create_callback(art, "communication_code"), 10)
            )
        
        # 외부에서 이 콜백을 등록하면, 새 데이터를 받을 때마다 호출함
        self.status_update_callback = None

        # ★ 여기부터: routine debug 관련 멤버
        self.routine_debug_data = None
        self.routine_update_callback = None

        self.routine_sub = self.create_subscription(
            String,
            "/debug/routine",
            self._routine_debug_callback,
            10
        )


    def create_callback(self, articulation, data_type):
        def callback(msg):
            # msg.data는 Int32MultiArray의 data (예, list of int)라고 가정
            self.latest_data[articulation][data_type] = msg.data
            # self.get_logger().info(f"Received {data_type} for {articulation}: {msg.data}")
            if self.status_update_callback is not None and data_type == "articulation_now":
                # articulation_now가 왔을때만 전체 데이터를 콜백에 전달 하자~
                self.status_update_callback(articulation, self.latest_data[articulation])
        return callback

    def publish_command(self, message_str):
        self.get_logger().info(f"Publish message: {message_str}")
        self.pub.publish(String(data=message_str))

    def _routine_debug_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().warn(
                f"Failed to parse /debug/routine JSON: {e}"
            )
            return

        self.routine_debug_data = data

        if self.routine_update_callback is not None:
            # 메인 스레드에서 처리되도록, 여기서는 단순 callback 호출만
            self.routine_update_callback(data)

