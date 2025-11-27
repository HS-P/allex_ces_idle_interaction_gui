#!/usr/bin/env python3
"""
비디오 파일을 ROS2 토픽으로 발행하는 노드
human.mp4 파일을 camera/color/image_raw/compressed 토픽으로 발행
"""
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


class VideoPublisher(Node):
    """비디오 파일을 ROS2 토픽으로 발행하는 노드"""
    
    def __init__(self, video_path: str = "human.mp4") -> None:
        super().__init__("video_publisher")
        
        # 비디오 파일 경로
        self.video_path = video_path
        if not os.path.exists(self.video_path):
            self.get_logger().error(f"비디오 파일을 찾을 수 없습니다: {self.video_path}")
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {self.video_path}")
        
        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            self.get_logger().error(f"비디오 파일을 열 수 없습니다: {self.video_path}")
            raise RuntimeError(f"비디오 파일을 열 수 없습니다: {self.video_path}")
        
        # 비디오 정보 가져오기
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.get_logger().info(f"비디오 파일 로드 완료: {self.video_path}")
        self.get_logger().info(f"해상도: {self.width}x{self.height}, FPS: {self.fps}, 프레임 수: {self.frame_count}")
        
        # QoS 설정: BEST_EFFORT + 큐 깊이 30
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # Publisher 생성
        self.publisher = self.create_publisher(
            CompressedImage,
            "/camera/color/image_raw/compressed",
            qos_profile,
        )
        
        # 타이머 생성 (비디오 FPS에 맞춰 발행)
        frame_interval = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(frame_interval, self.timer_callback)
        
        # 상태 변수
        self.loop_video = True  # 반복 재생 여부
        self.current_frame_num = 0
        self.last_publish_time = time.time()
        
        self.get_logger().info("비디오 발행 시작")
    
    def timer_callback(self) -> None:
        """타이머 콜백 - 비디오 프레임을 읽어서 발행"""
        ret, frame = self.cap.read()
        
        if not ret:
            # 비디오 끝에 도달
            if self.loop_video:
                # 처음부터 다시 재생
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_num = 0
                self.get_logger().info("비디오 재시작 (반복 재생)")
                ret, frame = self.cap.read()
                if not ret:
                    self.get_logger().error("비디오 재시작 실패")
                    return
            else:
                self.get_logger().info("비디오 종료")
                self.destroy_node()
                rclpy.shutdown()
                return
        
        # OpenCV 이미지를 JPEG로 압축
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, compressed_img = cv2.imencode('.jpg', frame, encode_param)
        
        # CompressedImage 메시지 생성
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.format = "jpeg"
        msg.data = compressed_img.tobytes()
        
        # 발행
        self.publisher.publish(msg)
        
        self.current_frame_num += 1
        
        # 주기적으로 상태 로그 (5초마다)
        current_time = time.time()
        if current_time - self.last_publish_time > 5.0:
            elapsed = current_time - self.last_publish_time
            actual_fps = self.current_frame_num / elapsed if elapsed > 0 else 0
            self.get_logger().info(
                f"발행 중: 프레임 {self.current_frame_num}/{self.frame_count} | "
                f"실제 FPS: {actual_fps:.1f}"
            )
            self.current_frame_num = 0
            self.last_publish_time = current_time
    
    def destroy_node(self) -> None:
        """노드 종료 시 리소스 정리"""
        if self.cap is not None:
            self.cap.release()
            self.get_logger().info("비디오 캡처 해제 완료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # 명령행 인자에서 비디오 경로 가져오기
    import sys
    video_path = "human.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    video_publisher = VideoPublisher(video_path=video_path)
    
    try:
        rclpy.spin(video_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        video_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

