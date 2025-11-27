#!/usr/bin/env python3
"""
test.png 이미지를 ROS2 토픽으로 발행하는 노드
test.png 파일을 /camera/color/image_raw/compressed 토픽으로 계속 발행
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


class ImagePublisher(Node):
    """이미지 파일을 ROS2 토픽으로 발행하는 노드"""
    
    def __init__(self, image_path: str = None) -> None:
        super().__init__("image_publisher")
        
        # 이미지 파일 경로 찾기
        if image_path is None:
            # 여러 경로에서 test.png 찾기
            current_file = Path(__file__).resolve()
            possible_paths = [
                current_file.parent.parent.parent.parent / "test.png",  # 프로젝트 루트
                current_file.parent.parent.parent.parent.parent / "test.png",  # 상위 디렉토리
            ]
            
            # 설치된 환경에서 프로젝트 루트 찾기
            parts = current_file.parts
            if 'install' in parts:
                install_idx = parts.index('install')
                if install_idx > 0:
                    project_root = Path(*parts[:install_idx])  # install 이전까지가 프로젝트 루트
                    possible_paths.insert(0, project_root / "test.png")
            
            # 소스 코드 위치에서도 찾기
            if 'src' in parts:
                src_idx = parts.index('src')
                if src_idx > 0:
                    project_root = Path(*parts[:src_idx])
                    possible_paths.insert(0, project_root / "test.png")
            
            image_path = None
            for path in possible_paths:
                if path.exists():
                    image_path = path
                    break
            
            if image_path is None:
                # 기본값: 프로젝트 루트
                image_path = current_file.parent.parent.parent.parent / "test.png"
        
        self.image_path = str(image_path)
        
        if not os.path.exists(self.image_path):
            self.get_logger().error(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {self.image_path}")
        
        # 이미지 로드
        self.image = cv2.imread(self.image_path)
        if self.image is None:
            self.get_logger().error(f"이미지 파일을 읽을 수 없습니다: {self.image_path}")
            raise RuntimeError(f"이미지 파일을 읽을 수 없습니다: {self.image_path}")
        
        self.height, self.width = self.image.shape[:2]
        self.get_logger().info(f"이미지 파일 로드 완료: {self.image_path}")
        self.get_logger().info(f"해상도: {self.width}x{self.height}")
        
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
        
        # 타이머 생성 (30 FPS로 발행)
        fps = 30.0
        frame_interval = 1.0 / fps
        self.timer = self.create_timer(frame_interval, self.timer_callback)
        
        # 상태 변수
        self.frame_count = 0
        self.last_publish_time = time.time()
        
        self.get_logger().info(f"이미지 발행 시작 (FPS: {fps})")
    
    def timer_callback(self) -> None:
        """타이머 콜백 - 이미지를 읽어서 발행"""
        # OpenCV 이미지를 JPEG로 압축
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, compressed_img = cv2.imencode('.jpg', self.image, encode_param)
        
        # CompressedImage 메시지 생성
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        msg.format = "jpeg"
        msg.data = compressed_img.tobytes()
        
        # 발행
        self.publisher.publish(msg)
        
        self.frame_count += 1
        
        # 주기적으로 상태 로그 (5초마다)
        current_time = time.time()
        if current_time - self.last_publish_time > 5.0:
            elapsed = current_time - self.last_publish_time
            actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.get_logger().info(
                f"발행 중: {self.frame_count} 프레임 | "
                f"실제 FPS: {actual_fps:.1f}"
            )
            self.frame_count = 0
            self.last_publish_time = current_time


def main(args=None):
    rclpy.init(args=args)
    
    # 명령행 인자에서 이미지 경로 가져오기
    import sys
    image_path = None
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    image_publisher = ImagePublisher(image_path=image_path)
    
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


