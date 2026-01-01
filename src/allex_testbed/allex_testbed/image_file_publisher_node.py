#!/usr/bin/env python3
"""
이미지 파일을 CompressedImage로 발행하는 노드
테스트용 이미지 파일을 카메라 이미지처럼 발행
"""
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from pathlib import Path


class ImageFilePublisherNode(Node):
    """이미지 파일을 CompressedImage로 발행하는 노드"""
    
    def __init__(self):
        super().__init__('image_file_publisher_node')
        
        # 파라미터: 이미지 파일 경로
        self.declare_parameter('image_file_path', '')
        self.declare_parameter('publish_rate', 30.0)  # 발행 주기 (Hz)
        self.declare_parameter('output_topic', '/camera/color/image_raw/compressed')
        
        image_file_path = self.get_parameter('image_file_path').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        
        # 이미지 파일 경로 확인
        if not image_file_path:
            # src 디렉토리 기준으로 찾기 (build/install이 아닌 src)
            current_file_path = os.path.abspath(__file__)
            
            # 경로에서 src/allex_testbed 찾기
            parts = current_file_path.split(os.sep)
            package_dir = None
            
            # build나 install 경로에서 src 찾기
            for i, part in enumerate(parts):
                if part in ['build', 'install']:
                    # build 또는 install 이전이 workspace root
                    workspace_root = os.sep.join(parts[:i])
                    src_package_dir = os.path.join(workspace_root, 'src', 'allex_testbed')
                    if os.path.exists(src_package_dir):
                        package_dir = src_package_dir
                        break
            
            if package_dir:
                image_dir = os.path.join(package_dir, 'image')
            else:
                image_dir = None
            
            # image 디렉토리에서 첫 번째 이미지 파일 찾기
            if image_dir and os.path.exists(image_dir):
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                    image_files.extend(Path(image_dir).glob(ext))
                if image_files:
                    image_file_path = str(image_files[0])
                    self.get_logger().info(f"기본 이미지 파일 사용: {image_file_path}")
                else:
                    self.get_logger().error(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
                    self.get_logger().error("사용법: image_file_path 파라미터로 이미지 파일 경로를 지정하세요")
                    raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_dir}")
            else:
                self.get_logger().error(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir if image_dir else 'src/allex_testbed/image'}")
                self.get_logger().error("사용법: image_file_path 파라미터로 이미지 파일 경로를 지정하세요")
                raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: src/allex_testbed/image")
        
        # 경로 확장 (상대 경로 지원)
        if not os.path.isabs(image_file_path):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            package_dir = os.path.dirname(current_file_dir)
            image_file_path = os.path.join(package_dir, image_file_path)
        
        if not os.path.exists(image_file_path):
            self.get_logger().error(f"이미지 파일을 찾을 수 없습니다: {image_file_path}")
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_file_path}")
        
        # 이미지 파일 읽기
        self.image = cv2.imread(image_file_path)
        if self.image is None:
            self.get_logger().error(f"이미지 파일을 읽을 수 없습니다: {image_file_path}")
            raise ValueError(f"이미지 파일을 읽을 수 없습니다: {image_file_path}")
        
        self.get_logger().info(f"이미지 파일 로드 완료: {image_file_path}")
        self.get_logger().info(f"이미지 크기: {self.image.shape[1]}x{self.image.shape[0]}")
        
        # QoS 설정 (카메라와 동일하게)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        
        # 이미지 발행자
        self.image_publisher = self.create_publisher(
            CompressedImage,
            output_topic,
            qos_profile
        )
        
        # 타이머 설정 (지정된 주기로 발행)
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Image File Publisher Node 시작")
        self.get_logger().info(f"이미지 파일: {image_file_path}")
        self.get_logger().info(f"발행 주기: {publish_rate} Hz")
        self.get_logger().info(f"출력 토픽: {output_topic}")
        self.get_logger().info("=" * 60)
    
    def timer_callback(self):
        """타이머 콜백: 이미지를 CompressedImage로 발행"""
        try:
            # OpenCV 이미지를 JPEG로 인코딩
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv2.imencode('.jpg', self.image, encode_param)
            
            if not result:
                self.get_logger().error("이미지 인코딩 실패")
                return
            
            # CompressedImage 메시지 생성
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_frame'
            msg.format = "jpeg"
            msg.data = encimg.tobytes()
            
            # 발행
            self.image_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"이미지 발행 오류: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageFilePublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

