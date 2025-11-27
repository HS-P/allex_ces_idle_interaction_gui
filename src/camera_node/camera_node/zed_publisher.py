#!/usr/bin/env python3
"""
ZED 카메라 (0번째 카메라)를 ROS2 토픽으로 발행하는 노드
cv2로 0번째 카메라를 열어서 /camera/color/image_raw/compressed 토픽으로 발행
"""
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


class ZedPublisher(Node):
    """ZED 카메라 (0번째 카메라)를 ROS2 토픽으로 발행하는 노드"""
    
    def __init__(self, camera_index: int = 0, fps: float = 30.0) -> None:
        super().__init__("zed_publisher")
        
        # 카메라 초기화 - V4L2 백엔드 명시적 사용
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"카메라 {self.camera_index}를 열 수 없습니다")
            raise RuntimeError(f"카메라 {self.camera_index}를 열 수 없습니다")
        
        # 카메라 해상도 설정 (1280x720)
        target_width = 1280
        target_height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        
        # 카메라 초기화를 위해 프레임을 몇 번 읽기
        self.get_logger().info("카메라 초기화 중...")
        for i in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                self.width = frame.shape[1]
                self.height = frame.shape[0]
                break
            time.sleep(0.1)
        
        # 프레임에서 해상도를 가져오지 못했으면 속성에서 가져오기
        if not hasattr(self, 'width') or self.width == 0:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"카메라 {self.camera_index} 열기 완료")
        if self.width == target_width and self.height == target_height:
            self.get_logger().info(f"해상도: {self.width}x{self.height} (설정 성공), FPS: {actual_fps}")
        else:
            self.get_logger().warn(f"해상도: {self.width}x{self.height} (요청: {target_width}x{target_height}), FPS: {actual_fps}")
        
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
        
        # 타이머 생성 (지정된 FPS로 발행)
        frame_interval = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self.timer = self.create_timer(frame_interval, self.timer_callback)
        
        # 상태 변수
        self.frame_count = 0
        self.last_publish_time = time.time()
        self.failed_read_count = 0  # 연속 실패 횟수
        self.last_warn_time = 0  # 마지막 경고 시간
        
        self.get_logger().info(f"카메라 발행 시작 (FPS: {fps})")
    
    def timer_callback(self) -> None:
        """타이머 콜백 - 카메라 프레임을 읽어서 발행"""
        ret, frame = self.cap.read()
        
        if not ret or frame is None:
            self.failed_read_count += 1
            # 1초에 한 번만 경고 메시지 출력
            current_time = time.time()
            if current_time - self.last_warn_time > 1.0:
                self.get_logger().warn(f"카메라에서 프레임을 읽을 수 없습니다 (연속 실패: {self.failed_read_count}회)")
                self.last_warn_time = current_time
            return
        
        # 성공적으로 읽었으면 실패 카운터 리셋
        self.failed_read_count = 0
        
        # ZED 스테레오 카메라: 좌우 이미지가 붙어있으므로 왼쪽 이미지만 crop
        h, w = frame.shape[:2]
        if w > h * 2:  # 스테레오 이미지인 경우 (너비가 높이의 2배 이상)
            # 왼쪽 절반만 사용
            frame = frame[:, :w // 2]
        
        # 1280x720으로 resize
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        
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
    
    def destroy_node(self) -> None:
        """노드 종료 시 리소스 정리"""
        if self.cap is not None:
            self.cap.release()
            self.get_logger().info("카메라 해제 완료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # 명령행 인자에서 카메라 인덱스와 FPS 가져오기
    import sys
    camera_index = 0
    fps = 30.0
    
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"잘못된 카메라 인덱스: {sys.argv[1]}, 기본값 0 사용")
    
    if len(sys.argv) > 2:
        try:
            fps = float(sys.argv[2])
        except ValueError:
            print(f"잘못된 FPS: {sys.argv[2]}, 기본값 30.0 사용")
    
    zed_publisher = ZedPublisher(camera_index=camera_index, fps=fps)
    
    try:
        rclpy.spin(zed_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            zed_publisher.destroy_node()
        except Exception as e:
            print(f"노드 종료 중 오류: {e}")
        try:
            rclpy.shutdown()
        except Exception as e:
            # shutdown이 이미 호출되었을 수 있음
            pass


if __name__ == '__main__':
    main()

