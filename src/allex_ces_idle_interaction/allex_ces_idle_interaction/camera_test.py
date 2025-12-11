#!/usr/bin/env python3
"""
카메라 ROS2 Publisher - OpenCV CPU 최적화 버전
IMX291 센서, 1280x720 @ 30fps, MJPEG
- 실제 카메라가 1920x1080으로 동작하더라도
  소프트웨어에서 1280x720으로 다운스케일 후
  90도 회전, JPEG 인코딩, ROS2 발행
"""

import os
os.environ.pop('QT_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import cv2
import numpy as np
import sys
import time
import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header


class CameraPublisherNode(Node):
    """카메라 프레임을 빠르게 ROS2로 발행"""

    def __init__(self, camera_index=0):
        super().__init__('camera_publisher_node')

        # 목표 해상도 (다운스케일 타겟)
        self.target_width = 1280
        self.target_height = 720

        # publisher
        self.image_publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            10
        )

        # camera init
        self.camera_index = camera_index
        if not self._init_camera():
            self.get_logger().error("카메라 초기화 실패")
            return

        # shared frame
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True

        # 프레임 캡처 스레드 시작
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()

        # timer callback: 30Hz 발행 요청
        self.timer = self.create_timer(1.0 / 60.0, self.publish_frame)

        # FPS 측정
        self.frame_count = 0
        self.fps_start_time = time.time()

        # 간단한 인코딩 시간 프로파일링
        self.encode_time_sum = 0.0
        self.encode_count = 0

        self.get_logger().info("카메라 Publisher 노드 시작!")

    # ---------------------------------------------------------
    # 카메라 초기화
    # ---------------------------------------------------------
    def _init_camera(self):
        self.get_logger().info(f"카메라 {self.camera_index} 열기 시도 중...")
        self.get_logger().info("   V4L2 백엔드 사용 중...")

        # 인덱스로 열어도 되고, /dev/video0 그대로 써도 됨
        self.cap = cv2.VideoCapture("/dev/widecam", cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.get_logger().error(f"카메라 {self.camera_index}를 열 수 없음")
            return False

        target_width = self.target_width
        target_height = self.target_height
        target_fps = 60

        # MJPEG 설정 시도
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        self.get_logger().info("🔥🔥🔥 camera_test NEW CODE IS RUNNING 🔥🔥🔥")


        # 버퍼 최소화
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # 자동 포커스 끄기 (지원 안 하는 카메라도 있음)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.get_logger().info(f"   실제 설정된 해상도: {width}x{height}")
        self.get_logger().info(f"   실제 설정된 FPS: {fps}")

        # 만약 여기서 width, height가 1920x1080으로 찍히면
        # 아래에서 소프트웨어 다운스케일로 처리함.
        if width != target_width or height != target_height:
            self.get_logger().warn(
                f"카메라가 요청한 해상도({target_width}x{target_height})를 따르지 않습니다. "
                f"캡처 후 소프트웨어에서 {target_width}x{target_height}로 리사이즈합니다."
            )

        return True

    # ---------------------------------------------------------
    # Capturing Thread (실제 30Hz 읽기 유지)
    # ---------------------------------------------------------
    def _capture_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("프레임을 읽을 수 없습니다.")
                time.sleep(0.005)
                continue

            # 필요 시 해상도 강제 다운스케일
            h, w = frame.shape[:2]
            if w != self.target_width or h != self.target_height:
                # INTER_AREA는 다운스케일에 적합한 필터
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )

            # rotate 90° CCW → 빠른 transpose + flip
            rotated = cv2.transpose(frame)
            rotated = cv2.flip(rotated, 0)

            # copy는 여기서 단 1번만 수행
            with self.frame_lock:
                self.latest_frame = rotated.copy()

    # ---------------------------------------------------------
    # Publish Frame
    # ---------------------------------------------------------
    def publish_frame(self):
        # 최신 프레임 가져오기
        with self.frame_lock:
            frame_to_publish = self.latest_frame

        if frame_to_publish is None:
            return

        # JPEG 인코딩: CPU에서 가장 무거운 부분
        start_t = time.time()
        # 품질 60 → 속도 더 확보 (필요하면 50까지도 가능)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        ok, enc = cv2.imencode('.jpg', frame_to_publish, encode_param)
        encode_dt = time.time() - start_t

        if not ok:
            self.get_logger().warn("JPEG 인코딩 실패")
            return

        # 인코딩 시간 누적 (성능 확인용)
        self.encode_time_sum += encode_dt
        self.encode_count += 1

        # CompressedImage 메시지 구성
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        msg.format = "jpeg"
        msg.data = enc.tobytes()

        self.image_publisher.publish(msg)

        # FPS 계산 (5초마다)
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 5.0:
            fps = self.frame_count / elapsed
            avg_enc = (self.encode_time_sum / self.encode_count) * 1000.0 if self.encode_count > 0 else 0.0
            self.get_logger().info(
                f"발행 FPS: {fps:.1f}  |  평균 JPEG 인코딩 시간: {avg_enc:.2f} ms"
            )
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.encode_time_sum = 0.0
            self.encode_count = 0

    # ---------------------------------------------------------
    # 종료 처리
    # ---------------------------------------------------------
    def destroy_node(self):
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)

        if hasattr(self, 'cap'):
            self.cap.release()

        self.get_logger().info("카메라 Publisher 종료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print("잘못된 인덱스, 기본값 0 사용")

    node = CameraPublisherNode(camera_index)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
