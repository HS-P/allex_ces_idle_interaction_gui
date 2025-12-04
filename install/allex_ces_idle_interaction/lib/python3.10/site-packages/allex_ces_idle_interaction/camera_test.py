#!/usr/bin/env python3
"""
카메라 ROS2 Publisher - OpenCV + CUDA 가속
IMX291 센서, 1280x720 @ 30fps, MJPEG 코덱
90도 회전된 이미지를 ROS2 토픽으로 발행
"""
import os
# Qt 플러그인 문제 해결 (OpenCV 창 표시용)
os.environ.pop('QT_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)

import cv2
import numpy as np
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header

# OpenCV CUDA 사용 가능 여부 확인
cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

def check_cuda_status():
    """CUDA 상태 확인"""
    print("=" * 60)
    print("OpenCV CUDA 상태 확인")
    print("=" * 60)
    
    if cuda_available:
        print(f"✅ OpenCV CUDA 사용 가능: {cv2.cuda.getCudaEnabledDeviceCount()}개 GPU")
        for i in range(cv2.cuda.getCudaEnabledDeviceCount()):
            try:
                device_info = cv2.cuda.DeviceInfo(i)
                print(f"   GPU {i}: {device_info.name()}")
            except:
                print(f"   GPU {i}: 정보 확인 불가")
    else:
        print("⚠️  OpenCV CUDA를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        print("   OpenCV가 CUDA 지원으로 빌드되어 있는지 확인하세요.")
    
    print("=" * 60)


class CameraPublisherNode(Node):
    """카메라 이미지를 ROS2 토픽으로 발행하는 노드"""
    
    def __init__(self, camera_index=0):
        super().__init__('camera_publisher_node')
        
        # CUDA 상태 확인
        check_cuda_status()
        
        # QoS 설정 (실시간 스트리밍용)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        
        # 이미지 Publisher
        self.image_publisher = self.create_publisher(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            qos_profile
        )
        
        # 카메라 인덱스
        self.camera_index = camera_index
        
        # 카메라 초기화
        if not self._init_camera():
            self.get_logger().error("카메라 초기화 실패!")
            return
        
        # CUDA 초기화
        self.use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.gpu_frame = None
        self.stream = None
        
        if self.use_cuda:
            try:
                self.stream = cv2.cuda.Stream()
                self.gpu_frame = cv2.cuda_GpuMat()
                self.get_logger().info("✅ CUDA 스트림 및 버퍼 생성 완료")
            except Exception as e:
                self.get_logger().warn(f"CUDA 스트림 생성 실패: {e}")
                self.use_cuda = False
        
        # 타이머로 주기적으로 프레임 발행 (30fps = 약 33ms 간격)
        self.timer = self.create_timer(1.0 / 30.0, self.publish_frame)
        
        # 성능 모니터링
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        self.get_logger().info("카메라 Publisher 노드 시작!")
    
    def _init_camera(self):
        """카메라 초기화"""
        self.get_logger().info(f"카메라 {self.camera_index} 열기 시도 중...")
        self.get_logger().info("   V4L2 백엔드 사용 중...")
        
        self.cap = cv2.VideoCapture("/dev/widecam", cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"❌ 카메라 {self.camera_index}를 열 수 없습니다!")
            return False
        
        # 해상도 및 FPS 설정
        target_width = 1920
        target_height = 1080
        target_fps = 30.0
        
        # MJPEG 코덱 설정 (카메라 사양에 맞춤)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        self.cap.set(cv2.CAP_PROP_FPS, target_fps)
        
        # 버퍼 크기 최소화 (지연 최소화)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 자동 포커스 비활성화
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # 실제 설정된 값 확인
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"✅ 카메라 {self.camera_index} 열기 성공!")
        self.get_logger().info(f"   해상도: {width}x{height}")
        self.get_logger().info(f"   FPS: {fps}")
        
        return True
    
    def publish_frame(self):
        """프레임을 읽어서 90도 회전 후 ROS2 토픽으로 발행"""
        # 프레임 읽기
        ret, frame = self.cap.read()
        
        if not ret:
            self.get_logger().warn("프레임을 읽을 수 없습니다.")
            return
        
        # 90도 회전 (시계 방향)
        # cv2.ROTATE_90_CLOCKWISE: 시계 방향 90도
        # cv2.ROTATE_90_COUNTERCLOCKWISE: 반시계 방향 90도
        frame_rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # CUDA 사용 시 GPU로 업로드 및 처리 (필요시)
        # 현재는 단순 회전만 하므로 CPU로 처리
        
        # OpenCV 이미지를 JPEG로 압축
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, encimg = cv2.imencode('.jpg', frame_rotated, encode_param)
        
        if not result:
            self.get_logger().warn("이미지 인코딩 실패")
            return
        
        # CompressedImage 메시지 생성
        msg = CompressedImage()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        msg.format = "jpeg"
        msg.data = encimg.tobytes()
        
        # 발행
        self.image_publisher.publish(msg)
        
        # 성능 모니터링
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 5.0:  # 5초마다 로그 출력
            fps = self.frame_count / elapsed
            self.get_logger().info(f"발행 FPS: {fps:.1f}")
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def destroy_node(self):
        """노드 종료 시 정리"""
        if hasattr(self, 'cap'):
            self.cap.release()
        
        if self.use_cuda and self.gpu_frame is not None:
            self.gpu_frame.release()
            if self.stream is not None:
                self.stream.waitForCompletion()
        
        self.get_logger().info("카메라 Publisher 노드 종료")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    # 카메라 인덱스 (기본값: 0)
    camera_index = 0
    if len(sys.argv) > 1:
        try:
            camera_index = int(sys.argv[1])
        except ValueError:
            print(f"잘못된 카메라 인덱스: {sys.argv[1]}, 기본값 0 사용")
    
    node = CameraPublisherNode(camera_index)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
