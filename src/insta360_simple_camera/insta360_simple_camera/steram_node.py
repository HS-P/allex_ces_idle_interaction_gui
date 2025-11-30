#!/usr/bin/env python3
import os
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, CompressedImage
from builtin_interfaces.msg import Time
import numpy as np
import torch
import threading
import time
from queue import Queue


class Insta360SimpleCamera(Node):
    """
    Insta360를 UVC 웹캠처럼 열어서 Image 토픽만 퍼블리시하는 단순 드라이버.
    - SDK 없이 /dev/videoX만 사용
    - CUDA GPU 가속 지원 (PyTorch 사용)
    """

    def __init__(self):
        super().__init__('insta360_simple_camera')

        # CUDA 사용 가능 여부 확인 (PyTorch 사용) - 필수
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA를 사용할 수 없습니다. GPU가 필수입니다.')
        
        self.device = torch.device('cuda:0')
        self.get_logger().info(f'CUDA 사용 가능: {torch.cuda.device_count()}개 GPU')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            self.get_logger().info(f'  GPU {i}: {props.name} (메모리: {props.total_memory / 1024**3:.1f} GB)')
        self.get_logger().info(f'현재 GPU: {torch.cuda.get_device_name(0)}')
        
        # GPU 스트림 생성 (비동기 처리용)
        self.stream = torch.cuda.Stream()
        
        # GPU 메모리 버퍼 미리 할당 (최적화)
        self.gpu_frame_buffer = None  # 나중에 초기화

        # 파라미터 선언
        self.declare_parameter('device', '/dev/video0')   # 0번 카메라
        self.declare_parameter('width', 1920)
        self.declare_parameter('height', 1080)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('frame_id', 'insta360_camera')
        self.declare_parameter('topic_name', '/insta360/image_raw/compressed')
        self.declare_parameter('undistort_enabled', False)  # fisheye undistortion 비활성화 (원본 출력)
        self.declare_parameter('output_width', 1280)  # 출력 이미지 너비 (undistort 시에만 사용)
        self.declare_parameter('output_height', 720)  # 출력 이미지 높이 (undistort 시에만 사용)
        self.declare_parameter('fov_degree', 120.0)  # 출력 화각 (undistort 시에만 사용)

        device = self.get_parameter('device').get_parameter_value().string_value
        width = self.get_parameter('width').get_parameter_value().integer_value
        height = self.get_parameter('height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.undistort_enabled = self.get_parameter('undistort_enabled').get_parameter_value().bool_value
        self.output_width = self.get_parameter('output_width').get_parameter_value().integer_value
        self.output_height = self.get_parameter('output_height').get_parameter_value().integer_value
        self.fov_degree = self.get_parameter('fov_degree').get_parameter_value().double_value
        
        # Fisheye undistortion 매핑 테이블 미리 계산 (한 번만)
        self.map_x = None
        self.map_y = None
        if self.undistort_enabled:
            self._init_undistort_maps(height, self.output_width, self.output_height, self.fov_degree)

        # Publisher (CompressedImage 사용, BEST_EFFORT QoS)
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.pub = self.create_publisher(CompressedImage, topic_name, qos_profile)
        
        # GPU 버퍼 초기화
        self.gpu_frame_buffer = torch.zeros((3, height, width), dtype=torch.uint8, device=self.device)

        # VideoCapture 열기
        self.cap = self._open_capture(device, width, height)
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error(f'Failed to open video device: {device}')
            raise RuntimeError('Cannot open Insta360 UVC device')

        self.get_logger().info(
            f'Insta360SimpleCamera started on {device} '
            f'({width}x{height} @ {self.fps} FPS)'
        )

        # 스레드 간 통신용 큐 (압축된 JPEG 바이트 전달)
        self.frame_queue = Queue(maxsize=2)  # 최대 2개 프레임만 버퍼링
        
        # 스레드 동기화
        self.lock = threading.Lock()
        self.running = True
        
        # 디버깅용 통계
        self.capture_frame_count = 0
        self.publish_frame_count = 0
        self.last_debug_time = time.monotonic()
        self.capture_times = []  # 최근 30프레임의 처리 시간
        self.publish_times = []  # 최근 30프레임의 발행 시간
        
        # 별도 스레드에서 카메라 읽기 시작
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # 타이머: 큐에서 프레임을 가져와서 발행
        timer_period = 1.0 / max(self.fps, 1.0)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _init_undistort_maps(self, fisheye_size: int, out_w: int, out_h: int, fov_deg: float):
        """
        Fisheye to Rectilinear 매핑 테이블 생성 (Equidistant projection)
        
        Insta360 어안 렌즈는 equidistant projection 사용
        r = f * theta (theta: 입사각, r: 이미지 평면에서의 거리)
        """
        self.get_logger().info(f'Fisheye undistortion 매핑 테이블 생성 중... (출력: {out_w}x{out_h}, FOV: {fov_deg}°)')
        
        # 어안 이미지 중심과 반지름
        cx_fish = fisheye_size / 2
        cy_fish = fisheye_size / 2
        r_fish = fisheye_size / 2  # 어안 이미지 반지름
        
        # 출력 이미지 중심
        cx_out = out_w / 2
        cy_out = out_h / 2
        
        # 출력 이미지의 focal length (화각으로 계산)
        fov_rad = np.radians(fov_deg)
        f_out = (out_w / 2) / np.tan(fov_rad / 2)
        
        # 매핑 테이블 생성
        self.map_x = np.zeros((out_h, out_w), dtype=np.float32)
        self.map_y = np.zeros((out_h, out_w), dtype=np.float32)
        
        # 벡터화된 계산 (빠름)
        y_coords, x_coords = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
        
        # 출력 이미지 좌표를 정규화된 3D 방향으로 변환
        x_norm = (x_coords - cx_out) / f_out
        y_norm = (y_coords - cy_out) / f_out
        
        # 3D 공간에서의 각도 (구면 좌표)
        r_norm = np.sqrt(x_norm**2 + y_norm**2)
        theta = np.arctan(r_norm)  # 광축에서 벗어난 각도
        
        # Equidistant projection: r_fish = (2 * r_fish_max / pi) * theta
        # Insta360은 약 200도 FOV이므로 pi에 가까움
        r_fish_proj = (r_fish / (np.pi / 2)) * theta
        
        # 어안 이미지 좌표로 변환
        # r_norm이 0인 경우 처리
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(r_norm > 0, r_fish_proj / r_norm, 0)
        
        self.map_x = (scale * x_norm + cx_fish).astype(np.float32)
        self.map_y = (scale * y_norm + cy_fish).astype(np.float32)
        
        self.get_logger().info('Fisheye undistortion 매핑 테이블 생성 완료')

    def _open_capture(self, device, width, height):
        # device가 숫자면 index로, 아니면 문자열 path로 처리
        cap = None
        try:
            if isinstance(device, str) and device.isdigit():
                idx = int(device)
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            elif isinstance(device, str) and os.path.exists(device):
                cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            else:
                # 그냥 전달
                cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        except Exception as e:
            self.get_logger().error(f'Error opening capture: {e}')
            return None

        if cap is None or not cap.isOpened():
            return None

        # 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # FPS 설정
        cap.set(cv2.CAP_PROP_FPS, 30.0)
        
        # 버퍼 크기 설정 (최신 프레임만 읽기 위해 버퍼를 작게 설정)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return cap

    def _capture_loop(self):
        """별도 스레드에서 카메라 읽기 및 JPEG 인코딩 (최적화 버전)"""
        
        while self.running:
            if self.cap is None:
                time.sleep(0.01)
                continue
            
            loop_start = time.monotonic()
            
            # 카메라에서 1번만 읽기 (버퍼 크기를 1로 설정했으므로 최신 프레임)
            read_start = time.monotonic()
            ret, frame = self.cap.read()
            read_time = time.monotonic() - read_start
            
            if not ret or frame is None:
                print(f"[DEBUG] 카메라 읽기 실패")
                time.sleep(0.001)
                continue

            # Fisheye undistortion (듀얼 어안 렌즈에서 전방 렌즈만 평면으로 변환)
            if self.undistort_enabled and self.map_x is not None and self.map_y is not None:
                h, w = frame.shape[:2]
                # 전방 렌즈만 사용 (2880x1440 -> 1440x1440, 왼쪽 절반)
                front_eye = frame[:, :h]
                
                # Fisheye to Rectilinear 변환 (미리 계산된 매핑 테이블 사용)
                frame = cv2.remap(front_eye, self.map_x, self.map_y, 
                                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # CPU에서 직접 JPEG 인코딩 (GPU 전송 오버헤드 제거)
            try:
                encode_start = time.monotonic()
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]  # 품질 80%로 약간 낮춤
                result, encimg = cv2.imencode('.jpg', frame, encode_param)
                encode_time = time.monotonic() - encode_start
                
                if result:
                    jpeg_bytes = encimg.tobytes()
                    jpeg_size = len(jpeg_bytes)
                    
                    # 압축된 JPEG 바이트를 큐에 추가
                    queue_start = time.monotonic()
                    try:
                        self.frame_queue.put_nowait(jpeg_bytes)
                        queue_dropped = False
                    except:
                        # 큐가 가득 차면 오래된 프레임 제거하고 새 프레임 추가
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put_nowait(jpeg_bytes)
                            queue_dropped = True
                        except:
                            queue_dropped = True
                    queue_time = time.monotonic() - queue_start
                    
                    self.capture_frame_count += 1
                    total_time = time.monotonic() - loop_start
                    self.capture_times.append(total_time)
                    if len(self.capture_times) > 30:
                        self.capture_times.pop(0)
                    
                    # 1초마다 디버깅 정보 출력
                    current_time = time.monotonic()
                    if current_time - self.last_debug_time >= 1.0:
                        avg_capture_time = sum(self.capture_times) / len(self.capture_times) if self.capture_times else 0
                        capture_fps = 1.0 / avg_capture_time if avg_capture_time > 0 else 0
                        queue_size = self.frame_queue.qsize()
                        
                        print(f"[DEBUG CAPTURE] FPS: {capture_fps:.2f} | "
                              f"읽기: {read_time*1000:.1f}ms | "
                              f"JPEG인코딩: {encode_time*1000:.1f}ms ({jpeg_size/1024:.1f}KB) | "
                              f"큐추가: {queue_time*1000:.1f}ms | "
                              f"큐크기: {queue_size}/2 | "
                              f"드롭: {queue_dropped} | "
                              f"총처리: {total_time*1000:.1f}ms")
                        self.last_debug_time = current_time
                else:
                    print(f"[DEBUG] JPEG 인코딩 실패")
                
            except Exception as e:
                print(f"[DEBUG ERROR] 처리 실패: {e}")
                import traceback
                traceback.print_exc()
    
    def timer_callback(self):
        """메인 스레드에서 큐에서 압축된 프레임을 가져와서 발행"""
        publish_start = time.monotonic()
        try:
            # 큐에서 압축된 JPEG 바이트 가져오기 (블로킹 없이)
            queue_get_start = time.monotonic()
            jpeg_bytes = self.frame_queue.get_nowait()
            queue_get_time = time.monotonic() - queue_get_start
            
            # CompressedImage 메시지 생성 및 발행
            msg_create_start = time.monotonic()
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()  # type: Time
            msg.header.frame_id = self.frame_id
            msg.format = 'jpeg'
            msg.data = jpeg_bytes
            msg_create_time = time.monotonic() - msg_create_start

            publish_msg_start = time.monotonic()
            self.pub.publish(msg)
            publish_msg_time = time.monotonic() - publish_msg_start
            
            self.publish_frame_count += 1
            total_publish_time = time.monotonic() - publish_start
            self.publish_times.append(total_publish_time)
            if len(self.publish_times) > 30:
                self.publish_times.pop(0)
            
            # 1초마다 디버깅 정보 출력
            current_time = time.monotonic()
            if current_time - self.last_debug_time >= 1.0:
                avg_publish_time = sum(self.publish_times) / len(self.publish_times) if self.publish_times else 0
                publish_fps = 1.0 / avg_publish_time if avg_publish_time > 0 else 0
                
                print(f"[DEBUG PUBLISH] FPS: {publish_fps:.2f} | "
                      f"큐가져오기: {queue_get_time*1000:.1f}ms | "
                      f"메시지생성: {msg_create_time*1000:.1f}ms | "
                      f"발행: {publish_msg_time*1000:.1f}ms | "
                      f"총처리: {total_publish_time*1000:.1f}ms | "
                      f"총프레임(캡처/발행): {self.capture_frame_count}/{self.publish_frame_count}")
            
        except:
            # 큐가 비어있으면 무시 (다음 프레임 기다림)
            pass

    def destroy_node(self):
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Insta360SimpleCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
