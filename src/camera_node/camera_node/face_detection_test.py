#!/usr/bin/env python3
"""
YOLOv11n Face Detection 테스트 노드
- /camera/color/image_raw/compressed 토픽 구독
- 얼굴 검출 및 시각화
- ESC 키로 종료
"""

import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from ultralytics import YOLO
from huggingface_hub import hf_hub_download


class FaceDetectionTestNode(Node):
    def __init__(self):
        super().__init__('face_detection_test')
        
        # 모델 로드
        face_model_path = hf_hub_download(
            repo_id="AdamCodd/YOLOv11n-face-detection", 
            filename="model.pt"
        )
        self.model = YOLO(face_model_path)
        
        # 이미지 구독
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            qos_profile
        )
        
        # FPS 계산
        self.fps_start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self.latest_frame = None
        
        # OpenCV 윈도우
        cv2.namedWindow("Face Detection Test", cv2.WINDOW_NORMAL)
        
        # 디스플레이 타이머
        self.display_timer = self.create_timer(1.0 / 30.0, self.display_callback)
    
    def image_callback(self, msg: CompressedImage) -> None:
        """이미지 수신 및 얼굴 검출"""
        # JPEG 디코딩
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        # 얼굴 검출
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        # 결과 시각화
        annotated_frame = frame.copy()
        face_count = 0
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                # 바운딩 박스
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 라벨
                label = f"{conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                face_count += 1
        
        # FPS 계산
        self.frame_count += 1
        if time.time() - self.fps_start_time > 1.0:
            self.fps = self.frame_count / (time.time() - self.fps_start_time)
            self.frame_count = 0
            self.fps_start_time = time.time()
        
        # 정보 표시
        info_text = f"Faces: {face_count} | FPS: {self.fps:.1f}"
        cv2.putText(annotated_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        self.latest_frame = annotated_frame
    
    def display_callback(self) -> None:
        """화면 업데이트"""
        if self.latest_frame is not None:
            cv2.imshow("Face Detection Test", self.latest_frame)
        
        # ESC 키로 종료
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            rclpy.shutdown()
    
    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionTestNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
