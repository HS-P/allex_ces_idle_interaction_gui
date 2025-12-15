#!/usr/bin/env python3
"""
Testbed Gaze Tracking GUI Node - BB BOX 표시, PID GAIN 수정
"""
import json
import os
import sys
import threading
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap
from typing import Optional, List, Dict


def setup_qt_plugin_path():
    """Qt 플러그인 경로 설정 (PyQt5)"""
    try:
        # OpenCV가 설정한 경로 제거
        os.environ.pop("QT_PLUGIN_PATH", None)
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
        
        import PyQt5
        pyqt5_path = os.path.dirname(PyQt5.__file__)
        
        possible_paths = []
        pyqt5_plugin_path = os.path.join(pyqt5_path, 'Qt5', 'plugins')
        if os.path.exists(pyqt5_plugin_path):
            possible_paths.append(pyqt5_plugin_path)
        
        # 시스템 경로들 추가
        system_paths = [
            "/usr/lib/x86_64-linux-gnu/qt5/plugins",
            "/usr/lib/qt5/plugins",
            "/usr/local/lib/qt5/plugins",
        ]
        for path in system_paths:
            if os.path.exists(path):
                possible_paths.append(path)
        
        if 'CONDA_PREFIX' in os.environ:
            import sysconfig
            conda_prefix = os.environ['CONDA_PREFIX']
            python_version = sysconfig.get_python_version()
            conda_plugin_path = os.path.join(
                conda_prefix, 'lib', f'python{python_version}', 
                'site-packages', 'PyQt5', 'Qt5', 'plugins'
            )
            if os.path.exists(conda_plugin_path):
                possible_paths.insert(0, conda_plugin_path)
        
        plugin_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.exists(os.path.join(path, 'platforms')):
                plugin_path = path
                break
        
        if plugin_path:
            os.environ['QT_PLUGIN_PATH'] = plugin_path
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
            print(f"Qt 플러그인 경로 설정: {plugin_path}")
        else:
            print("경고: Qt 플러그인 경로를 찾을 수 없습니다.")
        
        return plugin_path
    except Exception as e:
        print(f"Qt 플러그인 경로 설정 실패 (무시됨): {e}")
        return None


class GuiSignals(QObject):
    """GUI 시그널"""
    update_image = pyqtSignal(np.ndarray)
    update_tracking = pyqtSignal(dict)


class TestbedGazeTrackingGuiNode(Node, QMainWindow):
    """Testbed Gaze Tracking GUI Node"""
    
    def __init__(self):
        Node.__init__(self, "testbed_gaze_tracking_gui_node")
        QMainWindow.__init__(self)
        
        # 시그널 생성
        self.signals = GuiSignals()
        self.signals.update_image.connect(self._update_image_display)
        self.signals.update_tracking.connect(self._update_tracking_info)
        
        # QoS 설정
        qos_profile = QoSProfile(
            depth=30,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        # 토픽 파라미터
        self.declare_parameter('camera_image_topic', '/camera/color/image_raw/compressed')
        self.declare_parameter('tracking_result_topic', '/allex_camera/tracking_result')  # FSM 노드의 tracking_result 구독
        self.declare_parameter('tracking_image_topic', '/allex_testbed/tracking_image')
        self.declare_parameter('pid_tune_topic', '/allex_testbed/pid_tune')
        self.declare_parameter('control_topic', '/allex_testbed/control')
        
        camera_image_topic = self.get_parameter('camera_image_topic').get_parameter_value().string_value
        tracking_result_topic = self.get_parameter('tracking_result_topic').get_parameter_value().string_value
        tracking_image_topic = self.get_parameter('tracking_image_topic').get_parameter_value().string_value
        pid_tune_topic = self.get_parameter('pid_tune_topic').get_parameter_value().string_value
        control_topic = self.get_parameter('control_topic').get_parameter_value().string_value
        
        # 이미지 구독 (원본 + Annotated)
        self.image_subscription = self.create_subscription(
            CompressedImage,
            camera_image_topic,
            self.image_callback,
            qos_profile
        )
        self.tracking_image_subscription = self.create_subscription(
            CompressedImage,
            tracking_image_topic,
            self.tracking_image_callback,
            qos_profile
        )
        
        # 추적 결과 구독
        self.tracking_result_subscription = self.create_subscription(
            String,
            tracking_result_topic,
            self.tracking_result_callback,
            10
        )
        
        # PID 튜닝 발행
        self.pid_tune_publisher = self.create_publisher(
            String,
            pid_tune_topic,
            10
        )
        
        # 제어 명령 발행
        self.control_publisher = self.create_publisher(
            String,
            control_topic,
            10
        )
        
        # 상태 변수
        self.current_frame: Optional[np.ndarray] = None
        self.tracked_objects: List[Dict] = []
        self.target_info: Optional[Dict] = None
        self.current_state = "idle"
        self.last_image_time = 0.0
        self.last_tracking_image_time = 0.0  # annotated 이미지 마지막 수신 시간
        self.last_tracking_time = 0.0
        self.image_count = 0
        self.tracking_count = 0
        self.use_annotated_image = False  # 초기값: 원본 이미지 사용 (annotated 이미지 수신 시 자동 전환)
        
        # GUI 초기화
        self.init_ui()
        
        # 주기적 업데이트 타이머
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._periodic_update)
        self.update_timer.start(50)  # 20Hz 업데이트
        
        self.get_logger().info("Testbed Gaze Tracking GUI Node 초기화 완료")
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Testbed Gaze Tracking GUI")
        self.setGeometry(100, 100, 1400, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 왼쪽: 이미지 표시
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # 이미지 라벨
        self.image_label = QLabel()
        self.image_label.setMinimumSize(1280, 720)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 2px solid gray;")
        self.image_label.setText("이미지 대기 중...")
        left_layout.addWidget(self.image_label)
        
        # 상태 정보
        self.status_label = QLabel("상태: IDLE")
        self.status_label.setStyleSheet("font-size: 14pt; font-weight: bold; padding: 10px;")
        left_layout.addWidget(self.status_label)
        
        main_layout.addWidget(left_panel, 2)
        
        # 오른쪽: 제어 패널
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # RUN 버튼
        run_group = QGroupBox("시스템 제어")
        run_layout = QVBoxLayout()
        
        self.run_btn = QPushButton("RUN")
        self.run_btn.setCheckable(True)
        self.run_btn.setChecked(False)
        self.run_btn.setMinimumHeight(60)
        self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0;")
        self.run_btn.clicked.connect(self.on_run_clicked)
        run_layout.addWidget(self.run_btn)
        
        run_group.setLayout(run_layout)
        right_layout.addWidget(run_group)
        
        # PID 게인 제어
        pid_group = QGroupBox("PID 게인 조정")
        pid_layout = QGridLayout()
        
        # Yaw PID
        pid_layout.addWidget(QLabel("Yaw Kp:"), 0, 0)
        self.kp_yaw_spin = QDoubleSpinBox()
        self.kp_yaw_spin.setRange(0.0, 10.0)
        self.kp_yaw_spin.setSingleStep(0.01)
        self.kp_yaw_spin.setValue(1.55)
        self.kp_yaw_spin.setDecimals(2)
        self.kp_yaw_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.kp_yaw_spin, 0, 1)
        
        pid_layout.addWidget(QLabel("Yaw Ki:"), 1, 0)
        self.ki_yaw_spin = QDoubleSpinBox()
        self.ki_yaw_spin.setRange(0.0, 5.0)
        self.ki_yaw_spin.setSingleStep(0.01)
        self.ki_yaw_spin.setValue(0.2)
        self.ki_yaw_spin.setDecimals(2)
        self.ki_yaw_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.ki_yaw_spin, 1, 1)
        
        pid_layout.addWidget(QLabel("Yaw Kd:"), 2, 0)
        self.kd_yaw_spin = QDoubleSpinBox()
        self.kd_yaw_spin.setRange(0.0, 1.0)
        self.kd_yaw_spin.setSingleStep(0.001)
        self.kd_yaw_spin.setValue(0.02)
        self.kd_yaw_spin.setDecimals(3)
        self.kd_yaw_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.kd_yaw_spin, 2, 1)
        
        # Pitch PID
        pid_layout.addWidget(QLabel("Pitch Kp:"), 3, 0)
        self.kp_pitch_spin = QDoubleSpinBox()
        self.kp_pitch_spin.setRange(0.0, 10.0)
        self.kp_pitch_spin.setSingleStep(0.01)
        self.kp_pitch_spin.setValue(1.2)
        self.kp_pitch_spin.setDecimals(2)
        self.kp_pitch_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.kp_pitch_spin, 3, 1)
        
        pid_layout.addWidget(QLabel("Pitch Ki:"), 4, 0)
        self.ki_pitch_spin = QDoubleSpinBox()
        self.ki_pitch_spin.setRange(0.0, 5.0)
        self.ki_pitch_spin.setSingleStep(0.01)
        self.ki_pitch_spin.setValue(0.1)
        self.ki_pitch_spin.setDecimals(2)
        self.ki_pitch_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.ki_pitch_spin, 4, 1)
        
        pid_layout.addWidget(QLabel("Pitch Kd:"), 5, 0)
        self.kd_pitch_spin = QDoubleSpinBox()
        self.kd_pitch_spin.setRange(0.0, 1.0)
        self.kd_pitch_spin.setSingleStep(0.001)
        self.kd_pitch_spin.setValue(0.1)
        self.kd_pitch_spin.setDecimals(3)
        self.kd_pitch_spin.valueChanged.connect(self.on_pid_changed)
        pid_layout.addWidget(self.kd_pitch_spin, 5, 1)
        
        # Integral 리셋 버튼
        reset_btn = QPushButton("Integral 리셋")
        reset_btn.setMinimumHeight(40)
        reset_btn.clicked.connect(self.on_reset_integral)
        pid_layout.addWidget(reset_btn, 6, 0, 1, 2)
        
        pid_group.setLayout(pid_layout)
        right_layout.addWidget(pid_group)
        
        # 추적 정보
        info_group = QGroupBox("추적 정보")
        info_layout = QVBoxLayout()
        
        self.tracking_info_label = QLabel("타겟 ID: --\n상태: IDLE")
        self.tracking_info_label.setStyleSheet("font-size: 12pt; padding: 10px;")
        info_layout.addWidget(self.tracking_info_label)
        
        # 노드 상태 표시
        self.node_status_label = QLabel("노드 상태: 대기 중")
        self.node_status_label.setStyleSheet("font-size: 11pt; padding: 5px; color: orange;")
        info_layout.addWidget(self.node_status_label)
        
        # 이미지 수신 상태
        self.image_status_label = QLabel("이미지: 수신 대기 중")
        self.image_status_label.setStyleSheet("font-size: 10pt; padding: 5px; color: gray;")
        info_layout.addWidget(self.image_status_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        right_layout.addStretch()
        main_layout.addWidget(right_panel, 1)
    
    
    def image_callback(self, msg: CompressedImage):
        """원본 이미지 콜백 (Annotated 미수신 시 사용)"""
        # annotated 이미지를 사용하는 경우 원본 이미지는 무시
        if self.use_annotated_image:
            return
        
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.current_frame = frame
                self.last_image_time = time.time()
                self.image_count += 1
                # BB BOX 그리기 (tracking_result가 있으면)
                if self.tracked_objects:
                    self._draw_bboxes()
        except Exception as e:
            self.get_logger().error(f"이미지 처리 실패: {e}")

    def tracking_image_callback(self, msg: CompressedImage):
        """Annotated 이미지 콜백"""
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                self.current_frame = frame
                self.last_image_time = time.time()
                self.last_tracking_image_time = time.time()  # annotated 이미지 수신 시간 업데이트
                self.image_count += 1
                self.use_annotated_image = True  # annotated 이미지 수신 시 활성화
                
                # Annotated 이미지에 추가 정보 그리기 (타겟 포인트 등)
                frame_with_extras = self._draw_extra_info(frame.copy())
                # Annotated 이미지는 바로 표시 (BB BOX는 이미 그려져 있음)
                self.signals.update_image.emit(frame_with_extras)
        except Exception as e:
            self.get_logger().error(f"Tracking 이미지 처리 실패: {e}")
    
    def tracking_result_callback(self, msg: String):
        """추적 결과 콜백"""
        try:
            data = json.loads(msg.data)
            self.current_state = data.get('state', 'idle')
            self.target_info = data.get('target_info', {})
            self.tracked_objects = data.get('tracked_objects', [])
            self.last_tracking_time = time.time()
            self.tracking_count += 1
            
            # 시그널로 전달 (메인 스레드에서 처리)
            self.signals.update_tracking.emit(data)
            
            # 이미지 업데이트
            if self.current_frame is not None:
                if self.use_annotated_image:
                    # annotated 이미지에 타겟 포인트만 추가로 그려서 업데이트
                    frame_with_extras = self._draw_extra_info(self.current_frame.copy())
                    self.signals.update_image.emit(frame_with_extras)
                else:
                    # 원본 이미지에 BB BOX 그리기
                    self._draw_bboxes()
        except Exception as e:
            self.get_logger().error(f"추적 결과 파싱 실패: {e}")
    
    def _draw_extra_info(self, frame: np.ndarray) -> np.ndarray:
        """추가 정보 그리기 (타겟 포인트 등)"""
        # 타겟 포인트 표시
        if self.target_info and self.target_info.get('point'):
            point = self.target_info['point']
            if len(point) == 2:
                x, y = map(int, point)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                cv2.circle(frame, (x, y), 15, (0, 0, 255), 2)
        return frame
    
    def _draw_bboxes(self):
        """BB BOX 그리기"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # 모든 객체 BB BOX 그리기
        for obj in self.tracked_objects:
            bbox = obj.get('bbox', [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                track_id = obj.get('track_id', -1)
                conf = obj.get('confidence', 0.0)
                
                # 타겟인지 확인
                is_target = (self.target_info and 
                           self.target_info.get('track_id') == track_id)
                
                color = (0, 0, 255) if is_target else (0, 255, 0)
                thickness = 3 if is_target else 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # 라벨
                label = f"ID:{track_id} ({conf:.2f})"
                if is_target:
                    label = f"TARGET: {label}"
                
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 추가 정보 그리기
        frame = self._draw_extra_info(frame)
        
        # 이미지 업데이트
        self.signals.update_image.emit(frame)
    
    def _update_image_display(self, frame: np.ndarray):
        """이미지 표시 업데이트 (메인 스레드)"""
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
    
    def _update_tracking_info(self, data: dict):
        """추적 정보 업데이트 (메인 스레드)"""
        state = data.get('state', 'idle')
        target_info = data.get('target_info', {})
        target_id = target_info.get('track_id', '--')
        
        self.status_label.setText(f"상태: {state.upper()}")
        
        info_text = f"타겟 ID: {target_id}\n"
        info_text += f"상태: {state.upper()}\n"
        info_text += f"객체 수: {len(self.tracked_objects)}"
        
        self.tracking_info_label.setText(info_text)
    
    def _periodic_update(self):
        """주기적 업데이트 (상태 표시 등)"""
        current_time = time.time()
        
        # Annotated 이미지가 일정 시간(2초) 동안 수신되지 않으면 원본 이미지로 전환
        if self.use_annotated_image and self.last_tracking_image_time > 0:
            elapsed_annotated = current_time - self.last_tracking_image_time
            if elapsed_annotated > 2.0:  # 2초 동안 annotated 이미지 미수신
                self.use_annotated_image = False
                self.get_logger().info("Annotated 이미지 미수신: 원본 이미지로 전환")
        
        # 이미지 수신 상태 업데이트
        if self.last_image_time > 0:
            elapsed = current_time - self.last_image_time
            if elapsed < 1.0:
                image_source = "Annotated" if self.use_annotated_image else "원본"
                self.image_status_label.setText(f"이미지: 수신 중 ({image_source}, {self.image_count} 프레임)")
                self.image_status_label.setStyleSheet("font-size: 10pt; padding: 5px; color: green;")
            else:
                self.image_status_label.setText(f"이미지: 수신 없음 ({elapsed:.1f}초 전)")
                self.image_status_label.setStyleSheet("font-size: 10pt; padding: 5px; color: red;")
        else:
            self.image_status_label.setText("이미지: 수신 대기 중")
            self.image_status_label.setStyleSheet("font-size: 10pt; padding: 5px; color: gray;")
        
        # 추적 결과 수신 상태 업데이트
        if self.last_tracking_time > 0:
            elapsed = current_time - self.last_tracking_time
            if elapsed < 1.0:
                self.node_status_label.setText(f"노드 상태: 동작 중 (추적: {self.tracking_count}회)")
                self.node_status_label.setStyleSheet("font-size: 11pt; padding: 5px; color: green;")
            else:
                self.node_status_label.setText(f"노드 상태: 응답 없음 ({elapsed:.1f}초 전)")
                self.node_status_label.setStyleSheet("font-size: 11pt; padding: 5px; color: orange;")
        else:
            self.node_status_label.setText("노드 상태: 대기 중")
            self.node_status_label.setStyleSheet("font-size: 11pt; padding: 5px; color: orange;")
    
    def on_run_clicked(self):
        """RUN 버튼 클릭"""
        if self.run_btn.isChecked():
            msg = String()
            msg.data = json.dumps({'type': 'run'})
            self.control_publisher.publish(msg)
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #90EE90;")
            self.get_logger().info("RUN 시작")
        else:
            msg = String()
            msg.data = json.dumps({'type': 'stop'})
            self.control_publisher.publish(msg)
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0;")
            self.get_logger().info("RUN 중지")
    
    def on_pid_changed(self):
        """PID 게인 변경"""
        data = {
            'kp_yaw': float(self.kp_yaw_spin.value()),
            'kp_pitch': float(self.kp_pitch_spin.value()),
            'ki_yaw': float(self.ki_yaw_spin.value()),
            'ki_pitch': float(self.ki_pitch_spin.value()),
            'kd_yaw': float(self.kd_yaw_spin.value()),
            'kd_pitch': float(self.kd_pitch_spin.value())
        }
        
        msg = String()
        msg.data = json.dumps(data)
        self.pid_tune_publisher.publish(msg)
    
    def on_reset_integral(self):
        """Integral 리셋"""
        data = {'reset_integral': True}
        msg = String()
        msg.data = json.dumps(data)
        self.pid_tune_publisher.publish(msg)
        self.get_logger().info("Integral 리셋")
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        self.update_timer.stop()
        event.accept()


def main(args=None):
    """메인 함수"""
    try:
        setup_qt_plugin_path()
    except Exception as e:
        print(f"경고: Qt 플러그인 경로 설정 실패: {e}")
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    rclpy.init(args=args)
    node = TestbedGazeTrackingGuiNode()
    node.show()
    
    try:
        ros_thread_running = True
        
        def ros_spin():
            nonlocal ros_thread_running
            while ros_thread_running:
                rclpy.spin_once(node, timeout_sec=0.1)
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        
        app.exec_()
        ros_thread_running = False
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

