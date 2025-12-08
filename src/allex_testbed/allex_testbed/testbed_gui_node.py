#!/usr/bin/env python3
"""
테스트베드 GUI 노드
- PID 게인 실시간 튜닝
- 목표 명령 직접 설정
- 상태 모니터링
- 스무딩 파라미터 조정
"""
import json
import sys
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QPushButton, 
                               QDoubleSpinBox, QGroupBox, QTextEdit, QTabWidget)
from PySide6.QtCore import Qt, QTimer


class TestbedGUIWindow(QMainWindow):
    """테스트베드 GUI 윈도우"""
    
    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.setWindowTitle("ALLEX Testbed - PID 튜닝 및 제어")
        self.setGeometry(100, 100, 1000, 800)
        
        # 중앙 위젯
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # RUN/STOP 버튼 (전역 - 탭 위에 표시)
        control_group = QGroupBox("제어")
        control_layout = QHBoxLayout()
        
        self.run_btn = QPushButton("RUN")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._start_controller)
        control_layout.addWidget(self.run_btn)
        
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        self.stop_btn.clicked.connect(self._stop_controller)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)
        
        # 탭 위젯
        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        
        # 탭 1: PID 튜닝
        pid_tab = self._create_pid_tuning_tab()
        tabs.addTab(pid_tab, "PID 튜닝")
        
        # 탭 2: 목표 명령
        target_tab = self._create_target_command_tab()
        tabs.addTab(target_tab, "목표 명령")
        
        # 탭 3: 상태 모니터링
        status_tab = self._create_status_monitor_tab()
        tabs.addTab(status_tab, "상태 모니터링")
        
        # 상태 구독
        self.status_subscription = node.create_subscription(
            String,
            "/allex_testbed/status",
            self._status_callback,
            10
        )
        
        # PID 튜닝 Publisher
        self.pid_tune_publisher = node.create_publisher(
            String,
            "/allex_testbed/pid_tune",
            10
        )
        
        # 목표 명령 Publisher
        self.target_command_publisher = node.create_publisher(
            String,
            "/allex_testbed/target_command",
            10
        )
        
        # Controller 제어 명령 Publisher (RUN/STOP용)
        self.controller_control_publisher = node.create_publisher(
            String,
            "/allex_camera/controller_control",
            10
        )
        
        # 현재 상태 저장
        self.current_status = {}
        
        # 타이머로 상태 업데이트
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(100)  # 10Hz
    
    def _create_pid_tuning_tab(self):
        """PID 튜닝 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 목 PID 게인
        neck_group = QGroupBox("목 PID 게인")
        neck_layout = QVBoxLayout()
        
        # Yaw PID
        yaw_group = QGroupBox("Yaw (좌우)")
        yaw_layout = QVBoxLayout()
        
        self.kp_yaw_slider = self._create_slider_with_spinbox("Kp", 0.0, 5.0, 1.0, yaw_layout)
        self.ki_yaw_slider = self._create_slider_with_spinbox("Ki", 0.0, 2.0, 0.0, yaw_layout)
        self.kd_yaw_slider = self._create_slider_with_spinbox("Kd", 0.0, 1.0, 0.0, yaw_layout)
        
        yaw_group.setLayout(yaw_layout)
        neck_layout.addWidget(yaw_group)
        
        # Pitch PID
        pitch_group = QGroupBox("Pitch (상하)")
        pitch_layout = QVBoxLayout()
        
        self.kp_pitch_slider = self._create_slider_with_spinbox("Kp", 0.0, 5.0, 1.0, pitch_layout)
        self.ki_pitch_slider = self._create_slider_with_spinbox("Ki", 0.0, 2.0, 0.0, pitch_layout)
        self.kd_pitch_slider = self._create_slider_with_spinbox("Kd", 0.0, 1.0, 0.0, pitch_layout)
        
        pitch_group.setLayout(pitch_layout)
        neck_layout.addWidget(pitch_group)
        
        neck_group.setLayout(neck_layout)
        layout.addWidget(neck_group)
        
        # 허리 PID 게인
        waist_group = QGroupBox("허리 PID 게인")
        waist_layout = QVBoxLayout()
        
        self.kp_waist_slider = self._create_slider_with_spinbox("Kp", 0.0, 2.0, 0.5, waist_layout)
        self.ki_waist_slider = self._create_slider_with_spinbox("Ki", 0.0, 1.0, 0.0, waist_layout)
        self.kd_waist_slider = self._create_slider_with_spinbox("Kd", 0.0, 0.5, 0.0, waist_layout)
        
        waist_group.setLayout(waist_layout)
        layout.addWidget(waist_group)
        
        # 스무딩 파라미터
        smoothing_group = QGroupBox("스무딩")
        smoothing_layout = QVBoxLayout()
        
        self.smoothing_slider = self._create_slider_with_spinbox("Smoothing Factor", 0.0, 1.0, 1.0, smoothing_layout)
        smoothing_group.setLayout(smoothing_layout)
        layout.addWidget(smoothing_group)
        
        # 적용 버튼 (슬라이더가 자동 적용되므로 선택사항)
        apply_btn = QPushButton("PID 게인 수동 적용")
        apply_btn.clicked.connect(self._apply_pid_gains)
        layout.addWidget(apply_btn)
        
        info_label = QLabel("※ 슬라이더/스핀박스 값 변경 시 자동 적용됩니다")
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)
        
        # 리셋 버튼
        reset_btn = QPushButton("PID 상태 리셋")
        reset_btn.clicked.connect(self._reset_pid)
        layout.addWidget(reset_btn)
        
        layout.addStretch()
        
        return widget
    
    def _create_slider_with_spinbox(self, label_text: str, min_val: float, max_val: float, 
                                    default_val: float, parent_layout):
        """슬라이더와 스핀박스를 함께 생성"""
        h_layout = QHBoxLayout()
        
        label = QLabel(label_text)
        label.setMinimumWidth(80)
        h_layout.addWidget(label)
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(int(min_val * 1000))
        slider.setMaximum(int(max_val * 1000))
        slider.setValue(int(default_val * 1000))
        h_layout.addWidget(slider)
        
        spinbox = QDoubleSpinBox()
        spinbox.setMinimum(min_val)
        spinbox.setMaximum(max_val)
        spinbox.setSingleStep(0.01)
        spinbox.setValue(default_val)
        spinbox.setDecimals(3)
        h_layout.addWidget(spinbox)
        
        # 슬라이더와 스핀박스 동기화 및 자동 적용
        def slider_changed(value):
            spinbox.blockSignals(True)  # 무한 루프 방지
            spinbox.setValue(value / 1000.0)
            spinbox.blockSignals(False)
            # 슬라이더 값 변경 시 자동으로 PID 게인 적용
            self._apply_pid_gains()
        
        def spinbox_changed(value):
            slider.blockSignals(True)  # 무한 루프 방지
            slider.setValue(int(value * 1000))
            slider.blockSignals(False)
            # 스핀박스 값 변경 시 자동으로 PID 게인 적용
            self._apply_pid_gains()
        
        slider.valueChanged.connect(slider_changed)
        spinbox.valueChanged.connect(spinbox_changed)
        
        parent_layout.addLayout(h_layout)
        
        return {'slider': slider, 'spinbox': spinbox}
    
    def _create_target_command_tab(self):
        """목표 명령 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 목 각도 명령
        neck_group = QGroupBox("목 각도 명령 (도)")
        neck_layout = QVBoxLayout()
        
        yaw_layout = QHBoxLayout()
        yaw_layout.addWidget(QLabel("Yaw:"))
        self.target_yaw_spinbox = QDoubleSpinBox()
        self.target_yaw_spinbox.setRange(-80.0, 80.0)
        self.target_yaw_spinbox.setSingleStep(1.0)
        self.target_yaw_spinbox.setValue(0.0)
        yaw_layout.addWidget(self.target_yaw_spinbox)
        neck_layout.addLayout(yaw_layout)
        
        pitch_layout = QHBoxLayout()
        pitch_layout.addWidget(QLabel("Pitch:"))
        self.target_pitch_spinbox = QDoubleSpinBox()
        self.target_pitch_spinbox.setRange(-5.0, 215.0)
        self.target_pitch_spinbox.setSingleStep(1.0)
        self.target_pitch_spinbox.setValue(0.0)
        pitch_layout.addWidget(self.target_pitch_spinbox)
        neck_layout.addLayout(pitch_layout)
        
        neck_group.setLayout(neck_layout)
        layout.addWidget(neck_group)
        
        # 허리 각도 명령
        waist_group = QGroupBox("허리 각도 명령 (도)")
        waist_layout = QVBoxLayout()
        
        waist_yaw_layout = QHBoxLayout()
        waist_yaw_layout.addWidget(QLabel("Waist Yaw:"))
        self.target_waist_yaw_spinbox = QDoubleSpinBox()
        self.target_waist_yaw_spinbox.setRange(-85.0, 85.0)
        self.target_waist_yaw_spinbox.setSingleStep(1.0)
        self.target_waist_yaw_spinbox.setValue(0.0)
        waist_yaw_layout.addWidget(self.target_waist_yaw_spinbox)
        waist_layout.addLayout(waist_yaw_layout)
        
        waist_group.setLayout(waist_layout)
        layout.addWidget(waist_group)
        
        # 버튼들
        btn_layout = QHBoxLayout()
        
        set_target_btn = QPushButton("목표 설정")
        set_target_btn.clicked.connect(self._set_target_command)
        btn_layout.addWidget(set_target_btn)
        
        reset_target_btn = QPushButton("0도로 이동")
        reset_target_btn.clicked.connect(self._reset_target)
        btn_layout.addWidget(reset_target_btn)
        
        preset_40_btn = QPushButton("40도로 이동 (테스트)")
        preset_40_btn.clicked.connect(self._preset_40_degrees)
        btn_layout.addWidget(preset_40_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        
        return widget
    
    def _create_status_monitor_tab(self):
        """상태 모니터링 탭 생성"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # 현재 각도 표시
        current_group = QGroupBox("현재 각도")
        current_layout = QVBoxLayout()
        
        self.current_yaw_label = QLabel("Yaw: 0.0°")
        self.current_pitch_label = QLabel("Pitch: 0.0°")
        self.current_waist_label = QLabel("Waist Yaw: 0.0°")
        
        current_layout.addWidget(self.current_yaw_label)
        current_layout.addWidget(self.current_pitch_label)
        current_layout.addWidget(self.current_waist_label)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # 목표 각도 표시
        target_group = QGroupBox("목표 각도")
        target_layout = QVBoxLayout()
        
        self.target_yaw_label = QLabel("Yaw: 0.0°")
        self.target_pitch_label = QLabel("Pitch: 0.0°")
        self.target_waist_label = QLabel("Waist Yaw: 0.0°")
        
        target_layout.addWidget(self.target_yaw_label)
        target_layout.addWidget(self.target_pitch_label)
        target_layout.addWidget(self.target_waist_label)
        
        target_group.setLayout(target_layout)
        layout.addWidget(target_group)
        
        # 오차 표시
        error_group = QGroupBox("오차")
        error_layout = QVBoxLayout()
        
        self.error_yaw_label = QLabel("Yaw Error: 0.0°")
        self.error_pitch_label = QLabel("Pitch Error: 0.0°")
        self.error_waist_label = QLabel("Waist Error: 0.0°")
        
        error_layout.addWidget(self.error_yaw_label)
        error_layout.addWidget(self.error_pitch_label)
        error_layout.addWidget(self.error_waist_label)
        
        error_group.setLayout(error_layout)
        layout.addWidget(error_group)
        
        # 로그
        log_group = QGroupBox("로그")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        return widget
    
    def _apply_pid_gains(self):
        """PID 게인 적용"""
        # 슬라이더가 아직 생성되지 않았을 수 있음 (초기화 중)
        if not hasattr(self, 'kp_yaw_slider'):
            return
        
        data = {
            'kp_yaw': self.kp_yaw_slider['spinbox'].value(),
            'kp_pitch': self.kp_pitch_slider['spinbox'].value(),
            'ki_yaw': self.ki_yaw_slider['spinbox'].value(),
            'ki_pitch': self.ki_pitch_slider['spinbox'].value(),
            'kd_yaw': self.kd_yaw_slider['spinbox'].value(),
            'kd_pitch': self.kd_pitch_slider['spinbox'].value(),
            'kp_waist_yaw': self.kp_waist_slider['spinbox'].value(),
            'ki_waist_yaw': self.ki_waist_slider['spinbox'].value(),
            'kd_waist_yaw': self.kd_waist_slider['spinbox'].value(),
            'smoothing_factor': self.smoothing_slider['spinbox'].value(),
        }
        
        msg = String()
        msg.data = json.dumps(data)
        self.pid_tune_publisher.publish(msg)
        
        self._log(f"PID 게인 자동 적용: Kp_yaw={data['kp_yaw']:.3f}, Kp_pitch={data['kp_pitch']:.3f}")
    
    def _reset_pid(self):
        """PID 상태 리셋"""
        data = {'type': 'reset'}
        msg = String()
        msg.data = json.dumps(data)
        self.target_command_publisher.publish(msg)
        
        self._log("PID 상태 리셋")
    
    def _set_target_command(self):
        """목표 명령 설정"""
        data = {
            'type': 'set_target',
            'yaw_deg': self.target_yaw_spinbox.value(),
            'pitch_deg': self.target_pitch_spinbox.value(),
            'waist_yaw_deg': self.target_waist_yaw_spinbox.value(),
        }
        
        msg = String()
        msg.data = json.dumps(data)
        self.target_command_publisher.publish(msg)
        
        self._log(f"목표 명령 설정: Yaw={data['yaw_deg']:.1f}°, Pitch={data['pitch_deg']:.1f}°")
    
    def _reset_target(self):
        """0도로 이동"""
        self.target_yaw_spinbox.setValue(0.0)
        self.target_pitch_spinbox.setValue(0.0)
        self.target_waist_yaw_spinbox.setValue(0.0)
        self._set_target_command()
    
    def _preset_40_degrees(self):
        """40도로 이동 (SEARCHING 테스트용)"""
        self.target_yaw_spinbox.setValue(40.0)
        self.target_pitch_spinbox.setValue(0.0)
        self._set_target_command()
    
    def _start_controller(self):
        """Controller 시작"""
        data = {'type': 'run'}
        msg = String()
        msg.data = json.dumps(data)
        self.controller_control_publisher.publish(msg)
        self._log("Controller RUN 명령 전송")
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
    
    def _stop_controller(self):
        """Controller 중지"""
        data = {'type': 'stop'}
        msg = String()
        msg.data = json.dumps(data)
        self.controller_control_publisher.publish(msg)
        self._log("Controller STOP 명령 전송")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
    
    def _status_callback(self, msg: String):
        """상태 콜백"""
        try:
            self.current_status = json.loads(msg.data)
        except Exception as e:
            self.node.get_logger().error(f"상태 파싱 실패: {e}")
    
    def _update_display(self):
        """화면 업데이트"""
        if not self.current_status:
            return
        
        # 현재 각도
        current_yaw = math.degrees(self.current_status.get('current_yaw_rad', 0.0))
        current_pitch = math.degrees(self.current_status.get('current_pitch_rad', 0.0))
        current_waist = math.degrees(self.current_status.get('current_waist_yaw_rad', 0.0))
        
        self.current_yaw_label.setText(f"Yaw: {current_yaw:.2f}°")
        self.current_pitch_label.setText(f"Pitch: {current_pitch:.2f}°")
        self.current_waist_label.setText(f"Waist Yaw: {current_waist:.2f}°")
        
        # 목표 각도
        target_yaw = math.degrees(self.current_status.get('target_yaw_rad', 0.0))
        target_pitch = math.degrees(self.current_status.get('target_pitch_rad', 0.0))
        target_waist = math.degrees(self.current_status.get('target_waist_yaw_rad', 0.0))
        
        self.target_yaw_label.setText(f"Yaw: {target_yaw:.2f}°")
        self.target_pitch_label.setText(f"Pitch: {target_pitch:.2f}°")
        self.target_waist_label.setText(f"Waist Yaw: {target_waist:.2f}°")
        
        # 오차
        error_yaw = target_yaw - current_yaw
        error_pitch = target_pitch - current_pitch
        error_waist = target_waist - current_waist
        
        self.error_yaw_label.setText(f"Yaw Error: {error_yaw:.2f}°")
        self.error_pitch_label.setText(f"Pitch Error: {error_pitch:.2f}°")
        self.error_waist_label.setText(f"Waist Error: {error_waist:.2f}°")
        
        # PID 게인 업데이트 (상태에서 받은 값으로)
        pid_gains = self.current_status.get('pid_gains', {})
        if pid_gains:
            self.kp_yaw_slider['spinbox'].setValue(pid_gains.get('kp_yaw', 1.0))
            self.kp_pitch_slider['spinbox'].setValue(pid_gains.get('kp_pitch', 1.0))
            self.ki_yaw_slider['spinbox'].setValue(pid_gains.get('ki_yaw', 0.0))
            self.ki_pitch_slider['spinbox'].setValue(pid_gains.get('ki_pitch', 0.0))
            self.kd_yaw_slider['spinbox'].setValue(pid_gains.get('kd_yaw', 0.0))
            self.kd_pitch_slider['spinbox'].setValue(pid_gains.get('kd_pitch', 0.0))
            self.kp_waist_slider['spinbox'].setValue(pid_gains.get('kp_waist_yaw', 0.5))
            self.ki_waist_slider['spinbox'].setValue(pid_gains.get('ki_waist_yaw', 0.0))
            self.kd_waist_slider['spinbox'].setValue(pid_gains.get('kd_waist_yaw', 0.0))
        
        smoothing = self.current_status.get('smoothing_factor', 1.0)
        self.smoothing_slider['spinbox'].setValue(smoothing)
    
    def _log(self, message: str):
        """로그 추가"""
        self.log_text.append(f"[{self.node.get_clock().now().to_msg().sec}] {message}")


class TestbedGUINode(Node):
    """테스트베드 GUI 노드"""
    
    def __init__(self):
        super().__init__('testbed_gui_node')
        
        # GUI 애플리케이션 생성
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()
        
        # GUI 윈도우 생성
        self.window = TestbedGUIWindow(self)
        self.window.show()
        
        # ROS2 타이머로 GUI 이벤트 처리
        self.gui_timer = self.create_timer(0.01, self._process_gui_events)
        
        self.get_logger().info("Testbed GUI Node 초기화 완료")
    
    def _process_gui_events(self):
        """GUI 이벤트 처리"""
        self.app.processEvents()


def main(args=None):
    """메인 함수"""
    rclpy.init(args=args)
    node = TestbedGUINode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

