#!/usr/bin/env python3
"""
GUI Node - GUI를 관리하고 여러 Topic을 동적으로 구독하는 통합 Node
v1.10.2 - PySide6 사용

시스템 구조:
- SPARK 1 PC: Camera Publisher (카메라 + YOLO 추적)
- Laptop: GUI (이 노드)
"""
import os
import sys
import json
import math
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from collections import namedtuple

os.environ.pop("QT_PLUGIN_PATH", None)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, Duration
from std_msgs.msg import String, Float64MultiArray
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QGroupBox, QGridLayout, QScrollArea,
    QProgressBar, QLineEdit, QDoubleSpinBox, QStackedWidget
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QCoreApplication
from PySide6.QtGui import QMouseEvent

from .tracking_fsm_node import TrackingState

# TrackedObject를 위한 간단한 구조체
TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'centroid', 'state', 'confidence', 'age'
])
TargetInfo = namedtuple('TargetInfo', [
    'point', 'state', 'track_id'
])

# CLIP 분류 라벨
CLIP_LABELS = ("handshake", "highfive", "fist", "idle")
CLIP_LABEL_COLORS = {
    "handshake": "#ef5350", 
    "highfive": "#ffa726", 
    "fist": "#42a5f5", 
    "idle": "#78909c"
}
CLIP_LABEL_ICONS = {"handshake": "🤝", "highfive": "🙌", "fist": "👊", "idle": "😐"}


def setup_qt_plugin_path():
    """Qt 플러그인 경로 설정 (PySide6)"""
    try:
        import PySide6
        pyside6_path = os.path.dirname(PySide6.__file__)
        
        possible_paths = []
        pyside6_plugin_path = os.path.join(pyside6_path, 'Qt', 'plugins')
        if os.path.exists(pyside6_plugin_path):
            possible_paths.append(pyside6_plugin_path)
        
        if 'CONDA_PREFIX' in os.environ:
            import sysconfig
            conda_prefix = os.environ['CONDA_PREFIX']
            python_version = sysconfig.get_python_version()
            conda_plugin_path = os.path.join(
                conda_prefix, 'lib', f'python{python_version}', 
                'site-packages', 'PySide6', 'Qt', 'plugins'
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
        
        return plugin_path
    except Exception as e:
        print(f"Qt 플러그인 경로 설정 실패 (무시됨): {e}")
        return None


class TargetButton(QPushButton):
    """타겟 변경 버튼 - 클릭 이벤트 직접 처리"""
    clicked_with_id = Signal(int)
    
    def __init__(self, track_id: int, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self.setCheckable(False)
        self.setEnabled(True)
        if parent and hasattr(parent, '_on_target_button_clicked'):
            self.clicked_with_id.connect(parent._on_target_button_clicked)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked_with_id.emit(self.track_id)
        super().mousePressEvent(event)


class GuiSignals(QObject):
    """GUI와 메인 쓰레드 간 통신을 위한 시그널"""
    mode_changed = Signal(bool)
    state_changed = Signal(str)
    update_target_buttons = Signal()
    update_topic_buttons = Signal()
    update_mode_ui = Signal(bool)  # Manual/Auto 모드 UI 업데이트 (외부 명령 수신 시)
    update_run_ui = Signal(bool)  # RUN/STOP UI 업데이트 (외부 명령 수신 시)


class GuiNode(Node, QMainWindow):
    """GUI를 관리하고 여러 Topic을 동적으로 구독하는 통합 Node"""
    
    def __init__(self) -> None:
        Node.__init__(self, "gui_node")
        QMainWindow.__init__(self)
        
        # Topic 설정 파일 로드
        self.topic_config_path = self._get_topic_config_path()
        self.topic_config = self._load_topic_config()
        
        # 구독 관리 딕셔너리
        self.topic_subscriptions: Dict[str, Any] = {}
        
        # 시그널 생성
        self.signals = GuiSignals()
        
        # 상태 정보 저장
        self.current_state = TrackingState.IDLE
        self.current_target_info = None
        self.tracked_objects = []
        self.fps = 0.0
        self.process_time_ms = 0.0
        self.center_zone_elapsed_time = None
        self.center_zone_duration = 5.0
        
        # GUI 모드 관리
        self.is_running = False
        
        # CLIP 결과 저장
        self.clip_best_label = "idle"
        self.clip_probs = {name: 0.0 for name in CLIP_LABELS}
        self.clip_hz = 0.0
        
        # 목 각도 정보
        self.neck_current_yaw = 0.0
        self.neck_current_pitch = 0.0
        self.neck_target_yaw = 0.0
        self.neck_target_pitch = 0.0
        
        # 허리 각도 정보
        self.waist_current_yaw = 0.0
        self.waist_target_yaw = 0.0
        
        # Topic 구독 버튼 리스트
        self.topic_buttons = []
        self.topic_scroll_layout = None
        
        # 타겟 변경 버튼 리스트
        self.target_buttons = []
        
        # GUI 초기화
        self.init_ui()
        
        # 시그널 연결
        self.signals.update_target_buttons.connect(self._update_target_buttons)
        self.signals.update_topic_buttons.connect(self._update_topic_buttons)
        self.signals.update_mode_ui.connect(self._on_mode_ui_update)
        self.signals.update_run_ui.connect(self._on_run_ui_update)
        # Camera Publisher 데이터 구독
        self._setup_camera_subscription()
        
        # Manual 제어 Publisher
        self.manual_control_publisher = self.create_publisher(
            String,
            self._get_topic_name('camera', 'manual_control'),
            10
        )
        
        # Manual 제어 구독 (조이스틱 명령 수신 시 GUI 업데이트)
        self.manual_control_subscription = self.create_subscription(
            String,
            self._get_topic_name('camera', 'manual_control'),
            self._manual_control_received_callback,
            10
        )
        
        # 타이머로 주기적으로 정보 업데이트
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_info)
        self.update_timer.start(50)  # 20Hz 업데이트
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GUI Node v1.10.2 초기화 완료!")
        self.get_logger().info(f"토픽 설정 파일: {self.topic_config_path}")
        self.get_logger().info("=" * 60)
    
    def _get_topic_name(self, category: str, key: str) -> str:
        """설정 파일에서 토픽 이름 가져오기"""
        try:
            return self.topic_config.get(category, {}).get(key, {}).get('name', '')
        except Exception:
            return ''
    
    def _setup_camera_subscription(self):
        """Camera Publisher의 데이터 구독 설정"""
        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            deadline=Duration(seconds=0, nanoseconds=0),
        )
        
        tracking_topic = self._get_topic_name('camera', 'tracking_data')
        self.camera_data_subscription = self.create_subscription(
            String,
            tracking_topic,
            self._camera_data_callback,
            qos_profile
        )
        self.get_logger().info(f"Camera Publisher 데이터 구독 시작: {tracking_topic}")
        
        # 허리 위치 구독
        waist_topic = self._get_topic_name('robot', 'waist_position')
        self.waist_position_subscription = self.create_subscription(
            Float64MultiArray,
            waist_topic,
            self._waist_position_callback,
            10
        )
        self.get_logger().info(f"허리 위치 구독 시작: {waist_topic}")
    
    def _camera_data_callback(self, msg: String):
        """Camera Publisher 데이터 콜백"""
        try:
            data = json.loads(msg.data)
            
            state_str = data.get('state', 'idle')
            prev_state = self.current_state
            try:
                self.current_state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                self.current_state = TrackingState.IDLE
            
            # 상태 변경 시 콤보박스 업데이트 (RUN 중일 때만, Manual 모드가 아니어도 업데이트)
            if self.current_state != prev_state:
                state_display_str = self.current_state.value.upper()
                current_combo_text = self.state_combo.currentText()
                if current_combo_text != state_display_str:
                    self.state_combo.blockSignals(True)
                    self.state_combo.setCurrentText(state_display_str)
                    self.state_combo.blockSignals(False)
                    self.get_logger().info(f"[GUI] 상태 변경 감지: {state_display_str} (콤보박스 업데이트)")
            
            # Manual 모드 정보도 업데이트 (tracking_data에 포함되어 있다면)
            manual_mode = data.get('manual_mode', None)
            if manual_mode is not None:
                current_manual_state = self.manual_btn.isChecked()
                if manual_mode != current_manual_state:
                    # tracking_data에서 받은 모드 정보로 UI 업데이트 (시그널 없이 직접 업데이트하여 무한 루프 방지)
                    self.manual_btn.blockSignals(True)
                    self.auto_btn.blockSignals(True)
                    if manual_mode:
                        self.manual_btn.setChecked(True)
                        self.auto_btn.setChecked(False)
                        self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
                        self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                    else:
                        self.manual_btn.setChecked(False)
                        self.auto_btn.setChecked(True)
                        self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                        self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
                    self.manual_btn.blockSignals(False)
                    self.auto_btn.blockSignals(False)
                    self.get_logger().info(f"[GUI] tracking_data에서 모드 변경 감지: {'MANUAL' if manual_mode else 'AUTO'}")
            
            # RUN/STOP 상태 정보도 업데이트 (tracking_data에 포함되어 있다면)
            is_running = data.get('is_running', None)
            if is_running is not None:
                if is_running != self.is_running:
                    self.is_running = is_running
                    self.run_btn.blockSignals(True)
                    if is_running:
                        self.run_btn.setChecked(True)
                        self.run_btn.setText("STOP")
                        self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #90EE90; color: black;")
                    else:
                        self.run_btn.setChecked(False)
                        self.run_btn.setText("RUN")
                        self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                    self.run_btn.blockSignals(False)
                    self.get_logger().info(f"[GUI] tracking_data에서 RUN 상태 변경 감지: {'RUN' if is_running else 'STOP'}")
            
            target_info_data = data.get('target_info', {})
            if target_info_data:
                point = tuple(target_info_data.get('point')) if target_info_data.get('point') else None
                track_id = target_info_data.get('track_id')
                self.current_target_info = TargetInfo(
                    point=point,
                    state=self.current_state,
                    track_id=track_id
                )
            else:
                self.current_target_info = None
            
            objects_data = data.get('tracked_objects', [])
            self.tracked_objects = []
            for obj_data in objects_data:
                track_id = obj_data.get('track_id')
                if track_id is not None:
                    self.tracked_objects.append(
                        TrackedObject(
                            track_id=track_id,
                            centroid=tuple(obj_data.get('centroid', [0, 0])),
                            state=obj_data.get('state', 'tracking'),
                            confidence=obj_data.get('confidence', 0.0),
                            age=obj_data.get('age', 0)
                        )
                    )
            
            performance_data = data.get('performance', {})
            self.fps = performance_data.get('fps', 0.0)
            self.process_time_ms = performance_data.get('process_time_ms', 0.0)
            
            neck_angles = data.get('neck_angles', {})
            self.neck_current_yaw = neck_angles.get('current', {}).get('yaw_rad', 0.0)
            self.neck_current_pitch = neck_angles.get('current', {}).get('pitch_rad', 0.0)
            self.neck_target_yaw = neck_angles.get('target', {}).get('yaw_rad', 0.0)
            self.neck_target_pitch = neck_angles.get('target', {}).get('pitch_rad', 0.0)
            
            waist_angles = data.get('waist_angles', {})
            if waist_angles:
                self.waist_target_yaw = waist_angles.get('target', {}).get('yaw_rad', 0.0)
            
            center_zone_data = data.get('center_zone', {})
            self.center_zone_elapsed_time = center_zone_data.get('elapsed_time')
            self.center_zone_duration = center_zone_data.get('duration', 5.0)
            
            if self.target_buttons:
                self.signals.update_target_buttons.emit()
                QCoreApplication.processEvents()
            
        except json.JSONDecodeError as e:
            self.get_logger().warn(f"JSON 디코딩 오류: {e}")
        except Exception as e:
            self.get_logger().error(f"Camera 데이터 콜백 오류: {e}")
    
    def _waist_position_callback(self, msg: Float64MultiArray):
        """허리 위치 콜백"""
        try:
            if len(msg.data) >= 1:
                yaw_deg = msg.data[0]
                self.waist_current_yaw = math.radians(yaw_deg)
        except Exception as e:
            self.get_logger().warn(f"허리 위치 콜백 오류: {e}")
    
    def _get_topic_config_path(self) -> Path:
        """Topic 설정 파일 경로 반환"""
        possible_paths = [
            Path(__file__).parent.parent / "config" / "topics.json",
            Path(__file__).parent.parent.parent.parent / "config" / "topics.json",
        ]
        
        current_path = Path(__file__).resolve()
        parts = current_path.parts
        if 'install' in parts:
            install_idx = parts.index('install')
            if install_idx + 1 < len(parts):
                install_base = Path(*parts[:install_idx + 2])
                share_path = install_base / "share" / "allex_ces_idle_interaction" / "config" / "topics.json"
                possible_paths.insert(0, share_path)
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return Path(__file__).parent.parent / "config" / "topics.json"
    
    def _load_topic_config(self) -> dict:
        """Topic 설정 파일 로드"""
        try:
            with open(self.topic_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.get_logger().info(f"Topic 설정 파일 로드 완료: {self.topic_config_path}")
            return config
        except Exception as e:
            self.get_logger().error(f"Topic 설정 파일 로드 실패: {e}")
            return {
                "camera": {
                    "tracking_data": {"name": "/allex_camera/tracking_data", "type": "std_msgs/String"},
                    "manual_control": {"name": "/allex_camera/manual_control", "type": "std_msgs/String"}
                },
                "robot": {
                    "waist_position": {"name": "/robot_outbound_data/theOne_waist/joint_positions_deg", "type": "std_msgs/Float64MultiArray"}
                },
            }
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Person Tracking Control Panel v1.10.2")
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen.width(), screen.height())
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 상단 제어 영역 (좌우 분할)
        top_layout = QHBoxLayout()
        
        # 왼쪽: 모드 및 상태 제어
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # IDLE Mode 제어 그룹
        self.idle_control_group = QGroupBox("IDLE Mode 제어")
        idle_control_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        self.mode_label = QLabel("운영 모드:")
        mode_layout.addWidget(self.mode_label)
        
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setCheckable(True)
        self.auto_btn.setChecked(False)
        self.auto_btn.setMinimumHeight(40)
        self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.auto_btn.clicked.connect(lambda: self.set_mode(False))
        
        self.manual_btn = QPushButton("Manual")
        self.manual_btn.setCheckable(True)
        self.manual_btn.setChecked(True)
        self.manual_btn.setMinimumHeight(40)
        self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
        self.manual_btn.clicked.connect(lambda: self.set_mode(True))
        
        mode_layout.addWidget(self.auto_btn)
        mode_layout.addWidget(self.manual_btn)
        idle_control_layout.addLayout(mode_layout)
        
        # RUN 버튼
        self.run_btn = QPushButton("RUN")
        self.run_btn.setCheckable(True)
        self.run_btn.setChecked(False)
        self.run_btn.setMinimumHeight(60)
        self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.run_btn.clicked.connect(self.on_run_clicked)
        idle_control_layout.addWidget(self.run_btn)
        
        # Parameter 버튼 추가
        self.parameter_btn = QPushButton("Parameter")
        self.parameter_btn.setCheckable(True)
        self.parameter_btn.setChecked(False)
        self.parameter_btn.setMinimumHeight(40)
        self.parameter_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.parameter_btn.clicked.connect(self.on_parameter_clicked)
        idle_control_layout.addWidget(self.parameter_btn)
        
        self.idle_control_group.setLayout(idle_control_layout)
        left_layout.addWidget(self.idle_control_group)
        
        # 상태 제어 그룹 (Manual 모드용) - 먼저 생성해야 parameter_stack에서 참조 가능
        self.state_group = QGroupBox("상태 제어 (Manual 모드)")
        state_layout = QVBoxLayout()
        
        state_select_layout = QHBoxLayout()
        state_select_layout.addWidget(QLabel("State 선택:"))
        
        self.state_combo = QComboBox()
        self.state_combo.addItems(["IDLE", "WAITING", "TRACKING", "LOST", "SEARCHING", "HELLO", "HANDSHAKE"])
        self.state_combo.setMinimumHeight(40)
        self.state_combo.setStyleSheet("font-size: 12pt;")
        self.state_combo.currentTextChanged.connect(self.on_state_changed)
        self.state_combo.setEnabled(False)
        
        state_select_layout.addWidget(self.state_combo)
        state_layout.addLayout(state_select_layout)
        self.state_group.setLayout(state_layout)
        
        # 파라미터 제어 섹션 (스택 위젯으로 전환)
        self.parameter_stack = QStackedWidget()
        
        # 기본 제어 패널 (기존 state_group)
        default_widget = QWidget()
        default_layout = QVBoxLayout()
        default_layout.addWidget(self.state_group)
        default_widget.setLayout(default_layout)
        self.parameter_stack.addWidget(default_widget)
        
        # 파라미터 제어 패널
        parameter_widget = QWidget()
        parameter_layout = QVBoxLayout()
        
        # 파라미터 제어 그룹
        param_control_group = QGroupBox("파라미터 제어")
        param_layout = QGridLayout()
        
        # PID 파라미터
        param_layout.addWidget(QLabel("PID 파라미터:"), 0, 0)
        
        param_layout.addWidget(QLabel("KP Yaw:"), 1, 0)
        self.kp_yaw_spin = QDoubleSpinBox()
        self.kp_yaw_spin.setRange(0.0, 10.0)
        self.kp_yaw_spin.setSingleStep(0.1)
        self.kp_yaw_spin.setValue(1.1)
        self.kp_yaw_spin.setDecimals(2)
        param_layout.addWidget(self.kp_yaw_spin, 1, 1)
        
        param_layout.addWidget(QLabel("KI Yaw:"), 1, 2)
        self.ki_yaw_spin = QDoubleSpinBox()
        self.ki_yaw_spin.setRange(0.0, 1.0)
        self.ki_yaw_spin.setSingleStep(0.01)
        self.ki_yaw_spin.setValue(0.02)
        self.ki_yaw_spin.setDecimals(3)
        param_layout.addWidget(self.ki_yaw_spin, 1, 3)
        
        param_layout.addWidget(QLabel("KP Pitch:"), 2, 0)
        self.kp_pitch_spin = QDoubleSpinBox()
        self.kp_pitch_spin.setRange(0.0, 10.0)
        self.kp_pitch_spin.setSingleStep(0.1)
        self.kp_pitch_spin.setValue(1.2)
        self.kp_pitch_spin.setDecimals(2)
        param_layout.addWidget(self.kp_pitch_spin, 2, 1)
        
        param_layout.addWidget(QLabel("KI Pitch:"), 2, 2)
        self.ki_pitch_spin = QDoubleSpinBox()
        self.ki_pitch_spin.setRange(0.0, 1.0)
        self.ki_pitch_spin.setSingleStep(0.01)
        self.ki_pitch_spin.setValue(0.12)
        self.ki_pitch_spin.setDecimals(3)
        param_layout.addWidget(self.ki_pitch_spin, 2, 3)
        
        # 스무딩 파라미터
        param_layout.addWidget(QLabel("스무딩 파라미터:"), 3, 0)
        
        param_layout.addWidget(QLabel("Total Yaw Alpha:"), 4, 0)
        self.total_yaw_alpha_spin = QDoubleSpinBox()
        self.total_yaw_alpha_spin.setRange(0.0, 1.0)
        self.total_yaw_alpha_spin.setSingleStep(0.05)
        self.total_yaw_alpha_spin.setValue(0.6)
        self.total_yaw_alpha_spin.setDecimals(2)
        param_layout.addWidget(self.total_yaw_alpha_spin, 4, 1)
        
        param_layout.addWidget(QLabel("Neck Target Alpha:"), 4, 2)
        self.neck_target_alpha_spin = QDoubleSpinBox()
        self.neck_target_alpha_spin.setRange(0.0, 1.0)
        self.neck_target_alpha_spin.setSingleStep(0.05)
        self.neck_target_alpha_spin.setValue(0.85)
        self.neck_target_alpha_spin.setDecimals(2)
        param_layout.addWidget(self.neck_target_alpha_spin, 4, 3)
        
        param_layout.addWidget(QLabel("PID Smoothing Alpha:"), 5, 0)
        self.pid_smoothing_alpha_spin = QDoubleSpinBox()
        self.pid_smoothing_alpha_spin.setRange(0.0, 1.0)
        self.pid_smoothing_alpha_spin.setSingleStep(0.05)
        self.pid_smoothing_alpha_spin.setValue(0.65)
        self.pid_smoothing_alpha_spin.setDecimals(2)
        param_layout.addWidget(self.pid_smoothing_alpha_spin, 5, 1)
        
        # 허리 파라미터
        param_layout.addWidget(QLabel("허리 파라미터:"), 6, 0)
        
        param_layout.addWidget(QLabel("Tau Waist:"), 7, 0)
        self.tau_waist_spin = QDoubleSpinBox()
        self.tau_waist_spin.setRange(0.01, 5.0)
        self.tau_waist_spin.setSingleStep(0.01)
        self.tau_waist_spin.setValue(0.12)
        self.tau_waist_spin.setDecimals(2)
        param_layout.addWidget(self.tau_waist_spin, 7, 1)
        
        param_layout.addWidget(QLabel("Tau Waist Searching:"), 7, 2)
        self.tau_waist_searching_spin = QDoubleSpinBox()
        self.tau_waist_searching_spin.setRange(0.1, 10.0)
        self.tau_waist_searching_spin.setSingleStep(0.1)
        self.tau_waist_searching_spin.setValue(1.5)
        self.tau_waist_searching_spin.setDecimals(2)
        param_layout.addWidget(self.tau_waist_searching_spin, 7, 3)
        
        param_layout.addWidget(QLabel("Max Delta Waist:"), 8, 0)
        self.max_delta_waist_spin = QDoubleSpinBox()
        self.max_delta_waist_spin.setRange(0.01, 10.0)
        self.max_delta_waist_spin.setSingleStep(0.1)
        self.max_delta_waist_spin.setValue(2.6)
        self.max_delta_waist_spin.setDecimals(2)
        param_layout.addWidget(self.max_delta_waist_spin, 8, 1)
        
        # 적용 버튼
        apply_btn = QPushButton("파라미터 적용")
        apply_btn.setMinimumHeight(40)
        apply_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #4CAF50; color: white;")
        apply_btn.clicked.connect(self.on_parameter_apply)
        param_layout.addWidget(apply_btn, 9, 0, 1, 4)
        
        param_control_group.setLayout(param_layout)
        parameter_layout.addWidget(param_control_group)
        
        # 스크롤 영역 추가 (파라미터가 많을 경우)
        scroll_area = QScrollArea()
        scroll_area.setWidget(parameter_widget)
        scroll_area.setWidgetResizable(True)
        parameter_widget.setLayout(parameter_layout)
        
        self.parameter_stack.addWidget(scroll_area)
        
        left_layout.addWidget(self.parameter_stack)
        
        top_layout.addWidget(left_panel, 1)
        
        # 오른쪽: 현재 상태 표시
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # 현재 상태 표시 그룹
        status_group = QGroupBox("현재 상태")
        status_layout = QGridLayout()
        
        status_layout.addWidget(QLabel("State:"), 0, 0)
        self.state_label = QLabel("IDLE")
        self.state_label.setStyleSheet("font-weight: bold; font-size: 18pt; color: blue;")
        status_layout.addWidget(self.state_label, 0, 1)
        
        status_layout.addWidget(QLabel("FPS:"), 1, 0)
        self.fps_label = QLabel("--")
        self.fps_label.setStyleSheet("font-size: 14pt;")
        status_layout.addWidget(self.fps_label, 1, 1)
        
        status_layout.addWidget(QLabel("처리 시간:"), 2, 0)
        self.process_time_label = QLabel("--")
        self.process_time_label.setStyleSheet("font-size: 14pt;")
        status_layout.addWidget(self.process_time_label, 2, 1)
        
        status_layout.addWidget(QLabel("추적 객체 수:"), 3, 0)
        self.objects_count_label = QLabel("0")
        self.objects_count_label.setStyleSheet("font-size: 14pt;")
        status_layout.addWidget(self.objects_count_label, 3, 1)
        
        status_layout.addWidget(QLabel("타겟 Track ID:"), 4, 0)
        self.target_id_label = QLabel("--")
        self.target_id_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        status_layout.addWidget(self.target_id_label, 4, 1)
        
        status_layout.addWidget(QLabel("Center Zone:"), 5, 0)
        self.center_zone_label = QLabel("--")
        self.center_zone_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        status_layout.addWidget(self.center_zone_label, 5, 1)
        
        status_group.setLayout(status_layout)
        right_layout.addWidget(status_group)
        
        # 목 각도 정보 그룹
        neck_group = QGroupBox("목 각도 정보")
        neck_layout = QGridLayout()
        
        neck_layout.addWidget(QLabel("현재 Yaw:"), 0, 0)
        self.current_yaw_label = QLabel("--")
        self.current_yaw_label.setStyleSheet("font-size: 14pt;")
        neck_layout.addWidget(self.current_yaw_label, 0, 1)
        
        neck_layout.addWidget(QLabel("현재 Pitch:"), 1, 0)
        self.current_pitch_label = QLabel("--")
        self.current_pitch_label.setStyleSheet("font-size: 14pt;")
        neck_layout.addWidget(self.current_pitch_label, 1, 1)
        
        neck_layout.addWidget(QLabel("목표 Yaw:"), 2, 0)
        self.target_yaw_label = QLabel("--")
        self.target_yaw_label.setStyleSheet("font-size: 14pt;")
        neck_layout.addWidget(self.target_yaw_label, 2, 1)
        
        neck_layout.addWidget(QLabel("목표 Pitch:"), 3, 0)
        self.target_pitch_label = QLabel("--")
        self.target_pitch_label.setStyleSheet("font-size: 14pt;")
        neck_layout.addWidget(self.target_pitch_label, 3, 1)
        
        neck_group.setLayout(neck_layout)
        right_layout.addWidget(neck_group)
        
        # 허리 각도 정보 그룹
        waist_group = QGroupBox("허리 각도 정보")
        waist_layout = QGridLayout()
        
        waist_layout.addWidget(QLabel("현재 Yaw:"), 0, 0)
        self.current_waist_yaw_label = QLabel("--")
        self.current_waist_yaw_label.setStyleSheet("font-size: 14pt;")
        waist_layout.addWidget(self.current_waist_yaw_label, 0, 1)
        
        waist_layout.addWidget(QLabel("목표 Yaw:"), 1, 0)
        self.target_waist_yaw_label = QLabel("--")
        self.target_waist_yaw_label.setStyleSheet("font-size: 14pt;")
        waist_layout.addWidget(self.target_waist_yaw_label, 1, 1)
        
        waist_group.setLayout(waist_layout)
        right_layout.addWidget(waist_group)
        
        top_layout.addWidget(right_panel, 1)
        
        main_layout.addLayout(top_layout)
        
        # CLIP 결과 표시 그룹 (Interaction Mode에서만 표시)
        self.clip_result_group = QGroupBox("🎯 CLIP 분류 결과")
        clip_layout = QVBoxLayout()
        
        # 현재 분류 라벨
        self.clip_label_display = QLabel("--")
        self.clip_label_display.setAlignment(Qt.AlignCenter)
        self.clip_label_display.setStyleSheet("""
            font-size: 24pt; font-weight: bold; color: #4fc3f7;
            background-color: #1a1a1a; border: 3px solid #333;
            border-radius: 10px; padding: 10px; min-height: 50px;
        """)
        clip_layout.addWidget(self.clip_label_display)
        
        # 추론 Hz
        hz_layout = QHBoxLayout()
        hz_layout.addWidget(QLabel("추론 Hz:"))
        self.clip_hz_label = QLabel("0.0 Hz")
        self.clip_hz_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #81c784;")
        hz_layout.addWidget(self.clip_hz_label)
        hz_layout.addStretch()
        clip_layout.addLayout(hz_layout)
        
        # 라벨별 신뢰도 표시
        self.clip_label_bars = {}
        self.clip_percent_labels = {}
        
        for name in CLIP_LABELS:
            color = CLIP_LABEL_COLORS[name]
            icon = CLIP_LABEL_ICONS[name]
            row_layout = QHBoxLayout()
            
            name_label = QLabel(f"{icon} {name}")
            name_label.setFixedWidth(110)
            name_label.setStyleSheet(f"font-size: 11pt; font-weight: bold; color: {color};")
            row_layout.addWidget(name_label)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 200)  # 앙상블 합계 최대 200%
            progress_bar.setValue(0)
            progress_bar.setTextVisible(False)
            progress_bar.setFixedHeight(22)
            progress_bar.setStyleSheet(f"""
                QProgressBar {{ border: 2px solid #333; border-radius: 5px; background-color: #1a1a1a; }}
                QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}
            """)
            self.clip_label_bars[name] = progress_bar
            row_layout.addWidget(progress_bar)
            
            percent_label = QLabel("0.0%")
            percent_label.setFixedWidth(65)
            percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            percent_label.setStyleSheet(f"font-size: 12pt; font-weight: bold; color: {color};")
            self.clip_percent_labels[name] = percent_label
            row_layout.addWidget(percent_label)
            
            clip_layout.addLayout(row_layout)
        
        self.clip_result_group.setLayout(clip_layout)
        self.clip_result_group.setVisible(False)  # 초기에는 숨김
        main_layout.addWidget(self.clip_result_group)
        
        # 추적 객체 정보 그룹
        objects_group = QGroupBox("추적 객체 정보")
        objects_layout = QVBoxLayout()
        
        self.objects_info_label = QLabel("객체 정보가 여기에 표시됩니다.")
        self.objects_info_label.setWordWrap(True)
        self.objects_info_label.setStyleSheet("font-size: 12pt;")
        objects_layout.addWidget(self.objects_info_label)
        
        objects_group.setLayout(objects_layout)
        main_layout.addWidget(objects_group)
        
        # 타겟 변경 버튼 그룹
        target_group = QGroupBox("타겟 변경 (현재 추적 중인 객체 선택)")
        target_layout = QGridLayout()
        target_layout.setSpacing(10)
        target_layout.setContentsMargins(10, 10, 10, 10)
        
        self.target_buttons = []
        MAX_TARGET_BUTTONS = 10
        for i in range(MAX_TARGET_BUTTONS):
            btn = TargetButton(0, self)
            btn.setMinimumHeight(60)
            btn.setMinimumWidth(120)
            btn.setVisible(False)
            btn.setEnabled(True)
            btn.setCheckable(False)
            btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            
            row = i // 5
            col = i % 5
            target_layout.addWidget(btn, row, col)
            self.target_buttons.append(btn)
        
        target_group.setLayout(target_layout)
        main_layout.addWidget(target_group)
        
        main_layout.addStretch()
        
        QCoreApplication.processEvents()
        self._update_target_buttons()
        QCoreApplication.processEvents()
        
        self.get_logger().info(f"초기화 완료: 타겟 버튼={len(self.target_buttons)}개")
    
    def on_run_clicked(self):
        """RUN 버튼 클릭 이벤트 (IDLE Mode)"""
        if self.run_btn.isChecked():
            self.is_running = True
            self.run_btn.setText("STOP")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #90EE90; color: black;")
            
            manual = self.manual_btn.isChecked()
            self._send_manual_control({
                'type': 'run',
                'manual': manual
            })
            self.get_logger().info(f"RUN 시작: IDLE Mode ({'Manual' if manual else 'Auto'})")
        else:
            self.is_running = False
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self._send_manual_control({'type': 'stop'})
            self.get_logger().info("RUN 중지: IDLE 상태로 전환")
    
    def set_mode(self, manual: bool):
        """운영 모드 설정 (Auto/Manual) - 모드 변경 시 기존 RUN 상태는 STOP"""
        # 모드 변경 전, 실행 중이었다면 먼저 STOP
        if self.is_running:
            # RUN 상태를 STOP으로 변경
            self.is_running = False
            self.run_btn.setChecked(False)
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            
            # STOP 명령 전송
            self._send_manual_control({'type': 'stop'})
            self.get_logger().info(f"[GUI] 모드 변경: 기존 RUN 상태 STOP (새 모드: {'MANUAL' if manual else 'AUTO'})")
        
        # UI 업데이트
        if manual:
            self.manual_btn.setChecked(True)
            self.auto_btn.setChecked(False)
            self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.state_combo.setEnabled(True)
        else:
            self.manual_btn.setChecked(False)
            self.auto_btn.setChecked(True)
            self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.state_combo.setEnabled(False)
        
        # 모드 변경 명령 전송 (RUN 중이 아니면 모드만 변경)
        if not self.is_running:
            # STOP 상태이므로 모드만 변경 (RUN 중이 아니면 모드 변경 명령만 전송)
            pass  # 모드 변경은 UI만으로 충분, 백엔드는 다음 RUN 시 적용됨
        else:
            # 만약 실행 중이면 (이론적으로는 이 블록에 들어오지 않아야 함)
            self._send_manual_control({
                'type': 'set_mode',
                'manual': manual
            })
        
        self.signals.mode_changed.emit(manual)
    
    def on_parameter_clicked(self):
        """Parameter 버튼 클릭 이벤트 - 파라미터 제어 패널 전환"""
        if self.parameter_btn.isChecked():
            # Parameter 모드: 파라미터 제어 패널 표시
            self.parameter_stack.setCurrentIndex(1)
            self.parameter_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.get_logger().info("[GUI] 파라미터 제어 모드 활성화")
        else:
            # 기본 모드: 상태 제어 패널 표시
            self.parameter_stack.setCurrentIndex(0)
            self.parameter_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.get_logger().info("[GUI] 기본 제어 모드로 복귀")
    
    def on_parameter_apply(self):
        """파라미터 적용 버튼 클릭 이벤트"""
        params = {
            'type': 'set_parameters',
            'parameters': {
                'kp_yaw': self.kp_yaw_spin.value(),
                'ki_yaw': self.ki_yaw_spin.value(),
                'kp_pitch': self.kp_pitch_spin.value(),
                'ki_pitch': self.ki_pitch_spin.value(),
                'total_yaw_smoothing_alpha': self.total_yaw_alpha_spin.value(),
                'neck_target_alpha': self.neck_target_alpha_spin.value(),
                'pid_smoothing_alpha': self.pid_smoothing_alpha_spin.value(),
                'tau_waist': self.tau_waist_spin.value(),
                'tau_waist_searching': self.tau_waist_searching_spin.value(),
                'max_delta_waist': self.max_delta_waist_spin.value()
            }
        }
        self._send_manual_control(params)
        self.get_logger().info(f"[GUI] 파라미터 적용 요청 전송: {params['parameters']}")
    
    def on_state_changed(self, state_text: str):
        """Manual 모드에서 State 변경"""
        if not self.is_running or not self.manual_btn.isChecked():
            return
        
        target_id = self.current_target_info.track_id if self.current_target_info else None
        self._send_manual_control({
            'type': 'set_state',
            'state': state_text.lower(),
            'target_id': target_id
        })
        
        self.signals.state_changed.emit(state_text)
    
    def _on_target_button_clicked(self, target_id: int):
        """타겟 변경 버튼 클릭 이벤트"""
        if not self.is_running:
            self.get_logger().warn("RUN 버튼을 먼저 눌러주세요.")
            return
        
        self.current_target_info = TargetInfo(
            point=self.current_target_info.point if self.current_target_info else None,
            state=TrackingState.TRACKING,
            track_id=target_id
        )
        
        self._update_target_buttons()
        
        self.target_id_label.setText(str(target_id))
        self.target_id_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: green;")
        
        self._send_manual_control({
            'type': 'set_target',
            'target_id': target_id,
            'force': True
        })
        self.get_logger().info(f"[GUI] 타겟 변경 요청 전송: {target_id}")
    
    def _send_manual_control(self, command: dict):
        """Manual 제어 명령 전송"""
        try:
            msg = String()
            msg.data = json.dumps(command, ensure_ascii=False)
            self.manual_control_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Manual 제어 명령 전송 실패: {e}")
    
    def _manual_control_received_callback(self, msg: String):
        """Manual 제어 명령 수신 (조이스틱 등에서 온 명령 처리)"""
        try:
            command = json.loads(msg.data)
            cmd_type = command.get('type')
            
            # Qt 시그널로 메인 스레드에서 UI 업데이트 (스레드 안전)
            if cmd_type == 'run' or cmd_type == 'start':
                manual_mode = command.get('manual', False)
                self.get_logger().info(f"[GUI] 외부 명령 수신: RUN ({'MANUAL' if manual_mode else 'AUTO'})")
                self.signals.update_run_ui.emit(True)  # True = RUN
            
            elif cmd_type == 'stop':
                self.get_logger().info("[GUI] 외부 명령 수신: STOP")
                self.signals.update_run_ui.emit(False)  # False = STOP
            
            elif cmd_type == 'set_mode':
                manual_mode = command.get('manual', False)
                self.get_logger().info(f"[GUI] 외부 명령 수신: 모드 변경 → {'MANUAL' if manual_mode else 'AUTO'}")
                self.signals.update_mode_ui.emit(manual_mode)  # True = Manual, False = Auto
            
            elif cmd_type == 'set_state':
                # 상태 변경 명령 수신 시 UI 업데이트는 tracking_data 콜백에서 처리
                # (tracking_result가 발행되면 자동으로 상태가 업데이트됨)
                state_str = command.get('state', 'idle')
                self.get_logger().info(f"[GUI] 외부 명령 수신: 상태 변경 → {state_str.upper()}")
                        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Manual 제어 명령 파싱 실패: {e}")
        except Exception as e:
            self.get_logger().error(f"Manual 제어 명령 처리 실패: {e}")
    
    def _on_mode_ui_update(self, manual_mode: bool):
        """모드 UI 업데이트 슬롯 (메인 스레드에서 실행)"""
        current_manual_state = self.manual_btn.isChecked()
        if manual_mode == current_manual_state:
            self.get_logger().debug(f"[GUI] 이미 {'MANUAL' if manual_mode else 'AUTO'} 모드입니다")
            return
        
        # 모드 변경 전, 실행 중이었다면 먼저 STOP
        if self.is_running:
            self.is_running = False
            self.run_btn.setChecked(False)
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.get_logger().info(f"[GUI] 모드 변경: 기존 RUN 상태 STOP")
        
        # UI 업데이트 (메인 스레드에서 실행되므로 안전)
        if manual_mode:
            self.manual_btn.setChecked(True)
            self.auto_btn.setChecked(False)
            self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.state_combo.setEnabled(True)
        else:
            self.manual_btn.setChecked(False)
            self.auto_btn.setChecked(True)
            self.manual_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.auto_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.state_combo.setEnabled(False)
        
        self.get_logger().info(f"[GUI] 모드 UI 업데이트 완료: {'MANUAL' if manual_mode else 'AUTO'}")
    
    def _on_run_ui_update(self, is_running: bool):
        """RUN/STOP UI 업데이트 슬롯 (메인 스레드에서 실행)"""
        if is_running:
            if not self.run_btn.isChecked():
                self.is_running = True
                self.run_btn.setChecked(True)
                self.run_btn.setText("STOP")
                self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #90EE90; color: black;")
                self.get_logger().info("[GUI] RUN UI 업데이트 완료")
        else:
            if self.run_btn.isChecked():
                self.is_running = False
                self.run_btn.setChecked(False)
                self.run_btn.setText("RUN")
                self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                self.get_logger().info("[GUI] STOP UI 업데이트 완료")
    
    def _update_target_buttons(self):
        """타겟 버튼 업데이트"""
        if not self.target_buttons:
            return
        
        # 가로축 기준 좌측부터 정렬 (centroid의 x 좌표 기준)
        sorted_objects = sorted(self.tracked_objects, key=lambda obj: obj.centroid[0])
        tracked_ids = [obj.track_id for obj in sorted_objects][:10]
        current_target_id = self.current_target_info.track_id if self.current_target_info else None
        
        for btn in self.target_buttons:
            btn.setVisible(False)
            try:
                btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
        
        for idx, track_id in enumerate(tracked_ids):
            if idx >= len(self.target_buttons):
                break
            
            btn = self.target_buttons[idx]
            
            if isinstance(btn, TargetButton):
                try:
                    btn.clicked_with_id.disconnect()
                except (TypeError, RuntimeError):
                    pass
                btn.track_id = track_id
                btn.clicked_with_id.connect(self._on_target_button_clicked)
            
            is_current_target = track_id == current_target_id
            btn.setStyleSheet(
                "font-size: 16pt; font-weight: bold; "
                f"background-color: {'#90EE90' if is_current_target else '#E0E0E0'}; color: black;"
            )
            btn.setText(f"ID: {track_id}\n✓ (현재 타겟)" if is_current_target else f"ID: {track_id}")
            btn.setEnabled(True)
            btn.setCheckable(False)
            btn.setVisible(True)
    
    def update_info(self):
        """정보 업데이트 (주기적으로 호출)"""
        # State 표시 업데이트
        state_str = self.current_state.value.upper()
        self.state_label.setText(state_str)
        
        state_colors = {
            'IDLE': 'gray',
            'WAITING': 'lightblue',
            'TRACKING': 'green',
            'LOST': 'orange',
            'SEARCHING': 'yellow',
            'HELLO': 'cyan',
            'HANDSHAKE': 'magenta',
        }
        color = state_colors.get(state_str, 'black')
        self.state_label.setStyleSheet(f"font-weight: bold; font-size: 14pt; color: {color};")
        
        # ComboBox 동기화 (항상 업데이트 - 상태 변경 시 자동 반영)
        current_combo_text = self.state_combo.currentText()
        if current_combo_text != state_str:
            self.state_combo.blockSignals(True)
            self.state_combo.setCurrentText(state_str)
            self.state_combo.blockSignals(False)
            self.get_logger().debug(f"[GUI] update_info: 콤보박스 업데이트 {current_combo_text} → {state_str}")
        
        # FPS 및 처리 시간 표시
        self.fps_label.setText(f"{self.fps:.1f}" if self.fps > 0 else "--")
        self.process_time_label.setText(f"{self.process_time_ms:.1f} ms" if self.process_time_ms > 0 else "--")
        
        # 객체 수 표시
        self.objects_count_label.setText(str(len(self.tracked_objects)))
        
        # 타겟 Track ID 표시
        if self.current_target_info and self.current_target_info.track_id is not None:
            self.target_id_label.setText(str(self.current_target_info.track_id))
            color = "green" if self.current_state == TrackingState.TRACKING else "orange"
            self.target_id_label.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")
        else:
            self.target_id_label.setText("--")
            self.target_id_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: gray;")
        
        # Center Zone 시간 표시
        if self.center_zone_elapsed_time is not None:
            elapsed_str = f"{self.center_zone_elapsed_time:.2f}s / {self.center_zone_duration:.1f}s"
            progress = min(self.center_zone_elapsed_time / self.center_zone_duration, 1.0)
            if progress >= 0.5:
                color = "green"
            else:
                color = "orange"
            self.center_zone_label.setText(elapsed_str)
            self.center_zone_label.setStyleSheet(f"font-size: 14pt; font-weight: bold; color: {color};")
        else:
            self.center_zone_label.setText("--")
            self.center_zone_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: gray;")
        
        # 목 각도 정보 표시
        if self.neck_current_yaw == 0.0 and self.neck_current_pitch == 0.0:
            self.current_yaw_label.setText("Waiting...")
            self.current_pitch_label.setText("Waiting...")
        else:
            self.current_yaw_label.setText(f"{math.degrees(self.neck_current_yaw):.1f}°")
            self.current_pitch_label.setText(f"{math.degrees(self.neck_current_pitch):.1f}°")
        
        if self.neck_target_yaw == 0.0 and self.neck_target_pitch == 0.0:
            self.target_yaw_label.setText("No command")
            self.target_pitch_label.setText("No command")
        else:
            self.target_yaw_label.setText(f"{math.degrees(self.neck_target_yaw):.1f}°")
            self.target_pitch_label.setText(f"{math.degrees(self.neck_target_pitch):.1f}°")
        
        # 허리 각도 정보 표시
        if self.waist_current_yaw == 0.0:
            self.current_waist_yaw_label.setText("Waiting...")
        else:
            self.current_waist_yaw_label.setText(f"{math.degrees(self.waist_current_yaw):.1f}°")
        
        if self.waist_target_yaw == 0.0:
            self.target_waist_yaw_label.setText("No command")
        else:
            self.target_waist_yaw_label.setText(f"{math.degrees(self.waist_target_yaw):.1f}°")
        
        # 추적 객체 정보 업데이트
        if self.tracked_objects:
            info_lines = []
            for obj in self.tracked_objects[:5]:
                info_lines.append(
                    f"ID: {obj.track_id}, State: {obj.state}, "
                    f"Conf: {obj.confidence:.2f}, "
                    f"Centroid: ({obj.centroid[0]:.0f}, {obj.centroid[1]:.0f})"
                )
            if len(self.tracked_objects) > 5:
                info_lines.append(f"... 외 {len(self.tracked_objects) - 5}개")
            self.objects_info_label.setText("\n".join(info_lines))
        else:
            self.objects_info_label.setText("추적 객체 없음")
        
        # CLIP 결과 업데이트 (LLM 제거로 인해 비활성화)
        
        # 타겟 버튼 업데이트
        self._update_target_buttons()
    
    def _update_topic_buttons(self):
        """Topic 구독 버튼 업데이트 (호환성용)"""
        pass
    
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
    gui_node = GuiNode()
    gui_node.show()
    
    try:
        ros_thread_running = True
        
        def ros_spin():
            nonlocal ros_thread_running
            while ros_thread_running:
                rclpy.spin_once(gui_node, timeout_sec=0.1)
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        
        app.exec()
        ros_thread_running = False
    except KeyboardInterrupt:
        pass
    finally:
        gui_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
