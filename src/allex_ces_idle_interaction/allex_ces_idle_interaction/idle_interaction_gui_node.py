#!/usr/bin/env python3
"""
GUI Node - GUI를 관리하고 여러 Topic을 동적으로 구독하는 통합 Node
v1.2.0 - PySide6 사용, LLM Publisher 제어 기능 추가

시스템 구조:
- SPARK 1 PC: Camera Publisher (카메라 + YOLO 추적)
- SPARK 2 PC: LLM Publisher (CLIP 추론, RUN/STOP 대기)
- Laptop: GUI (이 노드) - IDLE/Interaction 모드 전환
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
    QProgressBar
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
    llm_result_received = Signal(str, dict, float)  # best_label, probs, hz


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
        self.interaction_mode = False
        self.is_running = False
        
        # LLM 관련 상태
        self.llm_subscribed = False
        self.llm_running = False
        self.llm_status = {'running': False, 'hz': 0.0, 'device': 'unknown'}
        
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
        self.signals.llm_result_received.connect(self._on_llm_result_received)
        
        # Camera Publisher 데이터 구독
        self._setup_camera_subscription()
        
        # Manual 제어 Publisher
        self.manual_control_publisher = self.create_publisher(
            String,
            self._get_topic_name('camera', 'manual_control'),
            10
        )
        
        # LLM 제어 Publisher
        self.llm_control_publisher = self.create_publisher(
            String,
            self._get_topic_name('llm', 'control'),
            10
        )
        
        # Launch 후 자동으로 LLM 토픽 구독
        self.subscribe_llm_topics()
        self.get_logger().info("LLM 토픽 자동 구독 완료 (Launch 후)")
        
        # 타이머로 주기적으로 정보 업데이트
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_info)
        self.update_timer.start(50)  # 20Hz 업데이트
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("GUI Node v1.2.0 초기화 완료!")
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
            try:
                self.current_state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                self.current_state = TrackingState.IDLE
            
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
                "llm": {
                    "response": {"name": "/llm/response", "type": "std_msgs/String"},
                    "control": {"name": "/llm/control", "type": "std_msgs/String"},
                    "status": {"name": "/llm/status", "type": "std_msgs/String"}
                }
            }
    
    def subscribe_llm_topics(self) -> bool:
        """LLM 관련 토픽 구독 시작"""
        try:
            qos_profile = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                deadline=Duration(seconds=0, nanoseconds=0),
            )
            
            # LLM 응답 토픽 구독
            response_topic = self._get_topic_name('llm', 'response')
            if response_topic and response_topic not in self.topic_subscriptions:
                subscription = self.create_subscription(
                    String,
                    response_topic,
                    self._llm_response_callback,
                    qos_profile
                )
                self.topic_subscriptions[response_topic] = {
                    'subscription': subscription,
                    'type': 'std_msgs/String',
                    'enabled': True
                }
                self.get_logger().info(f"LLM 응답 토픽 구독 시작: {response_topic}")
            
            # LLM 상태 토픽 구독
            status_topic = self._get_topic_name('llm', 'status')
            if status_topic and status_topic not in self.topic_subscriptions:
                subscription = self.create_subscription(
                    String,
                    status_topic,
                    self._llm_status_callback,
                    10
                )
                self.topic_subscriptions[status_topic] = {
                    'subscription': subscription,
                    'type': 'std_msgs/String',
                    'enabled': True
                }
                self.get_logger().info(f"LLM 상태 토픽 구독 시작: {status_topic}")
            
            self.llm_subscribed = True
            return True
            
        except Exception as e:
            self.get_logger().error(f"LLM 토픽 구독 실패: {e}")
            return False
    
    def unsubscribe_llm_topics(self) -> bool:
        """LLM 관련 토픽 구독 해제"""
        try:
            response_topic = self._get_topic_name('llm', 'response')
            status_topic = self._get_topic_name('llm', 'status')
            
            for topic in [response_topic, status_topic]:
                if topic in self.topic_subscriptions:
                    del self.topic_subscriptions[topic]
                    self.get_logger().info(f"LLM 토픽 구독 해제: {topic}")
            
            self.llm_subscribed = False
            return True
        except Exception as e:
            self.get_logger().error(f"LLM 토픽 구독 해제 실패: {e}")
            return False
    
    def _llm_response_callback(self, msg: String) -> None:
        """LLM 응답 메시지 콜백"""
        try:
            data = json.loads(msg.data)
            best = data.get("best", "idle")
            probs = data.get("probs", {})
            hz = data.get("hz", 0.0)
            
            # 시그널로 메인 쓰레드에 전달
            self.signals.llm_result_received.emit(best, probs, hz)
            
        except Exception as e:
            self.get_logger().error(f"LLM 응답 파싱 실패: {e}")
    
    def _llm_status_callback(self, msg: String) -> None:
        """LLM 상태 메시지 콜백"""
        try:
            data = json.loads(msg.data)
            self.llm_status = data
            self.llm_running = data.get('running', False)
            
            # 상태 표시 업데이트 (Interaction Mode일 때만)
            if hasattr(self, 'llm_status_label') and self.interaction_mode:
                if self.llm_running:
                    hz = data.get('hz', 0.0)
                    device = data.get('device', 'unknown')
                    self.llm_status_label.setText(f"🟢 LLM: 추론 중 ({hz:.1f} Hz, {device})")
                    self.llm_status_label.setStyleSheet("font-size: 11pt; color: #81c784;")
                else:
                    self.llm_status_label.setText("🟡 LLM: 대기 중 (토픽 구독 중)")
                    self.llm_status_label.setStyleSheet("font-size: 11pt; color: #ffa726;")
        except Exception as e:
            self.get_logger().warn(f"LLM 상태 파싱 실패: {e}")
    
    def _on_llm_result_received(self, best_label: str, probs: dict, hz: float):
        """LLM 결과 수신 시 호출 (메인 쓰레드)"""
        self.clip_best_label = best_label
        self.clip_probs = probs
        self.clip_hz = hz
    
    def send_llm_control(self, command: str):
        """LLM Publisher에 제어 명령 전송"""
        try:
            msg = String()
            msg.data = json.dumps({'type': command})
            self.llm_control_publisher.publish(msg)
            self.get_logger().info(f"LLM 제어 명령 전송: {command}")
        except Exception as e:
            self.get_logger().error(f"LLM 제어 명령 전송 실패: {e}")
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Person Tracking Control Panel v1.2.0")
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
        
        # Interaction/IDLE Mode 선택 그룹
        interaction_group = QGroupBox("시스템 모드")
        interaction_layout = QHBoxLayout()
        
        self.interaction_btn = QPushButton("Interaction Mode")
        self.interaction_btn.setCheckable(True)
        self.interaction_btn.setChecked(False)
        self.interaction_btn.setMinimumHeight(50)
        self.interaction_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #E0E0E0;")
        self.interaction_btn.clicked.connect(lambda: self.set_interaction_mode(True))
        
        self.idle_btn = QPushButton("IDLE Mode")
        self.idle_btn.setCheckable(True)
        self.idle_btn.setChecked(True)
        self.idle_btn.setMinimumHeight(50)
        self.idle_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #90EE90;")
        self.idle_btn.clicked.connect(lambda: self.set_interaction_mode(False))
        
        interaction_layout.addWidget(self.interaction_btn)
        interaction_layout.addWidget(self.idle_btn)
        interaction_group.setLayout(interaction_layout)
        left_layout.addWidget(interaction_group)
        
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
        
        self.idle_control_group.setLayout(idle_control_layout)
        left_layout.addWidget(self.idle_control_group)
        
        # 상태 제어 그룹 (Manual 모드용)
        self.state_group = QGroupBox("상태 제어 (Manual 모드)")
        state_layout = QVBoxLayout()
        
        state_select_layout = QHBoxLayout()
        state_select_layout.addWidget(QLabel("State 선택:"))
        
        self.state_combo = QComboBox()
        self.state_combo.addItems(["IDLE", "TRACKING", "LOST", "SEARCHING", "WAIST_FOLLOWER", "HELLO"])
        self.state_combo.setMinimumHeight(40)
        self.state_combo.setStyleSheet("font-size: 12pt;")
        self.state_combo.currentTextChanged.connect(self.on_state_changed)
        self.state_combo.setEnabled(False)
        
        state_select_layout.addWidget(self.state_combo)
        state_layout.addLayout(state_select_layout)
        self.state_group.setLayout(state_layout)
        left_layout.addWidget(self.state_group)
        
        # Interaction Mode 제어 그룹 (초기에는 숨김)
        self.interaction_control_group = QGroupBox("Interaction Mode 제어")
        interaction_control_layout = QVBoxLayout()
        
        # LLM 상태 표시
        self.llm_status_label = QLabel("🟡 LLM: 대기 중 (토픽 구독 중)")
        self.llm_status_label.setStyleSheet("font-size: 11pt; color: #ffa726;")
        interaction_control_layout.addWidget(self.llm_status_label)
        
        # Interaction RUN 버튼
        self.interaction_run_btn = QPushButton("RUN (Interaction)")
        self.interaction_run_btn.setCheckable(True)
        self.interaction_run_btn.setChecked(False)
        self.interaction_run_btn.setMinimumHeight(60)
        self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.interaction_run_btn.clicked.connect(self.on_interaction_run_clicked)
        interaction_control_layout.addWidget(self.interaction_run_btn)
        
        self.interaction_control_group.setLayout(interaction_control_layout)
        self.interaction_control_group.setVisible(False)
        left_layout.addWidget(self.interaction_control_group)
        
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
    
    def set_interaction_mode(self, interaction: bool):
        """Interaction/IDLE Mode 설정"""
        if self.is_running:
            self.run_btn.setChecked(False)
            self.interaction_run_btn.setChecked(False)
            self.is_running = False
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.interaction_run_btn.setText("RUN (Interaction)")
            self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self._send_manual_control({'type': 'stop'})
        
        self.interaction_mode = interaction
        if interaction:
            self.interaction_btn.setChecked(True)
            self.idle_btn.setChecked(False)
            self.interaction_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #90EE90;")
            self.idle_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #E0E0E0;")
            
            # IDLE Mode 제어 숨기기
            self.idle_control_group.setVisible(False)
            self.state_group.setVisible(False)
            
            # Interaction Mode 제어 표시
            self.interaction_control_group.setVisible(True)
            self.clip_result_group.setVisible(True)
            
            self._send_manual_control({
                'type': 'set_interaction_mode',
                'enabled': True
            })
            self.get_logger().info("Interaction Mode 선택됨")
        else:
            self.interaction_btn.setChecked(False)
            self.idle_btn.setChecked(True)
            self.interaction_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #E0E0E0;")
            self.idle_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #90EE90;")
            
            # Interaction Mode 제어 숨기기
            self.interaction_control_group.setVisible(False)
            self.clip_result_group.setVisible(False)
            
            # IDLE Mode 제어 표시
            self.idle_control_group.setVisible(True)
            self.state_group.setVisible(True)
            
            # LLM 추론 중지 (구독은 유지)
            if self.llm_subscribed:
                self.send_llm_control('stop')
                self.llm_status_label.setText("🟡 LLM: 대기 중 (토픽 구독 중)")
                self.llm_status_label.setStyleSheet("font-size: 11pt; color: #ffa726;")
            
            self._send_manual_control({
                'type': 'set_interaction_mode',
                'enabled': False
            })
            self.get_logger().info("IDLE Mode 선택됨")
    
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
    
    def on_interaction_run_clicked(self):
        """RUN 버튼 클릭 이벤트 (Interaction Mode)"""
        if self.interaction_run_btn.isChecked():
            self.is_running = True
            self.interaction_run_btn.setText("STOP (Interaction)")
            self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #90EE90; color: black;")
            
            # 추적 시작
            self._send_manual_control({
                'type': 'run',
                'manual': False
            })
            
            # LLM 추론 시작 (자동)
            if self.llm_subscribed:
                self.send_llm_control('run')
                self.llm_status_label.setText("🟢 LLM: 추론 중")
                self.llm_status_label.setStyleSheet("font-size: 11pt; color: #81c784;")
            
            self.get_logger().info("RUN 시작: Interaction Mode (BB Box 추적 + LLM 추론)")
        else:
            self.is_running = False
            self.interaction_run_btn.setText("RUN (Interaction)")
            self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            
            # 추적 중지
            self._send_manual_control({'type': 'stop'})
            
            # LLM 추론 중지 (자동)
            if self.llm_subscribed:
                self.send_llm_control('stop')
                self.llm_status_label.setText("🟡 LLM: 대기 중 (토픽 구독 중)")
                self.llm_status_label.setStyleSheet("font-size: 11pt; color: #ffa726;")
            
            self.get_logger().info("RUN 중지: Interaction Mode 종료 (LLM 추론 중지)")
    
    def set_mode(self, manual: bool):
        """운영 모드 설정 (Auto/Manual)"""
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
        
        if not self.is_running:
            return
        
        self._send_manual_control({
            'type': 'set_mode',
            'manual': manual
        })
        
        self.signals.mode_changed.emit(manual)
    
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
        """타겟 변경 버튼 클릭 이벤트 - tracking_result를 받을 때까지 기다림"""
        if not self.is_running:
            self.get_logger().warn("RUN 버튼을 먼저 눌러주세요.")
            return
        
        # 로컬 업데이트 제거: tracking_result를 받을 때 자동으로 동기화됨
        # 명령만 전송하고, 실제 업데이트는 _camera_data_callback에서 처리
        
        self._send_manual_control({
            'type': 'set_target',
            'target_id': target_id,
            'force': True
        })
        self.get_logger().info(f"[GUI] 타겟 변경 요청 전송: {target_id} (tracking_result 대기 중...)")
    
    def _send_manual_control(self, command: dict):
        """Manual 제어 명령 전송"""
        try:
            msg = String()
            msg.data = json.dumps(command, ensure_ascii=False)
            self.manual_control_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Manual 제어 명령 전송 실패: {e}")
    
    def _update_target_buttons(self):
        """타겟 버튼 업데이트"""
        if not self.target_buttons:
            return
        
        tracked_ids = sorted([obj.track_id for obj in self.tracked_objects])[:10]
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
            'TRACKING': 'green',
            'LOST': 'orange',
            'SEARCHING': 'yellow',
            'WAIST_FOLLOWER': 'purple',
            'HELLO': 'cyan',
            'INTERACTION': 'blue'
        }
        color = state_colors.get(state_str, 'black')
        self.state_label.setStyleSheet(f"font-weight: bold; font-size: 14pt; color: {color};")
        
        # ComboBox 동기화
        if self.manual_btn.isChecked():
            current_combo_text = self.state_combo.currentText()
            if current_combo_text != state_str:
                self.state_combo.blockSignals(True)
                self.state_combo.setCurrentText(state_str)
                self.state_combo.blockSignals(False)
        
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
        
        # CLIP 결과 업데이트 (Interaction Mode에서만)
        if self.interaction_mode and self.llm_subscribed:
            self.clip_label_display.setText(self.clip_best_label.upper())
            self.clip_hz_label.setText(f"{self.clip_hz:.1f} Hz")
            
            for name in CLIP_LABELS:
                prob = self.clip_probs.get(name, 0.0)
                self.clip_label_bars[name].setValue(int(prob))
                self.clip_percent_labels[name].setText(f"{prob:.1f}%")
        
        # 타겟 버튼 업데이트
        self._update_target_buttons()
    
    def _update_topic_buttons(self):
        """Topic 구독 버튼 업데이트 (호환성용)"""
        pass
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        # LLM Stop 명령 전송
        if self.llm_subscribed:
            self.send_llm_control('stop')
        
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
