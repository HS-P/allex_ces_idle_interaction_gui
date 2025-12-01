#!/usr/bin/env python3
"""
GUI Node - GUI를 관리하고 여러 Topic을 동적으로 구독하는 통합 Node
v1.1.0 - PySide6 사용, 통합 토픽 설정 파일 지원
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
    QLabel, QPushButton, QComboBox, QGroupBox, QGridLayout, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QCoreApplication
from PySide6.QtGui import QMouseEvent

from .tracker_manager import TrackingState

# TrackedObject를 위한 간단한 구조체
TrackedObject = namedtuple('TrackedObject', [
    'track_id', 'centroid', 'state', 'confidence', 'age'
])
TargetInfo = namedtuple('TargetInfo', [
    'point', 'state', 'track_id'
])


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
    clicked_with_id = Signal(int)  # track_id를 포함한 커스텀 시그널
    
    def __init__(self, track_id: int, parent=None):
        super().__init__(parent)
        self.track_id = track_id
        self.setCheckable(False)
        self.setEnabled(True)
        # 커스텀 시그널을 부모의 핸들러에 연결
        if parent and hasattr(parent, '_on_target_button_clicked'):
            self.clicked_with_id.connect(parent._on_target_button_clicked)
    
    def mousePressEvent(self, event: QMouseEvent):
        """마우스 클릭 이벤트 직접 처리"""
        if event.button() == Qt.LeftButton:
            self.clicked_with_id.emit(self.track_id)
        super().mousePressEvent(event)


class GuiSignals(QObject):
    """GUI와 메인 쓰레드 간 통신을 위한 시그널"""
    mode_changed = Signal(bool)  # True: Manual, False: Auto
    state_changed = Signal(str)  # State 이름
    update_target_buttons = Signal()  # 타겟 버튼 업데이트 시그널
    update_topic_buttons = Signal()  # Topic 버튼 업데이트 시그널


class GuiNode(Node, QMainWindow):
    """GUI를 관리하고 여러 Topic을 동적으로 구독하는 통합 Node"""
    
    def __init__(self) -> None:
        # ROS2 Node 초기화
        Node.__init__(self, "gui_node")
        # Qt Window 초기화
        QMainWindow.__init__(self)
        
        # Topic 설정 파일 로드
        self.topic_config_path = self._get_topic_config_path()
        self.topic_config = self._load_topic_config()
        
        # 구독 관리 딕셔너리
        self.topic_subscriptions: Dict[str, Any] = {}
        
        # 시그널 생성
        self.signals = GuiSignals()
        
        # 상태 정보 저장 (Camera Publisher에서 받은 데이터)
        self.current_state = TrackingState.IDLE
        self.current_target_info = None
        self.tracked_objects = []
        self.fps = 0.0
        self.process_time_ms = 0.0
        self.center_zone_elapsed_time = None
        self.center_zone_duration = 5.0
        
        # GUI 모드 관리
        self.interaction_mode = False  # True: Interaction Mode, False: IDLE Mode
        self.is_running = False  # IDLE Mode에서 RUN 버튼으로 시작 여부
        
        # LLM 관련 상태
        self.llm_subscribed = False  # LLM 토픽 구독 여부
        self.llm_started = False  # LLM 시작 여부
        
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
        
        # Camera Publisher 데이터 구독
        self._setup_camera_subscription()
        
        # Manual 제어 Publisher
        self.manual_control_publisher = self.create_publisher(
            String,
            self._get_topic_name('camera', 'manual_control'),
            10
        )
        
        # 타이머로 주기적으로 정보 업데이트
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_info)
        self.update_timer.start(50)  # 20Hz 업데이트
        
        self.get_logger().info("GUI Node v1.1.0 초기화 완료")
        self.get_logger().info(f"토픽 설정 파일: {self.topic_config_path}")
    
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
            
            # 상태 정보 업데이트
            state_str = data.get('state', 'idle')
            try:
                self.current_state = TrackingState[state_str.upper()]
            except (KeyError, AttributeError):
                self.current_state = TrackingState.IDLE
            
            # 타겟 정보 업데이트
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
            
            # 추적 객체 정보 업데이트
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
            
            # 성능 정보 업데이트
            performance_data = data.get('performance', {})
            self.fps = performance_data.get('fps', 0.0)
            self.process_time_ms = performance_data.get('process_time_ms', 0.0)
            
            # 목 각도 정보 저장
            neck_angles = data.get('neck_angles', {})
            self.neck_current_yaw = neck_angles.get('current', {}).get('yaw_rad', 0.0)
            self.neck_current_pitch = neck_angles.get('current', {}).get('pitch_rad', 0.0)
            self.neck_target_yaw = neck_angles.get('target', {}).get('yaw_rad', 0.0)
            self.neck_target_pitch = neck_angles.get('target', {}).get('pitch_rad', 0.0)
            
            # 허리 각도 정보 저장
            waist_angles = data.get('waist_angles', {})
            if waist_angles:
                self.waist_target_yaw = waist_angles.get('target', {}).get('yaw_rad', 0.0)
            
            # Center Zone 정보 저장
            center_zone_data = data.get('center_zone', {})
            self.center_zone_elapsed_time = center_zone_data.get('elapsed_time')
            self.center_zone_duration = center_zone_data.get('duration', 5.0)
            
            # 타겟 버튼 업데이트
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
            Path(__file__).parent.parent / "config" / "topics.json",  # 소스 코드 위치
            Path(__file__).parent.parent.parent.parent / "config" / "topics.json",  # 프로젝트 루트
        ]
        
        # 설치된 환경에서 share 디렉토리 찾기
        current_path = Path(__file__).resolve()
        parts = current_path.parts
        if 'install' in parts:
            install_idx = parts.index('install')
            if install_idx + 1 < len(parts):
                install_base = Path(*parts[:install_idx + 2])
                share_path = install_base / "share" / "camera_node" / "config" / "topics.json"
                possible_paths.insert(0, share_path)
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # 기본값: 소스 코드 위치
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
            # 기본 설정 반환
            return {
                "camera": {
                    "tracking_data": {"name": "/allex_camera/tracking_data", "type": "std_msgs/String"},
                    "manual_control": {"name": "/allex_camera/manual_control", "type": "std_msgs/String"}
                },
                "robot": {
                    "waist_position": {"name": "/robot_outbound_data/theOne_waist/joint_positions_deg", "type": "std_msgs/Float64MultiArray"}
                },
                "llm": {
                    "response": {"name": "/llm/response", "type": "std_msgs/String"}
                }
            }
    
    def subscribe_llm_topic(self) -> bool:
        """LLM 응답 토픽 구독 시작"""
        llm_topic = self._get_topic_name('llm', 'response')
        if not llm_topic:
            self.get_logger().error("LLM 토픽 설정을 찾을 수 없습니다.")
            return False
        
        if llm_topic in self.topic_subscriptions:
            self.get_logger().warn(f"LLM Topic '{llm_topic}'은 이미 구독 중입니다.")
            return False
        
        try:
            qos_profile = QoSProfile(
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                deadline=Duration(seconds=0, nanoseconds=0),
            )
            
            subscription = self.create_subscription(
                String,
                llm_topic,
                self._llm_response_callback,
                qos_profile
            )
            
            self.topic_subscriptions[llm_topic] = {
                'subscription': subscription,
                'type': 'std_msgs/String',
                'enabled': True
            }
            
            self.llm_subscribed = True
            self.get_logger().info(f"LLM Topic 구독 시작: {llm_topic}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"LLM Topic 구독 실패: {e}")
            return False
    
    def unsubscribe_llm_topic(self) -> bool:
        """LLM 응답 토픽 구독 해제"""
        llm_topic = self._get_topic_name('llm', 'response')
        if llm_topic not in self.topic_subscriptions:
            self.get_logger().warn(f"LLM Topic '{llm_topic}'은 구독 중이 아닙니다.")
            return False
        
        try:
            del self.topic_subscriptions[llm_topic]
            self.llm_subscribed = False
            self.get_logger().info(f"LLM Topic 구독 해제: {llm_topic}")
            return True
        except Exception as e:
            self.get_logger().error(f"LLM Topic 구독 해제 실패: {e}")
            return False
    
    def _llm_response_callback(self, msg: String) -> None:
        """LLM 응답 메시지 콜백"""
        self.get_logger().info(f"[LLM Response] {msg.data}")
        # TODO: LLM 응답 처리 로직 추가
    
    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle("Person Tracking Control Panel v1.1.0")
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
        
        # 운영 모드 선택 (Auto/Manual)
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
        
        # LLM Topic Subscribe 버튼
        self.llm_subscribe_btn = QPushButton("LLM Topic Subscribe")
        self.llm_subscribe_btn.setCheckable(True)
        self.llm_subscribe_btn.setChecked(False)
        self.llm_subscribe_btn.setMinimumHeight(50)
        self.llm_subscribe_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.llm_subscribe_btn.clicked.connect(self._on_llm_subscribe_clicked)
        interaction_control_layout.addWidget(self.llm_subscribe_btn)
        
        # LLM Start 버튼
        self.llm_start_btn = QPushButton("LLM Start")
        self.llm_start_btn.setCheckable(True)
        self.llm_start_btn.setChecked(False)
        self.llm_start_btn.setMinimumHeight(50)
        self.llm_start_btn.setEnabled(False)  # Subscribe 후에만 활성화
        self.llm_start_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.llm_start_btn.clicked.connect(self._on_llm_start_clicked)
        interaction_control_layout.addWidget(self.llm_start_btn)
        
        # Interaction RUN 버튼
        self.interaction_run_btn = QPushButton("RUN (Interaction)")
        self.interaction_run_btn.setCheckable(True)
        self.interaction_run_btn.setChecked(False)
        self.interaction_run_btn.setMinimumHeight(60)
        self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
        self.interaction_run_btn.clicked.connect(self.on_interaction_run_clicked)
        interaction_control_layout.addWidget(self.interaction_run_btn)
        
        self.interaction_control_group.setLayout(interaction_control_layout)
        self.interaction_control_group.setVisible(False)  # 초기에는 숨김
        left_layout.addWidget(self.interaction_control_group)
        
        # 왼쪽 패널을 메인 레이아웃에 추가
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
        
        # 오른쪽 패널을 메인 레이아웃에 추가
        top_layout.addWidget(right_panel, 1)
        
        # 상단 레이아웃을 메인에 추가
        main_layout.addLayout(top_layout)
        
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
        
        # 타겟 버튼 미리 생성 (최대 10개)
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
        
        # 레이아웃 설정 완료 후 업데이트
        QCoreApplication.processEvents()
        self._update_target_buttons()
        QCoreApplication.processEvents()
        
        self.get_logger().info(f"초기화 완료: 타겟 버튼={len(self.target_buttons)}개")
    
    def _on_llm_subscribe_clicked(self):
        """LLM Subscribe 버튼 클릭 이벤트"""
        if self.llm_subscribe_btn.isChecked():
            # 구독 시작
            success = self.subscribe_llm_topic()
            if success:
                self.llm_subscribe_btn.setText("✓ LLM Topic Subscribed")
                self.llm_subscribe_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
                self.llm_start_btn.setEnabled(True)  # Start 버튼 활성화
                self.get_logger().info("LLM Topic 구독 성공")
            else:
                self.llm_subscribe_btn.setChecked(False)
                self.get_logger().error("LLM Topic 구독 실패")
        else:
            # 구독 해제
            success = self.unsubscribe_llm_topic()
            if success:
                self.llm_subscribe_btn.setText("LLM Topic Subscribe")
                self.llm_subscribe_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                self.llm_start_btn.setEnabled(False)  # Start 버튼 비활성화
                self.llm_start_btn.setChecked(False)
                self.llm_start_btn.setText("LLM Start")
                self.llm_start_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                self.llm_started = False
                self.get_logger().info("LLM Topic 구독 해제")
            else:
                self.llm_subscribe_btn.setChecked(True)
    
    def _on_llm_start_clicked(self):
        """LLM Start 버튼 클릭 이벤트"""
        if self.llm_start_btn.isChecked():
            # LLM 시작
            self.llm_started = True
            self.llm_start_btn.setText("✓ LLM Started")
            self.llm_start_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #90EE90; color: black;")
            self.get_logger().info("LLM 시작됨")
            # TODO: LLM 시작 로직 추가
        else:
            # LLM 중지
            self.llm_started = False
            self.llm_start_btn.setText("LLM Start")
            self.llm_start_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.get_logger().info("LLM 중지됨")
            # TODO: LLM 중지 로직 추가
    
    def set_interaction_mode(self, interaction: bool):
        """Interaction/IDLE Mode 설정"""
        # 모드 전환 시 RUN 중이면 강제 STOP
        if self.is_running:
            self.run_btn.setChecked(False)
            self.interaction_run_btn.setChecked(False)
            self.is_running = False
            self.run_btn.setText("RUN")
            self.run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self.interaction_run_btn.setText("RUN (Interaction)")
            self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self._send_manual_control({'type': 'stop'})
            self.get_logger().info("모드 전환: 강제 STOP 수행")
        
        self.interaction_mode = interaction
        if interaction:
            self.interaction_btn.setChecked(True)
            self.idle_btn.setChecked(False)
            self.interaction_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #90EE90;")
            self.idle_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #E0E0E0;")
            self.get_logger().info("Interaction Mode 선택됨")
            
            # IDLE Mode 제어 숨기기
            self.idle_control_group.setVisible(False)
            self.state_group.setVisible(False)
            
            # Interaction Mode 제어 표시
            self.interaction_control_group.setVisible(True)
            
            # Camera Publisher에 Interaction Mode 명령 전송
            self._send_manual_control({
                'type': 'set_interaction_mode',
                'enabled': True
            })
        else:
            self.interaction_btn.setChecked(False)
            self.idle_btn.setChecked(True)
            self.interaction_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #E0E0E0;")
            self.idle_btn.setStyleSheet("font-size: 14pt; font-weight: bold; background-color: #90EE90;")
            self.get_logger().info("IDLE Mode 선택됨")
            
            # Interaction Mode 제어 숨기기
            self.interaction_control_group.setVisible(False)
            
            # IDLE Mode 제어 표시
            self.idle_control_group.setVisible(True)
            self.state_group.setVisible(True)
            
            # LLM 구독 해제
            if self.llm_subscribed:
                self.unsubscribe_llm_topic()
                self.llm_subscribe_btn.setChecked(False)
                self.llm_subscribe_btn.setText("LLM Topic Subscribe")
                self.llm_subscribe_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                self.llm_start_btn.setEnabled(False)
                self.llm_start_btn.setChecked(False)
                self.llm_start_btn.setText("LLM Start")
                self.llm_start_btn.setStyleSheet("font-size: 12pt; font-weight: bold; background-color: #E0E0E0; color: black;")
                self.llm_started = False
            
            # Camera Publisher에 IDLE Mode 명령 전송
            self._send_manual_control({
                'type': 'set_interaction_mode',
                'enabled': False
            })
    
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
            
            self._send_manual_control({
                'type': 'run',
                'manual': False  # Interaction Mode는 항상 Auto
            })
            self.get_logger().info("RUN 시작: Interaction Mode (BB Box 추적)")
        else:
            self.is_running = False
            self.interaction_run_btn.setText("RUN (Interaction)")
            self.interaction_run_btn.setStyleSheet("font-size: 16pt; font-weight: bold; background-color: #E0E0E0; color: black;")
            self._send_manual_control({'type': 'stop'})
            self.get_logger().info("RUN 중지: Interaction Mode 종료")
    
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
        
        self.get_logger().debug(f"타겟 버튼 업데이트: {len(tracked_ids)}개 표시 (현재 타겟: {current_target_id})")
    
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
        
        # 타겟 버튼 업데이트
        self._update_target_buttons()
    
    def _update_topic_buttons(self):
        """Topic 구독 버튼 업데이트 (사용하지 않음 - 호환성용)"""
        pass
    
    def closeEvent(self, event):
        """창 닫기 이벤트"""
        self.update_timer.stop()
        event.accept()


def main(args=None):
    """메인 함수"""
    # Qt 플러그인 경로 설정
    try:
        setup_qt_plugin_path()
    except Exception as e:
        print(f"경고: Qt 플러그인 경로 설정 실패: {e}")
    
    # QApplication 생성
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # ROS2 초기화
    rclpy.init(args=args)
    gui_node = GuiNode()
    gui_node.show()
    
    try:
        # 별도 쓰레드에서 ROS2 spin 실행
        ros_thread_running = True
        
        def ros_spin():
            nonlocal ros_thread_running
            while ros_thread_running:
                rclpy.spin_once(gui_node, timeout_sec=0.1)
        
        ros_thread = threading.Thread(target=ros_spin, daemon=True)
        ros_thread.start()
        
        # 메인 쓰레드에서 Qt 이벤트 루프 실행
        app.exec()  # PySide6에서는 exec_() 대신 exec() 사용
        ros_thread_running = False
    except KeyboardInterrupt:
        pass
    finally:
        gui_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
