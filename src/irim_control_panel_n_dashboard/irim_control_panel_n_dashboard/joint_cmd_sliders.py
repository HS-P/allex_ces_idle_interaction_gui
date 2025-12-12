#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from PyQt5 import QtWidgets, QtCore
import sys
from functools import partial
import random, time
import math


import sys
import os

# 현재 파일의 경로를 가져온 뒤, 그 경로를 sys.path에 추가
current_dir = os.path.dirname(os.path.realpath(__file__))  # 현재 파일의 디렉토리 경로
sys.path.append(current_dir)

import ik_solver



# Arm 관련된 데이터는 아직 아무것도 제대로 된게 없음.

# Thumb:  CMC yaw, CMC roll, MCP
# Index:  ab/ad, MCP flex/ext, PIP flex/ext
# Middle: ab/ad, MCP flex/ext, PIP flex/ext
# Ring:   ab/ad, MCP flex/ext, PIP flex/ext
# Little: ab/ad, MCP flex/ext, PIP flex/ext

# 부호!!!!
# Thumb: Retroposition이 +, Flexion이 +, Flexion이 +
# Index 기준 Adduction 이 + 방향. Flexion이 +, Flexion이 +
# addcution은 middle 기준 좌우 대칭이지만, 4개의 손가락 모두 index와 같은 부호로 움직임.


# -- Global configuration --
NAMES = [
    # 'test_5dof',
    'Arm_R_theOne',
    'Arm_L_theOne',
    'Hand_R_thumb_wir',
    'Hand_R_index_wir',
    'Hand_R_middle_wir',
    'Hand_R_ring_wir',
    'Hand_R_little_wir',
    'Hand_L_thumb_wir',
    'Hand_L_index_wir',
    'Hand_L_middle_wir',
    'Hand_L_ring_wir',
    'Hand_L_little_wir'
]

# NAMES = [
#     'n2_Arm_R',
#     'n2_Arm_L',
#     'n2_Hand_R_thumb',
#     'n2_Hand_R_index',
#     'n2_Hand_R_middle',
#     'n2_Hand_R_ring',
#     'n2_Hand_R_little',
#     'n2_Hand_L_thumb',
#     'n2_Hand_L_index',
#     'n2_Hand_L_middle',
#     'n2_Hand_L_ring',
#     'n2_Hand_L_little'
# ]
NUM_JOINTS = [7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# NUM_JOINTS = [3, 3, 3, 3, 3]
SLIDER_STEPS = 1000


SLIDER_MINS = [
    # [-100, -100, -100, -100, -100],
    [-3.0, -3.4, -1.7, -2.8, -0.3, -1.0, -1.3],
    [-3.0, -0.17, -3.2, -2.8, -4.4, -0.8, -1.3],
    [-2.617994,       0.0, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0],
    [0.0,       0.0, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0],
    [-0.5,      -0.174532, 0.0]
]
SLIDER_MAXS = [
    # [100, 100, 100, 100, 100],
    [ 1.7,  0.17,  3.0,  0.0,  4.5,  1.0,  1.3],
    [ 1.7,  3.4,  1.7,  0.0,  0.4,  0.5,  1.3],
    [ 0.0,  1.570796,  1.570796],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293],
    [ 2.617994,  1.570796,  1.570796],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293],
    [ 0.5,  1.570796,  1.7453293]
]

class JointCommandPublisherNode(Node):
    def __init__(self):
        super().__init__('joint_command_publisher')

        # publisher 맵
        self._publishers_map = {}
        # topic으로부터 받은 목표 각도를 저장할 맵
        self._target_angles_map = {}

        for name in NAMES:
            # inbound publisher
            in_topic = f'/robot_inbound/{name}/joint_command'
            pub = self.create_publisher(Float64MultiArray, in_topic, 10)
            self._publishers_map[name] = pub
            self.get_logger().info(f'Created publisher for {in_topic}')

            # outbound subscriber
            out_topic = f'/robot_outbound_data/{name}/joint_ang_target_deg'
            self._target_angles_map[name] = None
            self.create_subscription(
                Float64MultiArray,
                out_topic,
                partial(self._target_callback, name),
                10
            )
            self.get_logger().info(f'Created subscriber for {out_topic}')

    def _target_callback(self, name, msg):
        # 최신 목표 각도 저장 (radians)
        self._target_angles_map[name] = [math.radians(d) for d in msg.data]

    def get_target(self, name):
        # MainWindow에서 호출
        return self._target_angles_map.get(name)

    def publish(self, name, data):
        msg = Float64MultiArray()
        msg.data = data
        publisher = self._publishers_map.get(name)
        if publisher:
            publisher.publish(msg)
        else:
            self.get_logger().error(f'No publisher for {name}')


class MainWindow(QtWidgets.QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('Joint Command Publisher')
        self.sliders = {}
        self.spinboxes = {}
        self.trans_scales = {}
        self.rot_scales = {}
        self.cart_buttons = {}
        self.null_values = {}

        # self.null_dials = {}

        self._setup_ui()
        self._setup_timers()

        self.k = 50.0            # 스프링 상수 (튜닝 가능)
        self.c = 2.0 * math.sqrt(self.k)  # 임계 댐핑
        self.dt = 0.010          # publish_timer 간격(초)
        self.filter_states = {}
        for name in NAMES:
            # 초기 위치는 슬라이더 초기값으로, 속도는 0
            idx = NAMES.index(name)
            init = [self._compute_value(idx, j, slider.value())
                    for j, slider in enumerate(self.sliders.get(name, []))]
            self.filter_states[name] = {'x': init, 'v': [0.0]*len(init)}



    def _spring_damper(self, name, raw_angles):
        state = self.filter_states[name]
        x, v = state['x'], state['v']
        new_x, new_v = [], []
        for xi, vi, target in zip(x, v, raw_angles):
            a = self.k * (target - xi) - self.c * vi
            vi_next = vi + a * self.dt
            xi_next = xi + vi_next * self.dt
            new_v.append(vi_next)
            new_x.append(xi_next)
        state['x'], state['v'] = new_x, new_v
        return new_x


    def _setup_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # Tab widget for 4 categories
        self.tabs = QtWidgets.QTabWidget()
        main_layout.addWidget(self.tabs)

        categories = ['Arm_L', 'Arm_R', 'Hand_L', 'Hand_R']
        for cat in categories:
            tab = QtWidgets.QWidget()
            tab_layout = QtWidgets.QVBoxLayout(tab)

            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.verticalScrollBar().setStyleSheet("""
                QScrollBar:vertical {
                    width: 30px;
                }
            """)
            
            container = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(container)

            for idx, name in enumerate(NAMES):
                if cat in name:  # 'Arm_L', 'Arm_R', 'Hand_L', 'Hand_R' 이 포함되면 OK
                    group = self._create_control_group(name, idx)
                    vbox.addWidget(group)

            container.setLayout(vbox)
            scroll.setWidget(container)
            tab_layout.addWidget(scroll)
            self.tabs.addTab(tab, cat)

        # Global controls
        h_layout = QtWidgets.QHBoxLayout()
        zero_btn = QtWidgets.QPushButton('Zero All')
        zero_btn.clicked.connect(self._zero_all)
        self.sync_button = QtWidgets.QPushButton('Sync All')
        self.sync_button.setCheckable(True)
        self.sync_button.toggled.connect(self._on_sync_toggled)
        self.sync_button.setChecked(True)  # <-- 기본적으로 눌린 상태로 시작
        self.sync_button.setToolTip('ON: reflect topic data in GUI')
        h_layout.addWidget(zero_btn)
        h_layout.addWidget(self.sync_button)
        main_layout.addLayout(h_layout)

        self.setLayout(main_layout)

    def _create_control_group(self, name, idx):
        group = QtWidgets.QGroupBox(name)
        layout = QtWidgets.QVBoxLayout()

        sliders_list = []
        spinboxes_list = []

        for j in range(NUM_JOINTS[idx]):
            hl = QtWidgets.QHBoxLayout()
            minv, maxv = SLIDER_MINS[idx][j], SLIDER_MAXS[idx][j]

            min_label = QtWidgets.QLabel(f"{minv:.2f}")
            max_label = QtWidgets.QLabel(f"{maxv:.2f}")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, SLIDER_STEPS)
            zero_ratio = (0 - minv) / (maxv - minv) if minv <= 0 <= maxv else 0.5
            slider.setValue(int(zero_ratio * SLIDER_STEPS))

            spin = QtWidgets.QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(math.degrees(minv), math.degrees(maxv))
            spin.setSingleStep(1.0)
            spin.setValue(math.degrees(self._compute_value(idx, j, slider.value())))

            slider.valueChanged.connect(partial(self._update_spinbox, idx, j, spin))
            spin.valueChanged.connect(partial(self._on_spinbox_change, idx, j, spin))
            slider.sliderPressed.connect(self._disable_sync)
            spin.editingFinished.connect(self._disable_sync)

            hl.addWidget(min_label)
            hl.addWidget(slider, 1)
            hl.addWidget(spin)
            hl.addWidget(max_label)

            layout.addLayout(hl)
            sliders_list.append(slider)
            spinboxes_list.append(spin)

        self.sliders[name] = sliders_list
        self.spinboxes[name] = spinboxes_list

        # Cartesian controls omitted for brevity; include as in original
        # ...
        cart_group = QtWidgets.QGroupBox("Cartesian Control")
        grid = QtWidgets.QGridLayout()

        # 1) 스케일 콤보박스
        trans_cb = QtWidgets.QComboBox()
        for mm in (1, 5, 10):
            trans_cb.addItem(f"{mm} mm", mm / 1000.0)
        rot_cb = QtWidgets.QComboBox()
        for deg in (0.5, 1.0, 5.0):
            rot_cb.addItem(f"{deg}°", math.radians(deg))
        self.trans_scales[name] = trans_cb
        self.rot_scales[name] = rot_cb
        grid.addWidget(QtWidgets.QLabel("Trans step:"), 0, 0)
        grid.addWidget(trans_cb, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Rot step:"),   0, 2)
        grid.addWidget(rot_cb,   0, 3)

        # 2) 축별 +/– 버튼
        axes = ["X", "Y", "Z"]
        if NUM_JOINTS[idx] == 7:
            axes += ["Wx", "Wy", "Wz"]
        self.cart_buttons[name] = []
        for i, axis in enumerate(axes, start=1):
            btn_pos = QtWidgets.QPushButton(f"{axis}+")
            btn_neg = QtWidgets.QPushButton(f"{axis}-")
            btn_pos.clicked.connect(partial(self._on_cartesian_cmd, name, axis, +1))
            btn_neg.clicked.connect(partial(self._on_cartesian_cmd, name, axis, -1))
            grid.addWidget(btn_pos, i, 0)
            grid.addWidget(btn_neg, i, 1)
            self.cart_buttons[name].append((btn_pos, btn_neg))


        if 'Arm_' in name:
            # 초기값
            self.null_values[name] = 0.0

            btn_layout = QtWidgets.QHBoxLayout()
            btn_inc = QtWidgets.QPushButton('Null+')
            btn_dec = QtWidgets.QPushButton('Null–')
            lbl = QtWidgets.QLabel(f"{self.null_values[name]:.2f}")

            # 버튼 클릭 시 null 값 변경
            def make_handler(n, delta, label):
                def _handler():
                    v = self.null_values[n] + delta
                    # –1.0 ~ +1.0 로 클램프
                    v = max(-1.0, min(1.0, v))
                    self.null_values[n] = v
                    label.setText(f"{v:.2f}")
                return _handler

            btn_inc.clicked.connect(make_handler(name, +0.01, lbl))
            btn_dec.clicked.connect(make_handler(name, -0.01, lbl))

            btn_layout.addWidget(QtWidgets.QLabel("Null motion:"))
            btn_layout.addWidget(btn_dec)
            btn_layout.addWidget(lbl)
            btn_layout.addWidget(btn_inc)
            layout.addLayout(btn_layout)


        # # — Arm 전용 null‐motion 다이얼 —
        # if 'Arm_' in name:
        #     dial_layout = QtWidgets.QHBoxLayout()
        #     dial_label = QtWidgets.QLabel("Null motion:")
        #     dial = QtWidgets.QDial()
        #     dial.setRange(-100, 100)                # 예: -1.0 … +1.0 (스케일링)
        #     dial.setValue(0)
        #     dial.setNotchesVisible(True)
        #     dial_value_label = QtWidgets.QLabel("0.00")
        #     dial.valueChanged.connect(
        #         lambda val, lbl=dial_value_label: lbl.setText(f"{val/100:.2f}")
        #     )
        #     dial_layout.addWidget(dial_label)
        #     dial_layout.addWidget(dial)
        #     dial_layout.addWidget(dial_value_label)
        #     layout.addLayout(dial_layout)
        #     self.null_dials[name] = dial

        cart_group.setLayout(grid)
        layout.addWidget(cart_group)

        group.setLayout(layout)
        return group

    def _setup_timers(self):
        self.spin_timer = QtCore.QTimer()
        self.spin_timer.timeout.connect(self._ros_spin)
        self.spin_timer.start(1)

        self.publish_timer = QtCore.QTimer()
        self.publish_timer.timeout.connect(self._publish_all)
        self.publish_timer.start(10)

        self.sync_timer = QtCore.QTimer()
        self.sync_timer.timeout.connect(self._update_from_topics)
        self.sync_timer.start(50)


    def _on_cartesian_cmd(self, name, axis, direction):
        self._disable_sync()
        # 로봇 인덱스 계산
        idx = NAMES.index(name)
        # prepare delta
        step_trans = self.trans_scales[name].currentData()
        step_rot = self.rot_scales[name].currentData()
        delta = [0]*6
        idx_map = {'X':0, 'Y':1, 'Z':2, 'Wx':3, 'Wy':4, 'Wz':5}
        i = idx_map[axis]
        if i < 3:
            delta[i] = direction * step_trans
        else:
            delta[i] = direction * step_rot

        # 현재 목표 관절값 가져오기
        current = self.node.get_target(name)
        if current is None:
            # UI로부터 계산
            current = [
                self._compute_value(idx, j, self.sliders[name][j].value())
                for j in range(NUM_JOINTS[idx])
            ]

        # IK 계산
        # new_angles = ik_solver.solve_ik(name, current, delta)
        # Arm일 때만 null_scalar 읽어오기
        null_scalar = 0.0
        if 'Arm_' in name:
            # null_scalar = self.null_dials[name].value() / 100.0  # -1.0 ~ +1.0
            null_scalar = self.null_values.get(name, 0.0)

        # null 다이얼 입력 여부에 따라 method와 null_motion 분기
        if null_scalar != 0.0:
            # null 다이얼이 들어왔을 때 → SVD + null-space motion
            method = 'svd'
            # current_angles 길이에 맞춰 null_motion 벡터 생성
            null_motion = [null_scalar] * len(current)
        else:
            # 버튼 조작만 있을 때 → DLS, null_motion 없음
            method = 'dls'
            null_motion = None
    


        method = 'dls'
        new_angles = ik_solver.solve_ik(
            robot_name=name,
            current_angles=current,
            delta_x=delta,
            damping=0.1,
            method=method,
            # null_motion=null_motion
        )

        # print(null_motion)


        # 슬라이더 및 스핀박스 업데이트
        for j, angle in enumerate(new_angles):
            minv, maxv = SLIDER_MINS[idx][j], SLIDER_MAXS[idx][j]
            if maxv != minv:
                ratio = (angle - minv) / (maxv - minv)
                self.sliders[name][j].blockSignals(True)
                self.sliders[name][j].setValue(int(ratio * SLIDER_STEPS))
                self.sliders[name][j].blockSignals(False)
            self.spinboxes[name][j].blockSignals(True)
            self.spinboxes[name][j].setValue(math.degrees(angle))
            self.spinboxes[name][j].blockSignals(False)

        # Publish
        if not self.sync_button.isChecked():
            self.node.publish(name, new_angles)


    def _disable_sync(self, *args):
        """사용자 조작이 감지되면 Sync 토글 해제"""
        if self.sync_button.isChecked():
            self.sync_button.blockSignals(True)
            self.sync_button.setChecked(False)
            self.sync_button.blockSignals(False)

    def _update_from_topics(self):
        """Sync가 ON인 그룹에 대해 토픽값 → GUI 반영"""
        if not self.sync_button.isChecked():
            return

        for idx, name in enumerate(NAMES):
            data = self.node.get_target(name)
            if data is None:
                continue
            # programmatic update: signals 차단
            for j, val_rad in enumerate(data):
                # slider
                slider = self.sliders[name][j]
                slider.blockSignals(True)
                # ratio → slider value
                minv, maxv = SLIDER_MINS[idx][j], SLIDER_MAXS[idx][j]
                ratio = (val_rad - minv) / (maxv - minv) if maxv != minv else 0.0
                slider.setValue(int(ratio * SLIDER_STEPS))
                slider.blockSignals(False)
                # spinbox
                spinbox = self.spinboxes[name][j]
                spinbox.blockSignals(True)
                spinbox.setValue(math.degrees(val_rad))
                spinbox.blockSignals(False)


    def _compute_value(self, idx, j, slider_value):
        """슬라이더 값 → 실제 관절 값 계산"""
        ratio = slider_value / SLIDER_STEPS
        minv = SLIDER_MINS[idx][j]
        maxv = SLIDER_MAXS[idx][j]
        return minv + ratio * (maxv - minv)

    def _update_spinbox(self, idx, j, spinbox, slider_value):
        """슬라이더가 움직일 때 스핀박스 업데이트"""
        val_rad = self._compute_value(idx, j, slider_value)
        val_deg = math.degrees(val_rad)
        spinbox.blockSignals(True)
        spinbox.setValue(val_deg)
        spinbox.blockSignals(False)

    def _on_spinbox_change(self, idx, j, spinbox, deg_value):
        """스핀박스 값 변경 시 슬라이더 업데이트"""
        rad = math.radians(deg_value)
        minv = SLIDER_MINS[idx][j]
        maxv = SLIDER_MAXS[idx][j]
        ratio = (rad - minv) / (maxv - minv) if maxv != minv else 0.0
        slider_val = int(ratio * SLIDER_STEPS)
        slider = self.sliders[NAMES[idx]][j]
        slider.blockSignals(True)
        slider.setValue(slider_val)
        slider.blockSignals(False)

    def _publish_all(self):
        if self.sync_button.isChecked():
            return
        for idx, name in enumerate(NAMES):
            raw = [self._compute_value(idx, j, slider.value())
                   for j, slider in enumerate(self.sliders[name])]
            filtered = self._spring_damper(name, raw)
            self.node.publish(name, filtered)
    
    def _on_sync_toggled(self, checked: bool):
        self.sync_button.setText("현재 Target 값으로 update" if checked else "cmd publishing")


    def _ros_spin(self):
        rclpy.spin_once(self.node, timeout_sec=0)

    def _zero_all(self):
        """모든 슬라이더를 0 rad 위치로 설정"""
        for idx, name in enumerate(NAMES):
            for j, slider in enumerate(self.sliders[name]):
                minv = SLIDER_MINS[idx][j]
                maxv = SLIDER_MAXS[idx][j]
                if minv <= 0 <= maxv:
                    zero_ratio = (0 - minv) / (maxv - minv)
                    init_val = int(zero_ratio * SLIDER_STEPS)
                else:
                    init_val = SLIDER_STEPS // 2
                slider.setValue(init_val)


def main():
    rclpy.init()
    node = JointCommandPublisherNode()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(node)
    window.show()

    exit_code = app.exec_()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
