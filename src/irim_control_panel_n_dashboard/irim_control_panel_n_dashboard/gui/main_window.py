# gui/main_window.py

from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTabWidget

from PyQt5.QtCore import pyqtSignal, QTimer  # 시그널 + 업데이트 타이머
from irim_control_panel_n_dashboard.gui.command_panel import CommandPanel
from irim_control_panel_n_dashboard.gui.status_panel import StatusPanel
from irim_control_panel_n_dashboard.gui.routine_panel import RoutinePanel



class MainWindow(QWidget):
    status_update_signal = pyqtSignal(str, dict)
    routine_update_signal = pyqtSignal(dict)

    def __init__(self, ros_interface):
        super().__init__()
        self.ros_interface = ros_interface
        # 각 articulation별 상세 joint 데이터를 저장할 dict
        self.all_details = {art: [] for art in self.ros_interface.articulations}
        self.init_ui()
        # # ROSInterface의 콜백을 MainWindow의 핸들러로 연결
        self.ros_interface.status_update_callback = self.handle_status_update


        # ROS 콜백은 직접 handle_status_update를 호출하지 않고,
        # 시그널을 emit하여 메인 스레드에서 업데이트하도록 함
        
        # 시그널과 슬롯 연결: 시그널이 메인 스레드에서 handle_status_update를 호출하게 됨
        # status 업데이트
        self.status_update_signal.connect(self.handle_status_update)
        self.ros_interface.status_update_callback = self.status_update_signal.emit

        # routine 업데이트
        self.routine_update_signal.connect(self.handle_routine_update)
        self.ros_interface.routine_update_callback = self.routine_update_signal.emit


        # --- 추가: 업데이트 버퍼 및 30Hz 타이머 ---
        self._pending_status_updates = {}  # articulation -> latest data
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(33)  # 약 30Hz
        self._status_timer.timeout.connect(self._flush_pending_updates)
        self._status_timer.start()


    def init_ui(self):
        main_layout = QHBoxLayout()

        # 왼쪽: 명령 전송 패널
        self.command_panel = CommandPanel(self.publish_command, self.ros_interface.articulations)
        self.command_panel.setMinimumWidth(600)
        main_layout.addWidget(self.command_panel)

        # 오른쪽: 탭 위젯 (Status / Routine)
        self.status_panel = StatusPanel(self.ros_interface.articulations)
        self.status_panel.setMinimumWidth(600)

        self.routine_panel = RoutinePanel()
        self.routine_panel.setMinimumWidth(600)

        tabs = QTabWidget()
        tabs.addTab(self.status_panel, "Status")
        tabs.addTab(self.routine_panel, "Routine")
        # 필요하면 탭 위치/모양도 조정 가능:
        # tabs.setTabPosition(QTabWidget.North)
        # tabs.setMovable(True)

        main_layout.addWidget(tabs)


        self.setLayout(main_layout)
        self.setWindowTitle("Robot Command HMI")
        self.setMinimumWidth(1200)
        self.setMinimumHeight(600)

    def publish_command(self, message_str):
        self.ros_interface.publish_command(message_str)

    def handle_status_update(self, articulation, data):
        # 들어오는 업데이트는 즉시 UI에 반영하지 않고 최신 값만 버퍼링함.
        # 실제 UI 반영은 _status_timer에 의해 초당 ~30Hz로 이루어짐.
        self._pending_status_updates[articulation] = data

    def _flush_pending_updates(self):
        if not self._pending_status_updates:
            return
        pending = self._pending_status_updates.copy()
        self._pending_status_updates.clear()
        for articulation, data in pending.items():
            # 모든 topic에 대해 데이터가 수신된 경우에만 업데이트
            if (data.get("articulation_now") is not None and
                data.get("servo_status") is not None and
                data.get("control_mode") is not None and
                data.get("communication_code") is not None):

                status_list = data.get("articulation_now", [])
                if len(status_list) < 7:
                    status_list = status_list + ["N/A"] * (7 - len(status_list))

                self.status_panel.update_articulation_status(articulation, status_list)

                detail_rows = []
                for i in range(len(data.get("servo_status", []))):
                    row = {
                        "Name": f"{articulation}_{i}",
                        "servo": data["servo_status"][i] if i < len(data["servo_status"]) else "N/A",
                        "comm_code": data["communication_code"][i] if i < len(data["communication_code"]) else "N/A",
                        "Cont_mode": data["control_mode"][i] if i < len(data["control_mode"]) else "N/A"
                    }
                    detail_rows.append(row)

                self.all_details[articulation] = detail_rows
        # 모든 articulation의 상세를 합쳐서 한 번만 갱신
        combined_details = []
        for art, rows in self.all_details.items():
            combined_details.extend(rows)
        self.status_panel.update_detail_table(combined_details)


    def handle_routine_update(self, data: dict):
        """
        /debug/routine JSON dict 를 RoutinePanel 에 전달
        """
        if self.routine_panel is not None:
            self.routine_panel.update_routine(data)