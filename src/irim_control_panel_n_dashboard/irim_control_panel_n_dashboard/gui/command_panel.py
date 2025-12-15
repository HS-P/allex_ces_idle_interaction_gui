import os
import glob
from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem,
    QAbstractItemView, QLineEdit,
    QTableWidget, QTableWidgetItem, 
    QInputDialog,                   
)
from PyQt5.QtCore import Qt

from irim_control_panel_n_dashboard.config import (
    COMMAND_TYPES,
    ROBOT_STATUS,
    SPECIFIC_MODES,
    ROBOT_SCENARIO,
    ROUTINE_ACTIONS,
)
import textwrap

REMOTE_USER = "user1"
FILE_DIR_AT_REMOTE = f"~/IRIM_robot_ws/src/irim_robot_pkg/asset/trajectory"
REMOTE_HOST = "192.168.77.101"
PASSWORD = "11"
import subprocess
from typing import List, Iterable
ROUTINE_FILE_AT_REMOTE = (
    "~/IRIM_robot_ws/src/irim_robot_pkg/src/routine/routine_names.txt"
)


def get_humanoid_control_asset_files_via_ssh(
    remote_host: str,
    remote_user: str,
    password: str,
    port: int = 22
) -> List[str]:
    """
    SSH를 통해 원격지 IRIM_robot_ws/src/<pkg_name>/asset 디렉토리에서
    '*_traj*' 또는 '*group*' 파일 리스트를 반환한다.
    비밀번호 인증을 위해 sshpass 사용 (기본 password="11").
    sudo apt-get install sshpass 로 설치 필요.
    """
    remote_dir = FILE_DIR_AT_REMOTE
    ssh_target = f"{remote_user}@{remote_host}"

    cmd = [
        "sshpass", "-p", password,
        "ssh", "-p", str(port),
        ssh_target,
        f"ls -1 {remote_dir}"
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3   # [CHANGE] 1s -> 3s (불안정 완화)
    )
    if proc.returncode != 0:
        raise RuntimeError(f"SSH 에러 ({proc.stderr.strip()})")

    files = [
        os.path.join(remote_dir, fn.strip())
        for fn in proc.stdout.splitlines()
        if ("traj" in fn) or ("group" in fn)   # [FIX] 항상 True가 되는 버그 수정
    ]
    return files


# routine_names.txt를 SSH로 읽어서 루틴 이름 리스트 반환
def get_routine_names_via_ssh(
    remote_host: str,
    remote_user: str,
    password: str,
    remote_file: str,
    port: int = 22,
) -> List[str]:
    """
    SSH를 통해 remote_file(routine_names.txt)을 읽어
    줄 단위로 잘라서 루틴 이름 리스트를 반환한다.
    """
    ssh_target = f"{remote_user}@{remote_host}"
    cmd = [
        "sshpass", "-p", password,
        "ssh", "-p", str(port),
        ssh_target,
        f"cat {remote_file}",
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=3,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"SSH 에러 ({proc.stderr.strip()})")

    names = [
        line.strip()
        for line in proc.stdout.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return names


class CommandPanel(QWidget):
    def __init__(self, publish_callback, articulations):
        super().__init__()
        self.publish_callback = publish_callback
        self.articulations = list(articulations)  # 원본 순서 유지

        # 선택 정보들
        self.selected_robot_names = []   # 내부적으로 순서 유지, 중복 방지 유틸 사용
        self.selected_command_type = None
        self.selected_command_data = None
        self.selected_specific_mode = None
        self.selected_specific_option = None

        self.group_map = self._build_groups(self.articulations)  # {'Hand_R': [...], 'Hand_L': [...], ...}
        self.robot_individual_btn_map = {}  # 'Hand_R_thumb_wir' -> QPushButton
        self.robot_group_btn_map = {}       # 'Hand_R' -> QPushButton
        self.btn_all = None

        self.member_row_widget = None
        self.member_row_layout = None
        self.toggle_members_btn = None      # 👁 토글 버튼
        self.active_member_group_key = None # 현재 멤버 줄에 보여줄 그룹 (None이면 모두)

        # 약간의 시각 힌트 (partial 상태)
        self.setStyleSheet("""
            QPushButton[groupPartial="true"] {
                border: 2px dashed #888;
            }
        """)

        self.init_ui()
        # 초기 그룹 버튼 비주얼 동기화
        self._sync_group_button_state()

    # ---------------------------
    # UI
    # ---------------------------
    def init_ui(self):
        main_layout = QVBoxLayout()

        # 1. Robot Name 영역 (2줄 구성: 1줄 = ALL + 그룹, 2줄 = 그룹 멤버들)
        self.robot_name_group = QGroupBox("Robot Name")
        robot_name_vlayout = QVBoxLayout()

        # --- 1줄: ALL + 그룹 버튼들 + [👁 멤버 보기] 토글 ---
        row_groups = QHBoxLayout()

        # ALL 버튼
        self.btn_all = QPushButton("ALL")
        self.btn_all.setCheckable(True)
        self.btn_all.setFixedHeight(60)
        self.btn_all.setProperty("role", "all")
        self.btn_all.clicked.connect(self.on_robot_name_clicked)
        row_groups.addWidget(self.btn_all)

        # 그룹 버튼
        for gkey in sorted(self.group_map.keys()):
            gbtn = QPushButton(self._group_label(gkey))
            gbtn.setCheckable(True)
            gbtn.setFixedHeight(60)
            gbtn.setProperty("is_group", True)
            gbtn.setProperty("group_key", gkey)
            gbtn.clicked.connect(self.on_robot_name_clicked)
            self.robot_group_btn_map[gkey] = gbtn
            row_groups.addWidget(gbtn)

        row_groups.addStretch(1)

        # 👁 멤버 보기 토글 (기본: 숨김)
        self.toggle_members_btn = QPushButton("👁 멤버 보기")
        self.toggle_members_btn.setCheckable(True)
        self.toggle_members_btn.setToolTip("2번째 줄의 멤버 버튼을 표시/숨김")
        self.toggle_members_btn.clicked.connect(self._toggle_member_row)
        row_groups.addWidget(self.toggle_members_btn)

        robot_name_vlayout.addLayout(row_groups)

        # --- 2줄: 멤버 버튼들(작게 표시). 기본은 숨김 ---
        self.member_row_widget = QWidget()
        self.member_row_layout = QHBoxLayout(self.member_row_widget)
        self.member_row_layout.setContentsMargins(0, 0, 0, 0)

        grouped_members = set(sum(self.group_map.values(), []))
        ungrouped = []

        for name in self.articulations:
            btn = QPushButton(name)
            btn.setCheckable(True)
            self.robot_individual_btn_map[name] = btn

            if name in grouped_members:
                # 작은 버튼 스타일
                btn.setFixedHeight(36)
                f = btn.font()
                f.setPointSize(max(10, f.pointSize() - 2))
                btn.setFont(f)
                btn.setProperty("is_group", False)
                btn.clicked.connect(self.on_robot_name_clicked)
                self.member_row_layout.addWidget(btn)
            else:
                # 그룹이 없는 애들은 1줄에 원래 크기로
                btn.setFixedHeight(60)
                btn.setProperty("is_group", False)
                btn.clicked.connect(self.on_robot_name_clicked)
                ungrouped.append(btn)

        # 1줄 끝에 ungrouped 개별 버튼들 추가
        for btn in ungrouped:
            row_groups.addWidget(btn)

        # 2번째 줄은 기본 숨김
        self.member_row_widget.setVisible(False)
        robot_name_vlayout.addWidget(self.member_row_widget)

        self.robot_name_group.setLayout(robot_name_vlayout)
        main_layout.addWidget(self.robot_name_group)

        # 2. Command Type 영역
        self.command_type_group = QGroupBox("Command Type")
        self.command_type_btn_group = []
        command_type_layout = QHBoxLayout()
        for ctype in COMMAND_TYPES:
            btn = QPushButton(ctype)
            btn.setCheckable(True)
            btn.setFixedHeight(60)
            btn.clicked.connect(self.on_command_type_selected)
            self.command_type_btn_group.append(btn)
            command_type_layout.addWidget(btn)
        self.command_type_group.setLayout(command_type_layout)
        main_layout.addWidget(self.command_type_group)

        # 3. Command Data 영역 (3단계)
        self.command_data_group = QGroupBox("Command Data")
        self.command_data_layout = QHBoxLayout()
        self.command_data_group.setLayout(self.command_data_layout)
        main_layout.addWidget(self.command_data_group)

        # 4. Specific Option 영역 (4단계)
        self.specific_option_group = QGroupBox("Specific Option (4th Level)")
        self.specific_option_layout = QHBoxLayout()
        self.specific_option_group.setLayout(self.specific_option_layout)
        main_layout.addWidget(self.specific_option_group)

        # Publish 버튼
        self.pub_button = QPushButton("Publish")
        self.pub_button.setFixedHeight(60)
        self.pub_button.clicked.connect(self.on_publish_clicked)
        self.pub_button.setEnabled(False)
        main_layout.addWidget(self.pub_button)

        self.setLayout(main_layout)


    # ---------------------------
    # Small helpers
    # ---------------------------
    def _clear_layout(self, layout):
        """
        주어진 QLayout 안의 위젯/하위 레이아웃을 재귀적으로 모두 제거.
        """
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            child_layout = item.layout()
            if w is not None:
                w.setParent(None)
            elif child_layout is not None:
                self._clear_layout(child_layout)

    # ---------------------------
    # Selection / Group helpers
    # ---------------------------
    def _group_label(self, gkey: str, partial: bool = False) -> str:
        n = len(self.group_map.get(gkey, []))
        return f"{gkey} ({n})" + (" ◐" if partial else "")

    def _add_selection(self, name: str):
        if (name not in self.selected_robot_names) and (name in self.robot_individual_btn_map):
            self.selected_robot_names.append(name)

    def _remove_selection(self, name: str):
        if name in self.selected_robot_names:
            self.selected_robot_names.remove(name)

    def _select_members(self, members: Iterable[str], checked: bool):
        for m in members:
            btn = self.robot_individual_btn_map.get(m)
            if not btn:
                continue
            old = btn.blockSignals(True)
            btn.setChecked(checked)
            btn.blockSignals(old)
            if checked:
                self._add_selection(m)
            else:
                self._remove_selection(m)

    def _sync_group_button_state(self):
        """
        개별 버튼들의 선택 상황을 보고,
        각 그룹 버튼의 상태(전부선택, 일부선택, 해제)를 반영한다.
        """
        for gkey, gbtn in self.robot_group_btn_map.items():
            members = self.group_map.get(gkey, [])
            total = len(members)
            on_cnt = sum(1 for m in members
                         if (m in self.robot_individual_btn_map) and self.robot_individual_btn_map[m].isChecked())

            all_on = (on_cnt == total and total > 0)
            none_on = (on_cnt == 0)

            # 체크 상태 업데이트
            old = gbtn.blockSignals(True)
            gbtn.setChecked(all_on)
            # partial 시각화 속성/텍스트
            partial = (not all_on) and (not none_on)
            gbtn.setProperty("groupPartial", partial)
            gbtn.setText(self._group_label(gkey, partial))
            gbtn.blockSignals(old)

            # 스타일 재적용
            gbtn.style().unpolish(gbtn)
            gbtn.style().polish(gbtn)

    def _toggle_member_row(self):
        show = self.toggle_members_btn.isChecked()
        self.member_row_widget.setVisible(show)

    def _filter_member_row(self, group_key=None):
        """
        group_key가 주어지면 해당 그룹 멤버만 보여주고 나머지는 숨김.
        None이면 전부 표시.
        """
        if not self.member_row_layout:
            return
        visible_set = set(self.group_map.get(group_key, [])) if group_key else None
        for name, btn in self.robot_individual_btn_map.items():
            # 2번째 줄에 있는 버튼만 대상으로 처리
            if btn.parent() is self.member_row_widget:
                btn.setVisible(True if visible_set is None else (name in visible_set))

    # ---------------------------
    # Robot name click handlers
    # ---------------------------
    def on_robot_name_clicked(self):
        sender = self.sender()

        # ALL 버튼 처리
        if sender.property("role") == "all":
            if sender.isChecked():
                # ALL 선택 -> 나머지 전부 해제 및 선택 목록은 ["ALL"]만
                for btn in self.robot_group_btn_map.values():
                    old = btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(old)
                for btn in self.robot_individual_btn_map.values():
                    old = btn.blockSignals(True)
                    btn.setChecked(False)
                    btn.blockSignals(old)
                self.selected_robot_names = ["ALL"]
            else:
                if "ALL" in self.selected_robot_names:
                    self.selected_robot_names.remove("ALL")
            self._sync_group_button_state()
            self.update_pub_button_state()
            return

        # ALL이 켜져있다면 우선 끄기
        if self.btn_all and self.btn_all.isChecked():
            old = self.btn_all.blockSignals(True)
            self.btn_all.setChecked(False)
            self.btn_all.blockSignals(old)
            if "ALL" in self.selected_robot_names:
                self.selected_robot_names.remove("ALL")

        # 그룹 버튼 처리
        if sender.property("is_group"):
            gkey = sender.property("group_key")
            members = self.group_map.get(gkey, [])
            if sender.isChecked():
                # 그룹 켜면 구성원 모두 체크 & 선택목록 추가
                self._select_members(members, True)
            else:
                # 그룹 끄면 구성원 모두 해제 & 선택목록 제거
                self._select_members(members, False)

            # 멤버 줄 표시/필터링
            self.active_member_group_key = gkey
            if self.toggle_members_btn and not self.toggle_members_btn.isChecked():
                old = self.toggle_members_btn.blockSignals(True)
                self.toggle_members_btn.setChecked(True)
                self.toggle_members_btn.blockSignals(old)
                self.member_row_widget.setVisible(True)
            self._filter_member_row(gkey)

            self._sync_group_button_state()
            self.update_pub_button_state()
            return

        # 개별 버튼 처리
        name = sender.text()
        if sender.isChecked():
            self._add_selection(name)
        else:
            self._remove_selection(name)

        # 개별 버튼 조작 후 그룹 버튼 상태 동기화
        self._sync_group_button_state()
        if self.toggle_members_btn and self.toggle_members_btn.isChecked():
            self._filter_member_row(self.active_member_group_key)

        self.update_pub_button_state()

    # ---------------------------
    # Command type/data/options
    # ---------------------------
    def on_command_type_selected(self):
        sender = self.sender()
        # 단일 선택 처리 (다른 버튼은 해제)
        for btn in self.command_type_btn_group:
            if btn != sender:
                btn.setChecked(False)
        self.selected_command_type = sender.text() if sender.isChecked() else None
        self.build_command_data_buttons()
        self.clear_specific_option_buttons()
        self.update_pub_button_state()

    def build_command_data_buttons(self):
        # 이전 버튼 삭제
        self._clear_layout(self.command_data_layout)
        self.selected_command_data = None

        if self.selected_command_type == "PATH":
            try:
                path_files = get_humanoid_control_asset_files_via_ssh(
                    remote_host=REMOTE_HOST,
                    remote_user=REMOTE_USER,
                    password=PASSWORD
                )
            except Exception as e:
                # 1차로 실패 메시지 표시
                self.command_data_layout.addWidget(
                    QLabel(f"파일 목록 조회 실패: {e}")
                )
                # 2차로 IP 재설정 다이얼로그 호출
                self.prompt_new_ssh_ip(str(e))
                return

            info_search_widget = self._create_path_info_search_widget(REMOTE_HOST, REMOTE_USER, PASSWORD)
            self.command_data_layout.addWidget(info_search_widget)

            self.traj_list = QListWidget()
            self.traj_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.traj_list.setAlternatingRowColors(True)
            self.traj_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

            for file_path in path_files:
                filename = os.path.basename(file_path)
                item = QListWidgetItem(filename)
                item.setData(Qt.UserRole, filename)  # 실제 파일명 저장(요청: 문자열 그대로)
                self.traj_list.addItem(item)

            self.traj_list.itemClicked.connect(self.on_traj_selected)
            self.command_data_layout.addWidget(self.traj_list)

        elif self.selected_command_type == "STATUS":
            for status in ROBOT_STATUS:
                btn = QPushButton(status)
                btn.setFixedHeight(60)
                btn.setCheckable(True)
                btn.clicked.connect(self.on_command_data_selected)
                self.command_data_layout.addWidget(btn)

        elif self.selected_command_type == "SPECIFIC":
            for mode in SPECIFIC_MODES:
                btn = QPushButton(mode)
                btn.setFixedHeight(60)
                btn.setCheckable(True)
                btn.clicked.connect(self.on_command_data_selected)
                self.command_data_layout.addWidget(btn)

        elif self.selected_command_type == "SCENARIO":
            for scena in ROBOT_SCENARIO:
                btn = QPushButton(scena)
                btn.setFixedHeight(60)
                btn.setCheckable(True)
                btn.clicked.connect(self.on_command_data_selected)
                self.command_data_layout.addWidget(btn)

        elif self.selected_command_type == "ROUTINE":
            # 1) 위쪽 info + 검색 위젯
            info_widget = self._create_routine_info_search_widget()
            self.command_data_layout.addWidget(info_widget)

            # 2) SSH로 routine_names.txt 읽기
            try:
                routine_names = get_routine_names_via_ssh(
                    remote_host=REMOTE_HOST,
                    remote_user=REMOTE_USER,
                    password=PASSWORD,
                    remote_file=ROUTINE_FILE_AT_REMOTE,
                )
            except Exception as e:
                self.command_data_layout.addWidget(
                    QLabel(f"루틴 목록 조회 실패: {e}")
                )
                self.prompt_new_ssh_ip(str(e))   # ★ 여기서 IP 재입력
                return

            if not routine_names:
                self.command_data_layout.addWidget(
                    QLabel("사용 가능한 루틴이 없습니다.")
                )
                return

            # 3) QListWidget 생성 (PATH의 traj_list와 동일한 스타일)
            self.routine_list = QListWidget()
            self.routine_list.setSelectionMode(QAbstractItemView.SingleSelection)
            self.routine_list.setAlternatingRowColors(True)
            self.routine_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

            for rname in routine_names:
                item = QListWidgetItem(rname)
                # 필요하면 UserRole에 그대로 저장
                item.setData(Qt.UserRole, rname)
                self.routine_list.addItem(item)

            self.routine_list.itemClicked.connect(self.on_routine_selected)
            self.command_data_layout.addWidget(self.routine_list)


        else:
            label = QLabel("해당 Command Type은 지원되지 않습니다.")
            self.command_data_layout.addWidget(label)


    def _build_groups(self, names):
        """
        오직 이름에 'Hand'가 포함된 항목만 그룹화.
        'Hand', 'Hand_R', 'Hand_L' 등을 키로 사용하고,
        구성원이 2개 이상인 그룹만 반환.
        'Hand'가 포함되지 않으면 아무 그룹에도 속하지 않음(그대로 개별 항목).
        """
        from collections import defaultdict
        import re

        groups = defaultdict(list)
        hand_side_re = re.compile(r'(Hand(?:_[RL])?)')

        for n in names:
            # 'Hand'가 없으면 묶지 않음
            if 'Hand' not in n:
                continue

            m = hand_side_re.search(n)
            if m:
                key = m.group(1)  # 'Hand', 'Hand_R' 또는 'Hand_L'
            else:
                key = 'Hand'
            groups[key].append(n)

        return {k: sorted(v) for k, v in groups.items() if len(v) >= 2}


    def filter_traj_list(self, text):
        if not hasattr(self, 'traj_list'):
            return
        for i in range(self.traj_list.count()):
            item = self.traj_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())
            
    def on_routine_selected(self, item: QListWidgetItem):
        """
        ROUTINE 리스트에서 한 줄 클릭되면 선택 루틴 이름 설정.
        """
        if item is None:
            return

        # 이름 저장
        self.selected_command_data = item.text()

        # ROUTINE은 선택하자마자 4단계 옵션(액션)도 다시 구성
        if self.selected_command_type == "ROUTINE":
            self.build_specific_option_buttons(self.selected_command_data)

        self.update_pub_button_state()

    def filter_routine_list(self, text: str):
        """
        ROUTINE 검색 박스에서 호출: 텍스트에 맞지 않는 항목은 숨김.
        """
        if not hasattr(self, "routine_list"):
            return

        lower = text.lower()
        for i in range(self.routine_list.count()):
            item = self.routine_list.item(i)
            if item is None:
                continue
            item.setHidden(lower not in item.text().lower())

    def on_traj_selected(self, item: QListWidgetItem):
        self.selected_command_data = item.data(Qt.UserRole)
        self.update_pub_button_state()

    def prompt_new_ssh_ip(self, error_message: str = ""):
        """
        SSH 접속 실패 시 또는 사용자가 원할 때 IP를 다시 입력받는 다이얼로그.
        """
        from PyQt5.QtWidgets import QInputDialog

        msg = "새로운 SSH IP 주소를 입력하세요 (예: 192.168.0.101)"
        if error_message:
            msg = f"SSH 접속에 실패했습니다.\n{error_message}\n\n" + msg

        global REMOTE_HOST
        new_ip, ok = QInputDialog.getText(
            self,
            "SSH IP 설정",
            msg,
            text=REMOTE_HOST
        )
        if ok and new_ip.strip():
            REMOTE_HOST = new_ip.strip()

    def _create_path_info_search_widget(self, remote_host: str, remote_user: str, password: str):
        """
        주소/비밀번호 표시 + 검색 입력창을 포함하는 위젯을 반환.
        """
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)

        wrapped_path = "\n    ".join(textwrap.wrap(FILE_DIR_AT_REMOTE, width=27))
        info_label = QLabel(
            f"아래의 경로에서 파일 목록을 읽어옴.\n"
            f"주소: {remote_host}\n"
            f"경로: {wrapped_path}\n"
            f"비밀번호: {password}"
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft)
        info_layout.addWidget(info_label)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("파일 이름으로 검색...")
        self.search_box.textChanged.connect(self.filter_traj_list)
        info_layout.addWidget(self.search_box)

        return info_widget
    
    def prompt_new_ssh_ip(self, error_message: str = ""):
        """
        SSH 접속 실패 시 또는 사용자가 원할 때 IP를 다시 입력받는 다이얼로그.
        """
        from PyQt5.QtWidgets import QInputDialog

        msg = "새로운 SSH IP 주소를 입력하세요 (예: 192.168.0.101)"
        if error_message:
            msg = f"SSH 접속에 실패했습니다.\n{error_message}\n\n" + msg

        global REMOTE_HOST
        new_ip, ok = QInputDialog.getText(
            self,
            "SSH IP 설정",
            msg,
            text=REMOTE_HOST
        )
        if ok and new_ip.strip():
            REMOTE_HOST = new_ip.strip()
    
    def _create_routine_info_search_widget(self):
        """
        ROUTINE 목록용: 주소 표시 + 검색 입력창을 포함하는 위젯.
        """
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)

        info_label = QLabel(
            f"SSH로 routine_names.txt 를 읽어옵니다.\n"
            f"주소: {REMOTE_HOST}"
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignLeft)
        info_layout.addWidget(info_label)

        # 검색 박스
        self.routine_search_box = QLineEdit()
        self.routine_search_box.setPlaceholderText("루틴 이름 검색...")
        self.routine_search_box.textChanged.connect(self.filter_routine_list)
        info_layout.addWidget(self.routine_search_box)

        return info_widget

    def on_command_data_selected(self):
        sender = self.sender()
        # 단일 선택 구현
        for i in range(self.command_data_layout.count()):
            widget = self.command_data_layout.itemAt(i).widget()
            if widget != sender and hasattr(widget, "setChecked"):
                widget.setChecked(False)
        self.selected_command_data = sender.text() if sender.isChecked() else None

        # ★ ROUTINE도 4단계 옵션 사용
        if self.selected_command_type in ("SPECIFIC", "ROUTINE"):
            self.build_specific_option_buttons(self.selected_command_data)
        else:
            self.clear_specific_option_buttons()

        self.update_pub_button_state()

    def build_specific_option_buttons(self, mode_name):
        # 기존 위젯들 및 하위 레이아웃 모두 제거
        self._clear_layout(self.specific_option_layout)

        self.selected_specific_mode = mode_name
        self.selected_specific_option = None

        # ---------------- SPECIFIC 모드 ----------------
        if self.selected_command_type == "SPECIFIC":
            if mode_name == "TorqueLimits":
                h_layout = QHBoxLayout()
                h_layout.setContentsMargins(0, 0, 0, 0)

                self.torque_limit_input = QLineEdit()
                self.torque_limit_input.setPlaceholderText("Enter value")
                self.torque_limit_input.setFixedHeight(50)
                self.torque_limit_input.setMaximumWidth(250)
                self.torque_limit_input.setAlignment(Qt.AlignRight)
                self.torque_limit_input.textChanged.connect(self.on_torque_limit_changed)

                unit_label = QLabel("%")
                font = unit_label.font()
                font.setPointSize(14)
                unit_label.setFont(font)

                h_layout.addWidget(self.torque_limit_input)
                h_layout.addWidget(unit_label)

                self.specific_option_layout.addStretch(1)
                self.specific_option_layout.addLayout(h_layout)
                self.specific_option_layout.addStretch(1)

            else:
                options = []
                if mode_name == "ControlMode":
                    from irim_control_panel_n_dashboard.config import CONTROL_MODE
                    options = CONTROL_MODE
                elif mode_name == "ControlSpaceType":
                    from irim_control_panel_n_dashboard.config import CONTROL_SPACE_TYPE
                    options = CONTROL_SPACE_TYPE

                if options:
                    for opt in options:
                        btn = QPushButton(opt)
                        btn.setCheckable(True)
                        btn.setFixedHeight(60)
                        btn.clicked.connect(self.on_specific_option_selected)
                        self.specific_option_layout.addWidget(btn)
                else:
                    if mode_name:
                        label = QLabel("해당 모드의 세부 옵션이 없습니다.")
                        self.specific_option_layout.addWidget(label)

        # ---------------- ROUTINE 모드 ----------------
        elif self.selected_command_type == "ROUTINE":
            # routine 이름(mode_name)과 무관하게 동일한 액션 버튼 4개
            for action in ROUTINE_ACTIONS:
                btn = QPushButton(action)
                btn.setCheckable(True)
                btn.setFixedHeight(60)
                btn.clicked.connect(self.on_specific_option_selected)
                self.specific_option_layout.addWidget(btn)



    def clear_specific_option_buttons(self):
        self._clear_layout(self.specific_option_layout)
        self.selected_specific_mode = None
        self.selected_specific_option = None

    def on_torque_limit_changed(self, text):
        self.selected_specific_option = text
        self.update_pub_button_state()

    def on_specific_option_selected(self):
        sender = self.sender()
        for i in range(self.specific_option_layout.count()):
            widget = self.specific_option_layout.itemAt(i).widget()
            if widget != sender and hasattr(widget, "setChecked"):
                widget.setChecked(False)
        self.selected_specific_option = sender.text() if sender.isChecked() else None
        self.update_pub_button_state()

    # ---------------------------
    # Publish
    # ---------------------------
    def update_pub_button_state(self):
        if not self.selected_robot_names:
            self.pub_button.setEnabled(False)
            return
        if not self.selected_command_type or not self.selected_command_data:
            self.pub_button.setEnabled(False)
            return
        if self.selected_command_type in ("SPECIFIC", "ROUTINE") and not self.selected_specific_option:
            self.pub_button.setEnabled(False)
            return
        self.pub_button.setEnabled(True)

    def on_publish_clicked(self):
        if "ALL" in self.selected_robot_names:
            robot_name_str = "ALL"
        else:
            # articulations 순서대로 정렬된 문자열 생성
            ordered = [n for n in self.articulations if n in self.selected_robot_names]
            robot_name_str = ",".join(ordered)

        # ★ SPECIFIC, ROUTINE은 4개 필드, 그 외는 3개 필드
        if self.selected_command_type in ("SPECIFIC", "ROUTINE"):
            message_str = (
                f"{robot_name_str}::{self.selected_command_type}::"
                f"{self.selected_command_data}::{self.selected_specific_option}"
            )
        else:
            message_str = (
                f"{robot_name_str}::{self.selected_command_type}::{self.selected_command_data}"
            )

        self.publish_callback(message_str)