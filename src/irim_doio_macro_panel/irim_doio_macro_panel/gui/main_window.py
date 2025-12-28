from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from PyQt5 import QtCore, QtWidgets

from irim_doio_macro_panel.input.evdev_reader import EvdevHotplugReader
from irim_doio_macro_panel.config.io import save_config
from irim_doio_macro_panel.utils.robot_discovery import get_robots_from_ros2_topic_list

def _safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, cfg: Dict[str, Any], cfg_path: str, ros_publish_fn, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.cfg_path = cfg_path
        self.ros_publish = ros_publish_fn

        self.pressed: Set[str] = set()
        self.pending_suffix: str = ""
        self.last_published: str = ""


        # robot grouping / selection helpers
        self.robot_groups: Dict[str, List[str]] = {}
        self.robot_group_btns: Dict[str, QtWidgets.QPushButton] = {}
        self.robot_group_scroll: Optional[QtWidgets.QScrollArea] = None
        self.robot_group_container: Optional[QtWidgets.QWidget] = None
        self.robot_group_layout: Optional[QtWidgets.QHBoxLayout] = None
        self._robot_item_by_name: Dict[str, QtWidgets.QListWidgetItem] = {}

        # partial selection visual hint for group buttons
        self.setStyleSheet(self.styleSheet() + "\n"
                           "QPushButton[groupPartial=\"true\"] {\n"
                           "    border: 2px dashed #888;\n"
                           "}\n")
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_now)

        title = _safe_get(cfg, ["ui", "window_title"], "DOIO Macro → ROS2 Publisher")
        self.setWindowTitle(title)
        self.resize(1150, 720)

        self._build_ui()

        # Physical layout button mapping (visual helper)
        # NOTE: These are logical keycodes as seen by evdev.
        # If a specific key doesn't send a standard KEY_* code, keep it as "".
        self._layout_keymap: Dict[str, str] = {
            # 4x4 main pad
            "a": "KEY_A",
            "b": "KEY_B",
            "c": "KEY_C",
            "d": "KEY_D",
            "e": "KEY_E",
            "f": "KEY_F",
            "g": "KEY_G",
            "h": "KEY_H",
            "i": "KEY_I",
            "j": "KEY_J",
            "k": "KEY_K",
            "l": "KEY_L",
            # bottom row (device-specific)
            "func": str(_safe_get(cfg, ["behavior", "clear_pending_key"], "")) or "",
            "TO(n-1)": "",
            "TO(n+1)": "",
            "p": str(_safe_get(cfg, ["behavior", "publish_key"], "KEY_P")) or "KEY_P",
            # knobs: 3 knobs * (ccw/cw/push) -> 1..9
            "1": "KEY_1",
            "2": "KEY_2",
            "3": "KEY_3",
            "4": "KEY_4",
            "5": "KEY_5",
            "6": "KEY_6",
            "7": "KEY_7",
            "8": "KEY_8",
            "9": "KEY_9",
        }

        print("[BOOT] MainWindow created, cfg_path =", self.cfg_path)
        self._append_log(f"[BOOT] cfg_path = {self.cfg_path}")
        self._append_log(f"[BOOT] device = {self.cfg.get('device')}")

        # evdev reader
        dev_paths = _safe_get(cfg, ["device", "paths"], [])
        match = _safe_get(cfg, ["device", "match"], None)  # <-- NEW
        grab = bool(_safe_get(cfg, ["device", "grab"], True))
        reconn = int(_safe_get(cfg, ["device", "reconnect_period_ms"], 500))

        self._append_log(f"[DEBUG] cfg.device.paths = {dev_paths}")
        self._append_log(f"[DEBUG] cfg.device.match = {match}")

        self.reader = EvdevHotplugReader(
            device_paths=dev_paths,
            grab=grab,
            reconnect_period_ms=reconn,
            parent=self,
            match=match,
        )

        self.reader.key_event.connect(self._on_key_event)
        self.reader.device_status.connect(self._append_log)
        self._append_log("[INFO] evdev reader signals connected")


        # robot discovery timer
        self._last_robot_names: List[str] = []

        self._robot_timer = QtCore.QTimer(self)
        self._robot_timer.timeout.connect(self._refresh_robots)
        period = float(_safe_get(cfg, ["robot_discovery", "refresh_period_sec"], 2.0))
        self._robot_timer.start(int(period * 1000))
        self._refresh_robots()

        # init layer
        init_layer = int(_safe_get(cfg, ["layer", "initial"], 0))
        self._set_active_layer(init_layer, from_key=False)

        self._update_pending_ui()

    # ---------------- Selection command (string) ----------------

    def _set_group_or_robot(self, key: str, checked: bool) -> None:
        """Set (not toggle) a group or robot without clearing other selections."""
        k = (key or "").strip()
        if not k:
            return

        self.chk_all.setChecked(False)

        # group set
        if k in self.robot_groups:
            self._apply_group_selection(k, checked=checked, exclusive=False)
            return

        # robot set (exact / case-insensitive)
        it = self._robot_item_by_name.get(k)
        if it is None:
            lower_map = {name.lower(): name for name in self._robot_item_by_name.keys()}
            hit = lower_map.get(k.lower())
            it = self._robot_item_by_name.get(hit) if hit else None

        if it is None:
            self._append_log(f"[ROBOT] not found: {k}")
            return

        it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        self._sync_robot_group_buttons()

    def _apply_selection_cmd(self, cmd: str) -> None:
        """Apply selection changes from a free-form string.

        Supported tokens (comma/space separated):
          - all / clear
          - Arm_R (group) or Hand_L_thumb (robot)
          - +X : force ON,  -X : force OFF,  !X : toggle

        Examples:
          - "Arm_R,Hand_R" (toggle both)
          - "+Arm_R,-Hand_L" (set Arm_R ON, Hand_L OFF)
          - "all" (toggle all)
          - "clear" (clear all)
        """
        s = (cmd or "").strip()
        if not s:
            return

        # split by comma or whitespace
        parts = [p for p in re.split(r"[\s,]+", s) if p]
        for raw in parts:
            tok = raw.strip()
            if not tok:
                continue

            op = "toggle"
            if tok[0] in ("+", "-", "!"):
                op = {"+": "on", "-": "off", "!": "toggle"}.get(tok[0], "toggle")
                tok = tok[1:].strip()

            low = tok.lower()

            # special
            if low in ("all", "select_all"):
                if op == "on":
                    self._select_all_robots()
                elif op == "off":
                    self._clear_robots()
                else:
                    self._toggle_select_all()
                continue

            if low in ("clear", "none", "off"):
                self._clear_robots()
                continue

            if op == "on":
                self._set_group_or_robot(tok, True)
            elif op == "off":
                self._set_group_or_robot(tok, False)
            else:
                self._toggle_group_or_robot(tok)

    # ---------------- Robot toggle helpers ----------------

    def _robot_selection_active_for_layer(self, layer_idx: int) -> bool:
        """Whether *global* robot_selection hotkeys are active in this layer.

        This keeps the original "robot_selection.keys" feature, but lets you scope it
        to a specific layer (e.g., "Layer 2 is for robot selection").

        Supported config shapes (all optional):
          - robot_selection.layer: 2
          - robot_selection.layers / active_layers / only_layers: [2]

        If none is provided, the hotkeys stay active in all layers (legacy behavior).
        """
        rs = self.cfg.get("robot_selection", {}) or {}
        if not bool(rs.get("enabled", False)):
            return False

        # single layer
        if rs.get("layer") is not None:
            try:
                return int(rs.get("layer")) == int(layer_idx)
            except Exception:
                return False

        # multiple layers (accept several aliases)
        layers = rs.get("layers")
        if layers is None:
            layers = rs.get("active_layers")
        if layers is None:
            layers = rs.get("only_layers")

        if isinstance(layers, list) and layers:
            out: List[int] = []
            for x in layers:
                try:
                    out.append(int(x))
                except Exception:
                    pass
            return int(layer_idx) in set(out)

        # default: active in all layers
        return True

    def _toggle_select_all(self) -> None:
        """Toggle all robot checkboxes (if any unchecked -> check all, else clear all)."""
        if self.robot_list.count() <= 0:
            return
        self.chk_all.setChecked(False)

        any_off = False
        for i in range(self.robot_list.count()):
            if self.robot_list.item(i).checkState() != QtCore.Qt.Checked:
                any_off = True
                break

        self.robot_list.blockSignals(True)
        try:
            for i in range(self.robot_list.count()):
                self.robot_list.item(i).setCheckState(QtCore.Qt.Checked if any_off else QtCore.Qt.Unchecked)
        finally:
            self.robot_list.blockSignals(False)

        self._sync_robot_group_buttons()

    def _toggle_group_or_robot(self, key: str) -> None:
        """Toggle a group if key matches a group name; otherwise toggle a robot by exact name."""
        k = (key or "").strip()
        if not k:
            return

        # group toggle
        if k in self.robot_groups:
            members = self.robot_groups.get(k, [])
            if not members:
                return
            all_on = True
            for n in members:
                it = self._robot_item_by_name.get(n)
                if it is None or it.checkState() != QtCore.Qt.Checked:
                    all_on = False
                    break
            self._apply_group_selection(k, checked=(not all_on), exclusive=False)
            return

        # robot toggle (exact)
        it = self._robot_item_by_name.get(k)
        if it is None:
            # case-insensitive exact match fallback
            lower_map = {name.lower(): name for name in self._robot_item_by_name.keys()}
            hit = lower_map.get(k.lower())
            it = self._robot_item_by_name.get(hit) if hit else None

        if it is None:
            self._append_log(f"[ROBOT] not found: {k}")
            return

        self.chk_all.setChecked(False)
        new_state = QtCore.Qt.Unchecked if it.checkState() == QtCore.Qt.Checked else QtCore.Qt.Checked
        it.setCheckState(new_state)
        self._sync_robot_group_buttons()

    # ---------------- UI ----------------

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)

        # Left: robot selection + status
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 2)

        grp_robot = QtWidgets.QGroupBox("Robot selection")
        v = QtWidgets.QVBoxLayout(grp_robot)

        self.manual_robots = QtWidgets.QLineEdit()
        self.manual_robots.setPlaceholderText(
            "Manual robots (comma-separated). If set, this overrides the discovered list."
        )
        self.manual_robots.textChanged.connect(lambda _t: None)
        v.addWidget(self.manual_robots)

        self.robot_search = QtWidgets.QLineEdit()
        self.robot_search.setPlaceholderText("Filter robots...")
        self.robot_search.textChanged.connect(self._apply_robot_filter)
        v.addWidget(self.robot_search)

        # selection command input (string, like other macro strings)
        sel_row = QtWidgets.QHBoxLayout()
        self.selection_cmd = QtWidgets.QLineEdit()
        self.selection_cmd.setPlaceholderText("Selection cmd: Arm_R, +Hand_L, -Arm_L, all, clear")
        self.selection_cmd.returnPressed.connect(lambda: self._apply_selection_cmd(self.selection_cmd.text()))

        self.btn_apply_sel = QtWidgets.QPushButton("Apply")
        self.btn_apply_sel.clicked.connect(lambda: self._apply_selection_cmd(self.selection_cmd.text()))
        self.btn_clear_sel = QtWidgets.QPushButton("Clear")
        self.btn_clear_sel.clicked.connect(self._clear_robots)

        sel_row.addWidget(self.selection_cmd, 1)
        sel_row.addWidget(self.btn_apply_sel)
        sel_row.addWidget(self.btn_clear_sel)
        v.addLayout(sel_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_refresh_robot = QtWidgets.QPushButton("Refresh")
        self.btn_refresh_robot.clicked.connect(self._refresh_robots)
        self.btn_all_robot = QtWidgets.QPushButton("Select ALL")
        self.btn_all_robot.clicked.connect(self._select_all_robots)
        self.btn_clear_robot = QtWidgets.QPushButton("Clear")
        self.btn_clear_robot.clicked.connect(self._clear_robots)
        btn_row.addWidget(self.btn_refresh_robot)
        btn_row.addWidget(self.btn_all_robot)
        btn_row.addWidget(self.btn_clear_robot)
        v.addLayout(btn_row)

        # Group buttons (auto-group similar robot names)
        self.robot_group_scroll = QtWidgets.QScrollArea()
        self.robot_group_scroll.setWidgetResizable(True)
        self.robot_group_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.robot_group_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.robot_group_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.robot_group_container = QtWidgets.QWidget()
        self.robot_group_layout = QtWidgets.QHBoxLayout(self.robot_group_container)
        self.robot_group_layout.setContentsMargins(0, 0, 0, 0)
        self.robot_group_layout.setSpacing(6)
        self.robot_group_layout.addStretch(1)
        self.robot_group_scroll.setWidget(self.robot_group_container)
        v.addWidget(self.robot_group_scroll)

        self.robot_list = QtWidgets.QListWidget()
        self.robot_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.robot_list.itemChanged.connect(lambda _it: self._sync_robot_group_buttons())
        v.addWidget(self.robot_list, 1)

        self.chk_all = QtWidgets.QCheckBox("Publish to ALL")
        self.chk_all.stateChanged.connect(self._on_all_changed)
        v.addWidget(self.chk_all)

        left.addWidget(grp_robot, 5)

        grp_state = QtWidgets.QGroupBox("Macro state")
        g = QtWidgets.QGridLayout(grp_state)
        self.lbl_layer = QtWidgets.QLabel("-")
        self.lbl_pending = QtWidgets.QLabel("(none)")
        self.lbl_last = QtWidgets.QLabel("(none)")
        self.lbl_pending.setWordWrap(True)
        self.lbl_last.setWordWrap(True)

        g.addWidget(QtWidgets.QLabel("Active layer:"), 0, 0)
        g.addWidget(self.lbl_layer, 0, 1)
        g.addWidget(QtWidgets.QLabel("Pending suffix:"), 1, 0)
        g.addWidget(self.lbl_pending, 1, 1)
        g.addWidget(QtWidgets.QLabel("Last publish:"), 2, 0)
        g.addWidget(self.lbl_last, 2, 1)

        left.addWidget(grp_state, 2)

        grp_log = QtWidgets.QGroupBox("Log")
        lv = QtWidgets.QVBoxLayout(grp_log)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        lv.addWidget(self.log)
        left.addWidget(grp_log, 3)

        # Right: mapping editor
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 3)

        top_row = QtWidgets.QHBoxLayout()
        right.addLayout(top_row)

        self.layer_combo = QtWidgets.QComboBox()
        self.layer_combo.addItems([f"Layer {i}" for i in range(9)])
        self.layer_combo.currentIndexChanged.connect(lambda idx: self._set_active_layer(idx, from_key=False))
        top_row.addWidget(QtWidgets.QLabel("Edit layer:"))
        top_row.addWidget(self.layer_combo)

        self.btn_reload = QtWidgets.QPushButton("Reload JSON")
        self.btn_reload.clicked.connect(self._reload_config_from_disk)
        self.btn_save = QtWidgets.QPushButton("Save now")
        self.btn_save.clicked.connect(self._save_now)
        top_row.addStretch(1)
        top_row.addWidget(self.btn_reload)
        top_row.addWidget(self.btn_save)

        # Physical keyboard layout (visual helper)
        grp_layout = QtWidgets.QGroupBox("Keyboard layout (visual)")
        layout_v = QtWidgets.QVBoxLayout(grp_layout)
        layout_root = QtWidgets.QHBoxLayout()
        layout_v.addLayout(layout_root)

        # left: 4x4 pad
        pad = QtWidgets.QGridLayout()
        pad.setHorizontalSpacing(8)
        pad.setVerticalSpacing(8)
        layout_root.addLayout(pad, 2)

        def mk_btn(label: str) -> QtWidgets.QPushButton:
            # Button text is populated from the current layer's binding description.
            # We keep the physical label (a/b/c/..., func/TO/p, 1..9) only for internal lookup.
            btn = QtWidgets.QPushButton("")
            btn.setMinimumSize(92, 58)
            btn.setCheckable(False)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.setStyleSheet("text-align: center; padding: 6px;")
            btn.clicked.connect(lambda _=False, l=label: self._on_layout_key_clicked(l))
            btn.setToolTip(label)
            return btn

        self._layout_buttons: Dict[str, QtWidgets.QPushButton] = {}

        pad_rows = [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["func", "TO(n-1)", "TO(n+1)", "p"],
        ]
        for r, row in enumerate(pad_rows):
            for c, lab in enumerate(row):
                b = mk_btn(lab)
                pad.addWidget(b, r, c)
                self._layout_buttons[lab] = b

        # right: knobs (3 knobs x (ccw/cw/push))
        knobs = QtWidgets.QGridLayout()
        knobs.setHorizontalSpacing(8)
        knobs.setVerticalSpacing(8)
        layout_root.addLayout(knobs, 1)

        knobs.addWidget(QtWidgets.QLabel("ccw"), 0, 0)
        knobs.addWidget(QtWidgets.QLabel("cw"), 0, 1)
        knobs.addWidget(QtWidgets.QLabel("push"), 0, 2)

        knob_rows = [
            ["1", "2", "3"],
            ["4", "5", "6"],
            ["7", "8", "9"],
        ]
        for r, row in enumerate(knob_rows, start=1):
            for c, lab in enumerate(row):
                b = mk_btn(lab)
                knobs.addWidget(b, r, c)
                self._layout_buttons[lab] = b

        # hint_layout = QtWidgets.QLabel(
        #     "Tip: 클릭하면 해당 KEYCODE가 현재 Layer의 bindings 표에서 선택됩니다.\n"
        #     "TO 키가 표준 KEY_* 로 안 잡히면 mapping.json의 bindings에 보이는 keycode로 바꿔주세요."
        # )
        # hint_layout.setWordWrap(True)
        # layout_v.addWidget(hint_layout)

        right.addWidget(grp_layout, 2)

        # bindings table
        grp_bind = QtWidgets.QGroupBox("Key bindings (per layer)")
        bv = QtWidgets.QVBoxLayout(grp_bind)
        self.tbl_bind = QtWidgets.QTableWidget()
        self.tbl_bind.setColumnCount(4)
        self.tbl_bind.setHorizontalHeaderLabels(["Keycode", "Kind", "Suffix / Layer", "Description"])
        self.tbl_bind.horizontalHeader().setStretchLastSection(True)
        self.tbl_bind.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.tbl_bind.itemChanged.connect(self._on_binding_item_changed)

        bv.addWidget(self.tbl_bind, 1)

        hint = QtWidgets.QLabel(
            "Kind examples: set_pending, publish_pending, publish_suffix, clear_pending, set_layer, select_all, select_group, select_robot, robot_select_cmd\n"
            "Suffix is the part AFTER robot names (robots are auto-prefixed at publish time).\n"
            "Tip: If two physical keys share the same keycode (e.g., KEY_A twice), software cannot distinguish them."
        )
        hint.setWordWrap(True)
        bv.addWidget(hint)

        right.addWidget(grp_bind, 6)

        # chord table
        grp_chord = QtWidgets.QGroupBox("Chords (combinations)")
        cv = QtWidgets.QVBoxLayout(grp_chord)
        self.tbl_chord = QtWidgets.QTableWidget()
        self.tbl_chord.setColumnCount(5)
        self.tbl_chord.setHorizontalHeaderLabels(["Held (comma)", "Press", "Kind", "Suffix", "Description"])
        self.tbl_chord.horizontalHeader().setStretchLastSection(True)
        self.tbl_chord.itemChanged.connect(self._on_chord_item_changed)

        btns = QtWidgets.QHBoxLayout()
        self.btn_add_chord = QtWidgets.QPushButton("Add chord")
        self.btn_add_chord.clicked.connect(self._add_chord)
        self.btn_del_chord = QtWidgets.QPushButton("Delete chord")
        self.btn_del_chord.clicked.connect(self._del_chord)
        btns.addWidget(self.btn_add_chord)
        btns.addWidget(self.btn_del_chord)
        btns.addStretch(1)

        cv.addWidget(self.tbl_chord, 1)
        cv.addLayout(btns)

        right.addWidget(grp_chord, 3)

        # populate tables for initial layer (0) – will be updated by _set_active_layer
        self._populate_bindings_table(0)
        self._populate_chords_table()

    # ---------------- Config save/reload ----------------

    def _schedule_save(self):
        ms = int(_safe_get(self.cfg, ["ui", "auto_save_debounce_ms"], 250))
        self._save_timer.start(ms)

    def _save_now(self):
        try:
            save_config(self.cfg_path, self.cfg)
            self._append_log(f"[SAVE] {self.cfg_path}")
        except Exception as e:
            self._append_log(f"[ERROR] save failed: {e}")

    def _reload_config_from_disk(self):
        from irim_doio_macro_panel.config.io import load_config
        try:
            new_cfg = load_config(self.cfg_path)
        except Exception as e:
            self._append_log(f"[ERROR] reload failed: {e}")
            return
        self.cfg = new_cfg
        self._append_log("[RELOAD] reloaded config from disk")
        # refresh UI based on new config
        cur_layer = self.layer_combo.currentIndex()
        self._populate_bindings_table(cur_layer)
        self._populate_chords_table()

    # ---------------- Robot selection ----------------

    def _refresh_robots(self):
        enabled = bool(_safe_get(self.cfg, ["robot_discovery", "enabled"], True))
        if not enabled:
            return

        # 사용자가 Manual robots를 쓰면 discovery로 UI를 흔들 필요가 없음
        manual = (self.manual_robots.text() or "").strip()
        if manual:
            return

        topic_regex = str(_safe_get(self.cfg, ["robot_discovery", "topic_regex"], r"^/robot_outbound_data/([^/]+)/"))
        names = get_robots_from_ros2_topic_list(topic_regex)

        # 변화 없으면 UI 갱신 자체를 하지 않음 (깜빡임 핵심 원인 제거)
        if names == getattr(self, "_last_robot_names", []):
            return
        self._last_robot_names = list(names)

        # 기존 체크 상태 보존
        checked = {
            self.robot_list.item(i).text(): (self.robot_list.item(i).checkState() == QtCore.Qt.Checked)
            for i in range(self.robot_list.count())
        }

        # UI 업데이트 잠깐 중지(리페인트/깜빡임 감소)
        self.robot_list.setUpdatesEnabled(False)
        self.robot_list.blockSignals(True)
        try:
            # 현재 아이템 맵
            existing = {self.robot_list.item(i).text(): self.robot_list.item(i)
                        for i in range(self.robot_list.count())}

            desired_set = set(names)

            # 1) 없어질 항목 제거 (clear 금지)
            for name, it in list(existing.items()):
                if name not in desired_set:
                    row = self.robot_list.row(it)
                    self.robot_list.takeItem(row)
                    existing.pop(name, None)

            # 2) 원하는 순서대로 삽입/이동/추가
            for idx, name in enumerate(names):
                it = existing.get(name)
                if it is None:
                    it = QtWidgets.QListWidgetItem(name)
                    it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                    it.setCheckState(QtCore.Qt.Checked if checked.get(name, False) else QtCore.Qt.Unchecked)
                    self.robot_list.insertItem(idx, it)
                    existing[name] = it
                else:
                    # 체크 상태 복원
                    it.setCheckState(QtCore.Qt.Checked if checked.get(name, False) else QtCore.Qt.Unchecked)

                    # 순서가 달라졌으면 이동 (clear 없이)
                    cur_row = self.robot_list.row(it)
                    if cur_row != idx:
                        self.robot_list.takeItem(cur_row)
                        self.robot_list.insertItem(idx, it)

        finally:
            self.robot_list.blockSignals(False)
            self.robot_list.setUpdatesEnabled(True)

        self._apply_robot_filter(self.robot_search.text())
        # Rebuild group buttons based on current discovered list
        self._rebuild_robot_groups(names)
        self._sync_robot_group_buttons()


    def _apply_robot_filter(self, text: str):
        t = (text or "").lower().strip()
        for i in range(self.robot_list.count()):
            it = self.robot_list.item(i)
            it.setHidden(bool(t) and t not in it.text().lower())

    # ---------------- Robot grouping ----------------

    def _rebuild_robot_groups(self, names: List[str]):
        if self.robot_group_layout is None:
            return

        self.robot_groups = self._build_robot_groups(names)

        # clear old buttons
        for btn in self.robot_group_btns.values():
            try:
                btn.setParent(None)
                btn.deleteLater()
            except Exception:
                pass
        self.robot_group_btns.clear()

        # clear layout items
        while self.robot_group_layout.count():
            item = self.robot_group_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        # rebuild buttons
        for gkey in sorted(self.robot_groups.keys()):
            members = self.robot_groups[gkey]
            btn = QtWidgets.QPushButton(f"{gkey} ({len(members)})")
            btn.setCheckable(True)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            btn.setProperty("is_robot_group", True)
            btn.setProperty("group_key", gkey)
            btn.clicked.connect(self._on_robot_group_clicked)
            self.robot_group_btns[gkey] = btn
            self.robot_group_layout.addWidget(btn)

        self.robot_group_layout.addStretch(1)

        # rebuild item lookup
        self._robot_item_by_name.clear()
        for i in range(self.robot_list.count()):
            it = self.robot_list.item(i)
            self._robot_item_by_name[it.text()] = it

    def _build_robot_groups(self, names: List[str]) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {}
        used: Set[str] = set()

        # 1) group by "<prefix>_<L|R>_*" (greedy prefix can include underscores)
        lr_pat = re.compile(r"^(.+)_([LR])(?:_|$)", re.IGNORECASE)
        for n in names:
            m = lr_pat.match(n)
            if m:
                prefix = m.group(1)
                side = m.group(2).upper()
                gk = f"{prefix}_{side}"
                groups.setdefault(gk, []).append(n)
                used.add(n)

        # 2) fallback group by first token before '_' (only for remaining)
        token_groups: Dict[str, List[str]] = {}
        for n in names:
            if n in used:
                continue
            token = n.split("_", 1)[0] if "_" in n else n
            token_groups.setdefault(token, []).append(n)
        for token, members in token_groups.items():
            if len(members) >= 2:
                groups.setdefault(token, []).extend(members)

        # keep only useful groups (2+ members)
        groups = {k: v for k, v in groups.items() if len(v) >= 2}

        # stable order
        order = {n: i for i, n in enumerate(names)}
        for k, v in groups.items():
            v.sort(key=lambda x: order.get(x, 1_000_000))

        return groups

    def _on_robot_group_clicked(self, checked: bool):
        btn = self.sender()
        if not isinstance(btn, QtWidgets.QPushButton):
            return
        gkey = btn.property("group_key")
        if not gkey:
            return
        self._apply_group_selection(str(gkey), checked=checked, exclusive=False)

    def _apply_group_selection(self, group_key: str, checked: bool, exclusive: bool = False):
        members = self.robot_groups.get(group_key, [])
        if not members:
            self._append_log(f"[ROBOT] unknown group '{group_key}'")
            return

        self.chk_all.setChecked(False)

        self.robot_list.blockSignals(True)
        try:
            if exclusive:
                for i in range(self.robot_list.count()):
                    self.robot_list.item(i).setCheckState(QtCore.Qt.Unchecked)

            for name in members:
                it = self._robot_item_by_name.get(name)
                if it is not None:
                    it.setCheckState(QtCore.Qt.Checked if checked else QtCore.Qt.Unchecked)
        finally:
            self.robot_list.blockSignals(False)

        self._sync_robot_group_buttons()

    def _sync_robot_group_buttons(self):
        # refresh item map (robust against list updates)
        self._robot_item_by_name = {self.robot_list.item(i).text(): self.robot_list.item(i)
                                   for i in range(self.robot_list.count())}

        for gkey, btn in self.robot_group_btns.items():
            members = self.robot_groups.get(gkey, [])
            if not members:
                continue
            total = len(members)
            on_cnt = 0
            for n in members:
                it = self._robot_item_by_name.get(n)
                if it is not None and it.checkState() == QtCore.Qt.Checked:
                    on_cnt += 1

            all_on = (on_cnt == total and total > 0)
            none_on = (on_cnt == 0)
            partial = (not all_on) and (not none_on)

            old = btn.blockSignals(True)
            btn.setChecked(all_on)
            btn.setProperty("groupPartial", partial)
            btn.setText(f"{gkey} ({total})" + (" ◐" if partial else ""))
            btn.blockSignals(old)

            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _apply_robot_selection_action(self, action: Any):
        """
        Supports key-bind actions for robot selection.

        Examples:
          {"mode": "all"}
          {"mode": "clear"}
          {"mode": "group", "group": "Arm_L", "exclusive": true}
          {"mode": "robots", "robots": ["Hand_L_thumb", "Hand_L_index"]}
          "all" / "clear" / "group:Arm_L"
        """
        if action is None:
            return

        if isinstance(action, str):
            s = action.strip()
            if not s:
                return
            if ":" in s:
                head, tail = s.split(":", 1)
                head = head.strip().lower()
                tail = tail.strip()
                if head == "group":
                    action = {"mode": "group", "group": tail}
                else:
                    action = {"mode": head, "value": tail}
            else:
                action = {"mode": s.strip().lower()}

        if not isinstance(action, dict):
            return

        mode = str(action.get("mode", "")).strip().lower()

        # NOTE (policy): selection hotkeys should be *toggle* by default and *not exclusive*.
        # Users can opt back into exclusive/one-shot behavior with explicit flags.
        toggle = bool(action.get("toggle", True))
        exclusive = bool(action.get("exclusive", False))

        if mode in ("all", "select_all"):
            # toggle all by default
            if toggle:
                self._toggle_select_all()
            else:
                self._select_all_robots()
            self._sync_robot_group_buttons()
            return

        if mode in ("clear", "none"):
            self._clear_robots()
            self._sync_robot_group_buttons()
            return

        if mode == "group":
            gkey = str(action.get("group", "")).strip()
            if not gkey:
                return
            if toggle and not exclusive:
                # Toggle a group (or robot) without touching other selections.
                self._toggle_group_or_robot(gkey)
            else:
                # One-shot select, optionally exclusive.
                self._apply_group_selection(gkey, checked=True, exclusive=exclusive)
            return

        if mode in ("robots", "names"):
            robots = action.get("robots") or action.get("names") or []
            if isinstance(robots, str):
                robots = [r.strip() for r in robots.split(",") if r.strip()]
            if not isinstance(robots, list):
                return

            wanted = [str(r) for r in robots if str(r).strip()]
            self.chk_all.setChecked(False)

            if toggle and not exclusive:
                # Toggle each robot name without affecting other selections.
                for name in wanted:
                    self._toggle_group_or_robot(name)
                return

            # One-shot select, optionally exclusive.
            self.robot_list.blockSignals(True)
            try:
                if exclusive:
                    for i in range(self.robot_list.count()):
                        self.robot_list.item(i).setCheckState(QtCore.Qt.Unchecked)
                for name in wanted:
                    it = self._robot_item_by_name.get(name)
                    if it is not None:
                        it.setCheckState(QtCore.Qt.Checked)
                    else:
                        self._append_log(f"[ROBOT] not found: {name}")
            finally:
                self.robot_list.blockSignals(False)

            self._sync_robot_group_buttons()
            return

        # unknown -> noop
        self._append_log(f"[ROBOT] noop (unknown mode) {mode}")

    def _select_all_robots(self):
        self.chk_all.setChecked(False)
        for i in range(self.robot_list.count()):
            self.robot_list.item(i).setCheckState(QtCore.Qt.Checked)
        self._sync_robot_group_buttons()

    def _clear_robots(self):
        self.chk_all.setChecked(False)
        for i in range(self.robot_list.count()):
            self.robot_list.item(i).setCheckState(QtCore.Qt.Unchecked)
        self._sync_robot_group_buttons()

    def _on_all_changed(self, _state):
        checked = self.chk_all.isChecked()
        # disable individual/group selection visually (but keep states)
        self.robot_list.setEnabled(not checked)
        if self.robot_group_container is not None:
            self.robot_group_container.setEnabled(not checked)

    def _selected_robots_str(self) -> str:
        if self.chk_all.isChecked():
            return "ALL"

        manual = (self.manual_robots.text() or "").strip()
        if manual:
            return manual

        names: List[str] = []
        for i in range(self.robot_list.count()):
            it = self.robot_list.item(i)
            # NOTE: filtering should not affect selection
            if it.checkState() == QtCore.Qt.Checked:
                names.append(it.text())

        if self.robot_list.count() > 0 and len(names) == self.robot_list.count():
            return "ALL"

        # keep original list order
        return ",".join(names) if names else ""

    # ---------------- Tables ----------------

    def _populate_bindings_table(self, layer_idx: int):
        layer = self.cfg["layers"][layer_idx]
        bindings: Dict[str, Any] = layer.get("bindings", {})
        keys = sorted(bindings.keys())

        self.tbl_bind.blockSignals(True)
        self.tbl_bind.setRowCount(len(keys))
        for r, keycode in enumerate(keys):
            b = bindings[keycode] or {}
            kind = str(b.get("kind", "set_pending"))
            value = ""
            if kind == "set_layer":
                value = str(b.get("layer", layer_idx))
            elif kind in ("select_group", "robot_select_group"):
                value = str(b.get("group", ""))
            elif kind in ("select_robot", "robot_select_robot"):
                value = str(b.get("robot") or b.get("name") or "")
            elif kind in ("robot_select_cmd", "select_cmd"):
                value = str(b.get("cmd", ""))
            elif kind in ("select_all", "robot_select_all"):
                value = ""
            else:
                value = str(b.get("suffix", ""))

            desc = str(b.get("description", ""))

            self.tbl_bind.setItem(r, 0, QtWidgets.QTableWidgetItem(keycode))
            self.tbl_bind.setItem(r, 1, QtWidgets.QTableWidgetItem(kind))
            self.tbl_bind.setItem(r, 2, QtWidgets.QTableWidgetItem(value))
            self.tbl_bind.setItem(r, 3, QtWidgets.QTableWidgetItem(desc))

            # lock keycode column (but still selectable)
            self.tbl_bind.item(r, 0).setFlags(self.tbl_bind.item(r, 0).flags() & ~QtCore.Qt.ItemIsEditable)

        self.tbl_bind.blockSignals(False)

    def _on_binding_item_changed(self, item: QtWidgets.QTableWidgetItem):
        r = item.row()
        keycode = self.tbl_bind.item(r, 0).text()
        kind = self.tbl_bind.item(r, 1).text().strip()
        val = self.tbl_bind.item(r, 2).text()
        desc = self.tbl_bind.item(r, 3).text()

        layer_idx = self.layer_combo.currentIndex()
        layer = self.cfg["layers"][layer_idx]
        b = layer.setdefault("bindings", {}).setdefault(keycode, {})

        b["kind"] = kind
        b["description"] = desc

        if kind == "set_layer":
            try:
                b["layer"] = int(val)
            except Exception:
                b["layer"] = layer_idx
            # clean unrelated fields
            b.pop("suffix", None)
            b.pop("group", None)
            b.pop("robot", None)
            b.pop("name", None)
        elif kind in ("select_group", "robot_select_group"):
            b["group"] = (val or "").strip()
            b.pop("suffix", None)
            b.pop("layer", None)
            b.pop("robot", None)
            b.pop("name", None)
        elif kind in ("select_robot", "robot_select_robot"):
            b["robot"] = (val or "").strip()
            b.pop("suffix", None)
            b.pop("layer", None)
            b.pop("group", None)
            b.pop("name", None)
            b.pop("cmd", None)
        elif kind in ("robot_select_cmd", "select_cmd"):
            b["cmd"] = (val or "").strip()
            b.pop("suffix", None)
            b.pop("layer", None)
            b.pop("group", None)
            b.pop("robot", None)
            b.pop("name", None)
        elif kind in ("select_all", "robot_select_all"):
            # nothing to store besides kind/desc
            b.pop("suffix", None)
            b.pop("layer", None)
            b.pop("group", None)
            b.pop("robot", None)
            b.pop("name", None)
            b.pop("cmd", None)
        else:
            b["suffix"] = val
            # clean selection-only fields
            b.pop("group", None)
            b.pop("robot", None)
            b.pop("name", None)
            b.pop("layer", None)
            b.pop("cmd", None)

        self._schedule_save()
        self._refresh_layout_tooltips()

    # ---------------- Physical layout helper ----------------

    def _on_layout_key_clicked(self, label: str):
        keycode = (self._layout_keymap.get(label, "") or "").strip()
        if not keycode:
            self._append_log(f"[LAYOUT] '{label}' has no keycode mapping (edit _layout_keymap in code or use table).")
            return

        layer_idx = self.layer_combo.currentIndex()
        layer = self.cfg["layers"][layer_idx]
        bindings = layer.setdefault("bindings", {})

        # If missing, create a default row so user can immediately edit it.
        if keycode not in bindings:
            bindings[keycode] = {"kind": "set_pending", "suffix": "", "description": ""}
            self._populate_bindings_table(layer_idx)
            self._schedule_save()

        # select row in table
        for r in range(self.tbl_bind.rowCount()):
            it = self.tbl_bind.item(r, 0)
            if it and it.text() == keycode:
                self.tbl_bind.setCurrentCell(r, 1)
                self.tbl_bind.scrollToItem(it, QtWidgets.QAbstractItemView.PositionAtCenter)
                break

        self._refresh_layout_tooltips()

    def _refresh_layout_tooltips(self):
        """Update physical-layout button text/tooltips to reflect current layer bindings."""
        try:
            layer_idx = self.layer_combo.currentIndex()
            bindings: Dict[str, Any] = (self.cfg.get("layers", [{}])[layer_idx] or {}).get("bindings", {}) or {}
        except Exception:
            bindings = {}

        for label, btn in getattr(self, "_layout_buttons", {}).items():
            keycode = (self._layout_keymap.get(label, "") or "").strip()
            if not keycode:
                btn.setText("-")
                btn.setToolTip(f"{label} (unmapped)")
                continue
            b = bindings.get(keycode, {}) or {}
            kind = str(b.get("kind", ""))
            if kind == "set_layer":
                detail = f"layer={b.get('layer', '')}"
            elif kind in ("select_group", "robot_select_group"):
                detail = f"group={b.get('group', '')}"
            elif kind in ("select_robot", "robot_select_robot"):
                detail = f"robot={b.get('robot') or b.get('name') or ''}"
            elif kind in ("robot_select_cmd", "select_cmd"):
                detail = f"cmd={b.get('cmd', '')}"
            elif kind in ("select_all", "robot_select_all"):
                detail = "(toggle all)"
            else:
                detail = f"suffix={b.get('suffix', '')}"
            desc = str(b.get("description", "") or "").strip()

            # Button face: show binding description (user-requested). If empty, show a short fallback.
            if desc:
                btn.setText(desc)
            else:
                # Keep it compact; the tooltip contains full details.
                fallback = " " if kind else "(unbound)"
                btn.setText(fallback)

            tip = f"{label} → {keycode}\n{kind} / {detail}"
            if desc:
                tip += f"\n{desc}"
            btn.setToolTip(tip)

    def _populate_chords_table(self):
        chords = self.cfg.get("chords", [])
        self.tbl_chord.blockSignals(True)
        self.tbl_chord.setRowCount(len(chords))
        for r, ch in enumerate(chords):
            held = ",".join(ch.get("when_held", []))
            press = ch.get("then_press", "")
            action = ch.get("action", {}) or {}
            kind = action.get("kind", "publish_suffix")
            value = ""
            if kind in ("select_group", "robot_select_group"):
                value = str(action.get("group", ""))
            elif kind in ("select_robot", "robot_select_robot"):
                value = str(action.get("robot") or action.get("name") or "")
            elif kind in ("robot_select_cmd", "select_cmd"):
                value = str(action.get("cmd", ""))
            elif kind in ("select_all", "robot_select_all"):
                value = ""
            else:
                value = str(action.get("suffix", ""))
            desc = ch.get("description", "")

            self.tbl_chord.setItem(r, 0, QtWidgets.QTableWidgetItem(held))
            self.tbl_chord.setItem(r, 1, QtWidgets.QTableWidgetItem(press))
            self.tbl_chord.setItem(r, 2, QtWidgets.QTableWidgetItem(kind))
            self.tbl_chord.setItem(r, 3, QtWidgets.QTableWidgetItem(value))
            self.tbl_chord.setItem(r, 4, QtWidgets.QTableWidgetItem(desc))
        self.tbl_chord.blockSignals(False)

    def _on_chord_item_changed(self, item: QtWidgets.QTableWidgetItem):
        r = item.row()
        held = [s.strip() for s in (self.tbl_chord.item(r, 0).text() or "").split(",") if s.strip()]
        press = (self.tbl_chord.item(r, 1).text() or "").strip()
        kind = (self.tbl_chord.item(r, 2).text() or "").strip()
        val = (self.tbl_chord.item(r, 3).text() or "")
        desc = (self.tbl_chord.item(r, 4).text() or "")

        chords = self.cfg.setdefault("chords", [])
        if r >= len(chords):
            return
        action: Dict[str, Any] = {"kind": kind}
        if kind in ("select_group", "robot_select_group"):
            action["group"] = (val or "").strip()
        elif kind in ("select_robot", "robot_select_robot"):
            action["robot"] = (val or "").strip()
        elif kind in ("robot_select_cmd", "select_cmd"):
            action["cmd"] = (val or "").strip()
        elif kind in ("select_all", "robot_select_all"):
            pass
        else:
            action["suffix"] = val

        chords[r] = {
            "when_held": held,
            "then_press": press,
            "action": action,
            "description": desc,
        }
        self._schedule_save()

    def _add_chord(self):
        self.cfg.setdefault("chords", []).append({
            "when_held": ["KEY_B"],
            "then_press": "KEY_A",
            "action": {"kind": "publish_suffix", "suffix": "STATUS::ESTOP"},
            "description": "example chord",
        })
        self._populate_chords_table()
        self._schedule_save()

    def _del_chord(self):
        r = self.tbl_chord.currentRow()
        if r < 0:
            return
        chords = self.cfg.get("chords", [])
        if 0 <= r < len(chords):
            chords.pop(r)
        self._populate_chords_table()
        self._schedule_save()

    # ---------------- Macro logic ----------------

    def _set_active_layer(self, layer_idx: int, from_key: bool):
        layer_idx = max(0, min(8, int(layer_idx)))
        self.layer_combo.blockSignals(True)
        self.layer_combo.setCurrentIndex(layer_idx)
        self.layer_combo.blockSignals(False)

        self.lbl_layer.setText(f"{layer_idx}")
        # repopulate binding table (editor view)
        self._populate_bindings_table(layer_idx)
        self._refresh_layout_tooltips()

        if from_key:
            self._append_log(f"[LAYER] -> {layer_idx}")

    def _wrap_for_label(self, text: str) -> str:
        """Make long macro strings wrap-friendly in QLabel.

        QLabel wordWrap wraps reliably at whitespace. Many of our macro/status
        strings contain few/no spaces, so we insert zero-width spaces after common
        separators to allow wrapping without changing visible text.
        """
        if not text:
            return ""
        zwsp = "\u200b"
        out = str(text)
        out = out.replace("::", f"::{zwsp}")
        out = out.replace(",", f",{zwsp}")
        out = out.replace("/", f"/{zwsp}")
        out = out.replace("_", f"_{zwsp}")
        out = out.replace("-", f"-{zwsp}")
        return out

    def _update_pending_ui(self):
        self.lbl_pending.setText(self._wrap_for_label(self.pending_suffix) if self.pending_suffix else "(none)")
        self.lbl_last.setText(self._wrap_for_label(self.last_published) if self.last_published else "(none)")

    def _clear_pending(self):
        self.pending_suffix = ""
        self._update_pending_ui()

    def _set_pending(self, suffix: str):
        self.pending_suffix = suffix or ""
        self._append_log(f"[PENDING] {self.pending_suffix}")
        self._update_pending_ui()

    def _publish_suffix(self, suffix: str):
        robots = self._selected_robots_str()
        if not robots:
            self._append_log("[WARN] No robot selected.")
            return

        fmt = str(_safe_get(self.cfg, ["behavior", "message_format"], "{robots}::{suffix}"))
        msg = fmt.format(robots=robots, suffix=suffix)
        self.ros_publish(msg)
        self.last_published = msg
        self._update_pending_ui()

    def _publish_pending(self):
        if not self.pending_suffix:
            self._append_log("[WARN] pending is empty.")
            return
        self._publish_suffix(self.pending_suffix)

    def _run_action(self, kind: str, payload: Dict[str, Any], default_layer_idx: int):
        kind = (kind or "").strip()
        if kind == "set_pending":
            self._set_pending(str(payload.get("suffix", "")))
        elif kind == "clear_pending":
            self._clear_pending()
        elif kind == "publish_pending":
            self._publish_pending()
        elif kind == "publish_suffix":
            self._publish_suffix(str(payload.get("suffix", "")))
        elif kind == "set_layer":
            self._set_active_layer(int(payload.get("layer", default_layer_idx)), from_key=True)
        elif kind in ("select_all", "robot_select_all"):
            self._toggle_select_all()
        elif kind in ("select_group", "robot_select_group"):
            target = str(payload.get("group") or payload.get("name") or payload.get("robot") or "").strip()
            # If the given group name doesn't exist, we fall back to toggling a robot name directly.
            self._toggle_group_or_robot(target)
        elif kind in ("select_robot", "robot_select_robot"):
            # Supports either a single robot name or a comma-separated list.
            robots = payload.get("robot") or payload.get("name") or payload.get("robots") or payload.get("names")
            if isinstance(robots, str):
                parts = [p.strip() for p in robots.split(",") if p.strip()]
                if len(parts) == 1:
                    self._toggle_group_or_robot(parts[0])
                else:
                    for p in parts:
                        self._toggle_group_or_robot(p)
            elif isinstance(robots, list):
                for p in robots:
                    self._toggle_group_or_robot(str(p))
        elif kind in ("robot_select_cmd", "select_cmd"):
            # Free-form string command (same feel as other macro strings)
            cmd = payload.get("cmd")
            if cmd is None:
                cmd = payload.get("suffix")
            self._apply_selection_cmd(str(cmd or ""))
        elif kind == "robot_select":
            # payload can be {"action": {...}} or directly an action dict/string
            action = payload.get("action", payload)
            self._apply_robot_selection_action(action)
        else:
            # noop / unknown
            self._append_log(f"[INFO] noop kind={kind}")

    def _match_chord(self, press_code: str) -> Optional[Dict[str, Any]]:
        chords = self.cfg.get("chords", [])
        for ch in chords:
            held = set(ch.get("when_held", []))
            if not held.issubset(self.pressed):
                continue
            if ch.get("then_press", "") != press_code:
                continue
            return ch
        return None

    def _on_key_event(self, code: str, is_down: bool, is_up: bool):
        # track pressed set
        if is_down:
            self.pressed.add(code)
        elif is_up:
            self.pressed.discard(code)

        # only react on key_down
        if not is_down:
            return

        if bool(_safe_get(self.cfg, ["ui", "debug_log_keys"], False)):
            self._append_log(f"[KEY] {code}")

        layer_idx = self.layer_combo.currentIndex()

        # layer indicator keys (global)
        ind = self.cfg.get("layer", {}).get("indicator_keys", {}) or {}
        if code in ind:
            self._set_active_layer(int(ind[code]), from_key=True)
            if bool(_safe_get(self.cfg, ["layer", "suppress_base_action_on_indicator"], True)):
                return

        # chord priority
        chord = self._match_chord(code)
        if chord is not None:
            action = chord.get("action", {}) or {}
            self._append_log(f"[CHORD] held={chord.get('when_held')} + {code}")

            # Safety: treat chord actions like normal bindings.
            # If a publish_key is configured, require an explicit publish_key press
            # by converting immediate publish_suffix into set_pending.
            publish_key = str(_safe_get(self.cfg, ["behavior", "publish_key"], "")).strip()
            kind = action.get("kind", "publish_suffix")
            if publish_key and kind == "publish_suffix":
                self._set_pending(str(action.get("suffix", "")))
            else:
                self._run_action(kind, action, self.layer_combo.currentIndex())

            if bool(_safe_get(self.cfg, ["behavior", "suppress_base_action_on_chord"], True)):
                return

        # normal binding
        layer = self.cfg["layers"][layer_idx]
        b = (layer.get("bindings", {}) or {}).get(code)

        if b:
            # If a per-layer binding exists, it should take precedence over the
            # global behavior keys (publish_key/clear_pending_key). This makes it
            # possible to map KEY_P (or any key) to a real macro without it being
            # "stolen" by publish_key.
            kind = b.get("kind", "set_pending")
            payload = dict(b)
            if kind == "set_layer" and "layer" not in payload:
                payload["layer"] = layer_idx
            self._run_action(kind, payload, layer_idx)
            return

        # robot selection keys (legacy/global hotkeys)
        # IMPORTANT: This runs only when there is NO per-layer binding for the key.
        # That way, robot selection behaves like other bindings and won't "steal" keys
        # from other layers.
        rs = self.cfg.get("robot_selection", {}) or {}
        if self._robot_selection_active_for_layer(layer_idx):
            keys = rs.get("keys", {}) or {}
            action = keys.get(code)
            if action is not None:
                if isinstance(action, dict) and "kind" in action:
                    self._run_action(str(action.get("kind", "")), action, layer_idx)
                else:
                    self._run_action("robot_select", {"action": action}, layer_idx)
                if bool(rs.get("suppress_base_action", False)):
                    return

        # global behavior keys (fallback when there is NO binding)
        publish_key = str(_safe_get(self.cfg, ["behavior", "publish_key"], "")).strip()
        clear_key = str(_safe_get(self.cfg, ["behavior", "clear_pending_key"], "")).strip()
        if clear_key and code == clear_key:
            self._clear_pending()
            return
        if publish_key and code == publish_key:
            self._publish_pending()
            return

    # ---------------- log ----------------

    def _append_log(self, line: str):
        self.log.appendPlainText(line)

    # ---------------- close ----------------

    def closeEvent(self, event):
        try:
            self.reader.close()
        except Exception:
            pass
        try:
            self._save_now()
        except Exception:
            pass
        event.accept()
