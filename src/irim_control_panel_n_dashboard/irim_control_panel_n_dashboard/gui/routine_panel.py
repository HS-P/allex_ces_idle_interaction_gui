# gui/routine_panel.py

from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout,
    QLabel, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QProgressBar
)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt


class RoutinePanel(QWidget):
    """
    /debug/routine 에서 오는 JSON 스냅샷을 트리 형태로 표시하는 패널
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._column_ratios = [0.5, 0.2, 0.15, 0.15]
        self._init_ui()

    def _init_ui(self):
        # 전체 패널 레이아웃
        main_layout = QVBoxLayout(self)
        # 패널 바깥 여백 / 위아래 간격 줄이기
        main_layout.setContentsMargins(4, 4, 4, 4)  # (left, top, right, bottom)
        main_layout.setSpacing(4)

        group = QGroupBox("Routine Debug")
        v = QVBoxLayout(group)
        # 그룹박스 안쪽 여백 / 간격 줄이기
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(4)

        # self.info_label = QLabel("No routine debug data yet.")
        # self.info_label.setAlignment(Qt.AlignLeft)
        # v.addWidget(self.info_label)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Name", "Type", "Status", "Progress"])

        header = self.tree.header()
        # 열 너비를 우리가 직접 비율로 설정하기 위해 Fixed 사용
        header.setSectionResizeMode(QHeaderView.Fixed)

        # ▶ (삼각형) ~ Name 사이 들여쓰기 줄이기
        self.tree.setIndentation(10)  # 기본값보다 작게 (8~12 정도 취향대로 조절)

        # 행 높이/패딩 줄여서 세로로 더 많이 보이게
        self.tree.setStyleSheet("""
        QTreeView::item {
            height: 18px;    /* 행 높이: 더 줄이고 싶으면 16, 더 늘리려면 20 등으로 변경 */
            padding: 0 2px;  /* 좌우 패딩도 최소한만 */
        }
        """)

        v.addWidget(self.tree)
        main_layout.addWidget(group)

    # --------- public API: MainWindow에서 호출 ---------
    def update_routine(self, data: dict):
        """
        data: /debug/routine JSON 파싱한 dict
        {
          "timestamp_sec": float,
          "nodes": [ {id, parent, name, type, status, progress}, ... ]
        }
        """
        # ts = data.get("timestamp_sec", 0.0)
        # self.info_label.setText(f"timestamp: {ts:.3f} sec")

        nodes = data.get("nodes", [])
        self.tree.clear()
        if not nodes:
            return

        # id -> (item, node_dict)
        items = {}
        for n in nodes:
            name = str(n.get("name", ""))
            ntype = str(n.get("type", ""))
            status_val = n.get("status", -1)
            progress = float(n.get("progress", 0.0))

            status_text, color = self._status_text_color(status_val)

            item = QTreeWidgetItem([name, ntype, status_text, ""])
            # 상태 색
            brush = QBrush(color)
            for col in range(3):
                item.setBackground(col, brush)

            items[n.get("id")] = (item, n, progress)

        # 트리 구조 연결
        roots = []
        for nid, (item, n, progress) in items.items():
            parent_id = n.get("parent", -1)
            if parent_id is None or parent_id < 0 or parent_id not in items:
                roots.append(nid)
            else:
                parent_item, _, _ = items[parent_id]
                parent_item.addChild(item)

        for rid in roots:
            self.tree.addTopLevelItem(items[rid][0])

        # progress bar 설정
        for item, n, progress in items.values():
            bar = QProgressBar()
            bar.setRange(0, 100)
            p = max(0, min(1.0, progress))
            bar.setValue(int(p * 100))
            bar.setFormat(f"{p*100:.0f}%")
            self.tree.setItemWidget(item, 3, bar)

        self.tree.expandAll()

    # --------- status 색/텍스트 매핑 ---------
    def _status_text_color(self, status_val: int):
        """
        BT status 가
          0: IDLE
          1: RUNNING
          2: SUCCESS
          3: FAILURE
          4: HALTED
        이런 식이라고 가정하고 매핑.
        (필요하면 여기 숫자만 바꾸면 됨)
        """
        mapping = {
            0: ("IDLE", QColor("lightgray")),
            1: ("RUNNING", QColor("lightblue")),
            2: ("SUCCESS", QColor("palegreen")),
            3: ("FAILURE", QColor("lightcoral")),
            4: ("HALTED", QColor("khaki")),
        }
        text, color = mapping.get(
            status_val,
            (str(status_val), QColor("white"))
        )
        return text, color


    def resizeEvent(self, event):
        """
        윈도우 / 패널 크기 변경 시
        Name, Type, Status, Progress 열을 지정한 비율로 맞춰주는 함수
        """
        super().resizeEvent(event)

        # 트리의 실제 표시 영역 너비
        total_width = self.tree.viewport().width()
        if total_width <= 0:
            return

        # self._column_ratios 에 저장해 둔 비율로 열 너비 설정
        for i, r in enumerate(self._column_ratios):
            w = int(total_width * r)
            self.tree.setColumnWidth(i, w)