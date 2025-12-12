from PyQt5.QtWidgets import (
    QWidget, QGroupBox, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt
from irim_control_panel_n_dashboard.gui.utile import *


class StatusPanel(QWidget):
    def __init__(self, articulations):
        """
        :param articulations: 로봇에서 publish되는 articulation의 이름 리스트.
                              예: ["L_arm", "R_arm", ...]
        """
        super().__init__()
        self.articulations = articulations
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 상단: articulation 요약 정보를 위한 그룹 박스 (표 형식)
        summary_group = QGroupBox("Articulation Summary")
        summary_layout = QVBoxLayout(summary_group)

        self.articulation_table = QTableWidget(0, 7)
        self.articulation_table.setHorizontalHeaderLabels(
            ["Name", "servo", "status", "C_space", "C_mode", "Trq_lim", "traj_prog"]
        )
        self.articulation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.articulation_table.verticalHeader().setDefaultSectionSize(15)
        summary_layout.addWidget(self.articulation_table)
        summary_group.setMaximumHeight(380)
        main_layout.addWidget(summary_group)

        # 하단: 상세 데이터(각 joint의 데이터)를 표시할 테이블
        self.detail_table = QTableWidget(0, 4)
        self.detail_table.setHorizontalHeaderLabels(["Name", "servo", "comm_code", "Cont_mode"])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.detail_table.verticalHeader().setDefaultSectionSize(15)
        main_layout.addWidget(self.detail_table)

    def _create_item(self, text, font_size=8, alignment=Qt.AlignCenter):
        """
        QTableWidgetItem을 생성하는 헬퍼 함수.
        """
        item = QTableWidgetItem(str(text))
        font = QFont()
        font.setPointSize(font_size)
        item.setFont(font)
        item.setTextAlignment(alignment)
        return item

    def update_servo_status(self, status_text):
        """
        servo status 전용 라벨 업데이트 (필요 시 구현).
        """
        pass

    def update_loop_time(self, loop_time):
        """
        loop time 전용 라벨 업데이트 (필요 시 구현).
        """
        pass

    def _get_articulation_row(self, articulation):
        """
        주어진 articulation 이름에 해당하는 행(row)의 인덱스를 반환.
        없으면 None 반환.
        """
        row_count = self.articulation_table.rowCount()
        for row in range(row_count):
            item = self.articulation_table.item(row, 0)
            if item and item.text() == articulation:
                return row
        return None
    
    def update_articulation_status(self, articulation, status_list):
        """
        각 articulation의 요약 정보를 테이블에 업데이트.
        
        :param articulation: articulation의 이름
        :param status_list: array('i', [...]) 형식으로, 
                            인덱스 0~4: 각각 servo, arti_status, control_space_type, now_control_mode, traj_method
                            인덱스 5: 현재 진행 값 (value)
                            인덱스 6: 전체 진행 값 (total)
        """
        # articulation 이름으로 기존 행 찾기. 없으면 새 행 추가.
        row = self._get_articulation_row(articulation)
        if row is None:
            row = self.articulation_table.rowCount()
            self.articulation_table.insertRow(row)
        
        # 7번째 열("traj_prog")에 progress bar 업데이트
        progress_bar = self.articulation_table.cellWidget(row, 6)
        if progress_bar is None:
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            self.articulation_table.setCellWidget(row, 6, progress_bar)
        
        # 첫 번째 열에 articulation 이름 업데이트
        self.articulation_table.setItem(row, 0, self._create_item(articulation, font_size=8))
        
        # status_list의 0~4 인덱스에 대해 매핑하여 각 열(1~5) 업데이트
        mapping_functions = [map_servo_status, map_arti_status, map_control_space_type, map_now_control_mode, map_trq_lim]
        
        for idx in range(5):
            # status_list의 길이가 5 미만이면 "N/A" 처리
            status = status_list[idx] if idx < len(status_list) else "N/A"
            mapped_text, bg_color = mapping_functions[idx](status)
            
            item = self._create_item(mapped_text, font_size=8)
            
            item.setBackground(QColor(bg_color))
            self.articulation_table.setItem(row, idx + 1, item)
        
        # progress bar 업데이트: status_list[5] = 현재 진행 값, status_list[6] = 전체 진행 값
        if len(status_list) >= 7:
            current = status_list[5]
            total = status_list[6]
            
            # 6번째 열("traj_prog")에 progress bar 업데이트
            progress_bar = self.articulation_table.cellWidget(row, 6)
            if progress_bar is None:
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 100)
                self.articulation_table.setCellWidget(row, 6, progress_bar)

            if total > 0:
                progress_bar.setValue(int((current / total) * 100))
            else:
                progress_bar.setValue(0)
            progress_bar.setFormat(f"{current}/{total}")

    def update_detail_table(self, data):
        """
        상세 테이블을 업데이트.
        
        :param data: 각 행에 해당하는 데이터 리스트.
                     각 원소는 아래와 같은 키를 포함하는 dict여야 함.
                     {
                         "Name": "L_arm_0",
                         "servo": <value>,
                         "comm_code": <value>,
                         "Cont_mode": <value>
                     }
        """
        self.detail_table.setRowCount(len(data))
        for row, row_data in enumerate(data):
            for col, key in enumerate(["Name", "servo", "comm_code", "Cont_mode"]):
                item = self._create_item(row_data.get(key, ""), font_size=8)
                self.detail_table.setItem(row, col, item)
