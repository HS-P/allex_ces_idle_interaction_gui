

from PyQt5.QtGui import QColor


def map_trq_lim(value):
    """
    Trq_lim 값에 따라 색상과 텍스트를 반환.
    100 -> 흰색
    < 100 -> 파란색 계열 (낮을수록 진하게)
    > 100 -> 빨간색 계열 (높을수록 진하게)
    """
    try:
        value = int(value)
    except (ValueError, TypeError):
        return "N/A", "slategray"

    if value == -1:
        return "N/A", "slategray"

    text = f"{value}%"
    
    if value == 100:
        color = QColor("white")
    elif value < 100:
        # 0에 가까울수록 진한 파란색 (0,0,255)
        # 100에 가까울수록 흰색에 가까운 파란색
        ratio = max(0, value) / 100.0
        red = int(255 - 255 * (1 - ratio))
        green = int(255 - 255 * (1 - ratio))
        color = QColor(red, green, 255)
    else: # value > 100
        # 200 이상에서 가장 진한 빨간색 (255,0,0)
        # 100에 가까울수록 흰색에 가까운 빨간색
        ratio = min(1.0, (value - 100) / 100.0)
        green = int(255 * (1 - ratio))
        blue = int(255 * (1 - ratio))
        color = QColor(255, green, blue)
    
    return text, color


def map_servo_status(value):
    """
    servo_status 매핑:
      0 -> OFF
      1 -> ON
    그 외 숫자는 그대로 문자열로 반환하고, 배경색은 'red'로 지정.
    """
    defulat_color = "white"
    if value == 0:
        return "OFF", defulat_color
    elif value == 1:
        return "ON", "lightsteelblue"
    elif value == -1:
        return "N/A", "slategray"
    else:
        return str(value), "red"


def map_arti_status(value):
    """
    arti_status 매핑:
      1 -> SAFE
      2 -> IDLE_MODE
      3 -> HOMING
      4 -> READY
      5 -> RUN
      6 -> STOP
      7 -> DONE
    그 외 숫자는 그대로 문자열로 반환하고, 배경색은 'red'로 지정.
    """
    defulat_color = "white"
    if value == 1:
        return "SAFE", defulat_color
    elif value == 2:
        return "IDLE_MODE", defulat_color
    elif value == 3:
        return "HOMING", defulat_color
    elif value == 4:
        return "READY", defulat_color
    elif value == 5:
        return "RUN", "lightblue"
    elif value == 6:
        return "STOP", "lightcoral"
    elif value == 7:
        return "DONE", defulat_color
    elif value == -1:
        return "N/A", "slategray"
    else:
        return str(value), "red"

def map_control_space_type(value):
    """
    control_space_type 매핑:
      1 -> Joint
      2 -> Cartasian
    그 외는 숫자 그대로, 배경색은 기본 'white'
    """
    mapping = {1: "Joint", 2: "Cartasian"}
    if value == -1:
        return "N/A", "slategray"
    return mapping.get(value, str(value)), "white"

def map_now_control_mode(value):
    """
    now_control_mode 매핑:
      1 -> Position
      2 -> Torque
      3 -> Speed
    그 외는 숫자 그대로, 배경색은 기본 'white'
    """
    mapping = {1: "Position", 2: "Torque", 3: "Speed"}
    if value == -1 or value == 0:
        return "N/A", "slategray"
    return mapping.get(value, str(value)), "white"


from PyQt5.QtWidgets import (
    QProgressBar
)

class CustomProgressBar(QProgressBar):
    def __init__(self, parent=None, total_value=100):
        super().__init__(parent)
        self.total_value = total_value
        self.setRange(0, self.total_value)
        self.setTextVisible(True)

    def textFromValue(self, value):
        return f"{value}/{self.total_value}"