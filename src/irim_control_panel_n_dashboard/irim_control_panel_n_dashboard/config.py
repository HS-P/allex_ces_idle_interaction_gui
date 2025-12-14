# config.py

COMMAND_TYPES = ["STATUS", "SPECIFIC", "SCENARIO", "PATH", "ROUTINE"]

ROBOT_STATUS = [
    "SERVO_OFF",
    "SERVO_ON",
    "HOMING",
    "READY",
    "RUN",
    "STOP",
    "SAFE"
]

SPECIFIC_MODES = [
    "ControlMode",
    "ControlSpaceType",
    "TorqueLimits"
]

CONTROL_MODE = ["Position", "Velocity", "Torque"]
CONTROL_SPACE_TYPE = ["Joint", "Cartesian"]

ROBOT_SCENARIO = ["IMPEDANCE", "P2P", "GRAVITY", "TELEOP"]

# ★ 추가: ROUTINE에서 사용할 공통 액션들
ROUTINE_ACTIONS = ["START", "PAUSE", "RESUME", "RESET"]
