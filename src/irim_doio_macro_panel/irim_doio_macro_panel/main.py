from __future__ import annotations

import argparse
import os
import signal
import sys
import threading

import rclpy
from ament_index_python.packages import get_package_share_directory
from PyQt5 import QtWidgets

from irim_doio_macro_panel.config.io import load_config, save_config
from irim_doio_macro_panel.gui.main_window import MainWindow
from irim_doio_macro_panel.ros_interface.ros_node import MacroPublisherNode

def _default_user_config_path() -> str:
    return os.path.join(os.path.expanduser("~/ros2_ws/src/IRIM_robot/irim_doio_macro_panel/config"), "irim_doio_macro_panel", "mapping.json")
# /home/yeah/ros2_ws/src/IRIM_robot/irim_doio_macro_panel/config

def _ensure_default_config(dst_path: str) -> None:
    if os.path.exists(dst_path):
        return
    share = get_package_share_directory("irim_doio_macro_panel")
    src = os.path.join(share, "config", "default_mapping.json")
    with open(src, "r", encoding="utf-8") as f:
        cfg = f.read()
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(cfg)

def _ros_spin(node):
    rclpy.spin(node)

def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]

    parser = argparse.ArgumentParser(description="DOIO macro keyboard → ROS2 topic publisher (GUI)")
    parser.add_argument("--config", default=_default_user_config_path(), help="mapping json path")
    args = parser.parse_args(argv)

    _ensure_default_config(args.config)
    cfg = load_config(args.config)

    rclpy.init()

    topic = cfg["ros"]["topic"]
    depth = int(cfg["ros"].get("qos_depth", 10))
    node = MacroPublisherNode(topic=topic, qos_depth=depth)

    ros_thread = threading.Thread(target=_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    app = QtWidgets.QApplication(sys.argv)

    # SIGINT -> Qt quit
    signal.signal(signal.SIGINT, lambda *_: app.quit())

    win = MainWindow(cfg=cfg, cfg_path=args.config, ros_publish_fn=node.publish_text)
    win.show()

    exit_code = app.exec_()

    node.destroy_node()
    rclpy.shutdown()
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
