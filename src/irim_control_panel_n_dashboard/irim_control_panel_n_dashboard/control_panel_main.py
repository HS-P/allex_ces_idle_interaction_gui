# main.py

import sys
import threading
import subprocess
import re
import time

from PyQt5.QtWidgets import QApplication
import rclpy
from rclpy.executors import MultiThreadedExecutor

from irim_control_panel_n_dashboard.gui.main_window import MainWindow
from irim_control_panel_n_dashboard.ros_interface.control_panel_ros_node import ROSInterface


def ros_spin(ros_interface):
    '''
    별도 스레드에서 rclpy.spin을 돌리기 위한 함수
    '''
    rclpy.spin(ros_interface)


def get_articulations_from_topics():
    '''
    ros2 topic list 를 호출해서
    /robot_outbound_data/<name>/… 형태의 토픽에서 <name> 만 뽑아서
    중복 없이 순서대로 리스트로 반환
    '''
    try:
        result = subprocess.run(
            ['ros2', 'topic', 'list'],
            capture_output=True, text=True, check=True
        )
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] ros2 topic list 호출 실패: {e}', file=sys.stderr)
        return []

    lines = result.stdout.splitlines()

    pattern = re.compile(r'^/robot_outbound_data/([^/]+)/')
    seen = []
    for line in lines:
        stripped = line.strip()
        m = pattern.match(stripped)
        if m:
            name = m.group(1)
            if name not in seen:
                seen.append(name)
    return seen


def main():
    # ROS 초기화
    rclpy.init()

    # 토픽에서 자동 파싱 (재시도 로직 추가)
    while True:
        articulations = get_articulations_from_topics()
        if articulations:
            print(f'사용 가능한 articulations: {articulations}')
            break
        else:
            print("토픽에서 파싱에 실패했습니다.")
            print('제어 노드를 먼저 동작시키세요.\n 1초 후 재시도합니다.\n', file=sys.stderr)
            time.sleep(1)

    # 인터페이스 생성 및 ROS 스핀
    ros_interface = ROSInterface(articulations)
    ros_thread = threading.Thread(
        target=ros_spin,
        args=(ros_interface,),
        daemon=True
    )
    ros_thread.start()

    # Qt 애플리케이션 실행
    app = QApplication(sys.argv)
    main_window = MainWindow(ros_interface)
    main_window.show()
    exit_code = app.exec_()

    # 종료 시 정리
    ros_interface.destroy_node()
    rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
