#!/usr/bin/env python3
"""
THOR 노드 Launch 파일
- 카메라 스트리밍 (orbbec_camera)
- 조이스틱 제어 노드
- GUI 노드
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch 파일 생성"""
    
    # 1. 카메라 Launch (orbbec_camera 패키지의 femto_bolt.launch.py 포함)
    orbbec_camera_package = get_package_share_directory('orbbec_camera')
    femto_launch_file = os.path.join(orbbec_camera_package, 'launch', 'femto_bolt.launch.py')
    
    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(femto_launch_file),
    )
    
    # 2. 조이스틱 제어 노드
    joystick_control_node = Node(
        package='allex_ces_idle_interaction',
        executable='joystick_control_node',
        name='joystick_control_node',
        output='screen',
    )
    
    # 3. GUI 노드
    gui_node = Node(
        package='allex_ces_idle_interaction',
        executable='idle_interaction_gui_node',
        name='idle_interaction_gui_node',
        output='screen',
    )
    
    return LaunchDescription([
        orbbec_camera_launch,
        joystick_control_node,
        gui_node,
    ])

