#!/usr/bin/env python3
"""
PC1 Launch 파일 (노트북 - GUI 및 조이스틱 제어)
- GUI 노드
- 조이스틱 제어 노드
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch 파일 생성"""
    
    # GUI 노드
    gui_node = Node(
        package='allex_ces_idle_interaction',
        executable='idle_interaction_gui_node',
        name='idle_interaction_gui_node',
        output='screen',
    )
    
    # 조이스틱 제어 노드
    joystick_control_node = Node(
        package='allex_ces_idle_interaction',
        executable='joystick_control_node',
        name='joystick_control_node',
        output='screen',
    )
    
    return LaunchDescription([
        gui_node,
        joystick_control_node,
    ])

