#!/usr/bin/env python3
"""
THOR GUI Launch 파일
- GUI 노드만 실행
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
    
    return LaunchDescription([
        gui_node,
    ])

