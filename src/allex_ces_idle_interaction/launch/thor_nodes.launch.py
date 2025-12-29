#!/usr/bin/env python3
"""
THOR 노드 Launch 파일
- 카메라 스트리밍 (orbbec_camera)
- 조이스틱 제어 노드
- GUI 노드
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch 파일 생성"""
    
    # 1. 카메라 Launch (IncludeLaunchDescription 사용하지 않고 직접 노드 실행)
    # 주의: orbbec_camera 패키지의 femto_bolt.launch.py를 직접 실행하는 것이 아니라
    # 여기서는 조이스틱과 GUI만 포함하고, 카메라는 별도 스크립트에서 실행
    
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
        joystick_control_node,
        gui_node,
    ])

