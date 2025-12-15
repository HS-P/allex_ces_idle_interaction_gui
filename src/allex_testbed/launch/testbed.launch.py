from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch 파일 생성"""
    
    # 파라미터 선언
    tracking_result_topic = LaunchConfiguration('tracking_result_topic', default='/allex_camera/tracking_result')
    controller_control_topic = LaunchConfiguration('controller_control_topic', default='/allex_camera/controller_control')
    neck_angle_topic = LaunchConfiguration('neck_angle_topic', default='/allex_camera/neck_angle')
    
    # 테스트베드 제어 노드
    gaze_controller_testbed_node = Node(
        package='allex_testbed',
        executable='gaze_controller_testbed_node',
        name='gaze_controller_testbed_node',
        output='screen',
        parameters=[{
            'tracking_result_topic': tracking_result_topic,
            'controller_control_topic': controller_control_topic,
            'neck_angle_topic': neck_angle_topic,
        }],
    )
    
    # 테스트베드 GUI 노드
    testbed_gui_node = Node(
        package='allex_testbed',
        executable='testbed_gui_node',
        name='testbed_gui_node',
        output='screen',
    )
    
    return LaunchDescription([
        gaze_controller_testbed_node,
        testbed_gui_node,
    ])

