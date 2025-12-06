#!/usr/bin/env python3
"""
ALLEX Idle Interaction Launch 파일
모든 노드를 통합하여 실행
토픽명을 Launch 파일에서 관리
"""
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Launch 파일 생성"""
    
    # 토픽명 파라미터 선언
    camera_image_topic = LaunchConfiguration('camera_image_topic', default='/camera/color/image_raw/compressed')
    detections_topic = LaunchConfiguration('detections_topic', default='/allex_camera/detections')
    tracking_result_topic = LaunchConfiguration('tracking_result_topic', default='/allex_camera/tracking_result')
    tracking_data_topic = LaunchConfiguration('tracking_data_topic', default='/allex_camera/tracking_data')
    target_crop_topic = LaunchConfiguration('target_crop_topic', default='/allex_camera/target_crop/compressed')
    tracker_control_topic = LaunchConfiguration('tracker_control_topic', default='/allex_camera/tracker_control')
    tracker_state_request_topic = LaunchConfiguration('tracker_state_request_topic', default='/allex_camera/tracker_state_request')
    neck_angle_topic = LaunchConfiguration('neck_angle_topic', default='/allex_camera/neck_angle')
    manual_control_topic = LaunchConfiguration('manual_control_topic', default='/allex_camera/manual_control')
    llm_response_topic = LaunchConfiguration('llm_response_topic', default='/llm/response')
    llm_control_topic = LaunchConfiguration('llm_control_topic', default='/llm/control')
    llm_status_topic = LaunchConfiguration('llm_status_topic', default='/llm/status')
    
    # # 1. Camera Publisher Node (카메라 이미지 발행)
    # camera_publisher_node = Node(
    #     package='allex_ces_idle_interaction',
    #     executable='camera_test',
    #     name='camera_publisher_node',
    #     output='screen',
    #     parameters=[{
    #         'camera_image_topic': camera_image_topic,
    #     }],
    # )
    
    # 2. YOLO Detection Node (YOLO Detection)
    yolo_detection_node = Node(
        package='allex_ces_idle_interaction',
        executable='yolo_detection_node',
        name='yolo_detection_node',
        output='screen',
        parameters=[{
            'camera_image_topic': camera_image_topic,
            'detections_topic': detections_topic,
            'tracker_control_topic': tracker_control_topic,
        }],
    )
    
    # 3. Tracking FSM Node (FSM 처리)
    tracking_fsm_node = Node(
        package='allex_ces_idle_interaction',
        executable='tracking_fsm_node',
        name='tracking_fsm_node',
        output='screen',
        parameters=[{
            'detections_topic': detections_topic,
            'camera_image_topic': camera_image_topic,
            'tracking_result_topic': tracking_result_topic,
            'tracker_control_topic': tracker_control_topic,
            'tracker_state_request_topic': tracker_state_request_topic,
            'neck_angle_topic': neck_angle_topic,
        }],
    )
    
    # 4. Gaze Controller Node (로봇 제어)
    gaze_controller_node = Node(
        package='allex_ces_idle_interaction',
        executable='gaze_controller_neck_waist_node',
        name='gaze_controller_neck_waist_node',
        output='screen',
        parameters=[{
            'tracking_result_topic': tracking_result_topic,
            'neck_angle_topic': neck_angle_topic,
            'tracker_state_request_topic': tracker_state_request_topic,
        }],
    )
    
    # 5. ALLEX Idle Interaction Node (총괄 노드)
    allex_idle_interaction_node = Node(
        package='allex_ces_idle_interaction',
        executable='allex_idle_interaction_node',
        name='allex_idle_interaction_node',
        output='screen',
        parameters=[{
            'camera_image_topic': camera_image_topic,
            'tracking_result_topic': tracking_result_topic,
            'tracking_data_topic': tracking_data_topic,
            'target_crop_topic': target_crop_topic,
            'tracker_control_topic': tracker_control_topic,
            'manual_control_topic': manual_control_topic,
        }],
    )
    
    # 6. LLM Hand Gesture CLIP Node (LLM 판단)
    llm_hand_gesture_clip_node = Node(
        package='allex_ces_idle_interaction',
        executable='llm_hand_gesture_clip_node',
        name='llm_hand_gesture_clip_node',
        output='screen',
        parameters=[{
            'target_crop_topic': target_crop_topic,
            'llm_response_topic': llm_response_topic,
            'llm_control_topic': llm_control_topic,
            'llm_status_topic': llm_status_topic,
        }],
    )
    
    return LaunchDescription([
        # camera_publisher_node,
        yolo_detection_node,
        tracking_fsm_node,
        gaze_controller_node,
        allex_idle_interaction_node,
        llm_hand_gesture_clip_node,
    ])

