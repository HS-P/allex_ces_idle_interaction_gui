#!/usr/bin/env python3
"""
PC2 Launch 파일 (카메라 전용 PC - orbbec_camera)
- orbbec_camera 패키지의 femto_bolt.launch.py 실행
"""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Launch 파일 생성"""
    
    # 카메라 Launch (orbbec_camera 패키지의 femto_bolt.launch.py 포함)
    orbbec_camera_package = get_package_share_directory('orbbec_camera')
    femto_launch_file = os.path.join(orbbec_camera_package, 'launch', 'femto_bolt.launch.py')
    
    orbbec_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(femto_launch_file),
    )
    
    return LaunchDescription([
        orbbec_camera_launch,
    ])

