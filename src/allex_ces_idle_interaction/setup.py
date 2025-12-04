import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'allex_ces_idle_interaction'

config_files = [
    path for path in glob('config/**/*', recursive=True) if os.path.isfile(path)
]
config_data_files = [
    (
        os.path.join('share', package_name, os.path.dirname(path)),
        [path],
    )
    for path in config_files
]

# Launch 파일 추가
launch_files = [
    path for path in glob('launch/**/*', recursive=True) if os.path.isfile(path)
]
launch_data_files = [
    (
        os.path.join('share', package_name, os.path.dirname(path)),
        [path],
    )
    for path in launch_files
]

setup(
    name=package_name,
    version='1.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + config_data_files + launch_data_files,
    install_requires=['setuptools', 'ultralytics', 'pyyaml', 'lap', 'filterpy', 'scipy', 'numpy', 'PySide6', 'huggingface_hub', 'transformers', 'pillow'],
    zip_safe=True,
    maintainer='yeah2',
    maintainer_email='yeah2@todo.todo',
    description='Person Tracking Control System with LLM Integration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'allex_idle_interaction_node = allex_ces_idle_interaction.allex_idle_interaction_node:main',
            'yolo_detection_node = allex_ces_idle_interaction.yolo_detection_node:main',
            'tracking_fsm_node = allex_ces_idle_interaction.tracking_fsm_node:main',
            'gaze_controller_neck_waist_node = allex_ces_idle_interaction.gaze_controller_neck_waist_node:main',
            'idle_interaction_gui_node = allex_ces_idle_interaction.idle_interaction_gui_node:main',
            'llm_hand_gesture_clip_node = allex_ces_idle_interaction.llm_hand_gesture_clip_node:main',
            'gui_node = allex_ces_idle_interaction.idle_interaction_gui_node:main',
            'camera_test = allex_ces_idle_interaction.camera_test:main',
        ],
    },
)
