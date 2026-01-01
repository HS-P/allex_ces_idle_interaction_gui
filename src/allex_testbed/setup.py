import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'allex_testbed'

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

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + config_data_files,
    install_requires=['setuptools', 'pyyaml', 'numpy', 'PyQt5', 'opencv-python', 'torch', 'ultralytics', 'huggingface-hub'],
    zip_safe=True,
    maintainer='yeah2',
    maintainer_email='yeah2@todo.todo',
    description='ALLEX 얼굴/허리 제어 테스트베드 - PID 튜닝 및 알고리즘 테스트',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'gaze_controller_testbed_node = allex_testbed.gaze_controller_testbed_node:main',
            'testbed_gui_node = allex_testbed.testbed_gui_node:main',
            'testbed_gaze_tracking_node = allex_testbed.testbed_gaze_tracking_node:main',
            'testbed_gaze_tracking_gui_node = allex_testbed.testbed_gaze_tracking_gui_node:main',
            'target_selection_debug_node = allex_testbed.target_selection_debug_node:main',
            'image_file_publisher_node = allex_testbed.image_file_publisher_node:main',
        ],
    },
)

