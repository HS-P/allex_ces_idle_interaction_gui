import os
from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = 'camera_node'

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
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ] + config_data_files,
    install_requires=['setuptools', 'ultralytics', 'pyyaml', 'lap', 'filterpy', 'scipy', 'numpy', 'PySide6', 'huggingface_hub'],
    zip_safe=True,
    maintainer='yeah2',
    maintainer_email='yeah2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = camera_node.camera_publisher:main',
            'video_publisher = camera_node.video_publisher:main',
            'image_publisher = camera_node.image_publisher:main',
            'zed_publisher = camera_node.zed_publisher:main',
            'gui_node = camera_node.gui_node:main',
            'face_detection_test = camera_node.face_detection_test:main',
        ],
    },
)
