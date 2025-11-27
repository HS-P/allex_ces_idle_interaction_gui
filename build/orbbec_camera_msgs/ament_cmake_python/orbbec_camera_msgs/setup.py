from setuptools import find_packages
from setuptools import setup

setup(
    name='orbbec_camera_msgs',
    version='1.2.2',
    packages=find_packages(
        include=('orbbec_camera_msgs', 'orbbec_camera_msgs.*')),
)
