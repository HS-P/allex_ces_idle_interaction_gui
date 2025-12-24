from setuptools import find_packages, setup

package_name = 'irim_control_panel_n_dashboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yeah',
    maintainer_email='yeah@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'control_panel_main = irim_control_panel_n_dashboard.control_panel_main:main',
            'joint_cmd_sliders = irim_control_panel_n_dashboard.joint_cmd_sliders:main',
            'foot_sw = irim_control_panel_n_dashboard.foot_sw_node:main'
        ],
    },
)
