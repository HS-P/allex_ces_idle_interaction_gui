from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="irim_doio_macro_panel",
            executable="doio_macro_panel",
            name="doio_macro_panel",
            output="screen",
            arguments=[],
        )
    ])
