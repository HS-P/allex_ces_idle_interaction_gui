from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # Launch 인자 선언
    bash_a_path_arg = DeclareLaunchArgument(
        'bash_a_path',
        default_value='',
        description='Bash A 스크립트의 전체 경로'
    )
    
    bash_b_path_arg = DeclareLaunchArgument(
        'bash_b_path',
        default_value='',
        description='Bash B 스크립트의 전체 경로'
    )
    
    # Bash Controller Node
    bash_controller_node = Node(
        package='allex_ces_idle_interaction',
        executable='bash_controller_node',
        name='bash_controller_node',
        parameters=[{
            'bash_a_path': LaunchConfiguration('bash_a_path'),
            'bash_b_path': LaunchConfiguration('bash_b_path'),
        }],
        output='screen'
    )
    
    return LaunchDescription([
        bash_a_path_arg,
        bash_b_path_arg,
        bash_controller_node,
    ])


