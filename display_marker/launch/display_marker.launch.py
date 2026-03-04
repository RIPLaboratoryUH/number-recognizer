from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('robot_frame', default_value='base_link'),
        DeclareLaunchArgument('value_min', default_value='0.0'),
        DeclareLaunchArgument('value_max', default_value='10.0'),
        DeclareLaunchArgument('marker_scale', default_value='0.15'),
        DeclareLaunchArgument('max_markers', default_value='500'),

        Node(
            package='display_marker',
            executable='display_marker_node',
            name='display_marker_node',
            output='screen',
            parameters=[{
                'map_frame': LaunchConfiguration('map_frame'),
                'robot_frame': LaunchConfiguration('robot_frame'),
                'value_min': LaunchConfiguration('value_min'),
                'value_max': LaunchConfiguration('value_max'),
                'marker_scale': LaunchConfiguration('marker_scale'),
                'max_markers': LaunchConfiguration('max_markers'),
            }]
        ),
    ])
