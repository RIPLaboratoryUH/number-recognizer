from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='0',
                              description='Camera index or device path'),
        DeclareLaunchArgument('rate', default_value='3.0',
                              description='Samples per second'),
        DeclareLaunchArgument('rotation', default_value='180',
                              description='Image rotation in degrees'),
        DeclareLaunchArgument('exposure', default_value='-1.0',
                              description='Camera exposure (-1 = auto)'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='Path to TFLite model (empty = use installed)'),

        Node(
            package='display_reader',
            executable='display_reader_node',
            name='display_reader_node',
            output='screen',
            parameters=[{
                'camera': LaunchConfiguration('camera'),
                'rate': LaunchConfiguration('rate'),
                'rotation': LaunchConfiguration('rotation'),
                'exposure': LaunchConfiguration('exposure'),
                'model_path': LaunchConfiguration('model_path'),
            }],
        ),
    ])
