#!/usr/bin/env python3
"""
ROS2 node that visualizes DisplayReading messages as colored markers
at the robot's position when each reading was taken.
"""

import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration

import tf2_ros
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from display_reader_msgs.msg import DisplayReading


def value_to_color(value: float, vmin: float, vmax: float) -> ColorRGBA:
    """Map a value to a color gradient: green -> yellow -> red."""
    t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin))) if vmax > vmin else 0.5
    
    color = ColorRGBA()
    color.a = 1.0
    
    if t < 0.5:
        # Green to yellow
        color.r = t * 2.0
        color.g = 1.0
        color.b = 0.0
    else:
        # Yellow to red
        color.r = 1.0
        color.g = 1.0 - (t - 0.5) * 2.0
        color.b = 0.0
    
    return color


class DisplayMarkerNode(Node):

    def __init__(self):
        super().__init__('display_marker_node')

        # Declare parameters
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('value_min', 0.0)
        self.declare_parameter('value_max', 10.0)
        self.declare_parameter('marker_scale', 0.15)
        self.declare_parameter('max_markers', 500)
        self.declare_parameter('tf_timeout', 0.5)

        # Read parameters
        self.map_frame = self.get_parameter('map_frame').value
        self.robot_frame = self.get_parameter('robot_frame').value
        self.value_min = self.get_parameter('value_min').value
        self.value_max = self.get_parameter('value_max').value
        self.marker_scale = self.get_parameter('marker_scale').value
        self.max_markers = self.get_parameter('max_markers').value
        self.tf_timeout = self.get_parameter('tf_timeout').value

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Storage for markers (rolling buffer)
        self.markers = deque(maxlen=self.max_markers)
        self.marker_id = 0

        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, 'display_markers', 10)

        # Subscriber
        self.reading_sub = self.create_subscription(
            DisplayReading,
            'display_reading',
            self.reading_callback,
            10
        )

        self.get_logger().info(
            f'Display marker node started. Listening on /display_reading, '
            f'publishing to /display_markers (frames: {self.map_frame} -> {self.robot_frame})'
        )

    def reading_callback(self, msg: DisplayReading):
        """Handle incoming display reading."""
        # Skip NaN readings
        if math.isnan(msg.data):
            return

        # Lookup transform at message timestamp
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                msg.header.stamp,
                timeout=Duration(seconds=self.tf_timeout)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return

        # Extract position
        pos = transform.transform.translation

        # Create flat square marker
        square = Marker()
        square.header.frame_id = self.map_frame
        square.header.stamp = self.get_clock().now().to_msg()
        square.ns = 'display_readings'
        square.id = self.marker_id
        square.type = Marker.CUBE
        square.action = Marker.ADD
        square.pose.position.x = pos.x
        square.pose.position.y = pos.y
        square.pose.position.z = 0.005  # Flat on ground
        square.pose.orientation.w = 1.0
        square.scale.x = self.marker_scale
        square.scale.y = self.marker_scale
        square.scale.z = 0.01  # Very thin
        square.color = value_to_color(msg.data, self.value_min, self.value_max)

        # Create text marker
        text = Marker()
        text.header.frame_id = self.map_frame
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = 'display_labels'
        text.id = self.marker_id
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = pos.x
        text.pose.position.y = pos.y
        text.pose.position.z = 0.02 + self.marker_scale * 0.5  # Just above square
        text.pose.orientation.w = 1.0
        text.scale.z = self.marker_scale * 0.8  # Text height
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f'{msg.data:.2f}'

        # Store markers
        self.markers.append((square, text))
        self.marker_id += 1

        # Publish all markers
        self.publish_markers()

    def publish_markers(self):
        """Publish the full marker array."""
        marker_array = MarkerArray()
        
        # First, delete old markers that were removed from deque
        # (handled implicitly by republishing all current markers)
        
        for sphere, text in self.markers:
            marker_array.markers.append(sphere)
            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DisplayMarkerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
