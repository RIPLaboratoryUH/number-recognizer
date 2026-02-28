#!/usr/bin/env python3
"""
ROS2 node that reads a 7-segment display via USB camera and publishes
the value as a timestamped DisplayReading message.
"""

import math
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from display_reader.display_utils import (
    find_main_digits, find_decimal_position,
    is_dash_display, crop_digit, set_rotation
)


class DigitClassifier:
    """Lightweight digit classifier using TFLite."""

    def __init__(self, model_path):
        self.interpreter = None

        try:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=model_path)
        except ImportError:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import tensorflow as tf
            self.interpreter = tf.lite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, digit_crops):
        """Classify a list of 28x28 digit images. Returns list of digit strings."""
        results = []
        for crop in digit_crops:
            inp = crop.astype(np.float32).reshape(1, 28, 28, 1) / 255.0
            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            results.append(str(np.argmax(output)))
        return results


def read_frame(gray, classifier):
    """Read the display number from a grayscale frame."""
    if is_dash_display(gray):
        return '----'

    digits, binary = find_main_digits(gray)
    if not digits:
        return None

    digit_crops = []
    for d in digits:
        crop = crop_digit(binary, d['x'], d['y'], d['w'], d['h'])
        if crop is None:
            return None
        digit_crops.append(crop)

    predicted = classifier.predict(digit_crops)

    decimal_after = find_decimal_position(gray, digits)
    if decimal_after is not None and decimal_after < len(predicted) - 1:
        return ''.join(predicted[:decimal_after + 1]) + '.' + ''.join(predicted[decimal_after + 1:])
    else:
        return ''.join(predicted)


class DisplayReaderNode(Node):

    def __init__(self):
        super().__init__('display_reader_node')

        # Declare parameters
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter(
            'camera', '0',
            descriptor=ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter('rate', 3.0)
        self.declare_parameter('rotation', 180)
        self.declare_parameter('exposure', 15.0)
        self.declare_parameter('model_path', '')

        # Read parameters
        camera_param = str(self.get_parameter('camera').value)
        rate = self.get_parameter('rate').get_parameter_value().double_value
        rotation = self.get_parameter('rotation').get_parameter_value().integer_value
        exposure = self.get_parameter('exposure').get_parameter_value().double_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value

        if not model_path:
            pkg_share = get_package_share_directory('display_reader')
            model_path = os.path.join(pkg_share, 'models', 'digit_cnn.tflite')

        set_rotation(rotation)

        # Camera setup
        try:
            cam_source = int(camera_param)
        except ValueError:
            cam_source = camera_param

        self.get_logger().info(f'Loading TFLite model from {model_path}...')
        t0 = time.time()
        self.classifier = DigitClassifier(model_path)
        self.get_logger().info(f'Model loaded in {time.time() - t0:.1f}s')

        self.cap = cv2.VideoCapture(cam_source)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open camera {cam_source}')
            raise RuntimeError(f'Could not open camera {cam_source}')

        if exposure >= 0:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            self.get_logger().info(f'Exposure set to {exposure}')

        # Import the custom message
        from display_reader_msgs.msg import DisplayReading
        self.DisplayReading = DisplayReading

        # Publisher
        self.pub = self.create_publisher(DisplayReading, 'display_reading', 10)

        # Timer
        period = 1.0 / rate
        self.timer = self.create_timer(period, self.timer_callback)
        self.last_reading = None

        self.get_logger().info(
            f'Display reader started | camera={cam_source} rate={rate}Hz '
            f'rotation={rotation}° exposure={exposure}'
        )

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Could not read frame from camera')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reading = read_frame(gray, self.classifier)

        if reading is None:
            return

        msg = self.DisplayReading()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'display_reader'

        if reading == '----':
            msg.data = float('nan')
        else:
            try:
                msg.data = float(reading)
            except ValueError:
                self.get_logger().warn(f'Could not parse reading: {reading}')
                return

        self.pub.publish(msg)

        if reading != self.last_reading:
            self.get_logger().info(f'Reading: {reading}')
            self.last_reading = reading

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DisplayReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
