#!/usr/bin/env python3
"""
Live display reader — lightweight version for RPi5 deployment.
Uses TFLite instead of full TensorFlow for minimal resource usage.

Usage:
    python live_feed.py              # default camera, 3 Hz
    python live_feed.py --show       # with preview window
    python live_feed.py --rate 1     # 1 sample/sec (lower CPU)
    python live_feed.py --rotate 0   # no rotation
"""

import cv2
import numpy as np
import sys
import os
import time
import argparse

from display_utils import (
    find_main_digits, find_decimal_position,
    is_dash_display, crop_digit, set_rotation
)


class DigitClassifier:
    """Lightweight digit classifier using TFLite."""
    
    def __init__(self, model_path='models/digit_cnn.tflite'):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
        # Try tflite_runtime first (lightweight, ideal for RPi)
        try:
            from tflite_runtime.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=model_path)
        except ImportError:
            # Fall back to tensorflow.lite
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
            # Prepare input: float32, shape [1, 28, 28, 1]
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


def main():
    parser = argparse.ArgumentParser(description='Live display reader (lightweight)')
    parser.add_argument('--camera', default='0',
                        help='Camera index or device path (default: 0)')
    parser.add_argument('--rate', type=float, default=3.0,
                        help='Samples per second (default: 3)')
    parser.add_argument('--show', action='store_true',
                        help='Show camera preview window')
    parser.add_argument('--rotate', type=int, default=180,
                        help='Rotation in degrees (default: 180)')
    parser.add_argument('--exposure', type=float, default=None,
                        help='Camera exposure (lower = darker). Try -7 to -1 for USB cams.')
    args = parser.parse_args()

    set_rotation(args.rotate)

    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera

    sample_interval = 1.0 / args.rate

    print("Loading model (TFLite)...")
    t0 = time.time()
    classifier = DigitClassifier()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"Error: could not open camera {cam_source}")
        sys.exit(1)

    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        print(f"Exposure set to {args.exposure}")

    print(f"Sampling at {args.rate} Hz | Ctrl+C to stop")
    print("-" * 40)

    last_sample = 0
    last_reading = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            now = time.time()

            if now - last_sample >= sample_interval:
                last_sample = now
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                reading = read_frame(gray, classifier)

                if reading != last_reading:
                    timestamp = time.strftime('%H:%M:%S')
                    display = reading if reading else "NO READING"
                    print(f"  [{timestamp}] {display}")
                    last_reading = reading

            if args.show:
                display_frame = frame.copy()
                if last_reading:
                    cv2.putText(display_frame, f"Reading: {last_reading}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)
                cv2.imshow('Live Display Reader', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
