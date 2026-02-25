#!/usr/bin/env python3
"""
Read the main display number from ultrasonic gauge images.
Uses TFLite for lightweight inference.

Usage:
    python read_display.py captures/img_0079.jpeg
    python read_display.py captures/
    python read_display.py captures/ --verify labels_new.json
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

from display_utils import (
    find_main_digits, find_decimal_position,
    is_dash_display, crop_digit, set_rotation
)


class DigitClassifier:
    """Lightweight digit classifier using TFLite."""

    def __init__(self, model_path='models/digit_cnn.tflite'):
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
        results = []
        for crop in digit_crops:
            inp = crop.astype(np.float32).reshape(1, 28, 28, 1) / 255.0
            self.interpreter.set_tensor(self.input_details[0]['index'], inp)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            results.append(str(np.argmax(output)))
        return results


def read_display(image_path, classifier):
    """Read the display number from an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if is_dash_display(gray):
        return '----'

    digits, binary = find_main_digits(gray)
    if not digits:
        return None

    crops = []
    for d in digits:
        crop = crop_digit(binary, d['x'], d['y'], d['w'], d['h'])
        if crop is None:
            return None
        crops.append(crop)

    predicted = classifier.predict(crops)

    decimal_after = find_decimal_position(gray, digits)
    if decimal_after is not None and decimal_after < len(predicted) - 1:
        return ''.join(predicted[:decimal_after + 1]) + '.' + ''.join(predicted[decimal_after + 1:])
    else:
        return ''.join(predicted)


def main():
    parser = argparse.ArgumentParser(description='Read gauge display numbers')
    parser.add_argument('target', help='Image file or directory')
    parser.add_argument('--verify', help='Labels JSON for accuracy check')
    parser.add_argument('--rotate', type=int, default=180,
                        help='Rotation in degrees (default: 180)')
    args = parser.parse_args()

    set_rotation(args.rotate)
    target = Path(args.target)

    print("Loading model...")
    classifier = DigitClassifier()

    if target.is_dir():
        images = sorted(target.glob('*.jpeg')) + sorted(target.glob('*.jpg')) + sorted(target.glob('*.png'))
    else:
        images = [target]

    verify_labels = {}
    if args.verify:
        import json
        with open(args.verify) as f:
            verify_labels = json.load(f)

    correct = total_verified = 0
    results = {}

    for img_path in images:
        result = read_display(img_path, classifier)
        results[img_path.name] = result
        display = result if result else "NO READING"

        if verify_labels:
            expected = verify_labels.get(img_path.name)
            if expected:
                total_verified += 1
                match = (result == expected)
                if match:
                    correct += 1
                status = "✓" if match else f"✗ (expected {expected})"
                print(f"  {img_path.name}: {display}  {status}")
            else:
                print(f"  {img_path.name}: {display}  (no label)")
        else:
            print(f"  {img_path.name}: {display}")

    if verify_labels and total_verified > 0:
        print(f"\n{'='*50}")
        print(f"Accuracy: {correct}/{total_verified} ({correct/total_verified*100:.1f}%)")

        mismatches = [(n, results.get(n), verify_labels.get(n))
                      for n in [p.name for p in images]
                      if verify_labels.get(n) and results.get(n) != verify_labels.get(n)]
        if mismatches:
            print(f"\nMismatches ({len(mismatches)}):")
            for name, got, expected in mismatches:
                print(f"  {name}: got '{got}', expected '{expected}'")


if __name__ == '__main__':
    main()
