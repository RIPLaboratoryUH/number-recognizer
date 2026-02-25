#!/usr/bin/env python3
"""
Label captured images by typing the display reading for each.

Usage:
    python label.py                       # label images in captures/
    python label.py --dir my_captures     # custom directory
    python label.py --labels my_labels.json

Controls:
    Type the reading shown (e.g. "15.31") and press Enter
    Type "skip" to skip an image
    Type "dash" for ---- display
    Type "quit" to stop
"""

import cv2
import json
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Label captured images')
    parser.add_argument('--dir', default='captures',
                        help='Directory with images (default: captures)')
    parser.add_argument('--labels', default='labels_new.json',
                        help='Labels output file (default: labels_new.json)')
    args = parser.parse_args()

    img_dir = Path(args.dir)
    labels_file = Path(args.labels)

    # Load existing labels
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)

    # Find unlabeled images
    images = sorted(img_dir.glob('*.jpeg')) + sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
    unlabeled = [img for img in images if img.name not in labels]

    print(f"Images: {len(images)} total, {len(labels)} labeled, {len(unlabeled)} remaining")
    print("Type the reading, 'skip', 'dash', or 'quit'")
    print("-" * 40)

    for img_path in unlabeled:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        cv2.imshow('Label This Image', img)
        cv2.waitKey(100)

        value = input(f"  {img_path.name}: ").strip()

        if value.lower() == 'quit':
            break
        elif value.lower() == 'skip':
            continue
        elif value.lower() == 'dash':
            labels[img_path.name] = '----'
        elif value:
            labels[img_path.name] = value

        # Save after each label
        with open(labels_file, 'w') as f:
            json.dump(labels, f, indent=2)

    cv2.destroyAllWindows()
    print(f"\nLabeled: {len(labels)} images saved to {labels_file}")


if __name__ == '__main__':
    main()
