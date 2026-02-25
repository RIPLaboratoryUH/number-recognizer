#!/usr/bin/env python3
"""
Extract individual digit crops from labeled gauge display images.
Saves 28x28 grayscale digit images to data/digits/{0..9}/ for CNN training.
"""

import cv2
import json
import sys
import shutil
from pathlib import Path

from display_utils import find_main_digits, is_dash_display, crop_digit, set_rotation


def process_image(image_path, label=None, output_dir=None, debug=False):
    """Process a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if is_dash_display(gray):
        return {'type': 'dashes'}
    
    digits, binary = find_main_digits(gray)
    if not digits:
        print(f"  No digits found: {Path(image_path).name}")
        return None
    
    if label and output_dir:
        label_digits = label.replace('.', '')
        if len(label_digits) != len(digits):
            print(f"  Mismatch: found {len(digits)} but need {len(label_digits)} ('{label}') - {Path(image_path).name}")
            if debug:
                debug_img = img.copy()
                for d in digits:
                    cv2.rectangle(debug_img, (d['x'], d['y']),
                                (d['x']+d['w'], d['y']+d['h']), (0, 255, 0), 3)
                debug_path = Path(output_dir).parent / 'debug' / Path(image_path).name
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_path), debug_img)
            return None
        
        for i, d in enumerate(digits):
            digit_crop = crop_digit(binary, d['x'], d['y'], d['w'], d['h'])
            if digit_crop is not None:
                digit_dir = Path(output_dir) / label_digits[i]
                digit_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(digit_dir / f"{Path(image_path).stem}_d{i}.png"), digit_crop)
    
    return {'type': 'number', 'num_digits': len(digits)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract digit crops from labeled images')
    parser.add_argument('--dir', default='captures',
                        help='Image directory (default: captures)')
    parser.add_argument('--labels', default='labels_new.json',
                        help='Labels JSON file (default: labels_new.json)')
    parser.add_argument('--out', default='data/digits',
                        help='Output directory for digit crops (default: data/digits)')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images for mismatches')
    parser.add_argument('--rotate', type=int, default=180,
                        help='Rotation in degrees (default: 180)')
    args = parser.parse_args()

    set_rotation(args.rotate)
    image_dir = Path(args.dir)
    labels_file = Path(args.labels)
    output_dir = Path(args.out)
    
    with open(labels_file) as f:
        labels = json.load(f)
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = success = 0
    digit_counts = {}
    
    for filename, label in sorted(labels.items()):
        image_path = image_dir / filename
        if not image_path.exists():
            continue
        total += 1
        result = process_image(image_path, label, str(output_dir), debug=args.debug)
        if result and result['type'] == 'number':
            success += 1
            for d in label.replace('.', ''):
                digit_counts[d] = digit_counts.get(d, 0) + 1
    
    print(f"\nExtraction: {success}/{total} ({100*success//total}%)")
    print(f"Digits: {sum(digit_counts.values())} total")
    for d in sorted(digit_counts.keys()):
        print(f"  '{d}': {digit_counts[d]}")


if __name__ == '__main__':
    main()
