#!/usr/bin/env python3
"""
Capture images from a live camera feed for building a training dataset.

Usage:
    python capture.py                          # default camera
    python capture.py --camera 1               # specific camera
    python capture.py --exposure -6             # set initial exposure
    python capture.py --brightness 50           # set initial brightness

Controls:
    SPACE      - Save current frame
    e / d      - Increase / decrease exposure
    r / f      - Increase / decrease brightness
    q / ESC    - Quit
"""

import cv2
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Capture training images')
    parser.add_argument('--camera', default='0',
                        help='Camera index or device path (default: 0)')
    parser.add_argument('--out', default='captures',
                        help='Output directory (default: captures)')
    parser.add_argument('--exposure', type=float, default=None,
                        help='Initial exposure value (negative = darker, e.g. -6)')
    parser.add_argument('--brightness', type=float, default=None,
                        help='Initial brightness (0-255)')
    args = parser.parse_args()

    try:
        cam_source = int(args.camera)
    except ValueError:
        cam_source = args.camera

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob('*.jpeg'))
    count = len(existing)

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        print(f"Error: could not open camera {cam_source}")
        sys.exit(1)

    # Disable auto exposure so manual control works
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode on many drivers

    if args.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
    if args.brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, args.brightness)

    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)

    print(f"Camera opened. Saving to: {out_dir}/")
    print(f"  Exposure: {exposure}  |  Brightness: {brightness}")
    print(f"  SPACE=capture  e/d=exposure  r/f=brightness  q=quit")
    print(f"  {count} existing images in {out_dir}/")
    print("-" * 40)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            display = frame.copy()
            info = f"Exp:{exposure:.0f}  Brt:{brightness:.0f}  Saved:{count}"
            cv2.putText(display, info, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, "SPACE=save e/d=exp r/f=brt q=quit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow('Capture', display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                count += 1
                filename = f"img_{count:04d}.jpeg"
                cv2.imwrite(str(out_dir / filename), frame)
                print(f"  [{count}] Saved: {filename}")

            elif key == ord('e'):
                exposure += 1
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"  Exposure: {exposure}")

            elif key == ord('d'):
                exposure -= 1
                cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
                print(f"  Exposure: {exposure}")

            elif key == ord('r'):
                brightness = min(255, brightness + 10)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                print(f"  Brightness: {brightness}")

            elif key == ord('f'):
                brightness = max(0, brightness - 10)
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
                print(f"  Brightness: {brightness}")

            elif key in (ord('q'), 27):
                break

    except KeyboardInterrupt:
        print()
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Done. {count} total images in {out_dir}/")


if __name__ == '__main__':
    main()
