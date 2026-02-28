# Gauge Display Number Recognizer

Reads thickness values from an ultrasonic gauge LCD display using a camera feed.
Uses contour detection for digit segmentation and a small CNN (TFLite, 475 KB) for classification.

## Quick Start (Runtime)

```bash
pip install opencv-python-headless numpy
# Optional but recommended for RPi:
pip install tflite-runtime
```

### Live Camera Feed
```bash
python live_feed.py --show           # with preview window
python live_feed.py                  # headless (prints to stdout)
python live_feed.py --rate 1         # 1 sample/sec instead of 3
python live_feed.py --camera 1       # different camera
```

### Process Saved Images
```bash
python read_display.py image.jpeg
python read_display.py images_dir/
python read_display.py images_dir/ --verify labels.json
```

### Camera Orientation
The camera is assumed to be mounted upside-down (180° rotation, the default).
Override with `--rotate`:
```bash
python live_feed.py --rotate 0       # no rotation
python live_feed.py --rotate 90      # 90° clockwise
```

---

## Retraining the Model

If the camera setup changes or accuracy degrades, you can recollect data and retrain.

**Additional dependency for training:**
```bash
pip install tensorflow
```

### Step 1: Capture Images
```bash
python capture.py
```
- **SPACE** — save frame
- **e/d** — increase/decrease exposure
- **r/f** — increase/decrease brightness
- **q** — quit

### Step 2: Label Images
```bash
python label.py
```
Type the display reading for each image (e.g. `10.41`), `dash` for `----`, or `skip`.

### Step 3: Extract Digit Crops
```bash
python extract_digits.py
```
Extracts individual 28×28 digit crops from labeled images into `data/digits/{0-9}/`.

### Step 4: Train the CNN
```bash
python train_digit_cnn.py
```
Trains a small CNN with data augmentation. Saves `models/digit_cnn.keras` and `models/digit_cnn.tflite`.

### Step 5: Verify
```bash
python read_display.py captures/ --verify labels_new.json
```

---

## File Overview

| File | Purpose |
|------|---------|
| `live_feed.py` | Live camera reading (main runtime script) |
| `read_display.py` | Batch image processing + verification |
| `display_utils.py` | Core detection: thresholding, contours, decimal finding |
| `capture.py` | Capture training images from camera |
| `label.py` | Label captured images interactively |
| `extract_digits.py` | Extract 28×28 digit crops from labeled images |
| `train_digit_cnn.py` | Train the digit classifier CNN |
| `models/digit_cnn.tflite` | Trained model (475 KB) |
| `models/digit_cnn.keras` | Full Keras model (for retraining) |
| `labels_new.json` | Current training labels |

## ROS2 Package

The `display_reader` and `display_reader_msgs` packages wrap the live feed into a ROS2 node.

### Build
```bash
cd ~/GIT/number-recognizer
colcon build
source install/setup.bash
```

### Run
```bash
ros2 run display_reader display_reader_node
ros2 run display_reader display_reader_node --ros-args -p exposure:=10.0
ros2 launch display_reader display_reader.launch.py exposure:=10.0
```

### Topic

| Topic | Type | Description |
|-------|------|-------------|
| `/display_reading` | `display_reader_msgs/DisplayReading` | Timestamped display value |

### Custom Message (`DisplayReading.msg`)
```
std_msgs/Header header    # ROS2 timestamp for sensor fusion / odometry sync
float64 data              # Display reading (NaN when dashes shown)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `camera` | `0` | Camera index or device path |
| `rate` | `3.0` | Samples per second |
| `rotation` | `180` | Image rotation degrees |
| `exposure` | `-1.0` | Manual exposure (`-1` = auto) |
| `model_path` | *(installed model)* | Path to TFLite model |

### Package Structure

| Package | Build Type | Purpose |
|---------|-----------|---------|
| `display_reader_msgs` | ament_cmake | Custom `DisplayReading.msg` definition |
| `display_reader` | ament_python | Camera node + digit classification |

---

## RPi5 Deployment

Only these files are needed on the Pi:

```
live_feed.py
display_utils.py
models/digit_cnn.tflite
```

Install:
```bash
pip install opencv-python-headless numpy tflite-runtime
```

Estimated performance on RPi5: **~13ms per frame** (~4% CPU at 3 Hz).

## Pipeline Details

1. **Rotation** — rotates frame if camera is upside-down
2. **Thresholding** — binary threshold at 160 to isolate bright LCD segments
3. **Contour detection** — finds digit-shaped contours by height (15-45% of image)
4. **Clustering** — groups contours at similar y-position as the digit row
5. **Decimal detection** — finds small contours in gaps between digits
6. **CNN classification** — each digit crop is resized to 28×28 and classified (0-9)
