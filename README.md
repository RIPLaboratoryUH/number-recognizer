# Gauge Display Number Recognizer

Reads thickness values from an ultrasonic gauge LCD display using a camera feed.
Uses contour detection for digit segmentation and a small CNN (TFLite, 475 KB) for classification.

See the number-recognizer repo for more information on how this was made.

The `display_reader` and `display_reader_msgs` packages wrap the live feed into a ROS2 node.

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
You may need to install some opencv/numpy/tflite packages.

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


