"""
Core display digit detection and classification utilities.
Shared between extract_digits.py, train_digit_cnn.py, and read_display.py.
"""

import cv2
import numpy as np

# Global rotation setting (degrees: 0, 90, 180, 270)
_rotation = 0


def set_rotation(degrees):
    """Set global image rotation (0, 90, 180, 270)."""
    global _rotation
    _rotation = int(degrees) % 360


def apply_rotation(img):
    """Apply the configured rotation to an image."""
    if _rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif _rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif _rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def get_binary(gray, threshold=120):
    """Threshold a grayscale image to binary."""
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def split_merged_contour(binary, x, y, w, h, min_digit_h):
    """Split a merged contour into individual digits using vertical projection."""
    if w < h * 0.8:
        return [{'x': x, 'y': y, 'w': w, 'h': h}]
    
    roi = binary[y:y+h, x:x+w]
    col_sums = np.sum(roi, axis=0) / 255.0
    threshold = h * 0.05
    is_digit = col_sums > threshold
    
    digits = []
    in_digit = False
    start = 0
    
    for i, val in enumerate(is_digit):
        if val and not in_digit:
            start = i
            in_digit = True
        elif not val and in_digit:
            digit_w = i - start
            if digit_w > 3:
                digits.append({'x': x + start, 'y': y, 'w': digit_w, 'h': h})
            in_digit = False
    
    if in_digit:
        digit_w = len(is_digit) - start
        if digit_w > 3:
            digits.append({'x': x + start, 'y': y, 'w': digit_w, 'h': h})
    
    if len(digits) <= 1:
        return [{'x': x, 'y': y, 'w': w, 'h': h}]
    
    refined = []
    for d in digits:
        sub_roi = binary[d['y']:d['y']+d['h'], d['x']:d['x']+d['w']]
        row_sums = np.sum(sub_roi, axis=1) / 255.0
        rows_active = np.where(row_sums > 0)[0]
        if len(rows_active) > 0:
            new_y = d['y'] + rows_active[0]
            new_h = rows_active[-1] - rows_active[0] + 1
            if new_h > min_digit_h:
                refined.append({'x': d['x'], 'y': new_y, 'w': d['w'], 'h': new_h})
    
    return refined if refined else [{'x': x, 'y': y, 'w': w, 'h': h}]


def find_main_digits(gray):
    """
    Find the main display digit contours.
    Uses resolution-independent thresholds.
    Returns list of digit bounding boxes and the binary image.
    """
    gray = apply_rotation(gray)
    h, w = gray.shape
    binary = get_binary(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_margin = max(3, int(w * 0.005))
    min_digit_h = h * 0.15
    max_digit_h = h * 0.45
    min_area = int(h * w * 0.001)
    
    all_candidates = []
    for cnt in contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if (cx <= edge_margin or cy <= edge_margin or 
            (cx + cw) >= (w - edge_margin) or (cy + ch) >= (h - edge_margin)):
            continue
        if ch < min_digit_h or ch > max_digit_h:
            continue
        aspect = cw / max(ch, 1)
        if aspect > 3.0:
            continue
        
        if aspect > 0.8 and area > min_area * 5:
            split = split_merged_contour(binary, cx, cy, cw, ch, min_digit_h)
            for s in split:
                s['area'] = int(area / len(split))
                all_candidates.append(s)
        else:
            all_candidates.append({
                'x': int(cx), 'y': int(cy), 'w': int(cw), 'h': int(ch),
                'area': int(area)
            })
    
    all_candidates = [c for c in all_candidates if c['area'] > min_area]
    
    if not all_candidates:
        return [], binary
    
    all_candidates.sort(key=lambda c: c['y'])
    best_group = []
    for i, ref in enumerate(all_candidates):
        group = [ref]
        for j, other in enumerate(all_candidates):
            if i != j:
                y_diff = abs(other['y'] - ref['y'])
                h_ratio = other['h'] / max(ref['h'], 1)
                if y_diff < ref['h'] * 0.5 and 0.5 < h_ratio < 2.0:
                    group.append(other)
        if len(group) > len(best_group):
            best_group = group
    
    if not best_group:
        return [], binary
    
    best_group.sort(key=lambda c: c['x'])
    filtered = []
    for c in best_group:
        overlap = False
        for f in filtered:
            ox = max(0, min(c['x']+c['w'], f['x']+f['w']) - max(c['x'], f['x']))
            if ox > min(c['w'], f['w']) * 0.3:
                overlap = True
                if c.get('area', 0) > f.get('area', 0):
                    filtered.remove(f)
                    filtered.append(c)
                break
        if not overlap:
            filtered.append(c)
    
    filtered.sort(key=lambda c: c['x'])
    return filtered, binary


def find_decimal_position(gray, digits):
    """Find where the decimal goes. Returns index: decimal goes after digit[index]."""
    if not digits or len(digits) < 2:
        return 0
    
    gray = apply_rotation(gray)
    h, w = gray.shape
    binary = get_binary(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_bottom = max(d['y'] + d['h'] for d in digits)
    digit_top = min(d['y'] for d in digits)
    edge_margin = max(3, int(w * 0.005))
    
    # Build list of gaps between digits
    gaps = []
    for i in range(len(digits) - 1):
        gap_left = digits[i]['x'] + digits[i]['w']
        gap_right = digits[i + 1]['x']
        gaps.append((i, gap_left, gap_right))
    
    # Find the best decimal candidate for each gap
    best_gap = None
    best_area = 0
    
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        
        if (x <= edge_margin or y <= edge_margin or
            (x + bw) >= (w - edge_margin) or (y + bh) >= (h - edge_margin)):
            continue
        
        # Must be small, below digit midline
        if not (20 < area < 2000 and
                y > (digit_top + digit_bottom) / 2 and
                bh < h * 0.15):
            continue
        
        x_center = x + bw // 2
        
        # Check if this falls in any gap (with some tolerance)
        for gap_idx, gap_left, gap_right in gaps:
            tolerance = max(10, (gap_right - gap_left) * 0.3)
            if gap_left - tolerance <= x_center <= gap_right + tolerance:
                if area > best_area:
                    best_gap = gap_idx
                    best_area = area
                break
    
    return best_gap if best_gap is not None else 0


def is_dash_display(gray):
    """Detect dashes (no reading)."""
    gray = apply_rotation(gray)
    h, w = gray.shape
    binary = get_binary(gray)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    edge_margin = max(3, int(w * 0.005))
    mid_y_min, mid_y_max = h * 0.3, h * 0.7
    
    dash_count = 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (x <= edge_margin or y <= edge_margin or
            (x + bw) >= (w - edge_margin) or (y + bh) >= (h - edge_margin)):
            continue
        if area > 200:
            aspect = bw / max(bh, 1)
            if aspect > 2.5 and bh < h * 0.08 and mid_y_min < y < mid_y_max:
                dash_count += 1
    return dash_count >= 3


def crop_digit(binary, x, y, w, h, target_size=28):
    """Crop a digit and resize to 28x28."""
    pad = 5
    y1, y2 = max(0, y - pad), min(binary.shape[0], y + h + pad)
    x1, x2 = max(0, x - pad), min(binary.shape[1], x + w + pad)
    
    digit = binary[y1:y2, x1:x2]
    if digit.size == 0:
        return None
    
    dh, dw = digit.shape
    scale = min((target_size - 4) / max(dw, 1), (target_size - 4) / max(dh, 1))
    new_w, new_h = max(1, int(dw * scale)), max(1, int(dh * scale))
    
    resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    
    return canvas
