import numpy as np
import time

DOUBLE_BLINK_THRESHOLD = 0.8


def calculate_dynamic_blink_threshold(blink_timestamps):
    if len(blink_timestamps) < 2:
        return DOUBLE_BLINK_THRESHOLD

    time_diffs = np.diff(blink_timestamps)
    avg_blink_interval = np.mean(time_diffs)
    dynamic_threshold = avg_blink_interval * 1.5
    return max(DOUBLE_BLINK_THRESHOLD, dynamic_threshold)


def check_double_blink(blink_timestamps):
    if len(blink_timestamps) < 2:
        return False

    time_diff = blink_timestamps[-1] - blink_timestamps[-2]
    dynamic_threshold = calculate_dynamic_blink_threshold(blink_timestamps)

    if 0.3 <= time_diff <= dynamic_threshold:
        print(f"[BLINK] Double blink detected! Interval: {time_diff:.2f}s")
        return True
    return False


def check_eye_movement_during_blink(eye_landmarks, face_center, is_blinking=False, check_eye_movement=None):
    if is_blinking:
        return False  # skip during the blink

    if check_eye_movement is None:
        raise ValueError("check_eye_movement function must be provided")

    return check_eye_movement(eye_landmarks, face_center, is_blinking)