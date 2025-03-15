import numpy as np
import cv2
from scipy.spatial import ConvexHull

eye_position_history = []
EYE_MOVEMENT_THRESHOLD = 5
ear_history_blink = []
EAR_CHANGE_THRESHOLD = 0.15


def calculate_eye_size(eye_landmarks_np):
    hull = ConvexHull(eye_landmarks_np)
    return hull.volume

def check_eye_movement(eye_landmarks):
    global eye_position_history
    eye_landmarks_np = np.array(eye_landmarks)
    eye_center = np.mean(eye_landmarks_np, axis=0)
    eye_position_history.append(eye_center)

    if len(eye_position_history) > 5:
        eye_position_history.pop(0)

    if len(eye_position_history) >= 10:
        smoothed_eye_centers = np.mean(eye_position_history, axis=0)
        variance = np.var(eye_position_history, axis=0)

        eye_size = calculate_eye_size(eye_landmarks_np)

        normalized_variance = np.max(variance) / eye_size
        print(f"Normalized Variance: {normalized_variance:.4f}, Eye Size: {eye_size}")
        print(f"Smoothed Eye Centers: {smoothed_eye_centers}, Raw Variance: {variance}")

        if normalized_variance > 0.15:
            return True
    return False

def check_ear_consistency(ear):
    global ear_history_blink
    ear_history_blink.append(ear)

    if len(ear_history_blink) > 5:
        ear_history_blink.pop(0)

    if len(ear_history_blink) >= 5:
        differences = np.abs(np.diff(ear_history_blink))
        if np.max(differences) > EAR_CHANGE_THRESHOLD:
            return True
    return False

def analyze_reflections(eye_region):
    if eye_region.size == 0:
        return False

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, reflection_mask = cv2.threshold(gray_eye, 100, 255, cv2.THRESH_BINARY)
    reflection_count = np.sum(reflection_mask > 0)
    cv2.imshow("Reflection Mask", reflection_mask)

    if reflection_count < 2:
        return True
    return False