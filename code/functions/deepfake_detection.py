import numpy as np
import cv2

EYE_MOVEMENT_THRESHOLD = 5
EAR_CHANGE_THRESHOLD = 0.25
NORMALIZED_VARIANCE_THRESHOLD = 0.25
EAR_WINDOW = 10
MIN_EYE_SIZE = 100
VAR_THRESHOLD = 0.003
EYE_MOVEMENT_WINDOW = 30
FACE_CENTER_LANDMARK = 30  # nose landmark tip
MICROSACCADE_MIN = 0.001
MICROSACCADE_MAX = 0.003

ear_history_blink = []
eye_position_history = []
face_center_history = []
BASELINE_EYE_SIZE = None


def calculate_eye_size(eye_landmarks_np):
    x = eye_landmarks_np[:, 0]
    y = eye_landmarks_np[:, 1]
    return (np.max(x) - np.min(x)) * (np.max(y) - np.min(y))


def check_eye_movement(eye_landmarks, face_center, is_blinking=False):
    global eye_position_history, face_center_history

    debug_info = {
        'movement_detected': False,
        'microsaccade_detected': False,
        'reason': [],
        'vars': {}
    }

    if is_blinking:
        debug_info['reason'].append("Blink in progress - check skipped")
        print(f"[EYE MOVEMENT] {debug_info}")
        return debug_info

    eye_landmarks_np = np.array(eye_landmarks)
    if len(eye_landmarks_np) < 6:
        debug_info['reason'].append("Insufficient eye landmarks (need 6 points)")
        print(f"[EYE MOVEMENT] {debug_info}")
        return debug_info

    # eye center loc relative to face center
    valid_points = eye_landmarks_np[(eye_landmarks_np[:, 0] > 0) & (eye_landmarks_np[:, 1] > 0)]
    if len(valid_points) < 4:
        debug_info['reason'].append("Not enough valid eye points (need 4+)")
        print(f"[EYE MOVEMENT] {debug_info}")
        return debug_info

    absolute_eye_center = np.mean(valid_points, axis=0)
    relative_eye_center = absolute_eye_center - face_center

    eye_position_history.append(relative_eye_center)
    face_center_history.append(face_center)

    if len(eye_position_history) > EYE_MOVEMENT_WINDOW:
        eye_position_history.pop(0)
        face_center_history.pop(0)

    if len(eye_position_history) >= 10:
        movements = []
        for i in range(1, len(eye_position_history)):
            dx = eye_position_history[i][0] - eye_position_history[i - 1][0]
            dy = eye_position_history[i][1] - eye_position_history[i - 1][1]
            movements.append(np.sqrt(dx ** 2 + dy ** 2))

        if len(movements) == 0:
            debug_info['reason'].append("No movement data available")
            print(f"[EYE MOVEMENT] {debug_info}")
            return debug_info

        movement_variance = np.var(movements)
        current_eye_size = calculate_eye_size(eye_landmarks_np)

        # dynamic normalization based on eye size
        normalized_variance = movement_variance / current_eye_size if current_eye_size > 0 else 0

        debug_info['vars'] = {
            'movement_intensity': f"{movement_variance:.4f} pixels²",
            'eye_size': f"{current_eye_size} pixels²",
            'normalized_intensity': f"{normalized_variance:.2%}",
            'threshold': f"{VAR_THRESHOLD:.2%}"
        }

        if MICROSACCADE_MIN <= normalized_variance <= MICROSACCADE_MAX:
            if normalized_variance < MICROSACCADE_MIN:
                debug_info['reason'].append("No significant movement detected")
            elif MICROSACCADE_MIN <= normalized_variance <= MICROSACCADE_MAX:
                debug_info['microsaccade_detected'] = True
                debug_info['reason'].append("Normal microsaccade detected")
            elif MICROSACCADE_MAX < normalized_variance <= VAR_THRESHOLD:
                debug_info['photo_attack_suspected'] = True  # New flag
                debug_info['reason'].append("Suspicious movement pattern detected")
            elif normalized_variance > VAR_THRESHOLD:
                debug_info['movement_detected'] = True
                debug_info['reason'].append("Excessive eye movement detected")

        print(f"[EYE MOVEMENT] {'Microsaccade' if debug_info['microsaccade_detected'] else 'Movement'} | "
              f"Type: {'NORMAL' if debug_info['microsaccade_detected'] else 'UNNATURAL'} | "
              f"Intensity: {normalized_variance:.2%} (Threshold: {VAR_THRESHOLD:.2%}) | "
              f"Details: {', '.join(debug_info['reason']) if debug_info['reason'] else 'Normal movement'}")

        return debug_info

    print(f"[EYE MOVEMENT] Insufficient data (need {EYE_MOVEMENT_WINDOW} frames)")
    return debug_info


def check_ear_consistency(ear):
    global ear_history_blink
    ear_history_blink.append(ear)
    debug_info = {
        'abrupt_change': False,
        'vars': {}
    }

    if len(ear_history_blink) > EAR_WINDOW:
        ear_history_blink.pop(0)

    if len(ear_history_blink) >= 5:
        weights = np.array([0.3, 0.5, 0.7, 0.9])
        diffs = np.abs(np.diff(ear_history_blink[-5:]))
        weighted_diffs = diffs * weights
        max_diff = np.max(weighted_diffs)

        debug_info['vars'] = {
            'recent_changes': [f"{wd:.4f}" for wd in weighted_diffs],
            'max_change': f"{max_diff:.4f}",
            'threshold': f"{EAR_CHANGE_THRESHOLD:.2}"
        }

        if max_diff > EAR_CHANGE_THRESHOLD:
            debug_info['abrupt_change'] = True
            debug_info['reason'] = f"Sudden EAR change detected ({max_diff:.4f} > {EAR_CHANGE_THRESHOLD:.2})"

        status = "ABNORMAL" if debug_info['abrupt_change'] else "Normal"
        print(f"[EAR CONSISTENCY] {status} | "
              f"Max change: {max_diff:.4f} (Threshold: {EAR_CHANGE_THRESHOLD:.2}) | "
              f"Recent changes: {', '.join([f'{x:.4f}' for x in weighted_diffs])}")
        return debug_info['abrupt_change']

    print(f"[EAR CONSISTENCY] Need more data (only {len(ear_history_blink)} frames)")
    return False


def analyze_reflections(eye_region):
    debug_info = {
        'reflections_found': True,
        'count': 0
    }

    if eye_region.size == 0:
        debug_info['reason'] = "Empty eye region"
        print(f"[REFLECTIONS] {debug_info}")
        return False

    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, reflection_mask = cv2.threshold(gray_eye, 100, 255, cv2.THRESH_BINARY)
    reflection_count = np.sum(reflection_mask > 0)

    debug_info['count'] = reflection_count
    debug_info['reflections_found'] = reflection_count >= 2

    status = "OK" if debug_info['reflections_found'] else "WARNING"
    reflection_text = (f"{reflection_count} reflections (needs 2+) - {status} | "
                       f"{debug_info.get('reason', 'Normal reflections')}")
    print(f"[REFLECTIONS] {reflection_text}")
    return not debug_info['reflections_found']