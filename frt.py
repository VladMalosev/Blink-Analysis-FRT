from deepfake_detection import check_eye_movement, check_ear_consistency, analyze_reflections
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time


BASELINE_FRAMES = 30
SMOOTHING_WINDOW = 2
BLINK_COOLDOWN = 1
DOUBLE_BLINK_THRESHOLD = 0.8

blink_count = 0
blink_active = False
baseline_ear = None
frame_count = 0
ear_history = []
cooldown_counter = 0
blink_timestamps = []


def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# I won't be using fixed ear_threshold, because otherwise it would not be robust to individual differences in
# eye shape and size. Instead, I'll be calculating dynamic threshold.
def calculate_baseline_ear(avg_ear):
    global baseline_ear, frame_count
    if frame_count < BASELINE_FRAMES:
        if baseline_ear is None:
            baseline_ear = avg_ear
        else:
            baseline_ear = (baseline_ear * frame_count + avg_ear) / (frame_count + 1)
        frame_count += 1
    return baseline_ear


# smoothing is necessary because EAR calculation can be noisy due to
# small variations in facial landmarks.
def smooth_ear(ear):
    ear_history.append(ear)
    if len(ear_history) > SMOOTHING_WINDOW:
        ear_history.pop(0)
    return np.mean(ear_history)


detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    dynamic_threshold = None

    for face in faces:
        landmarks = shape_predictor(gray_frame, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        avg_ear = smooth_ear(avg_ear)

        if frame_count < BASELINE_FRAMES:
            baseline_ear = calculate_baseline_ear(avg_ear)
            cv2.putText(frame, f"Calculating baseline... {frame_count}/{BASELINE_FRAMES}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, "Keep your eyes open and do not blink!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            dynamic_threshold = baseline_ear * 0.75
            if avg_ear < dynamic_threshold:
                if not blink_active and cooldown_counter == 0:
                    blink_active = True
                    blink_timestamps.append(time.time())
                    print(f"Blink detected at: {blink_timestamps[-1]}")
            else:
                if blink_active:
                    blink_count += 1
                    blink_active = False
                    cooldown_counter = BLINK_COOLDOWN

            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Check for double blink
            if len(blink_timestamps) >= 2:
                time_diff = blink_timestamps[-1] - blink_timestamps[-2]
                print(f"Blink timestamps: {blink_timestamps}")
                print(f"Time difference between blinks: {time_diff:.2f} seconds")
                if 0.3 <= time_diff <= DOUBLE_BLINK_THRESHOLD:
                    cv2.putText(frame, "Double Blink Detected!", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    print("Double blink detected!")
                    blink_timestamps.clear()

        left_eye_np = np.array(left_eye)
        right_eye_np = np.array(right_eye)

        padding = 10
        left_eye_region = frame[
            max(0, min(left_eye_np[:, 1]) - padding):min(frame.shape[0], max(left_eye_np[:, 1]) + padding),
            max(0, min(left_eye_np[:, 0]) - padding):min(frame.shape[1], max(left_eye_np[:, 0]) + padding)
        ]
        right_eye_region = frame[
            max(0, min(right_eye_np[:, 1]) - padding):min(frame.shape[0], max(right_eye_np[:, 1]) + padding),
            max(0, min(right_eye_np[:, 0]) - padding):min(frame.shape[1], max(right_eye_np[:, 0]) + padding)
        ]

        if left_eye_region.size == 0 or right_eye_region.size == 0:
            continue

        if check_eye_movement(left_eye) or check_eye_movement(right_eye):
            cv2.putText(frame, "Unnatural Eye Movement Detected!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if blink_active and check_ear_consistency(avg_ear):
            cv2.putText(frame, "Abrupt EAR Change Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if analyze_reflections(left_eye_region) or analyze_reflections(right_eye_region):
            cv2.putText(frame, "No Reflections Detected!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if frame_count >= BASELINE_FRAMES:
            cv2.putText(frame, f"Baseline EAR: {baseline_ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
            if dynamic_threshold is not None:
                cv2.putText(frame, f"Dynamic Threshold: {dynamic_threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    cv2.imshow("Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()