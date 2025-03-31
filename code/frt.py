import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent))

from functions.blink_detection import (
    check_double_blink,
    check_eye_movement_during_blink,
)

from functions.deepfake_detection import (
    check_eye_movement,
    check_ear_consistency,
    analyze_reflections,
    FACE_CENTER_LANDMARK
)

from functions.face_validation import FaceValidator
from functions.interactive_test import InteractiveBlinkTest
from functions.photo_attack_detection import PhotoAttackDetector

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time


def main(demo_mode=False, interactive_mode=False):
    global blink_count, blink_active, baseline_ear, frame_count, ear_history, cooldown_counter, blink_timestamps

    BASELINE_FRAMES = 30
    SMOOTHING_WINDOW = 2
    BLINK_COOLDOWN = 1

    blink_count = 0
    blink_active = False
    baseline_ear = None
    frame_count = 0
    ear_history = []
    cooldown_counter = 0
    blink_timestamps = []
    face_validator = FaceValidator()
    photo_attack_detector = PhotoAttackDetector()

    if interactive_mode:
        blink_test = InteractiveBlinkTest()
        current_test = 0
        blink_test.start_test(current_test)
        test_start_time = time.time()
        last_blink_count = 0


    def calculate_ear(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    # I won't be using fixed ear_threshold, because otherwise it would not be robust to individual differences in
    # eye shape and size. Instead, I'll be calculating dynamic threshold.
    def calculate_baseline_ear(avg_ear):
        global baseline_ear, frame_count
        if frame_count < BASELINE_FRAMES:
            if baseline_ear is None:
                baseline_ear = avg_ear
                print(f"[BASELINE] Initial baseline EAR set to {baseline_ear:.4f}")
            else:
                prev = baseline_ear
                baseline_ear = (baseline_ear * frame_count + avg_ear) / (frame_count + 1)
                print(f"[BASELINE] Updated baseline EAR {prev:.4f} → {baseline_ear:.4f}")
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
    shape_predictor = dlib.shape_predictor("../dat/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(0)

    print(f"[CONFIG] Baseline frames: {BASELINE_FRAMES}, Blink cooldown: {BLINK_COOLDOWN}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        left_ear = right_ear = avg_ear = 0.0
        dynamic_threshold = None
        face_detected = False

        for face in faces:
            face_detected = True
            landmarks = shape_predictor(gray_frame, face)


            validation_results = face_validator.validate_face_position(face, landmarks)
            if validation_results['face_too_far']:
                cv2.putText(frame, "Move Closer!", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("[VALIDATION] Face too far from camera")
            elif validation_results['face_too_close']:
                cv2.putText(frame, "Move Back Slightly!", (10, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("[VALIDATION] Face too close to camera")
            if validation_results['head_tilted']:
                cv2.putText(frame, f"Straighten Your Head! ({validation_results['tilt_angle']:.1f} degrees)",
                           (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"[VALIDATION] Head tilted ({validation_results['tilt_angle']:.1f}°)")

            face_center = np.array([
                landmarks.part(FACE_CENTER_LANDMARK).x,
                landmarks.part(FACE_CENTER_LANDMARK).y
            ])

            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = smooth_ear((left_ear + right_ear) / 2.0)
            print(f"[METRICS] Frame {frame_count} - Left EAR: {left_ear:.4f}, Right EAR: {right_ear:.4f}, Avg: {avg_ear:.4f}")

            left_eye_movement = check_eye_movement_during_blink(left_eye, face_center, blink_active, check_eye_movement)
            right_eye_movement = check_eye_movement_during_blink(right_eye, face_center, blink_active, check_eye_movement)

            if frame_count < BASELINE_FRAMES:
                baseline_ear = calculate_baseline_ear(avg_ear)
            else:
                dynamic_threshold = baseline_ear * 0.75
                if avg_ear < dynamic_threshold:
                    if not blink_active and cooldown_counter == 0:
                        blink_active = True
                        photo_attack_detector.blink_start_time = time.time()
                        blink_timestamps.append(time.time())
                        print(f"[BLINK] Blink started at {time.strftime('%H:%M:%S')}")
                elif avg_ear >= dynamic_threshold and blink_active:
                    blink_active = False
                    blink_count += 1
                    cooldown_counter = BLINK_COOLDOWN
                    duration = time.time() - blink_timestamps[-1]
                    print(f"[BLINK] Blink completed (duration: {duration:.2f}s, total: {blink_count})")

                if check_double_blink(blink_timestamps):
                    print(f"[BLINK] Double blink detected!")
                    blink_timestamps.clear()
                    blink_active = False
                    cooldown_counter = BLINK_COOLDOWN

            left_eye_np = np.array(left_eye)
            right_eye_np = np.array(right_eye)
            padding = 10

            try:
                left_eye_region = frame[
                    max(0, int(min(left_eye_np[:, 1])) - padding):min(frame.shape[0], int(max(left_eye_np[:, 1])) + padding),
                    max(0, int(min(left_eye_np[:, 0])) - padding):min(frame.shape[1], int(max(left_eye_np[:, 0])) + padding)
                ]
                right_eye_region = frame[
                    max(0, int(min(right_eye_np[:, 1])) - padding):min(frame.shape[0], int(max(right_eye_np[:, 1])) + padding),
                    max(0, int(min(right_eye_np[:, 0])) - padding):min(frame.shape[1], int(max(right_eye_np[:, 0])) + padding)
                ]
            except Exception as e:
                print(f"[WARNING] Eye region extraction failed: {str(e)}")
                continue

            frame_data = {
                'blink_timestamps': blink_timestamps,
                'is_blinking': blink_active,
                'eye_landmarks': [(p[0], p[1]) for p in left_eye + right_eye],
                'eye_region': left_eye_region,
                'avg_ear': avg_ear,
                'left_ear': left_ear,
                'right_ear': right_ear,
                'frame': frame
            }

            attack_result = photo_attack_detector.check_photo_attack(frame_data)

            if attack_result['is_photo']:
                cv2.putText(frame, "PHOTO ATTACK DETECTED!", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for i, reason in enumerate(attack_result['reasons']):
                    cv2.putText(frame, reason, (10, 420 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                print(f"[ATTACK DETECTED] Reasons: {', '.join(attack_result['reasons'])}")


            if left_eye_movement or right_eye_movement:
                cv2.putText(frame, "Unnatural Eye Movement!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"[DETECTION] Unnatural eye movement detected (L: {left_eye_movement}, R: {right_eye_movement})")
            ear_consistency = check_ear_consistency(avg_ear)
            if ear_consistency:
                cv2.putText(frame, "Abrupt EAR Change Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            2)
                print(f"[DETECTION] Abrupt EAR change during blink")

            if interactive_mode and blink_test.all_tests_passed():
                cv2.putText(frame, "ALL TESTS PASSED! Press 'q' to exit", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Blink Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            cv2.imshow("Blink Detection", frame)

            if blink_active and ear_consistency:
                cv2.putText(frame, "Abrupt EAR Change Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f"[DETECTION] Abrupt EAR change during blink")

            reflections = analyze_reflections(left_eye_region) or analyze_reflections(right_eye_region)
            if reflections:
                cv2.putText(frame, "No Reflections Detected!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("[DETECTION] No reflections detected in eyes")

            for point in left_eye + right_eye:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

                if interactive_mode:
                    if current_test < len(blink_test.test_sequence):
                        if blink_count > last_blink_count:
                            result = blink_test.update_blink(time.time())
                            if result and result[0]:
                                blink_test.test_results[current_test] = True
                                current_test += 1
                                if current_test < len(blink_test.test_sequence):
                                    blink_test.start_test(current_test)
                                    test_start_time = time.time()

                        result = blink_test.update_eyes_open(avg_ear, baseline_ear)
                        if result and result[0]:
                            blink_test.test_results[current_test] = True
                            current_test += 1
                            if current_test < len(blink_test.test_sequence):
                                blink_test.start_test(current_test)
                                test_start_time = time.time()

                    last_blink_count = blink_count

                    blink_test.get_visual_feedback(frame, baseline_ear, avg_ear)
                    if blink_test.all_tests_passed():
                        cv2.putText(frame, "ALL TESTS PASSED! Press 'q' to exit", (200, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Blink Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        continue

        if cooldown_counter > 0:
            cooldown_counter -= 1

        if face_detected:
            cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if frame_count < BASELINE_FRAMES:
                cv2.putText(frame, f"Calculating baseline... {frame_count}/{BASELINE_FRAMES}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Keep your eyes open and do not blink!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Baseline EAR: {baseline_ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                if dynamic_threshold is not None:
                    cv2.putText(frame, f"Dynamic Threshold: {dynamic_threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SYSTEM] Shutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()