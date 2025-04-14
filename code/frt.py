import sys
from pathlib import Path
import random


sys.path.append(str(Path(__file__).parent.parent))

from functions.microsaccade_detection import MicrosaccadeDetector


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
    global blink_count, blink_active, baseline_ear, baseline_eye_height, frame_count, ear_history, cooldown_counter, blink_timestamps

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
    consecutive_low_ear_frames = 0
    face_validator = FaceValidator()
    photo_attack_detector = PhotoAttackDetector()

    CONSECUTIVE_FRAMES_THRESHOLD = 1
    EYE_CLOSURE_THRESHOLD = 0.8
    MIN_BLINK_DURATION = 0.05
    MAX_BLINK_DURATION = 0.4
    SYMMETRY_THRESHOLD = 0.2
    BLINK_COMPLETION_THRESHOLD = 0.80

    EYE_HEIGHT_CLOSURE_THRESHOLD = 0.70
    EYE_HEIGHT_OPEN_THRESHOLD = 0.70

    blink_start_time = 0
    min_ear_during_blink = 1.0
    blink_eye_positions = None



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
    def calculate_baseline_ear(avg_ear, left_eye, right_eye):
        global baseline_ear, baseline_eye_height, frame_count
        if frame_count < BASELINE_FRAMES:
            if baseline_ear is None:
                baseline_ear = avg_ear
                # Calculate initial eye height
                left_height = max(p[1] for p in left_eye) - min(p[1] for p in left_eye)
                right_height = max(p[1] for p in right_eye) - min(p[1] for p in right_eye)
                baseline_eye_height = (left_height + right_height) / 2
                print(
                    f"[BASELINE] Initial baseline EAR set to {baseline_ear:.4f}, eye height: {baseline_eye_height:.1f}px")
            else:
                prev_ear = baseline_ear
                baseline_ear = (baseline_ear * frame_count + avg_ear) / (frame_count + 1)

                # Update eye height baseline
                left_height = max(p[1] for p in left_eye) - min(p[1] for p in left_eye)
                right_height = max(p[1] for p in right_eye) - min(p[1] for p in right_eye)
                current_height = (left_height + right_height) / 2
                baseline_eye_height = (baseline_eye_height * frame_count + current_height) / (frame_count + 1)

                print(
                    f"[BASELINE] Updated baseline EAR {prev_ear:.4f} → {baseline_ear:.4f}, eye height: {baseline_eye_height:.1f}px")
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
    microsaccade_detector = MicrosaccadeDetector(sampling_rate=60)


    print(f"[CONFIG] Baseline frames: {BASELINE_FRAMES}, Blink cooldown: {BLINK_COOLDOWN}s")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        current_time = time.time()
        ear_consistency = False

        if interactive_mode:
            blink_test.update_test_state(current_time)

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

            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)

            # вetect microsaccades in both eyes
            left_detected, left_velocity, left_details = microsaccade_detector.detect_microsaccades(left_eye_center)
            right_detected, right_velocity, right_details = microsaccade_detector.detect_microsaccades(right_eye_center)

            natural_movement = microsaccade_detector.is_natural_movement()

            if not natural_movement:
                cv2.putText(frame, "Unnatural Eye Movement Pattern!", (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("[DETECTION] Unnatural eye movement pattern detected")


            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = smooth_ear((left_ear + right_ear) / 2.0)

            print(
                f"[METRICS] Frame {frame_count} - Left EAR: {left_ear:.4f}, Right EAR: {right_ear:.4f}, Avg: {avg_ear:.4f}")

            left_eye_movement = check_eye_movement_during_blink(left_eye, face_center, blink_active, check_eye_movement)
            right_eye_movement = check_eye_movement_during_blink(right_eye, face_center, blink_active,
                                                                 check_eye_movement)
            if frame_count < BASELINE_FRAMES:
                baseline_ear = calculate_baseline_ear(avg_ear, left_eye, right_eye)
            else:
                dynamic_threshold = baseline_ear * EYE_CLOSURE_THRESHOLD

                # calc eye height
                current_left_height = max(p[1] for p in left_eye) - min(p[1] for p in left_eye)
                current_right_height = max(p[1] for p in right_eye) - min(p[1] for p in right_eye)
                current_eye_height_avg = (current_left_height + current_right_height) / 2

                if not blink_active and cooldown_counter == 0:
                    if consecutive_low_ear_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                        # eye height check to start condition
                        if (left_ear < dynamic_threshold
                            and right_ear < dynamic_threshold
                            and abs(left_ear - right_ear) < SYMMETRY_THRESHOLD
                            and current_eye_height_avg < baseline_eye_height * EYE_HEIGHT_CLOSURE_THRESHOLD):
                            blink_active = True

                if avg_ear < dynamic_threshold:
                    consecutive_low_ear_frames += 1
                else:
                    consecutive_low_ear_frames = 0

                if not blink_active and cooldown_counter == 0:
                    if consecutive_low_ear_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                        if (left_ear < dynamic_threshold and right_ear < dynamic_threshold and
                                abs(left_ear - right_ear) < SYMMETRY_THRESHOLD):
                            blink_active = True
                            blink_start_time = time.time()
                            min_ear_during_blink = avg_ear
                            blink_eye_positions = (left_eye, right_eye)  # store initial eye positions
                            microsaccade_detector.register_blink(True, avg_ear)
                            print(f"[BLINK] Potential blink started (EAR: {avg_ear:.3f})")

                # track minimum EAR and check for completion, to prevent
                # adding up when eyes are squinted
                elif blink_active:
                    if avg_ear < min_ear_during_blink:
                        min_ear_during_blink = avg_ear

                    if avg_ear >= baseline_ear * BLINK_COMPLETION_THRESHOLD or (time.time() - blink_start_time) > MAX_BLINK_DURATION:
                        blink_duration = time.time() - blink_start_time

                        if (blink_duration >= MIN_BLINK_DURATION
                                and min_ear_during_blink <= baseline_ear * 0.7
                                and current_eye_height_avg >= baseline_eye_height * EYE_HEIGHT_OPEN_THRESHOLD):
                            microsaccade_detector.register_blink(False, avg_ear)  # Blink ended

                            # check if eye position changed much
                            if blink_eye_positions:
                                left_movement = check_eye_movement_during_blink(left_eye, face_center, blink_active,
                                                                                check_eye_movement)
                                right_movement = check_eye_movement_during_blink(right_eye, face_center, blink_active,
                                                                                 check_eye_movement)

                                if not left_movement and not right_movement:
                                    blink_count += 1
                                    blink_timestamps.append(time.time())
                                    print(
                                        f"[BLINK] Detected (duration: {blink_duration:.3f}s, min EAR: {min_ear_during_blink:.3f})")
                                else:
                                    print(f"[BLINK] Rejected due to eye movement during blink")
                            else:
                                blink_count += 1
                                blink_timestamps.append(time.time())
                                print(
                                    f"[BLINK] Detected (duration: {blink_duration:.3f}s, min EAR: {min_ear_during_blink:.3f})")

                        else:
                            print(
                                f"[BLINK] Rejected - Eyes didn't reopen fully (Height: {current_eye_height_avg:.1f} vs Baseline: {baseline_eye_height:.1f})")

                        blink_active = False
                        cooldown_counter = BLINK_COOLDOWN
                        consecutive_low_ear_frames = 0
                        min_ear_during_blink = 1.0
                        blink_eye_positions = None

                # check for double blink
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
                                  max(0, int(min(left_eye_np[:, 1])) - padding):min(frame.shape[0], int(max(
                                      left_eye_np[:, 1])) + padding),
                                  max(0, int(min(left_eye_np[:, 0])) - padding):min(frame.shape[1], int(max(
                                      left_eye_np[:, 0])) + padding)
                                  ]
                right_eye_region = frame[
                                   max(0, int(min(right_eye_np[:, 1])) - padding):min(frame.shape[0], int(max(
                                       right_eye_np[:, 1])) + padding),
                                   max(0, int(min(right_eye_np[:, 0])) - padding):min(frame.shape[1], int(max(
                                       right_eye_np[:, 0])) + padding)
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
                cv2.putText(frame, "Potential attack detected!", (10, 390),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                for i, reason in enumerate(attack_result['reasons']):
                    cv2.putText(frame, reason, (10, 420 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            elif time.time() - photo_attack_detector.last_blink_time > 15:
                cv2.putText(frame, "Please blink to verify liveness", (10, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if left_eye_movement or right_eye_movement:
                # Only show warning if movement_detected is True
                if (isinstance(left_eye_movement, dict) and left_eye_movement.get('movement_detected')) or \
                        (isinstance(right_eye_movement, dict) and right_eye_movement.get('movement_detected')):
                    cv2.putText(frame, "Unnatural Eye Movement!", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                                2)
                    print(
                        f"[DETECTION] Unnatural eye movement detected (L: {left_eye_movement}, R: {right_eye_movement})")

                    ear_consistency = check_ear_consistency(avg_ear)
            if ear_consistency:
                cv2.putText(frame, "Abrupt EAR Change Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            2)
                print(f"[DETECTION] Abrupt EAR change during blink")
            if interactive_mode and blink_test.any_test_failed():
                cv2.putText(frame, "TEST FAILED - Press any key to continue",
                            (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Blink Detection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[SYSTEM] Shutting down...")
                    break
                elif key != 255:
                    if interactive_mode and blink_test.any_test_failed():
                        current_test += 1
                        if current_test < len(blink_test.test_sequence):
                            blink_test.start_test(current_test)
                            test_start_time = time.time()

            if interactive_mode and (blink_test.all_tests_passed() or blink_test.any_test_failed()):
                results = blink_test.get_final_results()
                print("\nTest Results:")
                for i, result in enumerate(results['results']):
                    status = "PASSED" if result['passed'] else f"FAILED: {result['failure_message']}"
                    print(f"Test {i + 1}: {result['description']} - {status}")

            if interactive_mode and blink_test.all_tests_passed():
                cv2.putText(frame, "ALL TESTS PASSED! Press 'q' to exit", (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Blink Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            cv2.imshow("Blink Detection", frame)

            if blink_active and ear_consistency:
                cv2.putText(frame, "Abrupt EAR Change Detected!", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                            2)
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

                        result = blink_test.update_eyes_open(avg_ear, baseline_ear, time.time())
                        if result and result[0]:
                            blink_test.test_results[current_test] = True
                            current_test += 1
                            if current_test < len(blink_test.test_sequence):
                                blink_test.start_test(current_test)
                                test_start_time = time.time()

                    last_blink_count = blink_count

                    blink_test.get_visual_feedback(frame, baseline_ear, avg_ear, time.time())
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

            # Add these near your other status text displays
            if left_detected:
                cv2.putText(frame, f"L Microsaccade: {left_velocity:.1f} deg/s", (10, 450),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            if right_detected:
                cv2.putText(frame, f"R Microsaccade: {right_velocity:.1f} deg/s", (10, 470),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            saccade_rate = microsaccade_detector.get_microsaccade_rate()
            cv2.putText(frame, f"Microsaccade Rate: {saccade_rate:.1f}/s", (10, 490),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            if microsaccade_detector.calibration_complete:
                status = "NATURAL" if natural_movement else "WARNING"
                color = (0, 255, 0) if natural_movement else (0, 0, 255)
                cv2.putText(frame, f"Microsaccade Status: {status}", (350, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, "Calibrating Microsaccade Detector...", (350, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if microsaccade_detector.calibration_complete:
                photo_attack_detected = microsaccade_detector.detect_photo_attack()
                if photo_attack_detected:
                    cv2.putText(frame, "PHOTO ATTACK SUSPECTED!", (350, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    natural_movement = False


            if frame_count >= BASELINE_FRAMES:
                # Visualize blink detection state
                blink_status = "BLINKING" if blink_active else "READY"
                cv2.putText(frame, f"Status: {blink_status}", (450, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"EAR Ratio: {avg_ear / baseline_ear:.2f}", (450, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if frame_count < BASELINE_FRAMES:
                cv2.putText(frame, f"Calculating baseline... {frame_count}/{BASELINE_FRAMES}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "Keep your eyes open and do not blink!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"Baseline EAR: {baseline_ear:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
                if dynamic_threshold is not None:
                    cv2.putText(frame, f"Dynamic Threshold: {dynamic_threshold:.2f}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

        cv2.imshow("Blink Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[SYSTEM] Shutting down...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()