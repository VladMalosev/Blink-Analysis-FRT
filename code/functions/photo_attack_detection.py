import random
import cv2
import numpy as np
from collections import deque
import time


class PhotoAttackDetector:
    def __init__(self):
        self.landmark_history = deque(maxlen=30)
        self.left_ear_history = deque(maxlen=50)
        self.right_ear_history = deque(maxlen=50)
        self.pixel_history = deque(maxlen=10)
        self.texture_history = deque(maxlen=10)
        self.pupil_size_during_blink = deque(maxlen=10)
        self.pupil_size_normal = deque(maxlen=10)
        self.face_positions = deque(maxlen=30)
        self.position_variance_threshold = 2.0

        self.blink_history = deque(maxlen=15)
        self.reflection_history = deque(maxlen=20)
        self.blink_start_time = 0
        self.dynamic_threshold = 0.0
        self.EYE_REGION_SIZE = (50, 50)

        self.MIN_BLINK_DURATION = 0.07
        self.MAX_BLINK_DURATION = 0.5
        self.MIN_BLINK_INTERVAL = 0.2
        self.suspicion_score = 0
        self.detection_threshold = 2.5
        self.consecutive_detections = 0
        self.last_attack_time = 0
        self.attack_cooldown = 2
        self.last_blink_time = time.time()

    # normalizing eye region for consistent processing
    def preprocess_eye_region(self, eye_region):
        if eye_region.size == 0:
            return None

        if len(eye_region.shape) == 3:
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = eye_region

        resized = cv2.resize(gray, self.EYE_REGION_SIZE, interpolation=cv2.INTER_AREA)
        normalized = cv2.equalizeHist(resized) # histogram equalization for contrast normalization
        return normalized

    # analyze texture changes with size-consistent eye regions, compare pixel differences to determine static
    def analyze_texture_variation(self, eye_region):
        processed = self.preprocess_eye_region(eye_region)
        if processed is None:
            return 0.0

        if len(self.texture_history) > 5:
            # compare with stored values in prev frames
            diffs = []
            for prev_eye in list(self.texture_history)[-5:]:
                if prev_eye.shape == processed.shape:
                    diff = cv2.absdiff(processed, prev_eye)
                    diffs.append(np.mean(diff))

            if diffs:
                avg_diff = np.mean(diffs)
                # static textures(photo, display attacks will have low variation)
                return 1.5 if avg_diff < 1.2 else 0.0

        self.texture_history.append(processed)
        return 0.0

    def analyze_eye_movement(self, landmarks):
        if len(self.landmark_history) < 10:
            self.landmark_history.append(landmarks)
            return False

        # compute euclidean distance between landmarks over frames
        movements = []
        for i in range(1, min(10, len(self.landmark_history))):
            prev = self.landmark_history[-i]
            curr = landmarks
            movement = np.mean([np.linalg.norm(np.array(p1) - np.array(p2))
                                for p1, p2 in zip(prev, curr)])
            movements.append(movement)

        avg_movement = np.mean(movements)
        std_movement = np.std(movements)

        # photo attacks often have either no movement or very random movement
        return avg_movement < 0.1 or std_movement > 0.5

    def detect_screen_artifacts(self, eye_region):
        processed = self.preprocess_eye_region(eye_region)
        if processed is None:
            return 0.0

        if len(self.texture_history) < 5:
            self.texture_history.append(processed)
            return 0.0

        # check for high-frequency patterns
        fft = np.fft.fft2(processed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        magnitude[magnitude == 0] = 1e-10
        magnitude = 20 * np.log(magnitude)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        outer_band = magnitude[cy - 5:cy + 5, cx - 5:cx + 5]  # center reg
        outer_band_mean = np.mean(outer_band)

        # unnatural pixel patterns
        pixel_variance = np.var(processed)
        if pixel_variance < 10:  # low variance -> static image
            return 1.0

        # compare with prev frames
        diff_score = np.mean([cv2.absdiff(processed, prev).mean()
                              for prev in list(self.texture_history)[-5:]])

        if outer_band_mean > 38 and diff_score < 5:
            return 1.5
        elif outer_band_mean > 35 and diff_score < 8:
            return 1.0

        self.texture_history.append(processed)
        return 0.0

    # analyze blink timing patterns for naturalness
    def analyze_blink_pattern(self, blink_timestamps):
        if len(blink_timestamps) < 3:
            return 0.0

        intervals = [] # inter blink intervals
        for i in range(1, len(blink_timestamps)):
            intervals.append(blink_timestamps[i] - blink_timestamps[i - 1])

        suspicious_count = 0
        for interval in intervals:
            if interval < 0.5 or interval > 15:  # either too fast or very slow blinking
                suspicious_count += 1
            elif 0.5 <= interval <= 2.0 and random.random() > 0.3:
                suspicious_count += 0.5

        # if has perfect periodicity -> increase suspicious count
        if len(intervals) > 2:
            first_interval = intervals[0]
            if all(abs(interval - first_interval) < 0.1 for interval in intervals[1:]):
                return 2.0

        return min(1.0, suspicious_count / len(intervals))

    # analyze the distribution of blink durations for natural patterns
    def analyze_blink_duration_distribution(self):
        if len(self.blink_history) < 4:
            return 0.0

        durations = [duration for _, duration in self.blink_history]

        mean_duration = np.mean(durations)
        std_duration = np.std(durations)

        if std_duration < 0.02: # check for consistency
            # if all blinks have almost the same consistency -> mark as highly suspicious
            return 1.0

        unnatural_count = 0
        for duration in durations:
            if not (0.1 <= duration <= 0.4):
                unnatural_count += 1

        return min(1.0, unnatural_count / len(durations))

    # analyze eyelid movement dynamics during blinks
    def analyze_eyelid_movement(self, landmarks):
        if not landmarks or len(landmarks) < 48:
            return 0.0

        try:
            if isinstance(landmarks[0], (int, float)):
                landmarks = [(landmarks[i], landmarks[i + 1]) for i in range(0, len(landmarks), 2)]

            upper_lid_indices = [36, 37, 38, 43, 44, 45]
            valid_indices = [i for i in upper_lid_indices if i < len(landmarks)]

            if len(valid_indices) < 4:
                return 0.0

            current_upper = [landmarks[i] for i in valid_indices]
            current_avg = np.mean([p[1] for p in current_upper])

            # will measure eyelid movement speed
            velocity = 0.0
            if len(self.landmark_history) >= 2:
                try:
                    prev_landmarks = self.landmark_history[-1]
                    if isinstance(prev_landmarks[0], (int, float)):
                        prev_landmarks = [(prev_landmarks[i], prev_landmarks[i + 1])
                                          for i in range(0, len(prev_landmarks), 2)]

                    prev_upper = [prev_landmarks[i] for i in valid_indices
                                  if i < len(prev_landmarks)]
                    prev_avg = np.mean([p[1] for p in prev_upper])

                    # eyelid movement velocity (pixels per second)
                    time_diff = max(0.01, time.time() - self.last_blink_time)
                    velocity = abs(current_avg - prev_avg) / time_diff
                except (IndexError, TypeError) as e:
                    print(f"Landmark history processing error: {str(e)}")
                    velocity = 0.0
            self.landmark_history.append(landmarks)

            if velocity > 0:
                if not (40 < velocity < 250):
                    return 1.0
            return 0.0

        except Exception as e:
            print(f"Eyelid analysis error: {str(e)}")
            return 0.0

    def check_photo_attack(self, frame_data):
        if not all(key in frame_data for key in ['eye_landmarks', 'eye_region', 'avg_ear', 'blink_timestamps']):
            return {
                'is_photo': False,
                'reasons': [],
                'confidence': 0,
                'current_score': 0
            }

        current_score = 0.0
        reasons = []
        blink_timestamps = frame_data['blink_timestamps']

        if blink_timestamps and time.time() - blink_timestamps[-1] < 10:
            self.last_blink_time = blink_timestamps[-1]

        if 'face_position' in frame_data:
            self.face_positions.append(frame_data['face_position'])
            if len(self.face_positions) >= 20:
                positions = np.array(self.face_positions)
                variance = np.var(positions, axis=0).mean()
                if variance < self.position_variance_threshold:
                    current_score += 1.2
                    reasons.append("Minimal face movement detected")

        # 1. texture analysis
        texture_score = self.analyze_texture_variation(frame_data['eye_region'])
        if texture_score > 0.7:
            current_score += 2
            reasons.append("Static texture pattern")

        # 2. eye movement analysis
        if self.analyze_eye_movement(frame_data['eye_landmarks']):
            current_score += 1.2
            reasons.append("Unnatural eye movement")

        # 3. EAR consistency check
        self.left_ear_history.append(frame_data['left_ear'])
        self.right_ear_history.append(frame_data['right_ear'])

        if len(self.left_ear_history) > 15:
            left_ear_range = max(self.left_ear_history) - min(self.left_ear_history)
            right_ear_range = max(self.right_ear_history) - min(self.right_ear_history)

            if left_ear_range < 0.015 or right_ear_range < 0.015:
                current_score += 2.0
                reasons.append("Static EAR pattern")

        # 4. blink pattern analysis
        if frame_data['blink_timestamps']:
            blink_pattern_score = self.analyze_blink_pattern(frame_data['blink_timestamps'])
            if blink_pattern_score > 0.7:
                current_score += blink_pattern_score
                reasons.append("Unnatural blink pattern")

        no_blink_penalty = 0.0
        time_since_last_blink = time.time() - self.last_blink_time
        if not blink_timestamps or time_since_last_blink > 15:
            no_blink_penalty = min(2.0, time_since_last_blink / 6)
            reasons.append(f"No blinks detected in {int(time_since_last_blink)} seconds")
        current_score += no_blink_penalty

        # 5. blink duration distribution
        if len(self.blink_history) >= 5:
            duration_score = self.analyze_blink_duration_distribution()
            if duration_score > 0.7:
                current_score += duration_score
                reasons.append("Unnatural blink duration distribution")

        # 6. eyelid movement analysis
        eyelid_score = self.analyze_eyelid_movement(frame_data['eye_landmarks'])
        if eyelid_score > 0.5:
            current_score += eyelid_score
            reasons.append("Unnatural eyelid movement")

        # 7. screen artifact detection
        screen_score = 0.0
        if current_score > 1.5:
            screen_score = self.detect_screen_artifacts(frame_data['eye_region'])
            if screen_score > 1.2:
                current_score += screen_score
                reasons.append("Possible screen artifacts detected")

        self.suspicion_score = max(0, self.suspicion_score * 0.8 + current_score * 1.2)

        if time.time() - self.last_blink_time > 8:
            self.suspicion_score += 0.5

        if current_score > 2.5:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = max(0, self.consecutive_detections - 1)

        detection = (self.suspicion_score >= self.detection_threshold and
                     self.consecutive_detections >= 2)

        if detection:
            self.last_attack_time = time.time()

        return {
            'is_photo': detection,
            'reasons': reasons,
            'confidence': min(1.0, self.suspicion_score / self.detection_threshold),
            'current_score': current_score
        }