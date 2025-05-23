import time
import numpy as np
from collections import deque
from scipy.signal import savgol_filter
import cv2


class MicrosaccadeDetector:
    def __init__(self, sampling_rate=60):
            self.in_blink_period = False
            self.blink_end_time = 0
            self.velocity_threshold = 3
            self.min_duration = 3
            self.max_duration = 60
            self.min_amplitude = 0.05
            self.max_amplitude = 2.5
            self.sampling_rate = sampling_rate
            self.lambda_factor = 6
            self.pixel_to_degree = 0.05

            self.eye_position_history = deque(maxlen=60)
            self.time_history = deque(maxlen=60)
            self.microsaccade_history = deque(maxlen=40)
            self.velocity_history = deque(maxlen=30)

            self.direction_history = deque(maxlen=15)
            self.amplitude_history = deque(maxlen=15)
            self.inter_saccade_intervals = deque(maxlen=15)

            self.calibration_complete = False
            self.calibration_samples = 0
            self.calibration_window = 60
            self.baseline_std_x = 0
            self.baseline_std_y = 0

            self.blink_timestamps = []
            self.microsaccade_during_blink = 0
            self.post_blink_microsaccades = 0

            self.last_detection_time = 0
            self.detection_count = 0

            self.consecutive_similar_movements = 0
            self.eye_movement_variance = deque(maxlen=10)

            self.position_variance = 0
            self.interval_variability = 1.0
            self.direction_variability = 1.0
            self.last_analysis_time = 0
            self.analysis_interval = 2.0

            self.last_counter_reset_time = time.time()
            self.last_counter_reset = time.time()
            self.suspicious_patterns_count = 20
            self.natural_patterns_count = 20

    def calibrate(self, eye_position):
        if self.calibration_complete:
            return

        current_time = time.time()
        self.eye_position_history.append((eye_position[0], eye_position[1]))
        self.time_history.append(current_time)
        self.calibration_samples += 1

        if self.calibration_samples >= self.calibration_window:
            positions = np.array(self.eye_position_history)

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.iqr.html
            q1_x, q3_x = np.percentile(positions[:, 0], [25, 75])
            q1_y, q3_y = np.percentile(positions[:, 1], [25, 75])
            iqr_x = q3_x - q1_x
            iqr_y = q3_y - q1_y

            # https://medium.com/@BetterEverything/find-outliers-in-data-with-tukey-fences-iqr-method-in-python-data-science-guide-aa5cb11fd372
            valid_idx = ((positions[:, 0] >= q1_x - 1.5 * iqr_x) &
                         (positions[:, 0] <= q3_x + 1.5 * iqr_x) &
                         (positions[:, 1] >= q1_y - 1.5 * iqr_y) &
                         (positions[:, 1] <= q3_y + 1.5 * iqr_y))

            filtered_positions = positions[valid_idx]

            if len(filtered_positions) < 0.7 * len(positions):
                filtered_positions = positions

            self.baseline_std_x = np.sqrt(
                np.median((filtered_positions[:, 0] - np.median(filtered_positions[:, 0])) ** 2) * 1.4826)
            self.baseline_std_y = np.sqrt(
                np.median((filtered_positions[:, 1] - np.median(filtered_positions[:, 1])) ** 2) * 1.4826)

            # prevent small values so that would not be /0
            self.baseline_std_x = max(self.baseline_std_x, 0.001)
            self.baseline_std_y = max(self.baseline_std_y, 0.001)

            median_dispersion = np.median(np.sqrt(np.sum(np.diff(filtered_positions, axis=0) ** 2, axis=1)))
            this_pixel_to_degree = 0.15 / max(median_dispersion, 0.001)

            if this_pixel_to_degree < 0.01:
                this_pixel_to_degree = 0.05
            elif this_pixel_to_degree > 0.2:
                this_pixel_to_degree = 0.1

            self.pixel_to_degree = this_pixel_to_degree
            self.velocity_threshold = self.lambda_factor
            self.calibration_complete = True

            print(
                f"[MICROSACCADE] Calibration complete. Noise level: {self.baseline_std_x:.4f}x, {self.baseline_std_y:.4f}y pixels")
            print(f"[MICROSACCADE] Pixel-to-degree conversion: {self.pixel_to_degree:.5f} deg/px")

    def detect_microsaccades(self, current_eye_position):
        current_time = time.time()

        # Reset counters periodically to prevent unbounded growth
        if current_time - self.last_counter_reset > 30:
            if self.suspicious_patterns_count + self.natural_patterns_count > 1000:
                ratio = self.natural_patterns_count / max(1, self.suspicious_patterns_count)
                self.suspicious_patterns_count = 20
                self.natural_patterns_count = 20 * ratio
            self.last_counter_reset = current_time
        else:
            self.last_counter_reset_time = current_time

        if not self.calibration_complete:
            self.calibrate(current_eye_position)
            return False, 0, {}

        self.eye_position_history.append((current_eye_position[0], current_eye_position[1]))
        self.time_history.append(current_time)

        if len(self.eye_position_history) < 8:
            return False, 0, {}

        positions = np.array(list(self.eye_position_history))
        times = np.array(list(self.time_history))

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
        # filter for preserving peaks
        window_size = min(9, len(positions) - (len(positions) % 2 == 0))
        if window_size >= 5:
            try:
                smoothed_x = savgol_filter(positions[:, 0], window_size, 2)
                smoothed_y = savgol_filter(positions[:, 1], window_size, 2)
                positions = np.column_stack((smoothed_x, smoothed_y))
            except ValueError:
                pass

        # https://patrickwalls.github.io/mathematicalpython/differentiation/differentiation/
        velocities = np.zeros((len(positions) - 2, 2))
        for i in range(1, len(positions) - 1):
            dt1 = times[i] - times[i - 1]
            dt2 = times[i + 1] - times[i]
            dt = dt1 + dt2

            if dt > 0:
                weight1 = dt2 / dt
                weight2 = dt1 / dt
                velocities[i - 1, 0] = (positions[i + 1, 0] - positions[i - 1, 0]) / dt
                velocities[i - 1, 1] = (positions[i + 1, 1] - positions[i - 1, 1]) / dt

        for v in velocities:
            self.velocity_history.append(np.sqrt(v[0] ** 2 + v[1] ** 2) * self.pixel_to_degree)

        med_x = np.median(velocities[:, 0])
        med_y = np.median(velocities[:, 1])
        # https://www.numberanalytics.com/blog/practical-mad-applications-techniques
        std_x = np.sqrt(np.median((velocities[:, 0] - med_x) ** 2)) * 1.4826
        std_y = np.sqrt(np.median((velocities[:, 1] - med_y) ** 2)) * 1.4826
        std_x = max(std_x, self.baseline_std_x, 0.001)
        std_y = max(std_y, self.baseline_std_y, 0.001)
        norm_velocities = np.zeros_like(velocities)
        norm_velocities[:, 0] = velocities[:, 0] / std_x
        norm_velocities[:, 1] = velocities[:, 1] / std_y

        test_statistic = np.sqrt(norm_velocities[:, 0] ** 2 + norm_velocities[:, 1] ** 2)

        if len(test_statistic) > 0:
            self.eye_movement_variance.append(np.var(test_statistic))

        # For photos: typically have either unnaturally consistent or zero test statistics
        dynamic_threshold = self.velocity_threshold
        if len(self.eye_movement_variance) > 5:
            avg_variance = np.mean(self.eye_movement_variance)
            if avg_variance < 0.05:
                dynamic_threshold *= 0.8
            elif avg_variance > 2.0:
                dynamic_threshold *= 1.2

        saccade_indices = np.where(test_statistic > dynamic_threshold)[0]

        if len(saccade_indices) > 0:
            events = self._find_continuous_events(saccade_indices)

            for event in events:
                if len(event) >= 2:
                    start_idx = event[0] + 1
                    end_idx = event[-1] + 1

                    start_time = times[start_idx]
                    end_time = times[end_idx]
                    duration_ms = (end_time - start_time) * 1000

                    start_pos = positions[start_idx]
                    end_pos = positions[end_idx]
                    amplitude_px = np.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)
                    amplitude_deg = amplitude_px * self.pixel_to_degree

                    peak_vel_idx = np.argmax(test_statistic[event])
                    peak_velocity = test_statistic[event[peak_vel_idx]]
                    direction = np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])

                    self.direction_history.append(direction)
                    self.amplitude_history.append(amplitude_deg)

                    saccade = {
                        'timestamp': end_time,
                        'start_time': start_time,
                        'duration': duration_ms,
                        'amplitude': amplitude_deg,
                        'velocity': peak_velocity,
                        'direction': direction,
                        'x_start': start_pos[0],
                        'y_start': start_pos[1],
                        'x_end': end_pos[0],
                        'y_end': end_pos[1],
                        'normalized_velocity': peak_velocity / dynamic_threshold
                    }

                    if self._validate_microsaccade(saccade):

                        if len(self.microsaccade_history) > 0:
                            last_saccade = self.microsaccade_history[-1]
                            interval = saccade['timestamp'] - last_saccade['timestamp']

                            self.inter_saccade_intervals.append(interval)
                            self._check_movement_similarity(saccade, last_saccade)

                        self.microsaccade_history.append(saccade)
                        self.last_detection_time = current_time
                        self.detection_count += 1
                        return True, peak_velocity, saccade

        if self.calibration_samples % 10 == 0:
            self._update_adaptive_threshold()

        return False, 0, {}

    # test for perfectly repeating patterns, machine-like cons
    def _check_movement_similarity(self, current_saccade, previous_saccade):
        # Reset the counters if they get too high (prevents unbounded growth)
        if self.suspicious_patterns_count + self.natural_patterns_count > 1000:
            ratio = self.natural_patterns_count / max(1, self.suspicious_patterns_count)
            self.suspicious_patterns_count = 20
            self.natural_patterns_count = 20 * ratio

        dir_diff = abs(current_saccade['direction'] - previous_saccade['direction'])
        dir_diff = min(dir_diff, 2 * np.pi - dir_diff)

        amp_ratio = min(current_saccade['amplitude'], previous_saccade['amplitude']) / max(current_saccade['amplitude'],
                                                                                           previous_saccade[
                                                                                               'amplitude'])
        vel_ratio = min(current_saccade['velocity'], previous_saccade['velocity']) / max(current_saccade['velocity'],
                                                                                         previous_saccade['velocity'])

        if dir_diff < 0.01 and amp_ratio > 0.995 and vel_ratio > 0.995:
            self.consecutive_similar_movements += 1
            self.suspicious_patterns_count += 0.1
        else:
            self.consecutive_similar_movements = max(0, self.consecutive_similar_movements - 1)
            self.natural_patterns_count += 0.5
    def _find_continuous_events(self, indices):
        if len(indices) == 0:
            return []

        events = []
        current_event = [indices[0]]

        # handles small detection dropouts
        for i in range(1, len(indices)):
            if indices[i] <= indices[i - 1] + 2:
                current_event.append(indices[i])
            else:
                if len(current_event) >= 2:
                    events.append(current_event)
                current_event = [indices[i]]

        if len(current_event) >= 2:
            events.append(current_event)

        return events

    def _validate_microsaccade(self, candidate):
        if not (self.min_amplitude <= candidate['amplitude'] <= self.max_amplitude):
            return False

        if not (self.min_duration <= candidate['duration'] <= self.max_duration):
            return False

        expected_ratio = candidate['amplitude'] / candidate['duration'] * 1000  # deg/s
        if expected_ratio > 0:
            ratio = candidate['velocity'] / expected_ratio
            if not (0.2 <= ratio <= 5.0):
                return False

        if len(self.microsaccade_history) > 5:
            # 1. check for suspiciously periodic intervals
            if len(self.inter_saccade_intervals) >= 5:
                intervals = np.array(self.inter_saccade_intervals)
                interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                if interval_cv < 0.05:
                    self.suspicious_patterns_count += 0.5
                    return False

            # 2. for too-perfect pattern in directions
            if len(self.direction_history) >= 6:
                directions = np.array(self.direction_history)
                direction_changes = np.diff(np.unwrap(directions))
                if np.std(direction_changes) < 0.05:
                    self.suspicious_patterns_count += 0.5
                    return False

            # 3. unusual consistency in amplitude
            if len(self.amplitude_history) >= 6:
                amplitudes = np.array(self.amplitude_history)
                amp_cv = np.std(amplitudes) / np.mean(amplitudes) if np.mean(amplitudes) > 0 else 0
                if amp_cv < 0.05:
                    self.suspicious_patterns_count += 0.5
                    return False

        if len(self.microsaccade_history) > 0:
            last_saccade = self.microsaccade_history[-1]
            time_since_last = candidate['timestamp'] - last_saccade['timestamp']

            # detects square wave jerks
            if 0.02 <= time_since_last <= 0.3:
                current_dir = candidate['direction']
                prev_dir = last_saccade['direction']

                angle_diff = abs(np.mod(current_dir - prev_dir + np.pi, 2 * np.pi) - np.pi)
                if angle_diff < np.pi / 4:
                    amp_ratio = candidate['amplitude'] / last_saccade['amplitude']
                    if 0.4 <= amp_ratio <= 2.5:
                        self.natural_patterns_count += 1
                        return True

        return True

    def _update_adaptive_threshold(self):
        if len(self.velocity_history) < 10:
            return

        velocities = list(self.velocity_history)
        velocities.sort()
        baseline = velocities[int(len(velocities) * 0.75)]

        if len(self.eye_movement_variance) > 3:
            avg_variance = np.mean(self.eye_movement_variance)

            if avg_variance < 0.05:
                new_threshold = self.lambda_factor * 0.7
                print("[MICROSACCADE] Low movement variance detected - adjusting threshold")
            else:
                new_threshold = self.lambda_factor
        else:
            new_threshold = self.lambda_factor

        if new_threshold > 1.5 * self.velocity_threshold:
            new_threshold = 1.5 * self.velocity_threshold
        elif new_threshold < 0.6 * self.velocity_threshold:
            new_threshold = 0.6 * self.velocity_threshold

        self.velocity_threshold = new_threshold

    def get_microsaccade_rate(self, window_seconds=5):
        current_time = time.time()
        recent_saccades = [s for s in self.microsaccade_history
                           if current_time - s['timestamp'] <= window_seconds]
        return len(recent_saccades) / window_seconds

    def check_main_sequence(self):
        if len(self.microsaccade_history) < 10:
            return True

        if len(self.blink_timestamps) > 0 and time.time() - self.blink_timestamps[-1] < 0.8:
            return True

        amplitudes = np.array([s['amplitude'] for s in self.microsaccade_history])
        peak_velocities = np.array([s['velocity'] for s in self.microsaccade_history])

        valid_indices = (amplitudes > 0.05) & (peak_velocities > 0.05)
        amplitudes = amplitudes[valid_indices]
        peak_velocities = peak_velocities[valid_indices]

        if len(amplitudes) < 3:
            return True

        log_amplitudes = np.log10(amplitudes)
        log_velocities = np.log10(peak_velocities)

        unique_log_amps = np.unique(np.round(log_amplitudes, 3))
        if len(unique_log_amps) < 3:
            return True

        try:
            slope, intercept = np.polyfit(log_amplitudes, log_velocities, 1, rcond=1e-3)
            corr = np.corrcoef(log_amplitudes, log_velocities)[0, 1]

            is_valid = 0.3 <= slope <= 0.9 and corr > 0.5

            if corr > 0.995:
                is_valid = False
                self.suspicious_patterns_count += 0.5
            return is_valid

        except np.linalg.LinAlgError:
            print("[MICROSACCADE] Warning: Main sequence check failed")
            return True

    def analyze_eyelid_correlation(self, is_blinking):
        current_time = time.time()

        if is_blinking:
            self.in_blink_period = True
            self.blink_end_time = current_time + 0.6

            if len(self.blink_timestamps) > 3:
                if len(self.microsaccade_history) > 0:
                    last_saccade = self.microsaccade_history[-1]
                    if last_saccade['start_time'] > current_time - 0.05:
                        self.microsaccade_during_blink += 0.5

            self.blink_timestamps.append(current_time)
            return

        elif len(self.blink_timestamps) > 0:
            time_since_last_blink = current_time - self.blink_timestamps[-1]

            if 0.05 < time_since_last_blink < 0.4:
                if len(self.microsaccade_history) > 0 and current_time - self.microsaccade_history[-1]['timestamp'] < 0.2:  # Extended window
                    self.post_blink_microsaccades += 1
                    self.natural_patterns_count += 1

        self.blink_timestamps = [t for t in self.blink_timestamps if current_time - t < 10.0]

    def is_photo_detected(self):
        current_time = time.time()
        if current_time - self.last_analysis_time < self.analysis_interval:
            return hasattr(self, 'last_photo_detection_result') and self.last_photo_detection_result

        self.last_analysis_time = current_time

        if len(self.microsaccade_history) > 5 and len(self.blink_timestamps) == 0:
            time_elapsed = current_time - self.time_history[0] if self.time_history else 0
            if time_elapsed > 10:
                print(
                    f"[PHOTO_DETECTION] Early detection: No blinks after {time_elapsed:.1f} seconds with {len(self.microsaccade_history)} microsaccades")
                self.last_photo_detection_result = True
                return True

        # Check repeating patterns
        if self._has_repeating_patterns() and len(self.microsaccade_history) > 6:
            print("[PHOTO_DETECTION] Detected repeating patterns")
            self.last_photo_detection_result = True
            return True

        photo_detected = self.detect_photo_attack(confidence_threshold=2)
        self.last_photo_detection_result = photo_detected
        return photo_detected

    def is_natural_movement(self):
        print(f"\n[NATURAL CHECK] Microsaccades: {len(self.microsaccade_history)}")
        print(f"Blink count: {len(self.blink_timestamps)}")
        print(f"Suspicious patterns: {self.suspicious_patterns_count}")
        print(f"Natural patterns: {self.natural_patterns_count}")

        current_time = time.time()
        time_since_start = current_time - self.time_history[0] if self.time_history else 0

        if len(self.microsaccade_history) < 5 or not self.calibration_complete:
            return True

        if time_since_start > 20 and len(self.blink_timestamps) == 0:
            print(f"[PHOTO_DETECTION] No blinks detected after {time_since_start:.1f} seconds - likely a photo")
            return False

        if len(self.microsaccade_history) >= 5 and len(self.blink_timestamps) == 0:
            print(f"[PHOTO_DETECTION] No blinks after {len(self.microsaccade_history)} microsaccades - likely a photo")
            return False

        if len(self.blink_timestamps) >= 1:
            if self.suspicious_patterns_count > self.natural_patterns_count * 2 and len(self.microsaccade_history) > 8:
                print(f"[PHOTO_DETECTION] Very suspicious pattern ratio despite blinks")
                return False
            return True

        if self.suspicious_patterns_count + self.natural_patterns_count > 1000:
            ratio = self.natural_patterns_count / max(1, self.suspicious_patterns_count)
            self.suspicious_patterns_count = 20
            self.natural_patterns_count = 20 * ratio

        if self.suspicious_patterns_count > self.natural_patterns_count * 0.8 and len(self.microsaccade_history) > 8:
            print(f"[PHOTO_DETECTION] Suspicious patterns dominant")
            return False

        if self._has_repeating_patterns() and len(self.microsaccade_history) > 5:
            print(f"[PHOTO_DETECTION] Repeating patterns detected")
            return False

        return True

        # 1. Check microsaccade rate
        current_rate = self.get_microsaccade_rate(window_seconds=5)
        if not (0.1 <= current_rate <= 6.0):
            self.suspicious_patterns_count += 1
        else:
            self.natural_patterns_count += 0.5

        # 2. Check main sequence relationship
        if not self.check_main_sequence():
            self.suspicious_patterns_count += 1
        else:
            self.natural_patterns_count += 0.5

        # 3. Analyze temporal distribution
        timestamps = np.array([s['timestamp'] for s in self.microsaccade_history])
        intervals = np.diff(timestamps)

        if len(intervals) < 3:
            return True

        self.interval_variability = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        if not (0.2 <= self.interval_variability):
            self.suspicious_patterns_count += 1
        else:
            self.natural_patterns_count += 0.5

        # 4. Direction pattern analysis
        directions = np.array([s['direction'] for s in self.microsaccade_history])
        direction_changes = np.diff(np.unwrap(directions))
        self.direction_variability = np.std(direction_changes)

        if self.direction_variability < 0.1:
            self.suspicious_patterns_count += 1
        else:
            self.natural_patterns_count += 0.5

        # 5. Check for post-blink microsaccades
        if len(self.blink_timestamps) > 5 and self.post_blink_microsaccades == 0:
            self.suspicious_patterns_count += 0.5

        # 6. Decision
        natural_score = self.natural_patterns_count
        suspicious_score = self.suspicious_patterns_count

        if suspicious_score > natural_score * 5:
            return False
        return True

    def _has_repeating_patterns(self):
        if len(self.microsaccade_history) < 5:
            return False

        # 1. Check for similar amplitudes
        amplitudes = np.array([s['amplitude'] for s in self.microsaccade_history])
        unique_amps = np.unique(np.round(amplitudes, 3))
        if len(unique_amps) < max(3, len(amplitudes) * 0.3):
            return True

        # 2. regular timing intervals
        timestamps = np.array([s['timestamp'] for s in self.microsaccade_history])
        intervals = np.diff(timestamps)
        if len(intervals) >= 4:
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            if cv < 0.15:
                return True

        # 3. consistent direction changes
        directions = np.array([s['direction'] for s in self.microsaccade_history])
        dir_changes = np.diff(directions)
        if len(dir_changes) >= 3 and np.std(
                dir_changes) < 0.20:
            return True

        # 4. perfect amplitude/velocity correlation
        if len(self.microsaccade_history) >= 5:
            amplitudes = np.array([s['amplitude'] for s in self.microsaccade_history])
            velocities = np.array([s['velocity'] for s in self.microsaccade_history])

            if np.corrcoef(amplitudes, velocities)[0, 1] > 0.98:
                return True

        return False

    def detect_photo_attack(self, confidence_threshold=2):
        # Special case for early detection
        if len(self.microsaccade_history) > 5 and len(self.blink_timestamps) == 0:
            time_elapsed = time.time() - self.time_history[0] if self.time_history else 0
            if time_elapsed > 10:
                print(
                    f"[PHOTO_DETECTION] No blinks detected after {time_elapsed:.1f} seconds with {len(self.microsaccade_history)} microsaccades")
                return True

        if len(self.microsaccade_history) < 5:
            return False

        evidence_score = 0
        print(f"[PHOTO_DETECTION] Starting photo attack analysis with {len(self.microsaccade_history)} microsaccades")

        # 1. Check for blinks
        if len(self.blink_timestamps) == 0:
            evidence_score += min(3, len(self.microsaccade_history) // 3)
            print(f"[PHOTO_DETECTION] No blinks detected: +{min(3, len(self.microsaccade_history) // 3)} points")

        # 2. eye position variance
        positions = np.array([(s['x_start'], s['y_start']) for s in self.microsaccade_history])
        position_variance = np.var(positions, axis=0).sum()

        print(f"[PHOTO_DETECTION] Position variance: {position_variance:.4f}")

        # Low variance can mean static image
        if position_variance < 10:
            evidence_score += 1
            print(f"[PHOTO_DETECTION] Unusually low position variance: +1 point")
        elif position_variance > 40:
            evidence_score += 1
            print(f"[PHOTO_DETECTION] Unusually high position variance: +1 point")

        # 3. Check microsaccade rate consistency
        if len(self.microsaccade_history) > 8:
            timestamps = np.array([s['timestamp'] for s in self.microsaccade_history])
            segments = np.array_split(timestamps, min(3, len(timestamps) // 3))
            rates = []

            for segment in segments:
                if len(segment) >= 2:
                    duration = segment[-1] - segment[0]
                    if duration > 0:
                        rate = len(segment) / duration
                        rates.append(rate)

            if len(rates) >= 2:
                rate_variation = np.std(rates) / np.mean(rates) if np.mean(rates) > 0 else 0
                print(f"[PHOTO_DETECTION] Microsaccade rate variation: {rate_variation:.4f}")

                if rate_variation < 0.3:
                    evidence_score += 2
                    print(f"[PHOTO_DETECTION] Unusually consistent microsaccade rate: +2 points")

        # 4. Check amplitude distribution
        amplitudes = np.array([s['amplitude'] for s in self.microsaccade_history])
        if len(amplitudes) >= 5:
            try:
                from scipy.stats import gaussian_kde
                density = gaussian_kde(amplitudes)
                test_points = np.linspace(min(amplitudes), max(amplitudes), 20)
                density_vals = density(test_points)

                peaks = np.where((density_vals[1:-1] > density_vals[:-2]) &
                                 (density_vals[1:-1] > density_vals[2:]))[0] + 1

                if len(peaks) <= 1:
                    evidence_score += 1
                    print(f"[PHOTO_DETECTION] Limited amplitude diversity: +1 point")
            except:
                pass

        # 5. unnatural consistent movement patterns
        if len(self.microsaccade_history) >= 6:
            velocities = np.array([s['velocity'] for s in self.microsaccade_history])
            directions = np.array([s['direction'] for s in self.microsaccade_history])

            vel_diffs = np.diff(velocities)
            dir_diffs = np.diff(np.unwrap(directions))

            if len(vel_diffs) >= 3:
                vel_pattern_score = np.std(vel_diffs) / np.mean(np.abs(vel_diffs)) if np.mean(
                    np.abs(vel_diffs)) > 0 else 0
                if vel_pattern_score < 0.5:
                    evidence_score += 1
                    print(f"[PHOTO_DETECTION] Repetitive velocity pattern: +1 point")

            if len(dir_diffs) >= 3:
                dir_pattern_score = np.std(dir_diffs) / np.mean(np.abs(dir_diffs)) if np.mean(
                    np.abs(dir_diffs)) > 0 else 0
                if dir_pattern_score < 0.5:
                    evidence_score += 1
                    print(f"[PHOTO_DETECTION] Repetitive direction pattern: +1 point")

        # 6. suspicious vs natural patterns
        if len(self.microsaccade_history) > 5:
            nat_to_sus_ratio = self.natural_patterns_count / max(1, self.suspicious_patterns_count)

            if nat_to_sus_ratio < 1.0:
                evidence_score += 1
                print(f"[PHOTO_DETECTION] Low natural/suspicious pattern ratio ({nat_to_sus_ratio:.2f}): +1 point")

        # 7. lack of blink-related microsaccades
        if len(self.blink_timestamps) > 2 and self.post_blink_microsaccades == 0:
            evidence_score += 1
            print(f"[PHOTO_DETECTION] No post-blink microsaccades despite multiple blinks: +1 point")

        print(f"[PHOTO_DETECTION] Total evidence score: {evidence_score}/{confidence_threshold}")

        if evidence_score >= confidence_threshold:
            print(f"[PHOTO_DETECTION] PHOTO ATTACK DETECTED")
            return True

        return False
    def get_detection_stats(self):
        return {
            'total_detections': self.detection_count,
            'rate': self.get_microsaccade_rate(),
            'is_natural': self.is_natural_movement(),
            'calibration_complete': self.calibration_complete,
            'current_threshold': self.velocity_threshold,
            'time_since_last': time.time() - self.last_detection_time if self.last_detection_time > 0 else None,
            'suspicious_patterns': self.suspicious_patterns_count,
            'natural_patterns': self.natural_patterns_count,
            'blink_correlation': {
                'blinks_detected': len(self.blink_timestamps),
                'microsaccades_during_blink': self.microsaccade_during_blink,
                'post_blink_microsaccades': self.post_blink_microsaccades
            }
        }

    def register_blink(self, is_blinking, ear_value):
        current_time = time.time()

        if is_blinking:
            self.blink_timestamps.append(current_time)
            self.in_blink_period = True
            self.blink_end_time = current_time + 0.3
        elif hasattr(self, 'in_blink_period') and self.in_blink_period:
            if current_time >= self.blink_end_time:
                self.in_blink_period = False