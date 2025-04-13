import time
import numpy as np
from collections import deque


class MicrosaccadeDetector:
    def __init__(self):
        # have to play with vars for better detection
        self.velocity_threshold = 3.0
        self.min_duration = 5
        self.max_duration = 50
        self.min_amplitude = 0.05
        self.max_amplitude = 2.0
        self.sampling_rate = 60

        self.eye_position_history = deque(maxlen=15)
        self.microsaccade_history = deque(maxlen=20)
        self.velocity_history = deque(maxlen=10)

        # Calibration data
        self.calibration_complete = False
        self.calibration_samples = 0
        self.calibration_window = 30
        self.pixel_to_degree = 0.05


# calibrate the detector during initial fixation
    def calibrate(self, eye_position):
        if self.calibration_complete:
            return

        self.eye_position_history.append((eye_position[0], eye_position[1], time.time()))
        self.calibration_samples += 1

        if self.calibration_samples >= self.calibration_window:
            # baseline noise level
            positions = np.array([(x, y) for x, y, _ in self.eye_position_history])
            variances = np.var(positions, axis=0)
            avg_variance = np.mean(variances)

            # adjust thresholds
            self.velocity_threshold = max(3.0, 5.0 * (1 + avg_variance * 0.5))
            self.calibration_complete = True
            print(
                f"[MICROSACCADE] Calibration complete. Adjusted velocity threshold to {self.velocity_threshold:.2f} deg/s")

    def detect_microsaccades(self, current_eye_position):
        if not self.calibration_complete:
            self.calibrate(current_eye_position)
            return False, 0

        current_time = time.time()
        self.eye_position_history.append((current_eye_position[0], current_eye_position[1], current_time))

        if len(self.eye_position_history) < 3:
            return False, 0

        prev_pos = self.eye_position_history[-2]
        dx = current_eye_position[0] - prev_pos[0]
        dy = current_eye_position[1] - prev_pos[1]
        dt = max(0.001, current_time - prev_pos[2])

        velocity_px = np.sqrt(dx ** 2 + dy ** 2) / dt
        velocity_deg = velocity_px * self.pixel_to_degree

        self.velocity_history.append(velocity_deg)

        if velocity_deg > self.velocity_threshold:
            start_time = self.eye_position_history[-3][2]
            duration_ms = (current_time - start_time) * 1000
            amplitude_px = np.sqrt((current_eye_position[0] - self.eye_position_history[-3][0]) ** 2 +
                                   (current_eye_position[1] - self.eye_position_history[-3][1]) ** 2)
            amplitude_deg = amplitude_px * self.pixel_to_degree

            if (self.min_duration <= duration_ms <= self.max_duration and
                    self.min_amplitude <= amplitude_deg <= self.max_amplitude):
                self.microsaccade_history.append({
                    'timestamp': current_time,
                    'velocity': velocity_deg,
                    'duration': duration_ms,
                    'amplitude': amplitude_deg
                })
                return True, velocity_deg

        return False, 0

    def get_microsaccade_rate(self, window_seconds=5):
        current_time = time.time()
        recent_saccades = [s for s in self.microsaccade_history
                           if current_time - s['timestamp'] <= window_seconds]
        return len(recent_saccades) / window_seconds  # saccades per second

    def is_natural_movement(self):
        if len(self.microsaccade_history) < 5:
            return True

        timestamps = [s['timestamp'] for s in self.microsaccade_history]
        intervals = np.diff(timestamps)

        interval_variability = np.std(intervals) / np.mean(intervals)

        current_rate = self.get_microsaccade_rate()

        # not sure about vars
        return 0.3 <= current_rate <= 3.0