import random
import time
import cv2
from typing import List, Tuple, Optional, Dict


class InteractiveBlinkTest:
    def __init__(self):
        self.test_types: Dict[str, Dict] = {
            'single_blink': {
                'description': "Blink once normally",
                'required_blinks': 1,
                'max_time': 5.0,
                'speed': None
            },
            'double_quick': {
                'description': "Blink twice quickly (within 1 second)",
                'required_blinks': 2,
                'max_time': 3.0,
                'speed': 1.0
            },
            'eyes_open': {
                'description': "Keep eyes open for {duration} seconds",
                'required_blinks': 0,
                'duration': random.uniform(2, 5),
                'max_time': None
            },
            'triple_slow': {
                'description': "Blink three times slowly (at least 0.5s between blinks)",
                'required_blinks': 3,
                'max_time': 10.0,
                'min_interval': 0.5
            }
        }

        # 4 tests, changeable
        selected_tests = random.sample(list(self.test_types.keys()), min(4, len(self.test_types)))
        self.test_sequence = [self.test_types[test] for test in selected_tests]

        for test in self.test_sequence:
            if test['description'].count('{') > 0:
                test['description'] = test['description'].format(duration=f"{test['duration']:.1f}")

        self.current_test = 0
        self.test_start_time = 0
        self.last_blink_time = 0
        self.completed_blinks = 0
        self.test_results = [False] * len(self.test_sequence)
        self.blink_history = []
        self.eyes_open_start = 0
        self.eyes_open_duration = 0
        self.last_ear = 0
        self.last_eye_state = 'open'

    def start_test(self, test_index: int) -> str:
        self.current_test = test_index
        self.completed_blinks = 0
        self.blink_history = []
        self.eyes_open_duration = 0
        self.test_start_time = time.time()

        if self.test_sequence[test_index].get('duration'):
            self.eyes_open_start = time.time()

        return self.test_sequence[test_index]['description']

    def update_blink(self, timestamp: float) -> Optional[Tuple[bool, str]]:
        current_test = self.test_sequence[self.current_test]

        if current_test.get('duration'):
            return None

        self.blink_history.append(timestamp)
        self.completed_blinks = len(self.blink_history)

        if current_test.get('speed') and self.completed_blinks >= 2:
            time_between = self.blink_history[-1] - self.blink_history[-2]
            if time_between > current_test['speed']:
                self.blink_history = []
                self.completed_blinks = 0
                return (False, "Too slow! Try blinking faster")

        if current_test.get('min_interval') and self.completed_blinks >= 2:
            time_between = self.blink_history[-1] - self.blink_history[-2]
            if time_between < current_test['min_interval']:
                self.blink_history = []
                self.completed_blinks = 0
                return (False, "Too fast! Blink more slowly")

        if self.completed_blinks >= current_test['required_blinks']:
            return (True, "Test passed!")

        return None

    def update_eyes_open(self, current_ear: float, baseline_ear: float) -> Optional[Tuple[bool, str]]:
        current_test = self.test_sequence[self.current_test]

        if not current_test.get('duration'):
            return None

        # if eyes open > 80% of EAR baseline
        if current_ear > baseline_ear * 0.8:
            self.eyes_open_duration = time.time() - self.eyes_open_start
            if self.eyes_open_duration >= current_test['duration']:
                return (True, "Test passed!")
        else:
            # reset if eyes were closed
            self.eyes_open_start = time.time()
            self.eyes_open_duration = 0

        return None

    def get_visual_feedback(self, frame, baseline_ear: float, current_ear: float) -> None:
        current_test = self.test_sequence[self.current_test]
        instruction = current_test['description']

        cv2.putText(frame, f"Test {self.current_test + 1}/{len(self.test_sequence)}: {instruction}",
                    (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if current_test.get('duration'):
            progress = min(current_test['duration'], self.eyes_open_duration)
            cv2.putText(frame, f"Progress: {progress:.1f}/{current_test['duration']:.1f}s",
                        (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, f"Blinks: {self.completed_blinks}/{current_test['required_blinks']}",
                        (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # remaining time
        if current_test.get('max_time'):
            elapsed = time.time() - self.test_start_time
            remaining = max(0, current_test['max_time'] - elapsed)
            cv2.putText(frame, f"Time left: {remaining:.1f}s",
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    def all_tests_passed(self) -> bool:
        return all(self.test_results)

    def get_current_test_timeout(self) -> float:
        return self.test_sequence[self.current_test].get('max_time', float('inf'))