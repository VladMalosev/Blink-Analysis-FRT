import random
import time
import cv2
from typing import List, Tuple, Optional, Dict


class InteractiveBlinkTest:
    def __init__(self):
        self.test_types = {
            'single_blink': {
                'description': "Blink once normally",
                'required_blinks': 1,
                'max_time': 12.0,
                'prep_time': 3.0,
                'failure_message': "You didn't blink within the time limit!"
            },
            'double_quick': {
                'description': "Blink twice quickly",
                'required_blinks': 2,
                'max_time': 12.0,
                'prep_time': 3.0,
                'failure_message': "You didn't blink twice within the time limit!"
            },
            'triple_slow': {
                'description': "Blink three times slowly",
                'required_blinks': 3,
                'max_time': 15.0,
                'prep_time': 3.0,
                'failure_message': "You didn't blink three times within the time limit!"
            },
            'eyes_open': {
                'description': "Keep eyes open for {duration} seconds",
                'required_blinks': 0,
                'duration': random.uniform(3, 5),
                'max_time': 15.0,
                'prep_time': 3.0,
                'failure_message': "You blinked too soon! Keep eyes open."
            },
            'random_count': {
                'description': "Blink {count} times at your own pace",
                'required_blinks': random.randint(3, 5),
                'max_time': 15.0,
                'prep_time': 3.0,
                'failure_message': "You didn't blink the required number of times!"
            },
            'precise_timing': {
                'description': "Blink when you see 'NOW!' on screen",
                'max_time': 15.0,
                'prep_time': 3.0,
                'tolerance': 0.6,
                'failure_message': "Missed the timing window! Try again."
            },
            'rhythm_blinks': {
                'description': "Blink to the rhythm (when prompted)",
                'required_blinks': 3,
                'max_time': 15.0,
                'prep_time': 3.0,
                'tolerance': 0.4,
                'failure_message': "Didn't match the rhythm! Try again."
            }
        }

        test_categories = [
            ['single_blink', 'double_quick', 'triple_slow'],
            ['eyes_open', 'random_count'],
            ['precise_timing', 'rhythm_blinks']
        ]

        selected_tests = []
        for category in test_categories:
            selected_tests.append(random.choice(category))

        while len(selected_tests) < 4:
            test = random.choice(list(self.test_types.keys()))
            if test not in selected_tests:
                selected_tests.append(test)

        self.test_sequence = [self.test_types[test] for test in selected_tests]

        for test in self.test_sequence:
            if '{duration}' in test['description']:
                test['description'] = test['description'].format(duration=f"{test['duration']:.1f}")
            elif '{count}' in test['description']:
                test['description'] = test['description'].format(count=test['required_blinks'])
            elif '{interval}' in test['description']:
                test['description'] = test['description'].format(interval=f"{test['interval']:.1f}")

        self.current_test = 0
        self.test_start_time = 0
        self.prep_start_time = 0
        self.in_prep_period = True
        self.last_blink_time = 0
        self.completed_blinks = 0
        self.successful_rhythm_blinks = 0
        self.test_results = [None] * len(self.test_sequence)
        self.blink_history = []
        self.eyes_open_start = 0
        self.eyes_open_duration = 0
        self.last_feedback = ""
        self.last_feedback_time = 0
        self.failed_current_test = False
        self.test_passed = False
        self.beep_time = 0
        self.rhythm_markers = []
        self.warning_count = 0
        self.prompt_time = 0
        self.prompt_active = False
        self.blink_cooldown = 0
        self.rhythm_sync_window = 0.4
        self.currently_blinking = False
        self.counted_rhythm_markers = set()

    def start_test(self, test_index: int) -> str:
        self.current_test = test_index
        self.completed_blinks = 0
        self.successful_rhythm_blinks = 0
        self.blink_history = []
        self.eyes_open_duration = 0
        self.prep_start_time = time.time()
        self.test_start_time = 0
        self.in_prep_period = True
        self.last_feedback = ""
        self.failed_current_test = False
        self.test_passed = False
        self.warning_count = 0
        self.blink_cooldown = 0
        self.currently_blinking = False
        self.counted_rhythm_markers = set()

        current_test = self.test_sequence[test_index]
        return current_test['description']

    def reset_test_state(self):
        self.blink_history = []
        self.completed_blinks = 0
        self.successful_rhythm_blinks = 0
        self.eyes_open_start = time.time()
        self.eyes_open_duration = 0
        self.last_feedback = ""
        self.failed_current_test = False
        self.test_passed = False
        self.warning_count = 0
        self.test_start_time = time.time()
        self.prompt_active = False
        self.prompt_time = 0
        self.blink_cooldown = 0
        self.currently_blinking = False
        self.counted_rhythm_markers = set()

        current_test = self.test_sequence[self.current_test]
        if current_test.get('description') == "Blink when you see 'NOW!' on screen":
            self.prompt_time = time.time() + random.uniform(2, 4)
        elif current_test.get('description').startswith("Blink to the rhythm"):
            self.rhythm_markers = []
            self._setup_rhythm_markers()

    def update_test_state(self, current_time: float) -> bool:
        if self.test_passed:
            return False

        if not self.in_prep_period:
            current_test = self.test_sequence[self.current_test]

            if current_test.get('tolerance') and not current_test.get('interval'):
                if not self.prompt_active and self.prompt_time > 0 and current_time >= self.prompt_time:
                    self.prompt_active = True
                elif self.prompt_active and current_time - self.prompt_time > 2.0:
                    self.prompt_active = False
                    self.prompt_time = current_time + random.uniform(2, 4)
                    self.last_feedback = "Missed the prompt! Watch for the next one."
                    self.last_feedback_time = current_time

            if current_test.get('description').startswith("Blink to the rhythm"):
                if not self.rhythm_markers and not self.failed_current_test:
                    self._setup_rhythm_markers(current_time)

            if self.blink_cooldown > 0:
                self.blink_cooldown = max(0, self.blink_cooldown - (current_time - self.last_blink_time))
                self.last_blink_time = current_time

            return False

        current_test = self.test_sequence[self.current_test]
        prep_time = current_test.get('prep_time', 3.0)

        if current_time - self.prep_start_time >= prep_time:
            self.in_prep_period = False
            self.test_start_time = current_time
            self.last_blink_time = current_time

            if current_test.get('duration'):
                self.eyes_open_start = current_time
            elif current_test.get('tolerance') and not current_test.get('interval'):
                self.prompt_time = current_time + random.uniform(2, 4)
            elif current_test.get('description').startswith("Blink to the rhythm"):
                self._setup_rhythm_markers(current_time)

            return True
        return False

    def _setup_rhythm_markers(self, start_time=None):
        current_test = self.test_sequence[self.current_test]
        if not start_time:
            start_time = self.test_start_time

        base_interval = 1.8
        intervals = [
            base_interval,
            base_interval + random.uniform(0.3, 0.5),
            base_interval + random.uniform(0.5, 0.7)
        ]

        current_test['interval'] = base_interval

        self.rhythm_markers = []
        current_time = start_time
        for interval in intervals:
            current_time += interval
            self.rhythm_markers.append(current_time)

        self.counted_rhythm_markers = set()

    def is_blink_valid_for_rhythm(self, timestamp: float) -> bool:
        if not self.rhythm_markers:
            return False

        closest_marker = min(self.rhythm_markers, key=lambda m: abs(timestamp - m))
        if closest_marker in self.counted_rhythm_markers:
            return False

        closest_distance = abs(timestamp - closest_marker)
        tolerance = self.test_sequence[self.current_test].get('tolerance', 0.4)

        if closest_distance <= tolerance:
            self.counted_rhythm_markers.add(closest_marker)
            return True

        return False

    def update_blink_state(self, is_blinking: bool, timestamp: float) -> Optional[Tuple[bool, str]]:
        if is_blinking and not self.currently_blinking and self.blink_cooldown <= 0:
            self.currently_blinking = True
            return self.register_blink(timestamp)
        elif not is_blinking and self.currently_blinking:
            self.currently_blinking = False

        return None

    def register_blink(self, timestamp: float) -> Optional[Tuple[bool, str]]:
        if self.in_prep_period or self.failed_current_test or self.test_passed:
            return None

        current_test = self.test_sequence[self.current_test]

        if current_test.get('max_time') and (timestamp - self.test_start_time) > current_test['max_time']:
            self.test_results[self.current_test] = False
            self.failed_current_test = True
            return (False, current_test['failure_message'])

        if current_test.get('tolerance') and not current_test.get('interval'):
            if not self.prompt_active:
                self.last_feedback = "Wait for the 'NOW!' prompt"
                self.last_feedback_time = timestamp
                return None
            else:
                self.test_results[self.current_test] = True
                self.test_passed = True
                self.prompt_active = False
                return (True, "Test passed!")

        if current_test.get('description').startswith("Blink to the rhythm"):
            if self.is_blink_valid_for_rhythm(timestamp):
                self.successful_rhythm_blinks += 1
                self.last_feedback = f"Good! {self.successful_rhythm_blinks}/{current_test['required_blinks']}"
                self.last_feedback_time = timestamp

                if self.successful_rhythm_blinks >= current_test['required_blinks']:
                    self.test_results[self.current_test] = True
                    self.test_passed = True
                    return (True, "Test passed!")
            else:
                if self.rhythm_markers:
                    next_marker = min([m for m in self.rhythm_markers if m > timestamp], default=None)
                    if next_marker and timestamp < next_marker - current_test.get('tolerance', 0.4):
                        self.last_feedback = "Too early! Wait for the prompt"
                    else:
                        self.last_feedback = "Try to blink exactly when prompted"
                    self.last_feedback_time = timestamp
                return None

        self.blink_history.append(timestamp)
        self.blink_cooldown = 0.5
        self.last_blink_time = timestamp

        if current_test.get('required_blinks') and not current_test.get('description').startswith(
                "Blink to the rhythm"):
            self.completed_blinks += 1
            if self.completed_blinks >= current_test['required_blinks']:
                self.test_results[self.current_test] = True
                self.test_passed = True
                return (True, "Test passed!")

        return None

    def update_blink(self, timestamp: float) -> Optional[Tuple[bool, str]]:
        return self.register_blink(timestamp)

    def update_eyes_open(self, current_ear: float, baseline_ear: float, current_time: float) -> Optional[
        Tuple[bool, str]]:
        if self.failed_current_test or self.in_prep_period or self.test_passed:
            return None

        current_test = self.test_sequence[self.current_test]

        if not current_test.get('duration'):
            return None

        blink_threshold = 0.75
        is_blinking = current_ear < baseline_ear * blink_threshold

        blink_result = self.update_blink_state(is_blinking, current_time)
        if blink_result:
            return blink_result

        if is_blinking:
            self.warning_count += 1
            if self.warning_count >= 3:
                self.test_results[self.current_test] = False
                self.failed_current_test = True
                return (False, current_test['failure_message'])
            else:
                self.last_feedback = f"Warning {self.warning_count}/3: Keep eyes open!"
                self.last_feedback_time = current_time
        else:
            self.warning_count = max(0, self.warning_count - 1)

            self.eyes_open_duration = current_time - self.eyes_open_start
            if self.eyes_open_duration >= current_test['duration']:
                self.test_results[self.current_test] = True
                self.test_passed = True
                return (True, "Test passed!")

        return None

    def get_visual_feedback(self, frame, baseline_ear: float, current_ear: float, current_time: float) -> None:
        current_test = self.test_sequence[self.current_test]

        if self.in_prep_period:
            prep_time = current_test.get('prep_time', 3.0)
            remaining_prep = max(0, prep_time - (current_time - self.prep_start_time))
            cv2.putText(frame, f"Prepare: {remaining_prep:.1f}s",
                        (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return

        test_status = f"Test {self.current_test + 1}/{len(self.test_sequence)}"
        cv2.putText(frame, f"{test_status}: {current_test['description']}",
                    (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if self.test_passed:
            cv2.putText(frame, "Test Completed! Press any key to continue...",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return

        if not self.failed_current_test:
            if current_test.get('required_blinks') and not current_test.get('description').startswith(
                    "Blink to the rhythm"):
                progress = min(1.0, self.completed_blinks / current_test['required_blinks'])
                bar_width = int(200 * progress)
                cv2.rectangle(frame, (10, 430), (210, 450), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 430), (10 + bar_width, 450), (0, 255, 0), -1)
                cv2.putText(frame, f"Blinks: {self.completed_blinks}/{current_test['required_blinks']}",
                            (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif current_test.get('description').startswith("Blink to the rhythm"):
                progress = min(1.0, self.successful_rhythm_blinks / current_test['required_blinks'])
                bar_width = int(200 * progress)
                cv2.rectangle(frame, (10, 430), (210, 450), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 430), (10 + bar_width, 450), (0, 255, 0), -1)
                cv2.putText(frame, f"Rhythm: {self.successful_rhythm_blinks}/{current_test['required_blinks']}",
                            (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            elif current_test.get('duration'):
                progress = min(1.0, self.eyes_open_duration / current_test['duration'])
                bar_width = int(200 * progress)
                cv2.rectangle(frame, (10, 430), (210, 450), (100, 100, 100), -1)
                cv2.rectangle(frame, (10, 430), (10 + bar_width, 450), (0, 255, 0), -1)
                cv2.putText(frame, f"Time: {self.eyes_open_duration:.1f}/{current_test['duration']:.1f}s",
                            (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if current_test.get('tolerance') and not current_test.get('interval'):
            if self.prompt_active:
                cv2.putText(frame, "BLINK NOW!",
                            (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            elif self.prompt_time > 0 and not self.failed_current_test:
                time_to_prompt = max(0, self.prompt_time - current_time)
                if time_to_prompt < 1.0:
                    cv2.putText(frame, "Get ready...",
                                (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        elif current_test.get('description').startswith("Blink to the rhythm"):
            if self.rhythm_markers and not self.failed_current_test:
                upcoming_markers = [m for m in self.rhythm_markers if m > current_time]
                if upcoming_markers:
                    next_marker = min(upcoming_markers)
                    time_to_next = next_marker - current_time

                    if time_to_next < 0.5:
                        cv2.putText(frame, "BLINK NOW!",
                                    (frame.shape[1] // 2 - 100, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    elif time_to_next < 1.0:
                        cv2.putText(frame, "Get ready...",
                                    (frame.shape[1] // 2 - 80, frame.shape[0] // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.putText(frame,
                            f"Remaining: {current_test['required_blinks'] - self.successful_rhythm_blinks}/{current_test['required_blinks']}",
                            (10, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        if current_test.get('max_time'):
            elapsed = current_time - self.test_start_time
            remaining = max(0, current_test['max_time'] - elapsed)
            color = (0, 165, 255) if remaining < 3 else (255, 255, 0)
            cv2.putText(frame, f"Time left: {remaining:.1f}s",
                        (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if self.test_results[self.current_test] is False:
            cv2.putText(frame, current_test['failure_message'],
                        (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press any key to try again or ESC to skip",
                        (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

        if current_time - self.last_feedback_time < 2:
            cv2.putText(frame, self.last_feedback,
                        (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def all_tests_passed(self) -> bool:
        return all(result is True for result in self.test_results)

    def any_test_failed(self) -> bool:
        return any(result is False for result in self.test_results)

    def get_final_results(self) -> Dict[str, List[Dict]]:
        results = []
        for i, test in enumerate(self.test_sequence):
            results.append({
                'description': test['description'],
                'passed': self.test_results[i],
                'failure_message': test.get('failure_message', '') if self.test_results[i] is False else ''
            })
        return {
            'passed': self.all_tests_passed(),
            'results': results
        }