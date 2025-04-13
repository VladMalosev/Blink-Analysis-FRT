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
                'description': "Blink twice quickly (within 1 second)",
                'required_blinks': 2,
                'max_time': 12.0,
                'prep_time': 3.0,
                'speed': 1.0,
                'failure_message': "You didn't blink twice fast enough!"
            },
            'triple_slow': {
                'description': "Blink three times slowly (at least 0.5s between)",
                'required_blinks': 3,
                'max_time': 15.0,
                'prep_time': 3.0,
                'min_interval': 0.5,
                'failure_message': "You didn't blink three times with proper intervals!"
            },

            'eyes_open': {
                'description': "Keep eyes open for {duration} seconds",
                'required_blinks': 0,
                'duration': random.uniform(2, 5),
                'max_time': 12.0,
                'prep_time': 3.0,
                'failure_message': "You blinked too soon! Keep eyes open."
            },
            'eyes_open_hard': {
                'description': "Keep eyes open for {duration} seconds (no blinking!)",
                'required_blinks': 0,
                'duration': random.uniform(4, 7),
                'max_time': 15.0,
                'prep_time': 3.0,
                'strict': True,
                'failure_message': "You blinked! Try not to blink at all."
            },

            'random_count': {
                'description': "Blink {count} times at your own pace",
                'required_blinks': random.randint(2, 4),
                'max_time': 15.0,
                'prep_time': 3.0,
                'failure_message': "You didn't blink the required number of times!"
            },
            'surprise_challenge': {
                'description': "Follow the instructions on screen",
                'dynamic': True,
                'max_time': 15.0,
                'prep_time': 3.0,
                'failure_message': "Challenge not completed!"
            },

            'precise_timing': {
                'description': "Blink exactly {duration}s after the beep",
                'duration': random.uniform(1.5, 3.5),
                'max_time': 15.0,
                'prep_time': 3.0,
                'tolerance': 0.3,
                'failure_message': "Wrong timing! Try to match the beep."
            },
            'rhythm_blinks': {
                'description': "Blink to the rhythm (every {interval}s)",
                'interval': random.uniform(0.8, 1.5),
                'required_blinks': 3,
                'max_time': 15.0,
                'prep_time': 3.0,
                'tolerance': 0.2,
                'failure_message': "Didn't match the rhythm!"
            }
        }

        test_categories = [
            ['single_blink', 'double_quick', 'triple_slow'],
            ['eyes_open', 'eyes_open_hard'],
            ['random_count', 'surprise_challenge'],
            ['precise_timing', 'rhythm_blinks']
        ]

        selected_tests = []
        while len(selected_tests) < 4 and test_categories:
            category = random.choice(test_categories)
            test_categories.remove(category)
            selected_tests.append(random.choice(category))

        remaining_slots = 4 - len(selected_tests)
        if remaining_slots > 0:
            available_tests = [t for t in self.test_types.keys() if t not in selected_tests]
            selected_tests.extend(random.sample(available_tests, remaining_slots))

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
        self.test_results = [None] * len(self.test_sequence)
        self.blink_history = []
        self.eyes_open_start = 0
        self.eyes_open_duration = 0
        self.last_feedback = ""
        self.last_feedback_time = 0
        self.failed_current_test = False
        self.beep_time = 0
        self.rhythm_markers = []

    def start_test(self, test_index: int) -> str:
        self.current_test = test_index
        self.completed_blinks = 0
        self.blink_history = []
        self.eyes_open_duration = 0
        self.prep_start_time = time.time()
        self.test_start_time = 0
        self.in_prep_period = True
        self.last_feedback = ""
        self.failed_current_test = False

        current_test = self.test_sequence[test_index]
        return current_test['description']

# check if preparation period is over and start test if needed
    def update_test_state(self, current_time: float) -> bool:
        if not self.in_prep_period:
            return False

        current_test = self.test_sequence[self.current_test]
        prep_time = current_test.get('prep_time', 3.0)

        if current_time - self.prep_start_time >= prep_time:
            self.in_prep_period = False
            self.test_start_time = current_time

            if current_test.get('duration') and not current_test.get('strict'):
                self.eyes_open_start = current_time
            elif current_test.get('strict'):
                self.eyes_open_start = current_time
            elif current_test.get('dynamic'):
                self._setup_dynamic_test()
            elif current_test.get('interval'):
                total_duration = current_test['interval'] * (current_test['required_blinks'] + 1)
                if total_duration > current_test['max_time']:
                    current_test['interval'] = current_test['max_time'] / (current_test['required_blinks'] + 1)
                    current_test['description'] = f"Blink to the rhythm (every {current_test['interval']:.1f}s)"

                self.rhythm_markers = [self.test_start_time + (i * current_test['interval'])
                                       for i in range(1, current_test['required_blinks'] + 1)]
            elif current_test.get('description') == "Blink exactly {duration}s after the beep":
                self.beep_time = current_time + 2

            return True
        return False

    def _setup_dynamic_test(self):
        current_test = self.test_sequence[self.current_test]
        challenge_type = random.choice(['count', 'timing'])

        if challenge_type == 'count':
            current_test.update({
                'required_blinks': random.randint(3, 5),
                'max_time': 15.0,
                'dynamic_type': 'count'
            })
            current_test['description'] = f"Blink {current_test['required_blinks']} times quickly"
        else:
            current_test.update({
                'duration': random.uniform(1.5, 3.0),
                'max_time': 15.0,
                'tolerance': 0.4,
                'dynamic_type': 'timing'
            })
            current_test['description'] = f"Blink exactly {current_test['duration']:.1f}s after the beep"
            self.beep_time = time.time() + 2

    def update_blink(self, timestamp: float) -> Optional[Tuple[bool, str]]:
        if self.failed_current_test or self.in_prep_period:
            return None

        current_test = self.test_sequence[self.current_test]

        if current_test.get('max_time') and (timestamp - self.test_start_time) > current_test['max_time']:
            self.test_results[self.current_test] = False
            self.failed_current_test = True
            return (False, current_test['failure_message'])

        self.blink_history.append(timestamp)
        self.completed_blinks = len(self.blink_history)

        if current_test.get('tolerance') and not current_test.get('interval'):
            if self.beep_time > 0 and timestamp > self.beep_time:
                time_after_beep = timestamp - self.beep_time
                if abs(time_after_beep - current_test['duration']) > current_test['tolerance']:
                    self.last_feedback = f"Off by {abs(time_after_beep - current_test['duration']):.2f}s!"
                    self.last_feedback_time = timestamp
                    self.blink_history = []
                    self.completed_blinks = 0
                else:
                    self.test_results[self.current_test] = True
                    return (True, "Perfect timing!")

        elif current_test.get('interval'):
            if not self.rhythm_markers:
                return None

            if timestamp > self.rhythm_markers[0] + current_test['tolerance']:
                self.last_feedback = "Missed the rhythm!"
                self.last_feedback_time = timestamp
                self.blink_history = []
                self.completed_blinks = 0
                self.rhythm_markers = []
                return None

            if abs(timestamp - self.rhythm_markers[0]) <= current_test['tolerance']:
                self.rhythm_markers.pop(0)
                if not self.rhythm_markers:
                    self.test_results[self.current_test] = True
                    return (True, "Great rhythm!")
            else:
                self.last_feedback = "Close! Try to match the beat exactly"
                self.last_feedback_time = timestamp

        elif current_test.get('speed') and self.completed_blinks >= 2:
            time_between = self.blink_history[-1] - self.blink_history[-2]
            if time_between > current_test['speed']:
                self.last_feedback = "Too slow! Try blinking faster"
                self.last_feedback_time = timestamp
                self.blink_history = []
                self.completed_blinks = 0
                return None

        elif current_test.get('min_interval') and self.completed_blinks >= 2:
            time_between = self.blink_history[-1] - self.blink_history[-2]
            if time_between < current_test['min_interval']:
                self.last_feedback = "Too fast! Blink more slowly"
                self.last_feedback_time = timestamp
                self.blink_history = []
                self.completed_blinks = 0
                return None

        if (current_test.get('required_blinks') and
                not current_test.get('interval') and
                not current_test.get('tolerance')):
            if self.completed_blinks >= current_test['required_blinks']:
                self.test_results[self.current_test] = True
                return (True, "Test passed!")

        return None

    def update_eyes_open(self, current_ear: float, baseline_ear: float, current_time: float) -> Optional[
        Tuple[bool, str]]:
        if self.failed_current_test or self.in_prep_period:
            return None

        current_test = self.test_sequence[self.current_test]

        if not current_test.get('duration'):
            return None

        if current_test.get('strict'):
            if current_ear < baseline_ear * 0.85:
                self.test_results[self.current_test] = False
                self.failed_current_test = True
                return (False, current_test['failure_message'])
            else:
                self.eyes_open_duration = current_time - self.eyes_open_start
                if self.eyes_open_duration >= current_test['duration']:
                    self.test_results[self.current_test] = True
                    return (True, "Test passed!")

        elif current_test.get('duration'):
            if current_ear > baseline_ear * 0.8:
                self.eyes_open_duration = current_time - self.eyes_open_start
                if self.eyes_open_duration >= current_test['duration']:
                    self.test_results[self.current_test] = True
                    return (True, "Test passed!")
            else:
                self.test_results[self.current_test] = False
                self.failed_current_test = True
                return (False, current_test['failure_message'])

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

        # spec feedback for certain tests
        if current_test.get('interval'):
            if len(self.rhythm_markers) > 0:
                next_beat = self.rhythm_markers[0] - current_time
                if next_beat > 0:
                    cv2.putText(frame, f"Next beat in: {next_beat:.1f}s",
                                (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "Blink NOW!",
                                (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            completed = current_test['required_blinks'] - len(self.rhythm_markers)
            cv2.putText(frame, f"Progress: {completed}/{current_test['required_blinks']}",
                        (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if completed > 0 else (0, 255, 255), 2)

        elif current_test.get('tolerance') and not current_test.get('interval'):
            if self.beep_time > 0:
                if current_time < self.beep_time:
                    cv2.putText(frame, f"Beep in: {self.beep_time - current_time:.1f}s",
                                (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Blink now! ({current_test['duration']:.1f}s after beep)",
                                (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # show left time
        if current_test.get('max_time'):
            elapsed = current_time - self.test_start_time
            remaining = max(0, current_test['max_time'] - elapsed)
            color = (0, 165, 255) if remaining < 3 else (255, 255, 0)
            cv2.putText(frame, f"Time left: {remaining:.1f}s",
                        (10, 480 if current_test.get('tolerance') else 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # failure message
        if self.test_results[self.current_test] is False:
            cv2.putText(frame, current_test['failure_message'],
                        (10, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press any key to continue...",
                        (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        if current_time - self.last_feedback_time < 2:
            cv2.putText(frame, self.last_feedback,
                        (10, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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

