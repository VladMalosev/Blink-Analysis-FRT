import numpy as np
import cv2

class FaceValidator:
    def __init__(self):
        self.face_size_history = []
        self.FACE_SIZE_WINDOW = 10
        self.MIN_FACE_SIZE = 10000  # min face area
        self.MAX_FACE_SIZE = 50000   # max face area
        self.MAX_HEAD_TILT = 20
        self.face_too_far = False
        self.face_too_close = False
        self.head_tilted = False

    def validate_face_position(self, face_rect, landmarks):
        validation_results = {
            'face_too_far': False,
            'face_too_close': False,
            'head_tilted': False,
            'face_size': 0,
            'tilt_angle': 0
        }

        face_width = face_rect.right() - face_rect.left()
        face_height = face_rect.bottom() - face_rect.top()
        face_size = face_width * face_height
        validation_results['face_size'] = face_size

        if face_size < self.MIN_FACE_SIZE:
            validation_results['face_too_far'] = True
            self.face_too_far = True
            self.face_too_close = False
        elif face_size > self.MAX_FACE_SIZE:
            validation_results['face_too_close'] = True
            self.face_too_close = True
            self.face_too_far = False
        else:
            self.face_too_far = False
            self.face_too_close = False

        left_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
        mouth_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)], axis=0)

        eye_slope = (right_eye_center[1] - left_eye_center[1]) / (right_eye_center[0] - left_eye_center[0])
        tilt_angle = np.degrees(np.arctan(eye_slope))
        validation_results['tilt_angle'] = tilt_angle

        if abs(tilt_angle) > self.MAX_HEAD_TILT:
            validation_results['head_tilted'] = True
            self.head_tilted = True
        else:
            self.head_tilted = False

        return validation_results