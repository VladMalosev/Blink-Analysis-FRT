import cv2
import dlib


#https://dlib.net/files/
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
tracker = dlib.correlation_tracker()
detector = dlib.get_frontal_face_detector()



def detect_eyes_and_eyelids(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) > 0:
        face = faces[0]
        tracker.start_track(image, face)
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        for (x, y) in left_eye + right_eye:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

        tracking_quality = tracker.update(image)

        if tracking_quality < 7:
            tracker.start_track(image, face)

    cv2.imshow("Eye Tracking", image)

# acc webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_eyes_and_eyelids(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
