import cv2
import numpy as np
import psycopg2
import face_recognition

# CREATE TABLE faces (
#     id SERIAL PRIMARY KEY,
#     name VARCHAR(100) NOT NULL,
#     encoding BYTEA NOT NULL
# );

# database connection
conn = psycopg2.connect(
    dbname="face_recognition_db",
    user="postgres",
    password="postgres",
    host="localhost"
)
cursor = conn.cursor()

# Save the face encoding to the PostgreSQL database.
def save_face_to_db(name, face_encoding):
    cursor.execute(
        "INSERT INTO faces (name, encoding) VALUES (%s, %s)",
        (name, psycopg2.Binary(face_encoding.tobytes()))
    )
    conn.commit()

# Function to retrieve all face encodings from the PostgreSQL database
def get_all_faces_from_db():
    cursor.execute("SELECT name, encoding FROM faces")
    return cursor.fetchall()

# Function to recognize faces and optionally save unknown faces
def face_recognition_and_save(video_frame, known_face_encodings, known_face_names):
    rgb_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        else:
            # Save the new face to the database
            save_face_to_db(name, face_encoding)  # Save encoding as binary

        # Draw a rectangle and label around the detected face
        top, right, bottom, left = face_location
        cv2.rectangle(video_frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(video_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Access the Webcam
video_capture = cv2.VideoCapture(0)

# Load known faces from the database
known_face_encodings = []
known_face_names = []

for name, encoding in get_all_faces_from_db():
    known_face_names.append(name)
    known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))

# Real-time face recognition loop
while True:
    result, video_frame = video_capture.read()  # Capture frame-by-frame
    if not result:
        break

    # Perform face recognition and optionally save unknown faces
    face_recognition_and_save(video_frame, known_face_encodings, known_face_names)

    # Display the video feed with bounding boxes
    cv2.imshow("Real-Time Face Recognition", video_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
cursor.close()
conn.close()
