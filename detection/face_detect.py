import cv2

# Load Haar Cascade once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(frame):
    if frame is None:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,      # slightly more accurate
        minNeighbors=5,
        minSize=(100, 100)      # avoid very small false faces
    )

    return faces
