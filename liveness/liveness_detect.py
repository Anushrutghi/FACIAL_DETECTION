import cv2
import time

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

eyes_closed_start = None
ALERT_TIME = 3  # seconds

def check_liveness(face_bgr):
    global eyes_closed_start

    if face_bgr is None or face_bgr.size == 0:
        return False, 0.0, []

    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    upper_face = gray[0:int(h * 0.55), :]

    eyes = eye_cascade.detectMultiScale(
        upper_face,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(20, 20)
    )

    eyes_closed = True

    for (ex, ey, ew, eh) in eyes:
        # 👁️ if eye height is large enough → eye open
        if eh > 18:
            eyes_closed = False

    # ---------- TIMER LOGIC ----------
    if eyes_closed:
        if eyes_closed_start is None:
            eyes_closed_start = time.time()

        closed_time = time.time() - eyes_closed_start

        if closed_time >= ALERT_TIME:
            return True, closed_time, eyes

        return False, closed_time, eyes
    else:
        eyes_closed_start = None
        return False, 0.0, eyes
