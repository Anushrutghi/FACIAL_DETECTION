import cv2
import numpy as np

def detect_index_finger(frame):
    """
    Detects OPEN PALM (4 or more fingers).
    Returns True if palm is open, else False.
    """

    # Convert to HSV (better for skin detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Skin color range (can be tuned)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return False

    cnt = max(contours, key=cv2.contourArea)

    # Ignore small contours
    if cv2.contourArea(cnt) < 3000:
        return False

    hull = cv2.convexHull(cnt, returnPoints=False)

    if hull is None or len(hull) < 3:
        return False

    defects = cv2.convexityDefects(cnt, hull)

    finger_count = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > 10000:   # depth threshold
                finger_count += 1

    # ✅ OPEN PALM = 4 or more fingers
    return finger_count >= 4
