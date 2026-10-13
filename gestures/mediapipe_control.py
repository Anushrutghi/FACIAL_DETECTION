import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def hand_state(frame, draw=True):
    """
    Returns:
      state dict or None
      state = {
        'finger_count': int,
        'gesture': str,  # PALM, FIST, ONE, TWO, THREE, FOUR, THUMB_UP, UNKNOWN
        'index_tip': (x_px, y_px),
        'pinch': bool,
      }
    """
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]
    lm = hand.landmark

    if draw:
        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
        )

    # Fingers up (index/middle/ring/pinky): tip above PIP
    up = 0
    if lm[8].y < lm[6].y: up += 1
    if lm[12].y < lm[10].y: up += 1
    if lm[16].y < lm[14].y: up += 1
    if lm[20].y < lm[18].y: up += 1

    # Thumb up: compare thumb tip x to IP joint x (handedness varies, but good enough for control)
    thumb_up = lm[4].x < lm[3].x

    finger_count = up + (1 if thumb_up else 0)

    # Pinch = thumb tip close to index tip
    pinch = _dist(lm[4], lm[8]) < 0.05

    # Basic gesture label
    if finger_count == 0 and not pinch:
        gesture = "FIST"
    elif finger_count >= 5:
        gesture = "PALM"
    elif finger_count == 1:
        gesture = "ONE"
    elif finger_count == 2:
        gesture = "TWO"
    elif finger_count == 3:
        gesture = "THREE"
    elif finger_count == 4:
        gesture = "FOUR"
    elif thumb_up and up == 0:
        gesture = "THUMB_UP"
    else:
        gesture = "UNKNOWN"

    index_tip = (int(lm[8].x * w), int(lm[8].y * h))

    return {
        "finger_count": finger_count,
        "gesture": gesture,
        "index_tip": index_tip,
        "pinch": pinch,
    }
