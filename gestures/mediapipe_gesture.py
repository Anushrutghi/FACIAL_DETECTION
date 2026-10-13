import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def detect_hand_gesture(frame):
    """
    Returns:
    gesture (str): PALM, FIST, ONE, TWO, THUMB_UP, or None
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    hand = result.multi_hand_landmarks[0]

    # Finger tips
    tips = {
        "thumb": hand.landmark[4],
        "index": hand.landmark[8],
        "middle": hand.landmark[12],
        "ring": hand.landmark[16],
        "pinky": hand.landmark[20]
    }

    # PIP joints
    pip = {
        "index": hand.landmark[6],
        "middle": hand.landmark[10],
        "ring": hand.landmark[14],
        "pinky": hand.landmark[18]
    }

    fingers_up = 0

    if tips["index"].y < pip["index"].y:
        fingers_up += 1
    if tips["middle"].y < pip["middle"].y:
        fingers_up += 1
    if tips["ring"].y < pip["ring"].y:
        fingers_up += 1
    if tips["pinky"].y < pip["pinky"].y:
        fingers_up += 1

    # Thumb up (x-axis logic)
    thumb_up = tips["thumb"].x < hand.landmark[3].x

    # Gesture classification
    if fingers_up == 0 and not thumb_up:
        return "FIST"
    elif fingers_up == 4:
        return "PALM"
    elif fingers_up == 1:
        return "ONE"
    elif fingers_up == 2:
        return "TWO"
    elif thumb_up and fingers_up == 0:
        return "THUMB_UP"

    return None
