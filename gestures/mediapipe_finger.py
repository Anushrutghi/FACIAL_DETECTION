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

# Euclidean distance
def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def detect_fingers_and_draw(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    finger_count = 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,255,0), thickness=2)
            )

            lm = hand_landmarks.landmark

            # Thumb (x comparison)
            if lm[4].x < lm[3].x:
                finger_count += 1

            # Other 4 fingers (tip above PIP)
            fingers = [(8,6), (12,10), (16,14), (20,18)]
            for tip, pip in fingers:
                if lm[tip].y < lm[pip].y:
                    finger_count += 1

    return finger_count
