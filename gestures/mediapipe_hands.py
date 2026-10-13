import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)


def _thumb_up(lm, handedness_label):
    if handedness_label == "Right":
        return lm[4].x < lm[3].x
    return lm[4].x > lm[3].x


def _finger_up(lm, tip_idx, pip_idx):
    return lm[tip_idx].y < lm[pip_idx].y


def _mean_point(lm, indices, width, height):
    xs = [int(lm[i].x * width) for i in indices]
    ys = [int(lm[i].y * height) for i in indices]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


def detect_hand_states(frame, draw=True):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return []

    hand_states = []
    h, w, _ = frame.shape

    for index, hand_landmarks in enumerate(result.multi_hand_landmarks):
        handedness = "Unknown"
        if result.multi_handedness and len(result.multi_handedness) > index:
            handedness = result.multi_handedness[index].classification[0].label

        if draw:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
            )

        lm = hand_landmarks.landmark
        finger_count = 0
        if _thumb_up(lm, handedness):
            finger_count += 1

        for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
            if _finger_up(lm, tip, pip):
                finger_count += 1

        index_up = _finger_up(lm, 8, 6)
        palm = finger_count >= 4
        palm_center = _mean_point(lm, [0, 5, 9, 13, 17], w, h)
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))

        if finger_count == 0:
            gesture = "FIST"
        elif palm:
            gesture = "PALM"
        elif finger_count == 1:
            gesture = "ONE"
        elif finger_count == 2:
            gesture = "TWO"
        elif finger_count == 3:
            gesture = "THREE"
        elif finger_count == 4:
            gesture = "FOUR"
        elif finger_count == 5:
            gesture = "FIVE"
        else:
            gesture = "UNKNOWN"

        hand_states.append({
            "hand_index": index,
            "handedness": handedness,
            "finger_count": finger_count,
            "gesture": gesture,
            "palm": palm,
            "index_up": index_up,
            "index_tip": index_tip,
            "palm_center": palm_center,
        })

    return hand_states
