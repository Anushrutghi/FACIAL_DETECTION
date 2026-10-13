import time

import cv2
import numpy as np

from detection.face_detect import detect_faces
from gestures.mediapipe_hands import detect_hand_states
from recognition.face_recognize import load_known_faces, recognize_face

SMOOTH_ALPHA = 0.18
SMOOTH_ALPHA_DRAW = 0.22
TOGGLE_COOLDOWN = 0.8
BUTTON_COOLDOWN = 0.5
DRAW_COLOR = (0, 255, 0)
PALETTE_COLORS = [
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 0),
    (255, 255, 0),
    (255, 0, 255),
]
PALETTE_LABELS = ["GREEN", "ORANGE", "RED", "YELLOW", "PINK"]
BUTTON_SIZE = (90, 40)
BUTTON_MARGIN = 14
DRAW_THICKNESS = 10


def _rect_contains(rect, point):
    x, y, w, h = rect
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def _build_ui_buttons(frame_shape):
    h, w = frame_shape[:2]
    buttons = []
    x = w - BUTTON_MARGIN - BUTTON_SIZE[0]
    y = BUTTON_MARGIN
    for idx, color in enumerate(PALETTE_COLORS):
        buttons.append({
            "rect": (x, y, BUTTON_SIZE[0], BUTTON_SIZE[1]),
            "action": "color",
            "index": idx,
            "color": color,
            "label": PALETTE_LABELS[idx],
        })
        y += BUTTON_SIZE[1] + BUTTON_MARGIN

    buttons.append({
        "rect": (x, y, BUTTON_SIZE[0], BUTTON_SIZE[1]),
        "action": "erase",
        "label": "ERASE",
    })
    y += BUTTON_SIZE[1] + BUTTON_MARGIN

    buttons.append({
        "rect": (x, y, BUTTON_SIZE[0], BUTTON_SIZE[1]),
        "action": "clear",
        "label": "CLEAR",
    })
    return buttons


def _draw_ui_buttons(frame, buttons, selected_color_index, eraser_active, cursor_pos=None):
    for button in buttons:
        x, y, w, h = button["rect"]
        color = (50, 50, 50)
        if button["action"] == "color":
            color = button["color"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
        label = button["label"]
        if button["action"] == "erase" and eraser_active:
            label = "ERASING"
        cv2.putText(frame, label, (x + 8, y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if button["action"] == "color" and button["index"] == selected_color_index:
            cv2.rectangle(frame, (x + 4, y + 4), (x + w - 4, y + h - 4), (255, 255, 255), 2)

    if cursor_pos is not None:
        cv2.circle(frame, cursor_pos, 12, (255, 255, 255), 2)
        cv2.circle(frame, cursor_pos, 6, (0, 0, 0), -1)


def _draw_gesture_status(frame, hand_states, draw_mode):
    status_lines = [f"MODE: {'DRAW' if draw_mode else 'IDLE'}"]
    for state in hand_states:
        hand_label = f"H{state['hand_index'] + 1}:{state['handedness']}"
        palm_label = 'PALM' if state['palm'] else ''
        two_label = '2-FINGERS' if state['finger_count'] == 2 else ''
        gesture_label = state['gesture']
        line = f"{hand_label} | {state['finger_count']} | {gesture_label}"
        if palm_label:
            line += f" | {palm_label}"
        if two_label:
            line += f" | {two_label}"
        status_lines.append(line)

    x, y = 20, 40
    for text in status_lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 35


def _smooth_point(prev, target, alpha=SMOOTH_ALPHA):
    if prev is None:
        return target
    return (
        int(alpha * target[0] + (1.0 - alpha) * prev[0]),
        int(alpha * target[1] + (1.0 - alpha) * prev[1]),
    )


def _smooth_rect(prev_rect, target_rect, alpha=SMOOTH_ALPHA):
    if prev_rect is None:
        return target_rect
    px, py, pw, ph = prev_rect
    tx, ty, tw, th = target_rect
    return (
        int(alpha * tx + (1.0 - alpha) * px),
        int(alpha * ty + (1.0 - alpha) * py),
        int(alpha * tw + (1.0 - alpha) * pw),
        int(alpha * th + (1.0 - alpha) * ph),
    )


def _overlay_canvas(frame, canvas):
    mask = np.any(canvas != (0, 0, 0), axis=2)
    if not np.any(mask):
        return frame
    overlay = frame.copy()
    overlay[mask] = canvas[mask]
    return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


def _draw_gesture_status(frame, hand_states, draw_mode):
    status_lines = [f"MODE: {'DRAW' if draw_mode else 'IDLE'}"]
    for state in hand_states:
        hand_label = f"H{state['hand_index'] + 1}:{state['handedness']}"
        palm_label = 'PALM' if state['palm'] else ''
        two_label = '2-FINGERS' if state['finger_count'] == 2 else ''
        gesture_label = state['gesture']
        line = f"{hand_label} | {state['finger_count']} | {gesture_label}"
        if palm_label:
            line += f" | {palm_label}"
        if two_label:
            line += f" | {two_label}"
        status_lines.append(line)

    x, y = 20, 40
    for text in status_lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 35


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open camera")
        return

    cv2.namedWindow("Face + Hand Drawing", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face + Hand Drawing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    known_faces, labels = load_known_faces()
    smoothed_face_rect = None
    face_label = "Unknown"

    canvas = None
    draw_mode = False
    current_color_index = 0
    current_color = DRAW_COLOR
    eraser_active = False
    last_toggle_time = 0.0
    last_button_time = 0.0
    ten_fingers_held = False
    prev_draw_points = {}
    smoothed_draw_tips = {}

    print("Face + Hand Detection running. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        if canvas is None:
            canvas = np.zeros_like(frame)

        hand_states = detect_hand_states(frame, draw=not draw_mode)
        total_fingers = sum(state["finger_count"] for state in hand_states)
        cursor_pos = hand_states[0]["index_tip"] if hand_states else None
        pointer_active = any(state["index_up"] and not state["palm"] for state in hand_states)

        if total_fingers == 10:
            if not ten_fingers_held and time.time() - last_toggle_time > TOGGLE_COOLDOWN:
                draw_mode = not draw_mode
                eraser_active = False
                last_toggle_time = time.time()
            ten_fingers_held = True
        else:
            ten_fingers_held = False

        buttons = _build_ui_buttons(frame.shape)
        if draw_mode and cursor_pos is not None and pointer_active:
            if time.time() - last_button_time > BUTTON_COOLDOWN:
                for button in buttons:
                    if _rect_contains(button["rect"], cursor_pos):
                        if button["action"] == "color":
                            current_color_index = button["index"]
                            current_color = button["color"]
                            eraser_active = False
                        elif button["action"] == "erase":
                            eraser_active = not eraser_active
                        elif button["action"] == "clear":
                            canvas = np.zeros_like(frame)
                        last_button_time = time.time()
                        break

        if draw_mode:
            draw_color = (0, 0, 0) if eraser_active else current_color
            for state in hand_states:
                if state["index_up"] and not state["palm"]:
                    hand_id = state["hand_index"]
                    tip = state["index_tip"]
                    prev = prev_draw_points.get(hand_id)
                    smoothed_prev = smoothed_draw_tips.get(hand_id, tip)
                    smoothed_tip = _smooth_point(smoothed_prev, tip, alpha=SMOOTH_ALPHA_DRAW)
                    if prev is not None:
                        cv2.line(canvas, prev, smoothed_tip, draw_color, DRAW_THICKNESS)
                    else:
                        cv2.circle(canvas, smoothed_tip, DRAW_THICKNESS, draw_color, -1)
                    prev_draw_points[hand_id] = smoothed_tip
                    smoothed_draw_tips[hand_id] = smoothed_tip
                else:
                    prev_draw_points.pop(state["hand_index"], None)
                    smoothed_draw_tips.pop(state["hand_index"], None)
        else:
            faces = detect_faces(frame)
            if faces is not None and len(faces) > 0:
                x, y, w, h = faces[0]
                face_label = recognize_face(frame[y:y + h, x:x + w], known_faces, labels)
                smoothed_face_rect = _smooth_rect(smoothed_face_rect, (x, y, w, h))
            else:
                face_label = "Unknown"
                smoothed_face_rect = None

            if smoothed_face_rect is not None:
                x, y, w, h = smoothed_face_rect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
                cv2.putText(frame, face_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
                center = (x + w // 2, y + h // 2)
                cv2.circle(frame, center, 6, (0, 220, 0), -1)
                cv2.line(frame, (center[0] - 12, center[1]), (center[0] + 12, center[1]), (0, 220, 0), 1)
                cv2.line(frame, (center[0], center[1] - 12), (center[0], center[1] + 12), (0, 220, 0), 1)
            prev_draw_points.clear()

        frame = _overlay_canvas(frame, canvas)
        if draw_mode:
            _draw_ui_buttons(frame, buttons, current_color_index, eraser_active, cursor_pos)
        _draw_gesture_status(frame, hand_states, draw_mode)

        frame = _overlay_canvas(frame, canvas)
        _draw_gesture_status(frame, hand_states, draw_mode)

        cv2.imshow("Face + Hand Drawing", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
