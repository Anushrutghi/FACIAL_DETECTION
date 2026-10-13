import cv2
import time
import pyautogui

from controls.face_cursor import FaceCursor
from controls.blink_click import BlinkClicker
from controls.virtual_keyboard import VirtualKeyboard

from gestures.mediapipe_control import hand_state
from detection.face_detect import detect_faces
from recognition.face_recognize import load_known_faces, recognize_face

pyautogui.FAILSAFE = False

# ---------- SETTINGS ----------
PINCH_CLICK_COOLDOWN = 0.35
PINCH_DRAG_HOLD = 0.45
SCROLL_RATE = 30
SCROLL_INTERVAL = 0.12

HAND_SMOOTH_ALPHA = 0.22
hand_sx, hand_sy = None, None

# ---------- INIT ----------
known_faces, labels = load_known_faces()
face_cursor = FaceCursor(alpha=0.18, deadzone=0.07, gain=1.6)
blink_clicker = BlinkClicker(ear_thresh=0.20, cooldown=0.7)
vk = VirtualKeyboard()

cap = cv2.VideoCapture(0)

last_pinch_time = 0
pinch_start = None
dragging = False

last_scroll_time = 0
prev_index_y = None

# FPS
prev_frame_time = time.time()
fps = 0.0

print("HCI MODE RUNNING... Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()

    # ---------- FPS ----------
    dt = now - prev_frame_time
    prev_frame_time = now
    if dt > 0:
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

    # ---------- FACE TRACK + NAME ----------
    faces = detect_faces(frame)
    face_center_norm = None
    name = "Unknown"

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]

        face_img = frame[fy:fy + fh, fx:fx + fw]
        name = recognize_face(face_img, known_faces, labels)

        cx = (fx + fw / 2) / w
        cy = (fy + fh / 2) / h
        face_center_norm = (cx, cy)

        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Face: {name}",
            (fx, fy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # ---------- BLINK CLICK ----------
    clicked, ear = blink_clicker.update(frame)
    if clicked:
        pyautogui.click()

    # ---------- HAND STATE ----------
    state = hand_state(frame, draw=True)
    key_boxes = vk.draw(frame)

    hand_controls_cursor = False

    if state:
        gesture = state["gesture"]
        ix, iy = state["index_tip"]
        pinch = state["pinch"]

        cv2.putText(
            frame,
            f"Gesture: {gesture}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            3
        )

        # visual cursor + fps
        cv2.circle(frame, (ix, iy), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"{int(fps)} FPS",
            (ix + 10, iy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # ---------- TOGGLE KEYBOARD ----------
        if gesture == "THUMB_UP" and now - last_pinch_time > 0.8:
            vk.toggle()
            last_pinch_time = now

        # ---------- HAND CURSOR ----------
        if gesture == "ONE":
            hand_controls_cursor = True
            screen_w, screen_h = pyautogui.size()

            tx = int(ix / w * screen_w)
            ty = int(iy / h * screen_h)

            tx = max(0, min(screen_w - 1, tx))
            ty = max(0, min(screen_h - 1, ty))

            if hand_sx is None:
                hand_sx, hand_sy = tx, ty
            else:
                hand_sx = int(HAND_SMOOTH_ALPHA * tx + (1 - HAND_SMOOTH_ALPHA) * hand_sx)
                hand_sy = int(HAND_SMOOTH_ALPHA * ty + (1 - HAND_SMOOTH_ALPHA) * hand_sy)

            pyautogui.moveTo(hand_sx, hand_sy)

        # ---------- SCROLL ----------
        if gesture == "TWO":
            if prev_index_y is None:
                prev_index_y = iy

            dy = iy - prev_index_y
            if now - last_scroll_time > SCROLL_INTERVAL:
                if dy < -8:
                    pyautogui.scroll(SCROLL_RATE)
                elif dy > 8:
                    pyautogui.scroll(-SCROLL_RATE)
                last_scroll_time = now

            prev_index_y = iy
        else:
            prev_index_y = None

        # ---------- PINCH CLICK / DRAG ----------
        if pinch:
            if pinch_start is None:
                pinch_start = now
            elif now - pinch_start > PINCH_DRAG_HOLD and not dragging:
                pyautogui.mouseDown()
                dragging = True
        else:
            if pinch_start is not None:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                elif now - last_pinch_time > PINCH_CLICK_COOLDOWN:
                    if vk.visible:
                        key = vk.hit_test(key_boxes, ix, iy)
                        if key:
                            vk.press(key)
                    else:
                        pyautogui.click()
                    last_pinch_time = now
            pinch_start = None

    # ---------- FACE CURSOR (fallback) ----------
    if not hand_controls_cursor:
        face_cursor.update(face_center_norm)

    # ---------- HUD ----------
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("HCI System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
