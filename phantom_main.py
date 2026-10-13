#!/usr/bin/env python3
"""PHANTOM Vision – Modular Touchless Spatial-Computing System.

Phase 1 Core Engine:
- Dual-hand real-time 3D landmark tracking
- Precision fingertip cursor with One-Euro/EMA smoothing
- Scale-invariant pinch-to-click with hysteresis
- Air drawing with variable brush radius, multi-hand colors, and eraser
- Open-palm pause, fist clear, peace-sign gestures
- Futuristic holographic HUD with FPS & latency metrics
- Preserves face detection and recognition integration
"""

from __future__ import annotations
import os
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.config import PhantomConfig
from phantom.tracker import HandTracker
from phantom.gestures import GestureEngine
from phantom.canvas import SpatialCanvas
from phantom.ui import PhantomHUD

# Import existing face modules to preserve full functionality
try:
    from detection.face_detect import detect_faces
    from recognition.face_recognize import load_known_faces, recognize_face
    FACE_MODULES_AVAILABLE = True
except ImportError:
    FACE_MODULES_AVAILABLE = False


def main() -> None:
    print("=" * 60)
    print("           P H A N T O M   V I S I O N   v 1 . 0            ")
    print("      Modular Futuristic Touchless Spatial Computing        ")
    print("=" * 60)

    config = PhantomConfig()
    tracker = HandTracker(config=config)
    gesture_engine = GestureEngine(config=config)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera 0 could not be opened. Check permissions or camera index.")
        return

    # Warmup frame to determine resolution
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to read test frame from camera.")
        cap.release()
        return

    h, w = frame.shape[:2]
    canvas = SpatialCanvas((h, w, 3), config=config)
    hud = PhantomHUD((h, w, 3), config=config)

    # Face recognition setup
    known_faces, labels = ([], [])
    if FACE_MODULES_AVAILABLE:
        try:
            known_faces, labels = load_known_faces()
            print(f"[FACE] Loaded {len(labels)} known faces.")
        except Exception as e:
            print(f"[FACE] Notice: could not load faces database: {e}")

    window_name = "PHANTOM Vision - Spatial Computing System"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Performance and latency telemetry
    prev_time = time.perf_counter()
    fps = 30.0
    btn_cooldown_time = 0.0

    print("\n[CONTROLS]")
    print(" - Touchless: Pinch to Click buttons / Air Draw with Index Finger")
    print(" - Open Palm: Pause Drawing | Fist: Clear Canvas")
    print(" - Keys: [Q] Quit | [C] Calibrate | [E] Eraser | [X] Clear | [Z] Undo")
    print(" - Keys: [1-6] Palette Colors | [+][-] Brush Size | [S] Save Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Flip horizontally for natural mirror interaction
        frame = cv2.flip(frame, 1)
        now = time.perf_counter()

        # Telemetry calculations
        dt = now - prev_time
        prev_time = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # 1. Hand Tracking
        hand_states = tracker.process(frame, draw=False)
        latency_ms = tracker.last_inference_ms

        # 2. Gesture and Spatial Event Evaluation
        events = gesture_engine.update(hand_states)

        # Determine primary cursor (index tip of first hand, if present)
        cursor_pos: Optional[Tuple[int, int]] = None
        if hand_states:
            cursor_pos = hand_states[0]["index_tip"]

        # 3. Check for Open-Palm Pause
        is_paused = any(e["type"] == "PAUSE" for e in events)

        # 4. Handle Spatial Events (Drawing, Erasing, Button Clicks)
        if not is_paused and not hud.calibrating:
            for event in events:
                etype = event["type"]
                hid = event["hand_index"]
                pos = event["position"]

                if etype == "DRAW":
                    canvas.draw_point(hid, pos)
                elif etype == "CLEAR":
                    canvas.clear()
                elif etype == "ERASE":
                    canvas.toggle_eraser()
                elif etype == "HOVER" or etype == "DRAG_END":
                    canvas.stop_stroke(hid)
                elif etype == "CLICK":
                    # Check button interaction with fingertip pinch-click
                    btn = hud.draw_buttons(frame, pos, canvas.color_index_primary, canvas.eraser_active)
                    if btn and (time.time() - btn_cooldown_time > 0.35):
                        btype = btn["type"]
                        if btype == "color":
                            canvas.set_color_index(btn["index"])
                        elif btype == "eraser":
                            canvas.toggle_eraser()
                        elif btype == "clear":
                            canvas.clear()
                        elif btype == "undo":
                            canvas.undo()
                        elif btype == "calibrate":
                            hud.calibrating = not hud.calibrating
                        btn_cooldown_time = time.time()
        else:
            # When paused, stop active drawing strokes
            for s in hand_states:
                canvas.stop_stroke(s["hand_index"])

        # 5. Composite Drawing Canvas with Live Video
        frame = canvas.overlay(frame)

        # 6. Render Holographic Landmarks and Reticles
        hud.draw_holographic_landmarks(frame, hand_states)
        for state in hand_states:
            hud.draw_cursor_reticle(frame, state)

        # 7. Render On-Screen Buttons
        hovered_btn = hud.draw_buttons(
            frame,
            cursor_pos,
            canvas.color_index_primary,
            canvas.eraser_active,
        )

        # 8. Render HUD Telemetry
        hud.draw_telemetry(
            frame,
            fps=fps,
            latency_ms=latency_ms,
            hand_states=hand_states,
            brush_size=canvas.brush_size,
            paused=is_paused,
        )

        # 9. Calibration Overlay (if active)
        if hud.calibrating:
            hud.draw_calibration_overlay(frame, hand_states)

        # Display composite output
        cv2.imshow(window_name, frame)

        # 10. Keyboard Shortcut Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            break
        elif key == ord("c"):
            hud.calibrating = not hud.calibrating
        elif key == ord("e"):
            canvas.toggle_eraser()
        elif key == ord("x"):
            canvas.clear()
        elif key == ord("z"):
            canvas.undo()
        elif key in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            canvas.set_color_index(key - ord("1"))
        elif key == ord("=") or key == ord("+"):
            if hud.calibrating:
                hud.calibration_scale = min(2.5, hud.calibration_scale + 0.1)
            else:
                canvas.adjust_brush_size(2)
        elif key == ord("-") or key == ord("_"):
            if hud.calibrating:
                hud.calibration_scale = max(0.5, hud.calibration_scale - 0.1)
            else:
                canvas.adjust_brush_size(-2)
        elif key == ord("s"):
            # Save snapshot to database/snapshots/
            snap_dir = os.path.join(PROJECT_ROOT, "database", "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            fname = os.path.join(snap_dir, f"phantom_snap_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[SAVE] Snapshot saved to {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[PHANTOM] Shutdown clean. All resources released.")


if __name__ == "__main__":
    main()
