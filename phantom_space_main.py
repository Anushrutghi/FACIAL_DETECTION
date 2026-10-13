#!/usr/bin/env python3
"""PHANTOM Space – Interactive 3D Spatial Computing Workspace.

Phase 2 Core Application:
- Real-time 3D workspace with interactive holographic objects (Cube, Pyramid, GyroRing)
- Single-hand 3D translation & rotation via pinch-grab
- Two-hand dual 6-DOF manipulation (Midpoint translate, Distance scaling, Angle rotation)
- Floating 3D holographic action menus
- 3D spatial ribbon drawing in world coordinates
- Step-by-step guided onboarding demo tutorial
- Comprehensive diagnostic telemetry and holographic reticles
"""

from __future__ import annotations
import os
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.config import PhantomConfig
from phantom.tracker import HandTracker
from phantom.ui import PhantomHUD
from phantom.space import SpatialWorkspace


def main() -> None:
    print("=" * 65)
    print("           P H A N T O M   S P A C E   v 2 . 0           ")
    print("     Interactive 3D Spatial Computing & Virtual Objects   ")
    print("=" * 65)

    config = PhantomConfig()
    tracker = HandTracker(config=config)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera 0 could not be opened. Check permissions or camera index.")
        return

    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERROR] Failed to read test frame from camera.")
        cap.release()
        return

    h, w = frame.shape[:2]
    workspace = SpatialWorkspace(width=w, height=h, focal_length=600.0)
    hud = PhantomHUD((h, w, 3), config=config)

    window_name = "PHANTOM Space - 3D Interactive Workspace"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.perf_counter()
    fps = 30.0

    print("\n[3D INTERACTION CONTROLS]")
    print(" - Single-Hand Pinch: Grab & move virtual object in 3D (X, Y, Z)")
    print(" - Dual-Hand Pinch: Scale & rotate grabbed object with both hands")
    print(" - Floating 3D Buttons: Touch buttons with index fingertip")
    print(" - Keys: [Q] Quit | [R] Reset Objects | [D] Toggle 3D Spatial Draw")
    print(" - Keys: [T] Toggle Guided Tutorial | [C] Spawn Cube | [P] Spawn Pyramid")
    print(" - Keys: [S] Save Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame = cv2.flip(frame, 1)
        now = time.perf_counter()
        dt = max(0.001, now - prev_time)
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # 1. Hand Tracking
        hand_states = tracker.process(frame, draw=False)
        latency_ms = tracker.last_inference_ms

        # 2. Update 3D Workspace & Object Interactions
        workspace.update_and_interact(hand_states, dt=dt)

        # 3. Render 3D Scene (Perspective grid, 3D strokes, Objects, Floating buttons)
        workspace.render(frame)

        # 4. Render Holographic Landmarks & Reticles
        hud.draw_holographic_landmarks(frame, hand_states)
        for state in hand_states:
            hud.draw_cursor_reticle(frame, state)

        # 5. Render Diagnostic Telemetry
        active_mode = "3D DRAW" if workspace.spatial_draw_active else "OBJECT MANIPULATION"
        hud.draw_telemetry(
            frame,
            fps=fps,
            latency_ms=latency_ms,
            hand_states=hand_states,
            brush_size=int(workspace.objects[0].scale if workspace.objects else 60),
            paused=False,
        )

        # Display composite output
        cv2.imshow(window_name, frame)

        # Keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            workspace.reset_objects()
            print("[3D] Scene reset.")
        elif key == ord("d"):
            workspace.spatial_draw_active = not workspace.spatial_draw_active
            print(f"[3D] Spatial Drawing: {'ACTIVE' if workspace.spatial_draw_active else 'OFF'}")
        elif key == ord("t"):
            workspace.tutorial_active = not workspace.tutorial_active
            workspace.tutorial_step = 1
        elif key == ord("c"):
            workspace._trigger_button_action("btn_cube")
        elif key == ord("p"):
            workspace._trigger_button_action("btn_pyr")
        elif key == ord("s"):
            snap_dir = os.path.join(PROJECT_ROOT, "database", "snapshots")
            os.makedirs(snap_dir, exist_ok=True)
            fname = os.path.join(snap_dir, f"phantom_space_{int(time.time())}.jpg")
            cv2.imwrite(fname, frame)
            print(f"[SAVE] 3D Snapshot saved to {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[PHANTOM Space] Shutdown clean. All 3D resources released.")


if __name__ == "__main__":
    main()
