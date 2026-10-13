#!/usr/bin/env python3
"""PHANTOM Lab – Unified Spatial Computing Suite.

Phase 4 Complete Integration:
- Mode 1: PHANTOM Vision (Dual-hand 2D Air Drawing & Gestures)
- Mode 2: PHANTOM Space (3D Virtual Objects & 6-DOF Manipulation)
- Mode 3: PHANTOM Touch (Haptic Radiation Pressure & Acoustic Wavefront Simulator)
- Mode 4: Diagnostics & Device Detection (Camera, Audio, Serial Hardware, Logs)
- Accessibility Options: High-contrast HUD, Left-hand dominance, Tremor reduction
- Plugin Architecture with Telemetry Logging
- Automatic Video Dropout Error Recovery
"""

from __future__ import annotations
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.config import PhantomConfig
from phantom.tracker import HandTracker
from phantom.gestures import GestureEngine
from phantom.canvas import SpatialCanvas
from phantom.ui import PhantomHUD
from phantom.space import SpatialWorkspace
from phantom.touch import HapticDeviceInterface, create_haptic_device, HapticPattern
from phantom.lab import PluginManager, TelemetryLoggerPlugin, DeviceDetector, AccessibilityProfile


CYAN = (255, 220, 0)
NEON_GREEN = (0, 255, 128)
AMBER = (0, 165, 255)
WHITE = (255, 255, 255)
HUD_BG = (20, 24, 28)


class PhantomLabApp:
    """Master application controller for the unified PHANTOM suite."""

    def __init__(self) -> None:
        self.config = PhantomConfig()
        self.a11y = AccessibilityProfile()

        # Core subsystems
        self.tracker = HandTracker(config=self.config)
        self.gesture_engine = GestureEngine(config=self.config)
        self.haptics: HapticDeviceInterface = create_haptic_device()

        # Plugin subsystem
        self.plugin_manager = PluginManager()
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        telemetry_log = os.path.join(log_dir, "phantom_telemetry.log")
        self.plugin_manager.register(TelemetryLoggerPlugin(telemetry_log))

        # Runtime modes: 1=VISION, 2=SPACE, 3=TOUCH, 4=DIAGNOSTICS
        self.current_mode = 1
        self.mode_names = {
            1: "VISION // 2D AIR DRAW",
            2: "SPACE // 3D OBJECTS",
            3: "TOUCH // HAPTICS LAB",
            4: "DIAGNOSTICS // HARDWARE",
        }

        # Lazy initialized components
        self.canvas: Optional[SpatialCanvas] = None
        self.hud: Optional[PhantomHUD] = None
        self.workspace: Optional[SpatialWorkspace] = None

        # Camera and recovery state
        self.cap: Optional[cv2.VideoCapture] = None
        self.camera_index = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5

        # Telemetry
        self.fps = 30.0
        self.prev_time = time.perf_counter()
        self.btn_cooldown_time = 0.0

        # Cached hardware diagnostics
        self.cached_devices = DeviceDetector.get_system_summary()

    def init_camera(self) -> bool:
        """Initializes or recovers video capture."""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"[RECOVERY] Camera {self.camera_index} failed to open.")
            return False
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("[RECOVERY] Camera opened but frame read failed.")
            return False

        h, w = frame.shape[:2]
        if self.canvas is None:
            self.canvas = SpatialCanvas((h, w, 3), config=self.config)
            self.hud = PhantomHUD((h, w, 3), config=self.config)
            self.workspace = SpatialWorkspace(width=w, height=h, focal_length=600.0)

        print(f"[SYSTEM] Camera {self.camera_index} initialized ({w}x{h}).")
        self.reconnect_attempts = 0
        return True

    def run(self) -> None:
        print("=" * 70)
        print("          P H A N T O M   L A B   //   U L T I M A T E           ")
        print("    Unified Spatial Computing, 3D Workspace, Touch & Lab Suite  ")
        print("=" * 70)

        if not self.init_camera():
            print("[FATAL] Camera could not be initialized after probing. Exiting.")
            return

        window_name = "PHANTOM Lab - Unified Spatial Computing Suite"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n[NAVIGATION HOTKEYS]")
        print(" - [1] Vision (Air Draw) | [2] Space (3D Objects) | [3] Touch (Haptics) | [4] Diagnostics")
        print(" - [A] Cycle Accessibility Profile (High Contrast / Tremor Smoothing)")
        print(" - [Q] / [ESC] Safe Shutdown | [S] Save Snapshot Image\n")

        while True:
            ret, frame = self.cap.read()

            # Error recovery watchdog
            if not ret or frame is None:
                self.reconnect_attempts += 1
                print(f"[WATCHDOG] Frame dropout detected. Reconnecting ({self.reconnect_attempts}/{self.max_reconnect_attempts})...")
                time.sleep(0.5)
                if self.reconnect_attempts >= self.max_reconnect_attempts:
                    print("[WATCHDOG] Max reconnect attempts exceeded. Shutting down cleanly.")
                    break
                self.init_camera()
                continue

            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            dt = max(0.001, now - self.prev_time)
            self.prev_time = now
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

            # Plugin hook: pre-frame
            frame = self.plugin_manager.dispatch_frame(frame)

            # 1. Hand Tracking with Accessibility alpha and dominance reordering
            alpha = self.a11y.get_smoothing_alpha(self.config.get("tracking", "smoothing_alpha", 0.25))
            self.tracker.smooth_alpha = alpha
            hand_states = self.tracker.process(frame, draw=False)
            hand_states = self.a11y.prioritize_hand(hand_states)

            # Plugin hook: post-hands
            self.plugin_manager.dispatch_hands(hand_states)

            # 2. Gesture and Spatial Events
            events = self.gesture_engine.update(hand_states)
            for ev in events:
                self.plugin_manager.dispatch_event(ev)

            # Check primary cursor
            cursor_pos: Optional[Tuple[int, int]] = hand_states[0]["index_tip"] if hand_states else None

            # 3. Dispatch to Current Mode
            if self.current_mode == 1:
                self._update_mode_vision(frame, hand_states, events, cursor_pos)
            elif self.current_mode == 2:
                self._update_mode_space(frame, hand_states, dt)
            elif self.current_mode == 3:
                self._update_mode_touch(frame, hand_states, events, cursor_pos)
            elif self.current_mode == 4:
                self._update_mode_diagnostics(frame)

            # 4. Render Holographic Landmarks & Reticles across all modes
            self.hud.draw_holographic_landmarks(frame, hand_states)
            for state in hand_states:
                self.hud.draw_cursor_reticle(frame, state)

            # 5. Render Top Mode Navigation Bar
            self._render_mode_navbar(frame, cursor_pos, events)

            # 6. Render Diagnostic Telemetry
            self.hud.draw_telemetry(
                frame,
                fps=self.fps,
                latency_ms=self.tracker.last_inference_ms,
                hand_states=hand_states,
                brush_size=self.canvas.brush_size if self.canvas else 8,
                paused=any(e["type"] == "PAUSE" for e in events),
            )

            # Plugin hook: render
            self.plugin_manager.dispatch_render(frame)

            cv2.imshow(window_name, frame)

            # Handle Keyboard Input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                self.current_mode = key - ord("0")
                print(f"[MODE] Switched to {self.mode_names[self.current_mode]}")
            elif key == ord("a"):
                self.a11y.high_contrast = not self.a11y.high_contrast
                print(f"[A11Y] High Contrast Theme: {self.a11y.high_contrast}")
            elif key == ord("s"):
                snap_dir = os.path.join(PROJECT_ROOT, "database", "snapshots")
                os.makedirs(snap_dir, exist_ok=True)
                fname = os.path.join(snap_dir, f"phantom_lab_{int(time.time())}.jpg")
                cv2.imwrite(fname, frame)
                print(f"[SAVE] Snapshot saved to {fname}")
            elif self.current_mode == 1:
                # Vision mode shortcuts
                if key == ord("e"):
                    self.canvas.toggle_eraser()
                elif key == ord("x"):
                    self.canvas.clear()
                elif key == ord("z"):
                    self.canvas.undo()
            elif self.current_mode == 2:
                # Space mode shortcuts
                if key == ord("r"):
                    self.workspace.reset_objects()
                elif key == ord("d"):
                    self.workspace.spatial_draw_active = not self.workspace.spatial_draw_active

        # Cleanup on exit
        self.haptics.stop_emission()
        self.haptics.disconnect()
        self.plugin_manager.unload_all()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[PHANTOM Lab] Clean shutdown complete.")

    def _render_mode_navbar(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]],
        events: List[Dict[str, Any]],
    ) -> None:
        """Renders futuristic top navigation bar with touchless mode switcher."""
        h, w = frame.shape[:2]
        nav_h = 36
        nav_w = 480
        nx = (w - nav_w) // 2
        ny = 10

        # Semi-transparent background
        scrim = frame[ny:ny + nav_h, nx:nx + nav_w]
        tint = np.full_like(scrim, (15, 20, 25))
        cv2.addWeighted(tint, 0.75, scrim, 0.25, 0, scrim)
        cv2.rectangle(frame, (nx, ny), (nx + nav_w, ny + nav_h), CYAN, 1, cv2.LINE_AA)

        btn_w = nav_w // 4
        btn_labels = ["1:VISION", "2:SPACE", "3:TOUCH", "4:DIAGS"]

        is_click = any(e["type"] == "CLICK" for e in events)

        for i, lbl in enumerate(btn_labels):
            bx = nx + i * btn_w
            is_active = (self.current_mode == i + 1)
            is_hover = cursor_pos and (bx <= cursor_pos[0] <= bx + btn_w and ny <= cursor_pos[1] <= ny + nav_h)

            if is_hover and is_click and (time.time() - self.btn_cooldown_time > 0.4):
                self.current_mode = i + 1
                self.btn_cooldown_time = time.time()
                self.haptics.trigger_pattern(HapticPattern.CLICK, (float(cursor_pos[0]), float(cursor_pos[1]), 0.0))

            b_color = AMBER if is_active else (WHITE if is_hover else (160, 170, 180))
            if is_active:
                cv2.rectangle(frame, (bx + 2, ny + 2), (bx + btn_w - 2, ny + nav_h - 2), (40, 50, 60), -1)

            cv2.putText(frame, lbl, (bx + 14, ny + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_color, 1, cv2.LINE_AA)

    def _update_mode_vision(
        self,
        frame: np.ndarray,
        hand_states: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        cursor_pos: Optional[Tuple[int, int]],
    ) -> None:
        """Handles Mode 1: 2D Spatial Air Drawing."""
        is_paused = any(e["type"] == "PAUSE" for e in events)

        if not is_paused:
            for event in events:
                etype = event["type"]
                hid = event["hand_index"]
                pos = event["position"]

                if etype == "DRAW":
                    self.canvas.draw_point(hid, pos)
                elif etype == "CLEAR":
                    self.canvas.clear()
                    self.haptics.trigger_pattern(HapticPattern.BUZZ, (float(pos[0]), float(pos[1]), 0.0))
                elif etype == "ERASE":
                    self.canvas.toggle_eraser()
                elif etype == "HOVER" or etype == "DRAG_END":
                    self.canvas.stop_stroke(hid)
                elif etype == "CLICK":
                    btn = self.hud.draw_buttons(frame, pos, self.canvas.color_index_primary, self.canvas.eraser_active)
                    if btn and (time.time() - self.btn_cooldown_time > 0.35):
                        self.haptics.trigger_pattern(HapticPattern.CLICK, (float(pos[0]), float(pos[1]), 0.0))
                        btype = btn["type"]
                        if btype == "color":
                            self.canvas.set_color_index(btn["index"])
                        elif btype == "eraser":
                            self.canvas.toggle_eraser()
                        elif btype == "clear":
                            self.canvas.clear()
                        elif btype == "undo":
                            self.canvas.undo()
                        self.btn_cooldown_time = time.time()
        else:
            for s in hand_states:
                self.canvas.stop_stroke(s["hand_index"])

        frame[:] = self.canvas.overlay(frame)
        self.hud.draw_buttons(frame, cursor_pos, self.canvas.color_index_primary, self.canvas.eraser_active)

    def _update_mode_space(
        self,
        frame: np.ndarray,
        hand_states: List[Dict[str, Any]],
        dt: float,
    ) -> None:
        """Handles Mode 2: 3D Spatial Virtual Workspace."""
        self.workspace.update_and_interact(hand_states, dt=dt)
        self.workspace.render(frame)

    def _update_mode_touch(
        self,
        frame: np.ndarray,
        hand_states: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
        cursor_pos: Optional[Tuple[int, int]],
    ) -> None:
        """Handles Mode 3: Haptics Laboratory & Acoustic Wavefront Simulator."""
        h, w = frame.shape[:2]

        # Draw interactive tactile surface plate in center of screen
        cx, cy = w // 2, h // 2
        pw, ph = 260, 200
        px, py = cx - pw // 2, cy - ph // 2

        plate = frame[py:py + ph, px:px + pw]
        tint = np.full_like(plate, (30, 45, 60))
        cv2.addWeighted(tint, 0.65, plate, 0.35, 0, plate)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), CYAN, 2, cv2.LINE_AA)

        cv2.putText(frame, "PHANTOM ULTRASONIC TEST PAD", (px + 12, py + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
        cv2.putText(frame, "Touch pad area to feel simulated radiation pressure", (px + 12, py + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1, cv2.LINE_AA)

        # Check interaction with pad
        if cursor_pos and (px <= cursor_pos[0] <= px + pw and py <= cursor_pos[1] <= py + ph):
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), AMBER, 2, cv2.LINE_AA)
            self.haptics.emit_focal_point(float(cursor_pos[0]), float(cursor_pos[1]), 150.0, intensity=0.9, modulation_hz=200.0)

            # Spawn visual wavefront ripples on contact
            if hasattr(self.haptics, "spawn_visual_ripple"):
                self.haptics.spawn_visual_ripple(cursor_pos, intensity=0.8)

        # Render acoustic ripples if simulator
        if hasattr(self.haptics, "render_ripples"):
            self.haptics.render_ripples(frame)

        # Display Haptic Telemetry & Physical Hardware Safety status
        diag = self.haptics.get_diagnostics()
        diag_x, diag_y = 20, h - 130
        scrim = frame[diag_y - 20:diag_y + 110, diag_x - 10:diag_x + 380]
        tint = np.full_like(scrim, (15, 20, 25))
        cv2.addWeighted(tint, 0.8, scrim, 0.2, 0, scrim)
        cv2.rectangle(frame, (diag_x - 10, diag_y - 20), (diag_x + 380, diag_y + 110), NEON_GREEN, 1, cv2.LINE_AA)

        cv2.putText(frame, "HAPTIC EMISSION TELEMETRY", (diag_x, diag_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, NEON_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Backend Mode: {diag.get('mode', 'SIMULATION')}", (diag_x, diag_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Hardware Connected: {diag.get('is_hardware', False)} (Physical Transducer)", (diag_x, diag_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, AMBER if not diag.get('is_hardware') else NEON_GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Simulated Pressure: {diag.get('simulated_pressure_pa', 0.0)} Pa (at 200 Hz AM)", (diag_x, diag_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Audio Cues: {diag.get('audio_feedback_active', False)}", (diag_x, diag_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 210, 220), 1, cv2.LINE_AA)

    def _update_mode_diagnostics(self, frame: np.ndarray) -> None:
        """Handles Mode 4: Comprehensive Hardware Detection & System Health."""
        h, w = frame.shape[:2]
        bx, by = 40, 60
        bw, bh = w - 80, h - 100

        scrim = frame[by:by + bh, bx:bx + bw]
        tint = np.full_like(scrim, (15, 20, 25))
        cv2.addWeighted(tint, 0.85, scrim, 0.15, 0, scrim)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), CYAN, 1, cv2.LINE_AA)

        cv2.putText(frame, "=== PHANTOM LAB SYSTEM DIAGNOSTICS ===", (bx + 20, by + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CYAN, 1, cv2.LINE_AA)

        sys_info = self.cached_devices
        cams = sys_info.get("cameras", [])
        ser_devs = sys_info.get("serial_devices", [])
        audio = sys_info.get("audio", {})

        lines = [
            f"OS: {sys_info.get('os')} ({sys_info.get('arch')}) | Python: {sys_info.get('python_version')}",
            f"OpenCV: {sys_info.get('opencv_version')} | MediaPipe: {sys_info.get('mediapipe_version')}",
            f"Detected Video Cameras: {len(cams)} -> {[c['name'] for c in cams]}",
            f"Audio Subsystem: {audio.get('status')} (Device count: {audio.get('device_count', 0)})",
            f"Serial / USB Ports for Haptics: {len(ser_devs)} -> {[s['port'] for s in ser_devs] or 'None detected'}",
            f"Haptic Device Backend: {'PHYSICAL TRANSDUCER' if self.haptics.is_hardware() else 'HARDWARE-INDEPENDENT SIMULATOR'}",
            f"Plugin Extensions: {[p.name for p in self.plugin_manager.plugins]}",
            f"Active Accessibility: High-Contrast={self.a11y.high_contrast}, Left-Hand={self.a11y.left_hand_dominant}",
            f"Watchdog Frame Dropouts: {self.reconnect_attempts} events recovered",
        ]

        ty = by + 65
        for line in lines:
            cv2.putText(frame, line, (bx + 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
            ty += 28


def main() -> None:
    app = PhantomLabApp()
    app.run()


if __name__ == "__main__":
    main()
