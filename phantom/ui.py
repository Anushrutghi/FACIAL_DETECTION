"""PHANTOM Futuristic Spatial UI & HUD Renderer.

Renders glowing holographic landmarks, sci-fi reticles, touchless on-screen
action buttons, diagnostic telemetry (FPS, latency, confidence), and calibration guides.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from phantom.config import PhantomConfig


# Theme color constants (BGR)
CYAN_GLOW = (255, 220, 0)
NEON_GREEN = (0, 255, 128)
AMBER_ALERT = (0, 165, 255)
MAGENTA = (255, 0, 220)
HUD_BG = (20, 24, 28)
WHITE = (255, 255, 255)
DARK_GRAY = (45, 50, 55)


class PhantomHUD:
    """Renders cybernetic holographic UI and telemetry."""

    def __init__(self, shape: Tuple[int, int, int], config: Optional[PhantomConfig] = None) -> None:
        self.config = config or PhantomConfig()
        self.h, self.w = shape[:2]

        self.buttons: List[Dict[str, Any]] = self._build_buttons()
        self.button_cooldown = 0.4
        self.last_button_click_time = 0.0

        # Calibration state
        self.calibrating = False
        self.calibration_scale = 1.0

    def _build_buttons(self) -> List[Dict[str, Any]]:
        buttons = []
        palette = self.config.get("drawing", "palette", [])

        bw, bh = 85, 36
        margin = 12
        rx = self.w - margin - bw
        ry = margin + 10

        # Color palette buttons
        for idx, col in enumerate(palette):
            buttons.append({
                "id": f"color_{idx}",
                "rect": (rx, ry, bw, bh),
                "type": "color",
                "index": idx,
                "label": col["name"][:7],
                "color": tuple(col["bgr"]),
            })
            ry += bh + 8

        # Eraser button
        buttons.append({
            "id": "btn_eraser",
            "rect": (rx, ry, bw, bh),
            "type": "eraser",
            "label": "ERASER",
            "color": (180, 180, 180),
        })
        ry += bh + 8

        # Clear button
        buttons.append({
            "id": "btn_clear",
            "rect": (rx, ry, bw, bh),
            "type": "clear",
            "label": "CLEAR",
            "color": (60, 60, 220),
        })
        ry += bh + 8

        # Undo button
        buttons.append({
            "id": "btn_undo",
            "rect": (rx, ry, bw, bh),
            "type": "undo",
            "label": "UNDO",
            "color": (200, 140, 40),
        })
        ry += bh + 8

        # Calibration button
        buttons.append({
            "id": "btn_calib",
            "rect": (rx, ry, bw, bh),
            "type": "calibrate",
            "label": "CALIB",
            "color": (120, 200, 120),
        })

        return buttons

    def draw_holographic_landmarks(self, frame: np.ndarray, hand_states: List[Dict[str, Any]]) -> None:
        """Renders futuristic glowing skeleton overlay."""
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (5, 9), (9, 10), (10, 11), (11, 12),       # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),     # Ring
            (13, 17), (17, 18), (18, 19), (19, 20),    # Pinky
            (0, 17),                                  # Palm base
        ]

        for state in hand_states:
            pts = state.get("landmarks_px", [])
            if not pts or len(pts) < 21:
                continue

            # Draw bones
            for p1_idx, p2_idx in connections:
                cv2.line(frame, pts[p1_idx], pts[p2_idx], (120, 180, 230), 1, cv2.LINE_AA)

            # Draw joints
            for i, pt in enumerate(pts):
                # Highlight fingertips
                if i in (4, 8, 12, 16, 20):
                    cv2.circle(frame, pt, 5, CYAN_GLOW, -1, cv2.LINE_AA)
                    cv2.circle(frame, pt, 7, WHITE, 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, pt, 3, (180, 120, 80), -1, cv2.LINE_AA)

    def draw_cursor_reticle(self, frame: np.ndarray, state: Dict[str, Any]) -> None:
        """Draws dynamic sci-fi reticle on index fingertip with pinch progress ring."""
        pos = state["index_tip"]
        pinch = state["pinch"]
        strength = state.get("pinch_strength", 0.0)
        gesture = state.get("gesture", "UNKNOWN")

        color = AMBER_ALERT if pinch else CYAN_GLOW
        r = 16

        # Outer crosshair notches
        cv2.circle(frame, pos, r, color, 1, cv2.LINE_AA)
        cv2.line(frame, (pos[0] - r - 4, pos[1]), (pos[0] - r + 2, pos[1]), color, 1, cv2.LINE_AA)
        cv2.line(frame, (pos[0] + r - 2, pos[1]), (pos[0] + r + 4, pos[1]), color, 1, cv2.LINE_AA)
        cv2.line(frame, (pos[0], pos[1] - r - 4), (pos[0], pos[1] - r + 2), color, 1, cv2.LINE_AA)
        cv2.line(frame, (pos[0], pos[1] + r - 2), (pos[0], pos[1] + r + 4), color, 1, cv2.LINE_AA)

        # Center dot
        cv2.circle(frame, pos, 3, WHITE if not pinch else (0, 0, 255), -1, cv2.LINE_AA)

        # Pinch progress arc (0 to 360 deg)
        if strength > 0.05:
            end_angle = int(360 * strength)
            cv2.ellipse(frame, pos, (r + 4, r + 4), -90, 0, end_angle, (0, 255, 255), 2, cv2.LINE_AA)

        # Badge under cursor
        label = f"{state['handedness'][0]}:{gesture}"
        cv2.putText(frame, label, (pos[0] + 12, pos[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

    def draw_buttons(
        self,
        frame: np.ndarray,
        cursor_pos: Optional[Tuple[int, int]],
        active_color_idx: int,
        eraser_active: bool,
    ) -> Optional[Dict[str, Any]]:
        """Renders futuristic translucent buttons and checks for hover/click."""
        hovered_button = None

        for btn in self.buttons:
            x, y, w, h = btn["rect"]
            is_hover = False
            if cursor_pos and (x <= cursor_pos[0] <= x + w and y <= cursor_pos[1] <= y + h):
                is_hover = True
                hovered_button = btn

            # Semi-transparent backing
            sub = frame[y:y + h, x:x + w]
            bg_color = np.array(btn["color"], dtype=np.uint8)
            tint = np.full_like(sub, bg_color)
            alpha = 0.55 if not is_hover else 0.85
            cv2.addWeighted(tint, alpha, sub, 1.0 - alpha, 0, sub)

            # Border
            border_color = WHITE if is_hover else (150, 160, 170)
            thickness = 2 if is_hover else 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, thickness, cv2.LINE_AA)

            # Active indicators
            if btn["type"] == "color" and btn["index"] == active_color_idx:
                cv2.rectangle(frame, (x + 3, y + 3), (x + w - 3, y + h - 3), WHITE, 2, cv2.LINE_AA)
            elif btn["type"] == "eraser" and eraser_active:
                cv2.rectangle(frame, (x + 3, y + 3), (x + w - 3, y + h - 3), (0, 0, 255), 2, cv2.LINE_AA)

            # Text
            label = btn["label"]
            cv2.putText(frame, label, (x + 10, y + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        return hovered_button

    def draw_telemetry(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        hand_states: List[Dict[str, Any]],
        brush_size: int,
        paused: bool = False,
    ) -> None:
        """Renders cybernetic diagnostic telemetry bar in top-left corner."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (12, 12), (320, 125), HUD_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (12, 12), (320, 125), CYAN_GLOW, 1, cv2.LINE_AA)

        # Header
        cv2.putText(frame, "PHANTOM VISION // v1.0", (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN_GLOW, 1, cv2.LINE_AA)

        # FPS & Latency
        fps_str = f"FPS: {fps:.1f} | LATENCY: {latency_ms:.1f}ms"
        cv2.putText(frame, fps_str, (22, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        # Mode status
        mode_str = "STATUS: PAUSED (PALM)" if paused else "STATUS: ACTIVE TRACKING"
        mode_col = AMBER_ALERT if paused else NEON_GREEN
        cv2.putText(frame, mode_str, (22, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mode_col, 1, cv2.LINE_AA)

        # Hand detection count & brush size
        hands_str = f"HANDS: {len(hand_states)} | BRUSH: {brush_size}px"
        cv2.putText(frame, hands_str, (22, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 220), 1, cv2.LINE_AA)

        # Gestures per hand
        if hand_states:
            g_list = [f"H{s['hand_index']}({s['handedness'][0]}): {s['gesture']}" for s in hand_states]
            cv2.putText(frame, " | ".join(g_list), (22, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CYAN_GLOW, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "NO HANDS IN SPATIAL VOLUME", (22, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    def draw_calibration_overlay(self, frame: np.ndarray, hand_states: List[Dict[str, Any]]) -> None:
        """Renders interactive calibration targets and sensitivity indicators."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # Semi-transparent scrim
        scrim = frame.copy()
        cv2.rectangle(scrim, (0, 0), (w, h), (10, 15, 20), -1)
        cv2.addWeighted(scrim, 0.6, frame, 0.4, 0, frame)

        # Calibration target circle
        target_r = int(120 * self.calibration_scale)
        cv2.circle(frame, (cx, cy), target_r, CYAN_GLOW, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 8, AMBER_ALERT, -1, cv2.LINE_AA)
        cv2.line(frame, (cx - target_r - 20, cy), (cx + target_r + 20, cy), CYAN_GLOW, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - target_r - 20), (cx, cy + target_r + 20), CYAN_GLOW, 1, cv2.LINE_AA)

        # Instructions
        lines = [
            "=== PHANTOM CALIBRATION MODE ===",
            "Position open hand inside the reticle.",
            "Press [C] or button to exit calibration.",
            "Use [=] / [-] to adjust sensitivity scale.",
            f"Current Scale: {self.calibration_scale:.2f}x",
        ]
        ty = cy - target_r - 50
        for line in lines:
            cv2.putText(frame, line, (cx - 180, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
            ty += 24

        # If hand present, display measured scale
        if hand_states:
            measured_scale = hand_states[0].get("palm_scale", 0.0)
            cv2.putText(
                frame,
                f"Detected Palm Scale: {measured_scale:.1f}px",
                (cx - 120, cy + target_r + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                NEON_GREEN,
                1,
                cv2.LINE_AA,
            )
