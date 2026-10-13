"""PHANTOM Spatial Air-Drawing Canvas.

High-fidelity drawing surface with smooth stroke interpolation,
multi-hand dual-color drawing, variable brush radius, eraser mode,
and alpha-blended futuristic compositing.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from phantom.config import PhantomConfig


class SpatialCanvas:
    """Manages virtual spatial drawing strokes and rendering."""

    def __init__(self, shape: Tuple[int, int, int], config: Optional[PhantomConfig] = None) -> None:
        self.config = config or PhantomConfig()
        self.shape = shape
        self.h, self.w = shape[:2]

        # Canvas layers
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        # Drawing settings
        palette_conf = self.config.get("drawing", "palette", [])
        self.palette = [tuple(p["bgr"]) for p in palette_conf] or [(255, 220, 0), (0, 255, 128)]
        self.palette_names = [p["name"] for p in palette_conf] or ["CYAN", "NEON_GREEN"]

        self.color_index_primary = 0
        self.color_index_secondary = 1 if len(self.palette) > 1 else 0

        self.brush_size = self.config.get("drawing", "default_thickness", 8)
        self.min_size = self.config.get("drawing", "min_thickness", 2)
        self.max_size = self.config.get("drawing", "max_thickness", 36)

        self.eraser_active = False
        self.eraser_radius = 24

        # Stroke tracking: hand_id -> list of points in current continuous stroke
        self._prev_points: Dict[int, Optional[Tuple[int, int]]] = {}
        self._history_stack: List[np.ndarray] = []
        self._max_history = 10

    def get_primary_color(self) -> Tuple[int, int, int]:
        return self.palette[self.color_index_primary]

    def get_secondary_color(self) -> Tuple[int, int, int]:
        return self.palette[self.color_index_secondary]

    def set_color_index(self, idx: int, secondary: bool = False) -> None:
        idx = max(0, min(len(self.palette) - 1, idx))
        if secondary:
            self.color_index_secondary = idx
        else:
            self.color_index_primary = idx
            self.eraser_active = False

    def toggle_eraser(self) -> bool:
        self.eraser_active = not self.eraser_active
        return self.eraser_active

    def adjust_brush_size(self, delta: int) -> int:
        self.brush_size = max(self.min_size, min(self.max_size, self.brush_size + delta))
        return self.brush_size

    def clear(self) -> None:
        self._save_history()
        self.canvas[:] = 0
        self._prev_points.clear()

    def _save_history(self) -> None:
        if len(self._history_stack) >= self._max_history:
            self._history_stack.pop(0)
        self._history_stack.append(self.canvas.copy())

    def undo(self) -> bool:
        if self._history_stack:
            self.canvas = self._history_stack.pop()
            self._prev_points.clear()
            return True
        return False

    def draw_point(self, hand_id: int, current_point: Tuple[int, int]) -> None:
        """Adds a continuous stroke segment for the specified hand."""
        prev = self._prev_points.get(hand_id)

        if self.eraser_active:
            # Erase by clearing circular region around fingertip
            cv2.circle(self.canvas, current_point, self.eraser_radius, (0, 0, 0), -1)
            if prev is not None:
                cv2.line(self.canvas, prev, current_point, (0, 0, 0), self.eraser_radius * 2)
        else:
            color = self.get_primary_color() if hand_id == 0 else self.get_secondary_color()
            radius = self.brush_size

            if prev is not None:
                # Interpolate line segment with round caps
                cv2.line(self.canvas, prev, current_point, color, radius)
                cv2.circle(self.canvas, current_point, radius // 2, color, -1)
            else:
                self._save_history()
                cv2.circle(self.canvas, current_point, radius // 2, color, -1)

        self._prev_points[hand_id] = current_point

    def stop_stroke(self, hand_id: int) -> None:
        """Notifies the canvas that the current stroke has lifted."""
        self._prev_points.pop(hand_id, None)

    def overlay(self, frame: np.ndarray, alpha: float = 0.85) -> np.ndarray:
        """Alpha-blends glowing canvas content over the incoming video frame."""
        mask = np.any(self.canvas != 0, axis=2)
        if not np.any(mask):
            return frame

        out = frame.copy()
        # Create subtle glowing bloom on drawing
        blurred = cv2.GaussianBlur(self.canvas, (9, 9), 0)
        bloom = cv2.addWeighted(self.canvas, 0.7, blurred, 0.3, 0)

        out[mask] = cv2.addWeighted(frame[mask], 1.0 - alpha, bloom[mask], alpha, 0)
        return out
