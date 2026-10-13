"""PHANTOM Vision Hand Tracker.

High-performance real-time hand tracker wrapping MediaPipe with dual-hand support,
scale-invariant pinch detection with hysteresis, 21-landmark extraction,
exponential moving average smoothing, and gesture debouncing.
Fully backward-compatible with legacy `detect_hand_states` outputs.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from phantom.config import PhantomConfig


class HandTracker:
    """Robust multi-hand tracker and spatial analyzer."""

    def __init__(
        self,
        config: Optional[PhantomConfig] = None,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
    ) -> None:
        self.config = config or PhantomConfig()

        self.max_hands = self.config.get("tracking", "max_hands", max_num_hands)
        det_conf = self.config.get("tracking", "detection_confidence", min_detection_confidence)
        track_conf = self.config.get("tracking", "tracking_confidence", min_tracking_confidence)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Smoothing & hysteresis parameters
        self.smooth_alpha = self.config.get("tracking", "smoothing_alpha", 0.25)
        self.smooth_alpha_draw = self.config.get("tracking", "smoothing_alpha_draw", 0.35)
        self.debounce_frames = self.config.get("tracking", "debounce_frames", 3)

        self.pinch_thresh = self.config.get("gestures", "pinch_threshold_ratio", 0.32)
        self.pinch_release = self.config.get("gestures", "pinch_release_ratio", 0.42)

        # Internal state tracking per hand index (0 and 1)
        self._smoothed_points: Dict[str, Tuple[int, int]] = {}
        self._gesture_history: Dict[int, List[str]] = {0: [], 1: []}
        self._pinch_state: Dict[int, bool] = {0: False, 1: False}
        self._pinch_start_time: Dict[int, float] = {0: 0.0, 1: 0.0}

        # Telemetry
        self.last_inference_ms = 0.0

    def _smooth(self, key: str, target: Tuple[int, int], alpha: float) -> Tuple[int, int]:
        prev = self._smoothed_points.get(key)
        if prev is None:
            self._smoothed_points[key] = target
            return target
        sx = int(alpha * target[0] + (1.0 - alpha) * prev[0])
        sy = int(alpha * target[1] + (1.0 - alpha) * prev[1])
        res = (sx, sy)
        self._smoothed_points[key] = res
        return res

    def _debounce_gesture(self, hand_idx: int, gesture: str) -> str:
        history = self._gesture_history.setdefault(hand_idx, [])
        history.append(gesture)
        if len(history) > self.debounce_frames:
            history.pop(0)

        # Confirm gesture if consistent across debounce window
        if len(history) == self.debounce_frames and all(g == gesture for g in history):
            return gesture
        return history[-2] if len(history) >= 2 else gesture

    @staticmethod
    def _calc_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def process(self, frame: np.ndarray, draw: bool = False) -> List[Dict[str, Any]]:
        """Processes frame and returns comprehensive list of hand states."""
        t_start = time.perf_counter()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        self.last_inference_ms = (time.perf_counter() - t_start) * 1000.0

        if not results.multi_hand_landmarks:
            self._smoothed_points.clear()
            return []

        h, w = frame.shape[:2]
        hand_states: List[Dict[str, Any]] = []

        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            handedness = "Unknown"
            if results.multi_handedness and len(results.multi_handedness) > idx:
                handedness = results.multi_handedness[idx].classification[0].label

            lm = hand_lms.landmark

            # Extract 2D and 3D landmarks
            lms_px: List[Tuple[int, int]] = [(int(pt.x * w), int(pt.y * h)) for pt in lm]
            lms_norm: List[Tuple[float, float, float]] = [(pt.x, pt.y, pt.z) for pt in lm]

            # Scale-invariant hand scale: Wrist (0) to Middle MCP (9)
            palm_scale = max(10.0, self._calc_dist(lms_px[0], lms_px[9]))

            # Finger extension checks
            # Thumb: comparison depends on handedness
            if handedness == "Right":
                thumb_up = lm[4].x < lm[3].x
            else:
                thumb_up = lm[4].x > lm[3].x

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            fingers_up = [thumb_up, index_up, middle_up, ring_up, pinky_up]
            finger_count = sum(fingers_up)

            # Smooth key fingertips
            smoothed_index = self._smooth(f"h{idx}_idx", lms_px[8], self.smooth_alpha)
            smoothed_thumb = self._smooth(f"h{idx}_thumb", lms_px[4], self.smooth_alpha)
            smoothed_middle = self._smooth(f"h{idx}_mid", lms_px[12], self.smooth_alpha)

            # Pinch detection with hysteresis
            pinch_dist = self._calc_dist(lms_px[4], lms_px[8])
            pinch_ratio = pinch_dist / palm_scale

            curr_pinch = self._pinch_state.get(idx, False)
            if not curr_pinch:
                if pinch_ratio < self.pinch_thresh:
                    curr_pinch = True
                    self._pinch_start_time[idx] = time.time()
            else:
                if pinch_ratio > self.pinch_release:
                    curr_pinch = False
                    self._pinch_start_time[idx] = 0.0
            self._pinch_state[idx] = curr_pinch

            # Calculate pinch strength normalized (0.0 to 1.0)
            pinch_strength = max(0.0, min(1.0, 1.0 - (pinch_ratio - self.pinch_thresh) / (self.pinch_release - self.pinch_thresh + 1e-5)))

            # Palm center (mean of wrist and MCP joints)
            palm_pts = [lms_px[0], lms_px[5], lms_px[9], lms_px[13], lms_px[17]]
            palm_center = (
                int(sum(p[0] for p in palm_pts) / len(palm_pts)),
                int(sum(p[1] for p in palm_pts) / len(palm_pts)),
            )

            # Raw gesture classification
            palm = finger_count >= 4
            if curr_pinch:
                raw_gesture = "PINCH"
            elif finger_count == 0:
                raw_gesture = "FIST"
            elif palm:
                raw_gesture = "PALM"
            elif index_up and middle_up and not ring_up and not pinky_up:
                raw_gesture = "PEACE"
            elif finger_count == 1 and index_up:
                raw_gesture = "ONE"
            elif finger_count == 2 and index_up and middle_up:
                raw_gesture = "TWO"
            elif finger_count == 3:
                raw_gesture = "THREE"
            elif thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
                raw_gesture = "THUMB_UP"
            else:
                raw_gesture = "UNKNOWN"

            # Debounced gesture
            debounced_gesture = self._debounce_gesture(idx, raw_gesture)

            state: Dict[str, Any] = {
                "hand_index": idx,
                "handedness": handedness,
                "finger_count": finger_count,
                "fingers_up": fingers_up,
                "gesture": debounced_gesture,
                "raw_gesture": raw_gesture,
                "palm": palm,
                "index_up": index_up,
                "index_tip": smoothed_index,
                "thumb_tip": smoothed_thumb,
                "middle_tip": smoothed_middle,
                "wrist": lms_px[0],
                "palm_center": palm_center,
                "palm_scale": palm_scale,
                "pinch": curr_pinch,
                "pinch_ratio": pinch_ratio,
                "pinch_strength": pinch_strength,
                "pinch_start_time": self._pinch_start_time.get(idx, 0.0),
                "landmarks_px": lms_px,
                "landmarks_norm": lms_norm,
                "raw_landmarks": hand_lms,
            }

            if draw:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2),
                )

            hand_states.append(state)

        return hand_states


def detect_hand_states(frame: np.ndarray, draw: bool = True) -> List[Dict[str, Any]]:
    """Backward-compatible functional wrapper for legacy code."""
    global _default_tracker
    if "_default_tracker" not in globals() or _default_tracker is None:
        _default_tracker = HandTracker()
    return _default_tracker.process(frame, draw=draw)
