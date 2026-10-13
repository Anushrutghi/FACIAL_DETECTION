# PHANTOM Hand Tracker
"""Utility class that encapsulates MediaPipe hand detection and provides
smooth, debounced hand state information.

Typical usage::

    from phantom.tracker import HandTracker
    tracker = HandTracker()
    while True:
        hand_states = tracker.process(frame)
        # hand_states is a list of dicts compatible with the original
        # ``detect_hand_states`` output.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple

# Re‑use the original MediaPipe hand solution parameters
_mp_hands = mp.solutions.hands
_mp_draw = mp.solutions.drawing_utils

class HandTracker:
    """Wrap MediaPipe Hands and add smoothing / debouncing.

    Attributes
    ----------
    max_num_hands: int
        Maximum number of hands to track (default 2).
    smooth_alpha: float
        EMA coefficient for fingertip smoothing.
    debounce_frames: int
        Minimum number of consecutive frames a gesture must be stable before
        it is emitted.
    """

    def __init__(self, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.7,
                 smooth_alpha: float = 0.22,
                 debounce_frames: int = 3):
        self.hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.smooth_alpha = smooth_alpha
        self.debounce_frames = debounce_frames
        # internal state per hand index -> previous fingertip (x, y)
        self._prev_tip: Dict[int, Tuple[int, int]] = {}
        # gesture debounce counters
        self._gesture_counter: Dict[int, Tuple[str, int]] = {}

    def _smooth_point(self, prev: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
        if prev is None:
            return target
        return (
            int(self.smooth_alpha * target[0] + (1 - self.smooth_alpha) * prev[0]),
            int(self.smooth_alpha * target[1] + (1 - self.smooth_alpha) * prev[1]),
        )

    def _debounce_gesture(self, hand_id: int, gesture: str) -> str:
        """Return a stable gesture string after ``debounce_frames`` frames.
        If the gesture changes before the counter expires, reset.
        """
        prev_gesture, cnt = self._gesture_counter.get(hand_id, (gesture, 0))
        if gesture == prev_gesture:
            cnt = min(cnt + 1, self.debounce_frames)
        else:
            # reset counter for a new gesture
            prev_gesture = gesture
            cnt = 1
        self._gesture_counter[hand_id] = (prev_gesture, cnt)
        return prev_gesture if cnt >= self.debounce_frames else "UNKNOWN"

    def _thumb_up(self, lm, handedness_label: str) -> bool:
        if handedness_label == "Right":
            return lm[4].x < lm[3].x
        return lm[4].x > lm[3].x

    def _finger_up(self, lm, tip_idx, pip_idx) -> bool:
        return lm[tip_idx].y < lm[pip_idx].y

    def _mean_point(self, lm, indices, w, h):
        xs = [int(lm[i].x * w) for i in indices]
        ys = [int(lm[i].y * h) for i in indices]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def process(self, frame, draw: bool = True) -> List[Dict]:
        """Detect hands and return a list of state dictionaries.
        The output schema matches the legacy ``detect_hand_states`` function.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        h, w, _ = frame.shape
        hand_states = []
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            handedness = "Unknown"
            if result.multi_handedness and len(result.multi_handedness) > idx:
                handedness = result.multi_handedness[idx].classification[0].label
            if draw:
                _mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=3),
                    _mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2),
                )
            lm = hand_landmarks.landmark
            # Count fingers (thumb + 4 fingers)
            finger_count = 0
            if self._thumb_up(lm, handedness):
                finger_count += 1
            for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
                if self._finger_up(lm, tip, pip):
                    finger_count += 1
            # Determine basic gesture (mirrors original logic)
            if finger_count == 0:
                raw_gesture = "FIST"
            elif finger_count >= 4:
                raw_gesture = "PALM"
            elif finger_count == 1:
                raw_gesture = "ONE"
            elif finger_count == 2:
                raw_gesture = "TWO"
            elif finger_count == 3:
                raw_gesture = "THREE"
            elif finger_count == 4:
                raw_gesture = "FOUR"
            else:
                raw_gesture = "UNKNOWN"
            # Debounce gesture per hand
            gesture = self._debounce_gesture(idx, raw_gesture)
            # Index fingertip (used for cursor)
            tip_pt = (int(lm[8].x * w), int(lm[8].y * h))
            prev = self._prev_tip.get(idx)
            smoothed_tip = self._smooth_point(prev, tip_pt)
            self._prev_tip[idx] = smoothed_tip
            hand_states.append({
                "hand_index": idx,
                "handedness": handedness,
                "finger_count": finger_count,
                "gesture": gesture,
                "palm": finger_count >= 4,
                "index_up": self._finger_up(lm, 8, 6),
                "index_tip": smoothed_tip,
                "palm_center": self._mean_point(lm, [0, 5, 9, 13, 17], w, h),
            })
        return hand_states
