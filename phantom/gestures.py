"""PHANTOM Gesture Recognition and Event Engine.

Converts continuous hand telemetry and debounced gestures into discrete
and continuous spatial computing interaction events.
"""

from __future__ import annotations
import time
from typing import Any, Dict, List, Optional, Tuple

from phantom.config import PhantomConfig


class GestureEngine:
    """Dispatches spatial interaction events from hand states."""

    def __init__(self, config: Optional[PhantomConfig] = None) -> None:
        self.config = config or PhantomConfig()
        self.drag_delay = self.config.get("gestures", "pinch_drag_delay_sec", 0.35)
        self.palm_pause_enabled = self.config.get("gestures", "palm_pause_enabled", True)

        # Interaction state tracking per hand
        self._is_dragging: Dict[int, bool] = {0: False, 1: False}
        self._last_click_time: Dict[int, float] = {0: 0.0, 1: 0.0}
        self._click_cooldown = 0.25

    def update(self, hand_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluates hand states and generates interaction events.

        Returns a list of event dictionaries:
            {
                "type": "CLICK" | "DRAG_START" | "DRAG_MOVE" | "DRAG_END" | "DRAW" | "PAUSE" | "ERASE" | "CLEAR" | "HOVER",
                "hand_index": int,
                "position": (x, y),
                "data": dict
            }
        """
        events: List[Dict[str, Any]] = []
        now = time.time()

        for state in hand_states:
            idx = state["hand_index"]
            pos = state["index_tip"]
            pinch = state["pinch"]
            gesture = state["gesture"]
            pinch_start = state.get("pinch_start_time", 0.0)

            # 1. Open-palm pause check
            if self.palm_pause_enabled and gesture == "PALM":
                events.append({
                    "type": "PAUSE",
                    "hand_index": idx,
                    "position": pos,
                    "data": {"handedness": state["handedness"]},
                })
                continue

            # 2. Fist action (e.g., clear or grab)
            if gesture == "FIST":
                events.append({
                    "type": "CLEAR",
                    "hand_index": idx,
                    "position": pos,
                    "data": {},
                })
                continue

            # 3. Peace sign / Two fingers (e.g., eraser or menu)
            if gesture == "PEACE":
                events.append({
                    "type": "ERASE",
                    "hand_index": idx,
                    "position": pos,
                    "data": {},
                })
                continue

            # 4. Pinch click and drag state machine
            was_dragging = self._is_dragging.get(idx, False)

            if pinch:
                pinch_duration = now - pinch_start if pinch_start > 0 else 0.0
                if pinch_duration >= self.drag_delay:
                    if not was_dragging:
                        self._is_dragging[idx] = True
                        events.append({
                            "type": "DRAG_START",
                            "hand_index": idx,
                            "position": pos,
                            "data": {"pinch_strength": state["pinch_strength"]},
                        })
                    else:
                        events.append({
                            "type": "DRAG_MOVE",
                            "hand_index": idx,
                            "position": pos,
                            "data": {"pinch_strength": state["pinch_strength"]},
                        })
                else:
                    # Hovering in pre-drag pinch
                    events.append({
                        "type": "HOVER",
                        "hand_index": idx,
                        "position": pos,
                        "data": {"pinch_strength": state["pinch_strength"]},
                    })
            else:
                # Pinch was released
                if was_dragging:
                    self._is_dragging[idx] = False
                    events.append({
                        "type": "DRAG_END",
                        "hand_index": idx,
                        "position": pos,
                        "data": {},
                    })
                elif pinch_start > 0.0 and (now - pinch_start) < self.drag_delay:
                    # Quick pinch release = Click
                    if now - self._last_click_time.get(idx, 0.0) > self._click_cooldown:
                        self._last_click_time[idx] = now
                        events.append({
                            "type": "CLICK",
                            "hand_index": idx,
                            "position": pos,
                            "data": {},
                        })

                # Normal drawing check when index is up and not pinching
                if state["index_up"] and not state["palm"] and gesture in ("ONE", "UNKNOWN"):
                    events.append({
                        "type": "DRAW",
                        "hand_index": idx,
                        "position": pos,
                        "data": {"color_hint": idx},
                    })
                else:
                    events.append({
                        "type": "HOVER",
                        "hand_index": idx,
                        "position": pos,
                        "data": {},
                    })

        # Dual hand pinch check (two hands pinching at once = two-hand scale/rotate)
        if len(hand_states) >= 2:
            h0, h1 = hand_states[0], hand_states[1]
            if h0["pinch"] and h1["pinch"]:
                p0 = h0["index_tip"]
                p1 = h1["index_tip"]
                dist = ((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2) ** 0.5
                events.append({
                    "type": "DUAL_PINCH",
                    "hand_index": -1,
                    "position": ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2),
                    "data": {"distance": dist, "p0": p0, "p1": p1},
                })

        return events
