"""Automated Unit & Integration Test Suite for PHANTOM Vision (Phase 1)."""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.config import PhantomConfig
from phantom.tracker import HandTracker, detect_hand_states
from phantom.gestures import GestureEngine
from phantom.canvas import SpatialCanvas
from phantom.ui import PhantomHUD


class TestPhantomVision(unittest.TestCase):
    """Verifies all components of Phase 1 - PHANTOM Vision."""

    def setUp(self):
        self.config = PhantomConfig()
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_01_config_manager(self):
        """Test configuration loading, defaults, and overrides."""
        cfg = PhantomConfig()
        self.assertEqual(cfg.get("tracking", "max_hands"), 2)
        self.assertAlmostEqual(cfg.get("gestures", "pinch_threshold_ratio"), 0.32)
        self.assertTrue(cfg.get("gestures", "palm_pause_enabled"))
        self.assertEqual(cfg.get("drawing", "default_thickness"), 8)

        # Test set and get
        cfg.set("tracking", "test_param", 42)
        self.assertEqual(cfg.get("tracking", "test_param"), 42)

    def test_02_tracker_initialization_and_empty_frame(self):
        """Test tracker process on empty frame."""
        tracker = HandTracker(config=self.config)
        states = tracker.process(self.dummy_frame, draw=False)
        self.assertIsInstance(states, list)
        self.assertEqual(len(states), 0)
        self.assertGreaterEqual(tracker.last_inference_ms, 0.0)

    def test_03_tracker_smoothing_and_debouncing(self):
        """Test math and smoothing helpers."""
        tracker = HandTracker(config=self.config)
        p1 = tracker._smooth("test_key", (100, 100), alpha=0.5)
        self.assertEqual(p1, (100, 100))
        p2 = tracker._smooth("test_key", (200, 200), alpha=0.5)
        self.assertEqual(p2, (150, 150))

        # Test gesture debounce
        g1 = tracker._debounce_gesture(0, "PINCH")
        g2 = tracker._debounce_gesture(0, "PINCH")
        g3 = tracker._debounce_gesture(0, "PINCH")
        self.assertEqual(g3, "PINCH")

    def test_04_gesture_engine_events(self):
        """Test event dispatching for gestures."""
        engine = GestureEngine(config=self.config)

        # 1. Palm pause
        palm_state = [{
            "hand_index": 0,
            "handedness": "Right",
            "finger_count": 5,
            "gesture": "PALM",
            "palm": True,
            "index_up": True,
            "index_tip": (320, 240),
            "pinch": False,
        }]
        events = engine.update(palm_state)
        self.assertTrue(any(e["type"] == "PAUSE" for e in events))

        # 2. Air draw event
        draw_state = [{
            "hand_index": 0,
            "handedness": "Right",
            "finger_count": 1,
            "gesture": "ONE",
            "palm": False,
            "index_up": True,
            "index_tip": (320, 240),
            "pinch": False,
        }]
        events = engine.update(draw_state)
        self.assertTrue(any(e["type"] == "DRAW" for e in events))

        # 3. Fist clear event
        fist_state = [{
            "hand_index": 0,
            "handedness": "Right",
            "finger_count": 0,
            "gesture": "FIST",
            "palm": False,
            "index_up": False,
            "index_tip": (320, 240),
            "pinch": False,
        }]
        events = engine.update(fist_state)
        self.assertTrue(any(e["type"] == "CLEAR" for e in events))

        # 4. Dual hand pinch event
        dual_state = [
            {
                "hand_index": 0,
                "handedness": "Right",
                "finger_count": 2,
                "gesture": "PINCH",
                "palm": False,
                "index_up": True,
                "index_tip": (100, 100),
                "pinch": True,
                "pinch_strength": 1.0,
                "pinch_start_time": 0.0,
            },
            {
                "hand_index": 1,
                "handedness": "Left",
                "finger_count": 2,
                "gesture": "PINCH",
                "palm": False,
                "index_up": True,
                "index_tip": (200, 200),
                "pinch": True,
                "pinch_strength": 1.0,
                "pinch_start_time": 0.0,
            },
        ]
        events = engine.update(dual_state)
        self.assertTrue(any(e["type"] == "DUAL_PINCH" for e in events))

    def test_05_spatial_canvas(self):
        """Test drawing canvas, undo, eraser, and color selection."""
        canvas = SpatialCanvas((480, 640, 3), config=self.config)

        # Draw a stroke
        canvas.draw_point(0, (100, 100))
        canvas.draw_point(0, (110, 110))
        self.assertTrue(np.any(canvas.canvas != 0))

        # Test overlay blending
        out = canvas.overlay(self.dummy_frame)
        self.assertEqual(out.shape, self.dummy_frame.shape)

        # Test brush resizing
        s1 = canvas.brush_size
        s2 = canvas.adjust_brush_size(4)
        self.assertEqual(s2, s1 + 4)

        # Test eraser
        self.assertFalse(canvas.eraser_active)
        canvas.toggle_eraser()
        self.assertTrue(canvas.eraser_active)

        # Test clear and undo
        canvas.clear()
        self.assertFalse(np.any(canvas.canvas != 0))
        canvas.undo()
        self.assertTrue(np.any(canvas.canvas != 0))

    def test_06_hud_and_reticles(self):
        """Test UI button generation, reticle, and telemetry rendering."""
        hud = PhantomHUD((480, 640, 3), config=self.config)
        self.assertGreater(len(hud.buttons), 0)

        frame = self.dummy_frame.copy()

        # Telemetry
        hud.draw_telemetry(frame, fps=30.0, latency_ms=12.5, hand_states=[], brush_size=8)

        # Reticle on synthetic hand state
        syn_hand = {
            "hand_index": 0,
            "handedness": "Right",
            "index_tip": (300, 200),
            "pinch": True,
            "pinch_strength": 0.8,
            "gesture": "PINCH",
            "landmarks_px": [(300, 200)] * 21,
        }
        hud.draw_holographic_landmarks(frame, [syn_hand])
        hud.draw_cursor_reticle(frame, syn_hand)

        # Button hover
        hovered = hud.draw_buttons(frame, cursor_pos=hud.buttons[0]["rect"][:2], active_color_idx=0, eraser_active=False)
        self.assertIsNotNone(hovered)

        # Calibration overlay
        hud.calibrating = True
        hud.draw_calibration_overlay(frame, [syn_hand])

    def test_07_legacy_compatibility(self):
        """Ensure backward compatibility with detect_hand_states signature."""
        states = detect_hand_states(self.dummy_frame, draw=False)
        self.assertIsInstance(states, list)


if __name__ == "__main__":
    unittest.main()
