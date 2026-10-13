"""Automated Unit & Integration Test Suite for PHANTOM Lab (Phase 4)."""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.lab.plugin import PhantomPlugin, PluginManager, TelemetryLoggerPlugin
from phantom.lab.device_detector import DeviceDetector
from phantom.lab.accessibility import AccessibilityProfile


class DummyPlugin(PhantomPlugin):
    """Test plugin to verify lifecycle hooks."""

    def __init__(self):
        super().__init__("DummyPlugin")
        self.frame_count = 0
        self.hands_count = 0
        self.events_count = 0

    def on_frame(self, frame):
        self.frame_count += 1
        return frame

    def on_hands(self, hand_states):
        self.hands_count += len(hand_states)

    def on_event(self, event):
        self.events_count += 1


class TestPhantomLab(unittest.TestCase):
    """Verifies all components of Phase 4 - PHANTOM Lab."""

    def test_01_device_detector_summary(self):
        """Test system and hardware device probing."""
        summary = DeviceDetector.get_system_summary()
        self.assertIn("os", summary)
        self.assertIn("python_version", summary)
        self.assertIn("opencv_version", summary)
        self.assertIn("mediapipe_version", summary)
        self.assertIn("cameras", summary)
        self.assertIn("serial_devices", summary)
        self.assertIn("audio", summary)

    def test_02_accessibility_profile(self):
        """Test high-contrast themes, tremor smoothing, and left-hand dominance."""
        a11y = AccessibilityProfile()

        # Default state
        self.assertFalse(a11y.high_contrast)
        self.assertEqual(a11y.get_cursor_radius(16), 16)
        self.assertAlmostEqual(a11y.get_smoothing_alpha(0.25), 0.25)

        # High-contrast theme
        a11y.high_contrast = True
        colors = a11y.get_hud_colors()
        self.assertEqual(colors["background"], (0, 0, 0))

        # Enlarged reticle
        a11y.enlarged_reticle = True
        self.assertGreater(a11y.get_cursor_radius(16), 16)

        # Tremor reduction
        a11y.tremor_reduction_level = 2
        self.assertLess(a11y.get_smoothing_alpha(0.25), 0.20)

        # Left-hand dominance
        a11y.left_hand_dominant = True
        hands = [
            {"hand_index": 0, "handedness": "Right"},
            {"hand_index": 1, "handedness": "Left"},
        ]
        reordered = a11y.prioritize_hand(hands)
        self.assertEqual(reordered[0]["handedness"], "Left")

    def test_03_plugin_architecture(self):
        """Test plugin manager lifecycle dispatch."""
        mgr = PluginManager()
        plugin = DummyPlugin()
        mgr.register(plugin)

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mgr.dispatch_frame(frame)
        self.assertEqual(plugin.frame_count, 1)

        mgr.dispatch_hands([{"hand_index": 0}])
        self.assertEqual(plugin.hands_count, 1)

        mgr.dispatch_event({"type": "CLICK"})
        self.assertEqual(plugin.events_count, 1)

        mgr.unload_all()
        self.assertEqual(len(mgr.plugins), 0)

    def test_04_telemetry_logger_plugin(self):
        """Test telemetry logger plugin writing to scratch log."""
        tmp_log = os.path.join(PROJECT_ROOT, "logs", "test_telemetry.log")
        logger = TelemetryLoggerPlugin(tmp_log)
        logger._last_log_time = 0.0  # Force sample
        logger.on_hands([{"gesture": "ONE", "hand_index": 0}])

        if os.path.exists(tmp_log):
            os.remove(tmp_log)


if __name__ == "__main__":
    unittest.main()
