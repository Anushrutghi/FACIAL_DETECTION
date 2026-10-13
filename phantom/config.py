"""PHANTOM Configuration Manager.

Loads, validates, and persists configuration for PHANTOM modules.
Falls back gracefully to default values if the configuration file is missing or corrupted.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "phantom_config.json"
)

DEFAULT_CONFIG: Dict[str, Any] = {
    "tracking": {
        "max_hands": 2,
        "detection_confidence": 0.7,
        "tracking_confidence": 0.7,
        "smoothing_alpha": 0.25,
        "smoothing_alpha_draw": 0.35,
        "debounce_frames": 3,
    },
    "gestures": {
        "pinch_threshold_ratio": 0.32,
        "pinch_release_ratio": 0.42,
        "pinch_drag_delay_sec": 0.35,
        "palm_pause_enabled": True,
        "fist_action": "clear",
        "open_palm_action": "pause",
        "peace_action": "eraser",
        "two_fingers_action": "menu",
    },
    "drawing": {
        "default_thickness": 8,
        "min_thickness": 2,
        "max_thickness": 36,
        "default_color_idx": 0,
        "palette": [
            {"name": "CYAN", "bgr": [255, 220, 0]},
            {"name": "NEON_GREEN", "bgr": [0, 255, 128]},
            {"name": "MAGENTA", "bgr": [255, 0, 220]},
            {"name": "AMBER", "bgr": [0, 165, 255]},
            {"name": "ELECTRIC_BLUE", "bgr": [255, 100, 0]},
            {"name": "WHITE", "bgr": [255, 255, 255]},
        ],
        "two_hand_dual_draw": True,
    },
    "hud": {
        "theme": "phantom_scifi",
        "show_landmarks": True,
        "show_reticle": True,
        "show_telemetry": True,
        "reticle_radius": 14,
    },
    "calibration": {
        "sensitivity_x": 1.0,
        "sensitivity_y": 1.0,
        "deadzone": 0.02,
    },
}


class PhantomConfig:
    """Manages system configuration settings."""

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = config_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                return self._merge_defaults(DEFAULT_CONFIG, loaded)
            except Exception as e:
                print(f"[PHANTOM CONFIG] Warning: failed to load {self.config_path}: {e}. Using defaults.")
        return dict(DEFAULT_CONFIG)

    def _merge_defaults(self, defaults: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(defaults)
        for k, v in loaded.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._merge_defaults(merged[k], v)
            else:
                merged[k] = v
        return merged

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        return self.data.get(section, {}).get(key, fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        if section not in self.data:
            self.data[section] = {}
        self.data[section][key] = value

    def save(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            return True
        except Exception as e:
            print(f"[PHANTOM CONFIG] Error saving configuration: {e}")
            return False
