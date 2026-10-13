"""PHANTOM Lab Accessibility Profile & Inclusive Spatial Controls.

Supports high-contrast visual themes, enlarged reticles, left-hand dominance,
tremor-reduction filtering, and auditory tactile substitution.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class AccessibilityProfile:
    """Configurable accessibility options for inclusive touchless interaction."""
    high_contrast: bool = False
    left_hand_dominant: bool = False
    enlarged_reticle: bool = False
    audio_cues_enabled: bool = True
    tremor_reduction_level: int = 1  # 0: None, 1: Normal, 2: High Smoothing

    def get_cursor_radius(self, default_radius: int = 16) -> int:
        return int(default_radius * 1.6) if self.enlarged_reticle else default_radius

    def get_hud_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Returns color scheme based on contrast settings."""
        if self.high_contrast:
            return {
                "primary": (0, 255, 255),    # Vibrant Yellow
                "secondary": (255, 255, 255), # Pure White
                "accent": (0, 128, 255),     # High-vis Orange
                "background": (0, 0, 0),     # Solid Black
                "text": (255, 255, 255),
            }
        return {
            "primary": (255, 220, 0),        # Cyan Glow
            "secondary": (0, 255, 128),      # Neon Green
            "accent": (0, 165, 255),         # Amber Alert
            "background": (20, 24, 28),      # Cyber Dark
            "text": (255, 255, 255),
        }

    def get_smoothing_alpha(self, default_alpha: float = 0.25) -> float:
        """Calculates adjusted EMA alpha factor based on tremor reduction level."""
        if self.tremor_reduction_level == 2:
            return 0.12  # Very smooth (filters tremor)
        elif self.tremor_reduction_level == 0:
            return 0.45  # Direct / raw response
        return default_alpha

    def prioritize_hand(self, hand_states: list) -> list:
        """Re-orders hand list so the user's dominant hand comes first."""
        if not self.left_hand_dominant or len(hand_states) <= 1:
            return hand_states

        # Sort so Left hand is index 0
        return sorted(hand_states, key=lambda h: 0 if h.get("handedness") == "Left" else 1)
