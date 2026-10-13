"""PHANTOM Touch – Software Haptic Simulator & Acoustic Field Visualizer.

Provides visual, auditory, and mathematical simulation of midair ultrasonic haptics
without requiring physical phased-array hardware.
"""

from __future__ import annotations
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from phantom.touch.interface import HapticDeviceInterface, HapticPattern


# Acoustic simulation physical constants (air at 20 deg C)
SPEED_OF_SOUND_AIR = 343.0  # m/s
AIR_DENSITY = 1.2041        # kg/m^3


class HapticSimulator(HapticDeviceInterface):
    """Simulates midair tactile sensations using visual ripples and synthesized audio cues."""

    def __init__(self, enable_audio_feedback: bool = True) -> None:
        self._connected = True
        self._active = False
        self._focal_point: Optional[Tuple[float, float, float]] = None
        self._intensity = 0.0
        self._modulation_hz = 200.0
        self._simulated_pressure_pa = 0.0

        # Ripple animation states for visual overlay: list of {center, radius, max_radius, color, alpha}
        self._active_ripples: List[Dict[str, Any]] = []

        # Diagnostics counters
        self._total_emissions = 0
        self._last_emission_time = 0.0
        self.enable_audio = enable_audio_feedback

        # Test if sounddevice is available for audio tactile substitution
        self._sound_available = False
        try:
            import sounddevice as sd
            self._sd = sd
            self._sound_available = True
        except Exception:
            self._sd = None

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        self.stop_emission()

    def is_connected(self) -> bool:
        return self._connected

    def is_hardware(self) -> bool:
        """Explicitly returns False: this is a software simulator, not physical hardware."""
        return False

    def emit_focal_point(
        self,
        x: float,
        y: float,
        z: float,
        intensity: float = 1.0,
        modulation_hz: float = 200.0,
    ) -> bool:
        if not self._connected:
            return False

        intensity = max(0.0, min(1.0, intensity))
        self._active = intensity > 0.01
        self._focal_point = (x, y, z)
        self._intensity = intensity
        self._modulation_hz = modulation_hz

        # Physical acoustic radiation pressure calculation:
        # P = 2 * I / c. At peak 140 dB SPL (~2000 W/m^2), pressure on skin is ~11.6 Pa
        self._simulated_pressure_pa = intensity * 12.5

        if self._active:
            self._total_emissions += 1
            self._last_emission_time = time.time()

        return True

    def trigger_pattern(
        self,
        pattern: HapticPattern | str,
        pos_3d: Tuple[float, float, float],
        intensity: float = 1.0,
    ) -> bool:
        pat_str = str(pattern.value if isinstance(pattern, HapticPattern) else pattern).upper()

        # Map pattern to acoustic modulation parameters
        mod_freq = 200.0
        if pat_str == "CLICK":
            mod_freq = 250.0
        elif pat_str == "TICK":
            mod_freq = 150.0
        elif pat_str == "BUZZ":
            mod_freq = 100.0
        elif pat_str == "MAGNETIC_SNAP":
            mod_freq = 220.0

        self.emit_focal_point(pos_3d[0], pos_3d[1], pos_3d[2], intensity=intensity, modulation_hz=mod_freq)

        # Trigger visual ripple
        self.spawn_visual_ripple((int(pos_3d[0]), int(pos_3d[1])), intensity=intensity)

        # Play synthesized micro-click audio feedback if enabled
        if self.enable_audio and self._sound_available and intensity > 0.1:
            self._play_tactile_audio_cue(pat_str, intensity)

        return True

    def spawn_visual_ripple(self, screen_center: Tuple[int, int], intensity: float = 1.0) -> None:
        """Spawns an animated acoustic wavefront ripple on screen."""
        self._active_ripples.append({
            "center": screen_center,
            "radius": 4.0,
            "max_radius": 28.0 * intensity,
            "alpha": 1.0,
            "color": (0, 255, 255) if intensity > 0.5 else (0, 200, 100),
        })

    def stop_emission(self) -> None:
        self._active = False
        self._intensity = 0.0
        self._simulated_pressure_pa = 0.0

    def render_ripples(self, frame: np.ndarray) -> None:
        """Updates and renders animated acoustic wavefront ripples on the frame."""
        surviving = []
        for rip in self._active_ripples:
            r = int(rip["radius"])
            alpha = rip["alpha"]
            color = rip["color"]
            cx, cy = rip["center"]

            if 0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]:
                # Draw outer wave
                cv2.circle(frame, (cx, cy), r, color, 1, cv2.LINE_AA)
                # Draw inner wave if expanded
                if r > 10:
                    cv2.circle(frame, (cx, cy), r - 8, color, 1, cv2.LINE_AA)

            # Expand ripple
            rip["radius"] += 2.5
            rip["alpha"] -= 0.08
            if rip["radius"] < rip["max_radius"] and rip["alpha"] > 0.05:
                surviving.append(rip)

        self._active_ripples = surviving

    def _play_tactile_audio_cue(self, pattern: str, intensity: float) -> None:
        """Synthesizes a short 20ms tactile confirmation audio chirp."""
        try:
            sample_rate = 22050
            duration = 0.025  # 25 ms
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

            freq = 280.0 if pattern == "CLICK" else 180.0
            # Sine wave with exponential decay envelope
            envelope = np.exp(-t * 120.0)
            wave = (np.sin(2 * np.pi * freq * t) * envelope * (intensity * 0.15)).astype(np.float32)

            self._sd.play(wave, samplerate=sample_rate, blocking=False)
        except Exception:
            pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "mode": "SOFTWARE_SIMULATION",
            "is_hardware": False,
            "connected": self._connected,
            "active": self._active,
            "focal_point": self._focal_point,
            "intensity": self._intensity,
            "modulation_hz": self._modulation_hz,
            "simulated_pressure_pa": round(self._simulated_pressure_pa, 2),
            "total_emissions": self._total_emissions,
            "audio_feedback_active": self.enable_audio and self._sound_available,
        }
