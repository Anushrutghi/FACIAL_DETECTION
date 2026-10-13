"""PHANTOM Touch – Hardware-Independent Haptic Interface.

Defines the abstract contract for physical and simulated tactile feedback devices.
Provides unified haptic emission methods, acoustic field queries, and safety status.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class HapticPattern(str, Enum):
    CLICK = "CLICK"
    TICK = "TICK"
    PULSE = "PULSE"
    BUZZ = "BUZZ"
    TEXTURE_ROUGH = "TEXTURE_ROUGH"
    MAGNETIC_SNAP = "MAGNETIC_SNAP"
    BOUNDARY_RESISTANCE = "BOUNDARY_RESISTANCE"


class HapticDeviceInterface(ABC):
    """Abstract interface for all PHANTOM tactile transducers and simulators."""

    @abstractmethod
    def connect(self) -> bool:
        """Initializes connection to the device or simulator."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Closes device connection and releases hardware resources."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Returns True if the backend is actively connected and operational."""
        pass

    @abstractmethod
    def is_hardware(self) -> bool:
        """Returns True ONLY when physical transducers are verified and active."""
        pass

    @abstractmethod
    def emit_focal_point(
        self,
        x: float,
        y: float,
        z: float,
        intensity: float = 1.0,
        modulation_hz: float = 200.0,
    ) -> bool:
        """Focuses acoustic/simulated radiation pressure at a 3D coordinate (mm).

        Args:
            x, y, z: Target focus position in millimeters relative to array center.
            intensity: Normalized power from 0.0 to 1.0.
            modulation_hz: Amplitude modulation frequency in Hertz (typically 100-300 Hz for human tactile sensitivity).
        """
        pass

    @abstractmethod
    def trigger_pattern(
        self,
        pattern: HapticPattern | str,
        pos_3d: Tuple[float, float, float],
        intensity: float = 1.0,
    ) -> bool:
        """Emits a predefined spatiotemporal tactile sensation."""
        pass

    @abstractmethod
    def stop_emission(self) -> None:
        """Immediately silences all acoustic emissions."""
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns current operational status, telemetry, and safety limits."""
        pass
