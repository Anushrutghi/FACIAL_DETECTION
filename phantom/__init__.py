"""PHANTOM – Modular Futuristic Touchless Spatial-Computing System.

Exports:
    HandTracker – Robust multi-hand 3D tracker and landmark analyzer.
    GestureEngine – Spatial event and interaction dispatcher.
    SpatialCanvas – Multi-layer air-drawing canvas with interpolation.
    PhantomHUD – Cybernetic holographic UI, reticles, and telemetry.
    PhantomConfig – Configuration manager.
"""

from phantom.config import PhantomConfig
from phantom.tracker import HandTracker, detect_hand_states
from phantom.gestures import GestureEngine
from phantom.canvas import SpatialCanvas
from phantom.ui import PhantomHUD

__all__ = [
    "PhantomConfig",
    "HandTracker",
    "detect_hand_states",
    "GestureEngine",
    "SpatialCanvas",
    "PhantomHUD",
]
