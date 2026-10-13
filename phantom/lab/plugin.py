"""PHANTOM Lab Plugin Architecture.

Provides an extensible lifecycle hook system allowing third-party spatial computing
modules, custom gesture detectors, telemetry loggers, and hardware drivers to be attached.
"""

from __future__ import annotations
from abc import ABC
import time
from typing import Any, Dict, List, Optional
import numpy as np


class PhantomPlugin(ABC):
    """Base class for all PHANTOM runtime extensions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.enabled = True

    def on_load(self, app_context: Any) -> None:
        """Called once when plugin is initialized."""
        pass

    def on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Called on raw incoming frame before tracking."""
        return frame

    def on_hands(self, hand_states: List[Dict[str, Any]]) -> None:
        """Called after hand landmarks and gestures are calculated."""
        pass

    def on_event(self, event: Dict[str, Any]) -> None:
        """Called on discrete spatial interaction events (clicks, drags, pauses)."""
        pass

    def on_render(self, frame: np.ndarray) -> None:
        """Called during final HUD compositing for custom rendering."""
        pass

    def on_unload(self) -> None:
        """Called during application shutdown."""
        pass


class TelemetryLoggerPlugin(PhantomPlugin):
    """Sample plugin: Logs real-time spatial FPS and hand confidence to a file."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        super().__init__("TelemetryLogger")
        self.log_path = log_path
        self._last_log_time = 0.0

    def on_hands(self, hand_states: List[Dict[str, Any]]) -> None:
        now = time.time()
        if now - self._last_log_time > 1.0:  # Sample every second
            self._last_log_time = now
            if self.log_path and hand_states:
                try:
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hands: {len(hand_states)} | Gestures: {[h['gesture'] for h in hand_states]}\n")
                except Exception:
                    pass


class PluginManager:
    """Manages loaded plugins and dispatches lifecycle events."""

    def __init__(self) -> None:
        self.plugins: List[PhantomPlugin] = []

    def register(self, plugin: PhantomPlugin, app_context: Optional[Any] = None) -> None:
        self.plugins.append(plugin)
        plugin.on_load(app_context)

    def dispatch_frame(self, frame: np.ndarray) -> np.ndarray:
        for p in self.plugins:
            if p.enabled:
                frame = p.on_frame(frame)
        return frame

    def dispatch_hands(self, hand_states: List[Dict[str, Any]]) -> None:
        for p in self.plugins:
            if p.enabled:
                p.on_hands(hand_states)

    def dispatch_event(self, event: Dict[str, Any]) -> None:
        for p in self.plugins:
            if p.enabled:
                p.on_event(event)

    def dispatch_render(self, frame: np.ndarray) -> None:
        for p in self.plugins:
            if p.enabled:
                p.on_render(frame)

    def unload_all(self) -> None:
        for p in self.plugins:
            p.on_unload()
        self.plugins.clear()
