"""PHANTOM Lab – Integrated Systems, Diagnostics, Plugins, and Accessibility.

Exports:
    PhantomPlugin – Base class for modular extensions.
    PluginManager – Lifecycle event dispatcher.
    DeviceDetector – Probes video cameras, audio, and serial hardware.
    AccessibilityProfile – Inclusive interaction profiles.
"""

from phantom.lab.plugin import PhantomPlugin, PluginManager, TelemetryLoggerPlugin
from phantom.lab.device_detector import DeviceDetector
from phantom.lab.accessibility import AccessibilityProfile

__all__ = [
    "PhantomPlugin",
    "PluginManager",
    "TelemetryLoggerPlugin",
    "DeviceDetector",
    "AccessibilityProfile",
]
