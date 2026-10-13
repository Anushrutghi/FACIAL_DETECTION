"""PHANTOM Touch – Haptics and Midair Tactile Interaction Architecture.

Exports:
    HapticDeviceInterface – Abstract hardware-independent haptic protocol.
    HapticPattern – Standard tactile sensations (CLICK, TICK, PULSE, etc.).
    HapticSimulator – Software simulator with acoustic field and audio rendering.
    UltrasonicPhasedArrayDriver – Physical 40 kHz phased-array hardware driver.
    create_haptic_device – Factory function returning hardware if detected, or simulator fallback.
"""

from phantom.touch.interface import HapticDeviceInterface, HapticPattern
from phantom.touch.simulator import HapticSimulator
from phantom.touch.hardware import UltrasonicPhasedArrayDriver


def create_haptic_device(force_simulator: bool = False) -> HapticDeviceInterface:
    """Factory helper that checks for physical hardware and gracefully falls back to the simulator."""
    if not force_simulator:
        hw_driver = UltrasonicPhasedArrayDriver()
        if hw_driver.connect():
            print("[PHANTOM Touch] Physical ultrasonic phased array connected!")
            return hw_driver

    # Default to hardware-independent software simulator
    sim = HapticSimulator()
    sim.connect()
    return sim


__all__ = [
    "HapticDeviceInterface",
    "HapticPattern",
    "HapticSimulator",
    "UltrasonicPhasedArrayDriver",
    "create_haptic_device",
]
