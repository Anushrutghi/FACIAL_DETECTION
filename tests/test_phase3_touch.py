"""Automated Unit & Integration Test Suite for PHANTOM Touch (Phase 3)."""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phantom.touch.interface import HapticDeviceInterface, HapticPattern
from phantom.touch.simulator import HapticSimulator
from phantom.touch.hardware import UltrasonicPhasedArrayDriver, WAVELENGTH_MM
from phantom.touch import create_haptic_device


class TestPhantomTouch(unittest.TestCase):
    """Verifies all components of Phase 3 - PHANTOM Touch."""

    def test_01_simulator_interface_and_properties(self):
        """Verify HapticSimulator adheres to interface and marks software simulation."""
        sim = HapticSimulator(enable_audio_feedback=False)
        self.assertTrue(sim.connect())
        self.assertTrue(sim.is_connected())

        # CRITICAL SAFETY & INTEGRITY CHECK: Must NEVER claim to be hardware!
        self.assertFalse(sim.is_hardware(), "Simulator must never claim to be physical hardware!")

        # Diagnostics check
        diag = sim.get_diagnostics()
        self.assertEqual(diag["mode"], "SOFTWARE_SIMULATION")
        self.assertFalse(diag["is_hardware"])

    def test_02_simulator_emission_and_pressure_math(self):
        """Verify radiation pressure calculations in software simulation."""
        sim = HapticSimulator(enable_audio_feedback=False)
        sim.emit_focal_point(x=10.0, y=20.0, z=150.0, intensity=0.8, modulation_hz=200.0)

        diag = sim.get_diagnostics()
        self.assertTrue(diag["active"])
        self.assertAlmostEqual(diag["intensity"], 0.8)
        self.assertEqual(diag["focal_point"], (10.0, 20.0, 150.0))
        self.assertGreater(diag["simulated_pressure_pa"], 0.0)

        # Trigger pattern
        sim.trigger_pattern(HapticPattern.CLICK, pos_3d=(50.0, 50.0, 200.0), intensity=1.0)
        self.assertEqual(len(sim._active_ripples), 1)

        # Render ripples on frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sim.render_ripples(frame)
        self.assertTrue(np.any(frame != 0))

        sim.stop_emission()
        self.assertFalse(sim.get_diagnostics()["active"])

    def test_03_hardware_driver_disconnected_safety(self):
        """Verify UltrasonicPhasedArrayDriver safely handles disconnected state without crashing."""
        driver = UltrasonicPhasedArrayDriver(port_pattern="/dev/nonexistent_phantom_device*")
        connected = driver.connect()

        # Must report False when no hardware is plugged in
        self.assertFalse(connected)
        self.assertFalse(driver.is_connected())
        self.assertFalse(driver.is_hardware())

        # Diagnostics must clearly indicate hardware is disconnected
        diag = driver.get_diagnostics()
        self.assertEqual(diag["mode"], "HARDWARE_DISCONNECTED")
        self.assertIn("NO PHYSICAL HARDWARE DETECTED", diag["status_message"])

    def test_04_beamforming_phase_delay_algorithm(self):
        """Verify mathematical calculation of beamforming phase delays for 256 transducers."""
        driver = UltrasonicPhasedArrayDriver(array_rows=16, array_cols=16, pitch_mm=10.0)
        self.assertEqual(driver.num_elements, 256)

        # Focus target at 150mm on the Z axis normal
        target = (0.0, 0.0, 150.0)
        phases = driver.compute_phase_delays(target)

        self.assertEqual(len(phases), 256)
        # All phases must be quantized within 8-bit unsigned range [0, 255]
        for p in phases:
            self.assertGreaterEqual(p, 0)
            self.assertLessEqual(p, 255)

        # Geometric symmetry check:
        # Transducers at symmetric locations relative to (0, 0) should receive identical phase delays
        # Corner (0, 0) vs Corner (15, 15)
        phase_top_left = phases[0]
        phase_bottom_right = phases[255]
        self.assertEqual(phase_top_left, phase_bottom_right)

    def test_05_factory_device_creation(self):
        """Verify factory returns simulator fallback when no physical hardware is present."""
        device = create_haptic_device(force_simulator=True)
        self.assertIsInstance(device, HapticSimulator)
        self.assertFalse(device.is_hardware())


if __name__ == "__main__":
    unittest.main()
