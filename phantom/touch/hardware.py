"""PHANTOM Touch – Physical Ultrasonic Phased-Array Hardware Driver Architecture.

Hardware driver for 40 kHz airborne ultrasonic phased-array transducers.
Calculates beamforming phase delays per transducer element and communicates via serial/USB.
NEVER claims hardware is operational unless an authentic physical controller handshakes.
"""

from __future__ import annotations
import glob
import math
import time
from typing import Any, Dict, List, Optional, Tuple

from phantom.touch.interface import HapticDeviceInterface, HapticPattern


# Physical ultrasound constants
CARRIER_FREQ_HZ = 40000.0  # 40 kHz
SPEED_OF_SOUND_AIR = 343.0  # m/s
WAVELENGTH_M = SPEED_OF_SOUND_AIR / CARRIER_FREQ_HZ  # ~0.008575 m (8.575 mm)
WAVELENGTH_MM = WAVELENGTH_M * 1000.0


class UltrasonicPhasedArrayDriver(HapticDeviceInterface):
    """Driver and beamforming engine for real ultrasonic phased-array hardware."""

    def __init__(
        self,
        array_rows: int = 16,
        array_cols: int = 16,
        pitch_mm: float = 10.0,
        port_pattern: str = "/dev/cu.usbserial*",
        baud_rate: int = 115200,
    ) -> None:
        self.rows = array_rows
        self.cols = array_cols
        self.num_elements = array_rows * array_cols
        self.pitch_mm = pitch_mm
        self.port_pattern = port_pattern
        self.baud_rate = baud_rate

        # Hardware connection state
        self._is_hardware_connected = False
        self._active_serial_port: Optional[str] = None
        self._serial_handle = None

        # Safety cutoff parameters
        self.max_continuous_emission_sec = 5.0
        self.emission_start_time = 0.0
        self.emergency_cutoff_triggered = False

        # Precompute transducer element 3D coordinates on the array surface (z=0)
        self.transducer_positions: List[Tuple[float, float, float]] = []
        origin_x = -((array_cols - 1) * pitch_mm) / 2.0
        origin_y = -((array_rows - 1) * pitch_mm) / 2.0
        for r in range(array_rows):
            for c in range(array_cols):
                px = origin_x + c * pitch_mm
                py = origin_y + r * pitch_mm
                self.transducer_positions.append((px, py, 0.0))

        # Current phase states (0 to 255 representing 0 to 2*pi)
        self.phases = [0] * self.num_elements

    def connect(self) -> bool:
        """Probes for connected physical phased array controller."""
        matching_ports = glob.glob(self.port_pattern) + glob.glob("/dev/ttyUSB*")
        if not matching_ports:
            self._is_hardware_connected = False
            self._active_serial_port = None
            return False

        # Attempt handshake with detected port
        try:
            import serial
            for port in matching_ports:
                try:
                    ser = serial.Serial(port, self.baud_rate, timeout=0.2)
                    ser.write(b"PHANTOM_PING\n")
                    response = ser.readline().decode("utf-8", errors="ignore").strip()
                    if "PHANTOM_ACK" in response:
                        self._serial_handle = ser
                        self._active_serial_port = port
                        self._is_hardware_connected = True
                        return True
                    ser.close()
                except Exception:
                    continue
        except ImportError:
            pass

        self._is_hardware_connected = False
        return False

    def disconnect(self) -> None:
        self.stop_emission()
        if self._serial_handle:
            try:
                self._serial_handle.close()
            except Exception:
                pass
            self._serial_handle = None
        self._is_hardware_connected = False

    def is_connected(self) -> bool:
        return self._is_hardware_connected

    def is_hardware(self) -> bool:
        """Returns True ONLY if verified physical serial hardware is connected."""
        return self._is_hardware_connected

    def compute_phase_delays(
        self,
        target_focus: Tuple[float, float, float],
    ) -> List[int]:
        """Calculates phase delay $\Delta\phi_i$ for each transducer element to form focal point."""
        fx, fy, fz = target_focus
        delays = []

        for tx, ty, tz in self.transducer_positions:
            dist_mm = math.sqrt((tx - fx)**2 + (ty - fy)**2 + (tz - fz)**2)
            # Phase in radians
            phase_rad = (2.0 * math.pi * (dist_mm / WAVELENGTH_MM)) % (2.0 * math.pi)
            # Invert phase so acoustic waves arrive simultaneously in constructive interference
            inv_phase_rad = (2.0 * math.pi - phase_rad) % (2.0 * math.pi)
            # Map to 8-bit quantization [0, 255]
            byte_val = int((inv_phase_rad / (2.0 * math.pi)) * 255.0) % 256
            delays.append(byte_val)

        return delays

    def emit_focal_point(
        self,
        x: float,
        y: float,
        z: float,
        intensity: float = 1.0,
        modulation_hz: float = 200.0,
    ) -> bool:
        # Acoustic safety guard
        if self.emergency_cutoff_triggered:
            return False

        if not self._is_hardware_connected or not self._serial_handle:
            # Physical hardware is not connected
            return False

        # Check continuous emission timeout (thermal limit)
        now = time.time()
        if self.emission_start_time > 0 and (now - self.emission_start_time > self.max_continuous_emission_sec):
            self.stop_emission()
            self.emergency_cutoff_triggered = True
            return False

        if self.emission_start_time == 0:
            self.emission_start_time = now

        # Compute phases and transmit frame packet to controller
        self.phases = self.compute_phase_delays((x, y, z))
        try:
            # Packet: [0xAA, 0x55, num_elements, intensity_byte, mod_freq_high, mod_freq_low, ...phases...]
            header = bytes([0xAA, 0x55, self.num_elements % 256, int(intensity * 255)])
            payload = bytes(self.phases)
            self._serial_handle.write(header + payload)
            return True
        except Exception:
            self.disconnect()
            return False

    def trigger_pattern(
        self,
        pattern: HapticPattern | str,
        pos_3d: Tuple[float, float, float],
        intensity: float = 1.0,
    ) -> bool:
        return self.emit_focal_point(pos_3d[0], pos_3d[1], pos_3d[2], intensity=intensity)

    def stop_emission(self) -> None:
        self.emission_start_time = 0.0
        if self._serial_handle and self._is_hardware_connected:
            try:
                # Command 0x00: zero all transducer amplitudes
                self._serial_handle.write(bytes([0xAA, 0x00, 0x00, 0x00]))
            except Exception:
                pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "mode": "PHYSICAL_HARDWARE" if self._is_hardware_connected else "HARDWARE_DISCONNECTED",
            "is_hardware": self._is_hardware_connected,
            "connected": self._is_hardware_connected,
            "port": self._active_serial_port,
            "num_transducers": self.num_elements,
            "carrier_freq_khz": CARRIER_FREQ_HZ / 1000.0,
            "wavelength_mm": round(WAVELENGTH_MM, 3),
            "emergency_cutoff": self.emergency_cutoff_triggered,
            "status_message": (
                "PHYSICAL TRANSDUCER ARRAY VERIFIED & ACTIVE"
                if self._is_hardware_connected
                else "NO PHYSICAL HARDWARE DETECTED: Running in software simulation fallback"
            ),
        }
