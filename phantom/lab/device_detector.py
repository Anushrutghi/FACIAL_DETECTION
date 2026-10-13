"""PHANTOM Lab Hardware & Device Detector.

Probes host system for connected video cameras, audio output interfaces,
serial/USB ports (for ultrasonic phased arrays), and platform compute capabilities.
"""

from __future__ import annotations
import glob
import os
import platform
import sys
from typing import Any, Dict, List

import cv2


class DeviceDetector:
    """Probes system hardware resources and peripherals."""

    @staticmethod
    def detect_cameras(max_probe_index: int = 3) -> List[Dict[str, Any]]:
        """Probes video capture indexes 0 to max_probe_index."""
        available_cameras = []
        for idx in range(max_probe_index):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if ret and frame is not None:
                    available_cameras.append({
                        "index": idx,
                        "name": f"Camera {idx}",
                        "resolution": (w, h),
                        "nominal_fps": fps if fps > 0 else 30.0,
                        "status": "OPERATIONAL",
                    })
        return available_cameras

    @staticmethod
    def detect_serial_ports() -> List[Dict[str, Any]]:
        """Scans for USB/serial ports that could host ultrasonic haptic arrays."""
        patterns = [
            "/dev/cu.usbserial*",
            "/dev/cu.usbmodem*",
            "/dev/ttyUSB*",
            "/dev/ttyACM*",
        ]
        ports = []
        for pat in patterns:
            for p in glob.glob(pat):
                ports.append({
                    "port": p,
                    "type": "USB-SERIAL",
                    "status": "DETECTED_UNBOUND",
                })
        return ports

    @staticmethod
    def detect_audio() -> Dict[str, Any]:
        """Checks for audio subsystem for auditory tactile substitution."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_out = sd.default.device[1] if isinstance(sd.default.device, tuple) else None
            return {
                "available": True,
                "device_count": len(devices),
                "default_output": default_out,
                "status": "OPERATIONAL",
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "status": "UNAVAILABLE",
            }

    @staticmethod
    def get_system_summary() -> Dict[str, Any]:
        """Summarizes host platform, libraries, and available devices."""
        import numpy as np
        import mediapipe as mp

        return {
            "os": platform.system(),
            "os_release": platform.release(),
            "arch": platform.machine(),
            "python_version": sys.version.split()[0],
            "opencv_version": cv2.__version__,
            "mediapipe_version": mp.__version__,
            "numpy_version": np.__version__,
            "cameras": DeviceDetector.detect_cameras(),
            "serial_devices": DeviceDetector.detect_serial_ports(),
            "audio": DeviceDetector.detect_audio(),
        }
