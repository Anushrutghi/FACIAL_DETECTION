# PHANTOM: Modular Touchless Spatial-Computing System

**PHANTOM** is a modular, futuristic touchless spatial-computing platform built upon a preserved OpenCV & MediaPipe computer vision foundation. It upgrades desktop interaction into a multi-layered spatial computing environment spanning 2D air-drawing, interactive 3D virtual object manipulation, acoustic haptics simulation, and hardware diagnostics.

---

## 🏗️ Architecture & Modules

The system is decoupled into isolated, high-performance packages:

```
facial_recognition_system/
├── phantom/
│   ├── config.py              # Configuration manager (JSON schema & persistence)
│   ├── tracker.py             # Dual-hand 3D tracking, EMA smoothing, hysteresis debouncing
│   ├── gestures.py            # Spatial event engine (click, drag, draw, pause, clear)
│   ├── canvas.py              # Multi-layer air-drawing canvas with stroke interpolation
│   ├── ui.py                  # Cybernetic holographic HUD, reticles & calibration
│   ├── space/                 # Phase 2: 3D Spatial Computing Engine
│   │   ├── math3d.py          # Perspective projection, 3D rotation, ray-sphere intersection
│   │   ├── objects.py         # Virtual primitives (Cube, Pyramid, Gyro-Ring, 6-DOF controls)
│   │   └── workspace.py       # 3D scene graph, spatial ribbon draw & guided tutorial
│   ├── touch/                 # Phase 3: Haptics & Midair Tactile Interaction
│   │   ├── interface.py       # Abstract hardware-independent haptic protocol
│   │   ├── simulator.py       # Software acoustic radiation pressure & ripple visualizer
│   │   └── hardware.py        # 40 kHz ultrasonic phased-array beamforming driver
│   └── lab/                   # Phase 4: Unified Suite, Diagnostics & Extensibility
│       ├── plugin.py          # Extensible plugin lifecycle manager & telemetry logger
│       ├── device_detector.py # Hardware scanner (Cameras, Audio, USB Serial ports)
│       └── accessibility.py   # High contrast, left-hand dominance, tremor reduction
├── config/
│   └── phantom_config.json    # Central user-customizable configuration file
├── docs/
│   └── ULTRASONIC_HAPTICS_HARDWARE.md # Complete ultrasonic prototype engineering spec & safety
├── tests/                     # 23 automated unit & integration tests
│   ├── test_phase1_vision.py  # Phase 1 test suite
│   ├── test_phase2_space.py   # Phase 2 test suite
│   ├── test_phase3_touch.py   # Phase 3 test suite
│   └── test_phase4_lab.py     # Phase 4 test suite
├── phantom_lab_main.py        # Master unified application (All 4 phases integrated)
├── phantom_main.py            # Dedicated Phase 1 Vision runner
├── phantom_space_main.py      # Dedicated Phase 2 Space 3D runner
├── main.py                    # Preserved original Face + Hand drawing application
├── hci_main.py                # Preserved original OS cursor & virtual keyboard HCI application
└── requirements.txt           # Verified Python dependencies
```

---

## ⚡ The Four Progressive Phases

### Phase 1: PHANTOM Vision
- **Dual-Hand 3D Tracking:** Tracks both left and right hands with 21 landmarks per hand.
- **Fingertip Cursor:** Precision reticle with One-Euro / Exponential Moving Average (EMA) jitter reduction.
- **Scale-Invariant Pinch-to-Click:** Hysteresis thresholds (`pinch_threshold_ratio` vs. `pinch_release_ratio`) calculated relative to palm size ($d_{\text{wrist} \to \text{middleMCP}}$) to eliminate click flickering.
- **Air Drawing:** Continuous interpolated strokes with round caps in multi-color palette (Cyan, Neon Green, Magenta, Amber, Electric Blue, White). Hand 0 draws in primary color; Hand 1 draws in secondary color.
- **Tactile Gesture Controls:**
  - **Open Palm:** Pauses drawing / interaction to rest hands.
  - **Fist:** Triggers canvas clear.
  - **Peace Sign (2 fingers):** Toggles eraser radius tool.
- **Calibration Overlay:** Press `C` to enter on-screen calibration mode with interactive scaling.
- **Diagnostics HUD:** Displays real-time FPS, MediaPipe tracking latency in milliseconds, active hand labels, and gesture badges.

### Phase 2: PHANTOM Space
- **Interactive 3D Workspace:** Perspective projection camera matrix projecting 3D world coordinates onto the 2D video plane.
- **Holographic Virtual Primitives:**
  - `SpatialCube` (wireframe cube with glowing nodes and depth badges)
  - `SpatialPyramid` (square-based 3D pyramid)
  - `SpatialHoloRing` (concentric gyroscopic rings rotating in space)
- **Single-Hand 3D Manipulation:** Pinch-grab near any object to translate along X, Y, and Z (depth adjusted by hand scale proxy).
- **Two-Hand 6-DOF Manipulation:** Pinch an object with both hands simultaneously:
  - **Midpoint:** Translates object in 3D.
  - **Distance Ratio:** Dynamically scales object larger or smaller.
  - **Angle Delta:** Rotates object around the Z/Y axes.
- **3D Spatial Ribbon Drawing:** Stores drawing points as 3D world vectors $(X, Y, Z)$ that rotate and perspective-shift in 3D space.
- **Floating 3D Action Menus:** Touchless floating holographic buttons to spawn cubes, pyramids, or reset the scene.
- **Guided Onboarding Demo:** 4-step interactive tutorial that detects user actions and guides them from hovering to 6-DOF dual-hand manipulation.

### Phase 3: PHANTOM Touch
- **Hardware-Independent Interface:** Standardized `HapticDeviceInterface` ensuring clean separation between simulated and physical hardware.
- **Software Acoustic Simulator (`HapticSimulator`):**
  - Calculates physical acoustic radiation pressure on human skin:
    $$P_{\text{rad}} = \frac{2 \cdot I}{c} \approx 1\text{--}16\text{ Pa}$$
  - Renders animated holographic acoustic wavefront ripples on contact.
  - Generates synthesized micro-chirp audio cues for auditory-tactile confirmation.
- **Ultrasonic Phased-Array Driver (`UltrasonicPhasedArrayDriver`):**
  - Implements 40 kHz phase delay beamforming for $16 \times 16$ (256 element) transducer arrays:
    $$\phi_i = 2\pi - \left(\frac{2\pi}{\lambda} d_i \pmod{2\pi}\right)$$
  - Handshake serial protocol detecting real USB hardware (`/dev/cu.usbserial*` or `/dev/ttyUSB*`).
  - **Integrity Rule:** Explicitly distinguishes simulated from physical hardware. Never claims hardware is functional unless verified physical hardware is connected.
- **Complete Hardware Engineering Spec (`docs/ULTRASONIC_HAPTICS_HARDWARE.md`):** Complete schematics, BOM, acoustic exposure limits (OSHA, ACGIH, ICNIRP), and safety interlocks.

### Phase 4: PHANTOM Lab
- **Flagship Application (`phantom_lab_main.py`):** Integrates all modules into a cybernetic command suite with a touchless top navigation bar:
  - `[1] VISION (2D AIR DRAW)`
  - `[2] SPACE (3D WORKSPACE)`
  - `[3] TOUCH (HAPTICS LAB)`
  - `[4] DIAGNOSTICS & SYSTEM`
- **Device Detection:** Automatically probes available camera indexes, audio output devices, and serial hardware ports.
- **Accessibility Profiles (`AccessibilityProfile`):**
  - High-contrast visual theme (Yellow/White on deep black)
  - Left-hand dominance mode (prioritizes left hand as primary cursor)
  - Enlarged cursor reticle
  - Tremor-reduction smoothing levels (0: Raw, 1: Normal, 2: High Smoothing)
- **Plugin Architecture:** Extensible `PhantomPlugin` lifecycle hooks (`on_frame`, `on_hands`, `on_event`, `on_render`). Includes `TelemetryLoggerPlugin` writing to `logs/phantom_telemetry.log`.
- **Automatic Error Recovery:** Video dropout watchdog with automatic camera reconnection loops.

---

## 🚀 Installation & Running

### 1. Prerequisites
- macOS (Apple Silicon or Intel), Linux, or Windows.
- Python 3.10 or 3.11 with an active camera.

### 2. Virtual Environment Setup
```bash
# Activate existing virtual environment
source venv/bin/activate

# Or install dependencies from requirements.txt
pip install -r requirements.txt
```

### 3. Run Commands

#### Master Application (All Phases Integrated):
```bash
./venv/bin/python phantom_lab_main.py
```

#### Dedicated Phase 1 Runner (PHANTOM Vision):
```bash
./venv/bin/python phantom_main.py
```

#### Dedicated Phase 2 Runner (PHANTOM Space):
```bash
./venv/bin/python phantom_space_main.py
```

#### Preserved Legacy Applications:
```bash
# Preserved Face Recognition + Hand Drawing
./venv/bin/python main.py

# Preserved HCI OS Cursor + Virtual Keyboard
./venv/bin/python hci_main.py
```

---

## ⌨️ Controls & Hotkeys

| Key | Action |
|:---:|:---|
| **1** | Switch to **PHANTOM Vision** (2D Air Drawing) |
| **2** | Switch to **PHANTOM Space** (3D Virtual Workspace) |
| **3** | Switch to **PHANTOM Touch** (Haptics Lab & Wavefronts) |
| **4** | Switch to **Diagnostics & Hardware** |
| **A** | Cycle **Accessibility Profile** (High-Contrast / Tremor Filter) |
| **C** | Toggle **Calibration Mode** (Vision) / Spawn Cube (Space) |
| **E** | Toggle **Eraser Tool** |
| **X** | **Clear Canvas** |
| **Z** | **Undo** Last Stroke |
| **R** | **Reset** 3D Virtual Objects |
| **D** | Toggle **3D Spatial Ribbon Drawing** |
| **T** | Toggle **Guided 3D Tutorial** |
| **S** | Save Snapshot image to `database/snapshots/` |
| **Q** / **ESC** | Safe graceful shutdown |

---

## 🧪 Automated Testing

PHANTOM includes a comprehensive 23-test unit and integration test suite across all 4 phases:

```bash
PYTHONPATH=. ./venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

Result:
```
Ran 23 tests in 2.172s
OK
```
