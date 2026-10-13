# PHANTOM Touch: Ultrasonic Phased-Array Midair-Haptics Prototype Specification

> **HARDWARE STATUS DISCLAIMER:**  
> The PHANTOM core software operates using a **hardware-independent software simulator** (`HapticSimulator`) with visual acoustic wavefront ripple rendering and auditory tactile substitution. Physical acoustic feedback requires the physical ultrasonic phased-array prototype described below. **Physical midair tactile sensations must NEVER be claimed unless real verified hardware is connected, calibrated, and tested.**

---

## 1. Operating Principle & Physical Foundations

Midair ultrasonic haptics generates focused acoustic radiation pressure without gloves, wearables, or physical contact. When multiple ultrasonic transducers (typically at **40 kHz**, wavelength $\lambda \approx 8.575\text{ mm}$ in air at 20°C) emit sound waves whose phase delays are precisely calculated, constructive interference creates a localized high-intensity acoustic focal point in 3D space:

$$P_{\text{rad}} = \frac{2 \cdot I}{c}$$

where:
- $P_{\text{rad}}$ is the acoustic radiation pressure ($\text{Pa}$),
- $I$ is the acoustic intensity ($\text{W/m}^2$),
- $c \approx 343\text{ m/s}$ is the speed of sound in air.

Because human cutaneous mechanoreceptors (specifically **Meissner corpuscles** and **Pacinian corpuscles**) respond to vibrations between $40\text{ Hz}$ and $300\text{ Hz}$, the 40 kHz ultrasonic carrier wave is **amplitude-modulated (AM)** at $100\text{--}250\text{ Hz}$ to produce perceivable tactile sensations on glabrous skin.

---

## 2. Hardware Architecture & Bill of Materials (BOM)

### 2.1 Transducer Array Grid
- **Transducer Elements:** 256× (16×16 square grid) or 64× (8×8 development grid) 40 kHz piezoceramic transducers (e.g., *Murata MA40S4S* or *Manorshi MSO-P1040H*).
- **Resonant Frequency:** $40.0 \pm 1.0\text{ kHz}$.
- **Directivity / Beam Angle:** $\approx 80^\circ$ at $-6\text{ dB}$.
- **Grid Pitch:** $10.0\text{ mm}$ (center-to-center spacing, slightly larger than $\lambda \approx 8.575\text{ mm}$ to balance grating lobe minimization and mechanical packaging).

### 2.2 Control & Driver Electronics
- **Logic & Beamforming Processor:**
  - *Option A (High Performance):* Xilinx Spartan-7 or Intel Cyclone IV FPGA generating 256 independent phase-shifted square-wave PWM signals at 40 kHz.
  - *Option B (Embedded MCU):* STM32H7 dual-core ARM Cortex-M7 (480 MHz) driving multichannel shift registers or specialized ultrasound beamforming ASICs.
- **MOSFET Half-Bridge Gate Drivers:** High-speed push-pull drivers (e.g., *Microchip TC4427A* or *Maxim MAX14808*) supplying 0–18V peak-to-peak into the piezoceramic capacitive load.
- **Power Supply (PSU):** 12V to 24V DC regulated laboratory power supply capable of supplying 8A peak (96W–192W peak acoustic power across 256 transducers).
- **Host Communications:** High-speed USB-to-UART / FTDI FT232H bridge operating at 115,200 baud up to 12 Mbaud for streaming 60–120 Hz focal point updates.

---

## 3. Beamforming Phase Delay Algorithm

For an array of $N$ transducers located at $(x_i, y_i, 0)$ and a target focal point at $(x_f, y_f, z_f)$:

1. **Calculate distance from each transducer to focus:**
   $$d_i = \sqrt{(x_i - x_f)^2 + (y_i - y_f)^2 + z_f^2}$$

2. **Compute phase delay:**
   $$\phi_i = 2\pi - \left(\frac{2\pi}{\lambda} d_i \pmod{2\pi}\right)$$

3. **8-Bit Quantization for Hardware Counters:**
   $$\text{PhaseByte}_i = \left\lfloor \frac{\phi_i}{2\pi} \times 256 \right\rfloor \pmod{256}$$

---

## 4. Calibration & Spatial Alignment

1. **Camera-to-Array Extrinsic Calibration:**
   - Mount an ArUco fiducial marker or optical calibration target flush with the center of the ultrasonic array ($z = 0$).
   - Use OpenCV `solvePnP` to compute the rotation matrix $R$ and translation vector $T$ between camera coordinate space and transducer array space.
2. **Hydrophone / Acoustic Microphone Alignment:**
   - Place a calibrated ultrasonic hydrophone or PCB MEMS microphone (e.g., *Knowles SPU0410HR5H*) on a 3-axis CNC gantry above the array.
   - Sweep the array focus across a $100\text{ mm} \times 100\text{ mm}$ test plane at $z = 150\text{ mm}$ to confirm focal spot sharpness ($< 10\text{ mm}$ beam waist) and calibrate transducer element manufacturing variances.

---

## 5. Acoustic & Ultrasonic Safety Precautions

> **CRITICAL SAFETY DIRECTIVE:** Airborne ultrasound at high sound pressure levels (SPL) carries specific biological and auditory risks. Hardware builds MUST comply with all safety standards below:

1. **Occupational Exposure Limits:**
   - **ACGIH & ICNIRP Limits:** Maximum permissible SPL at 40 kHz in air is **110 dB SPL** for continuous 8-hour occupational exposure, and **140 dB SPL** for short-term localized beam exposure on skin.
   - Operating focal points must never exceed **145 dB SPL** at the focal spot.
2. **Thermal & Tissue Protection:**
   - Skin absorption at 40 kHz in dry air is modest, but continuous focal exposure can cause localized thermal warming.
   - **Mandatory Software Watchdog:** Continuous focal emissions on stationary skin are automatically clamped to a maximum duration of **5.0 seconds** (`max_continuous_emission_sec = 5.0`).
   - If tracking detects no hand movement or lost tracking for $> 300\text{ ms}$, the array immediately halts emission (`stop_emission()`).
3. **Sub-Harmonic Acoustic Noise & Hearing Protection:**
   - Amplitude modulation at 200 Hz inherently creates audible sideband acoustics (audible buzzing sound).
   - Operators within 2 meters during high-intensity prototyping runs should wear calibrated acoustic ear protection rated for high-frequency noise.
4. **Hardware Emergency Stop:**
   - An active physical hardware E-Stop push button on the primary 24V DC power rail must cut driver power independently of software or USB states.

---

## 6. Software Architecture in PHANTOM

The PHANTOM system isolates hardware through `phantom.touch.interface.HapticDeviceInterface`:

```
               +-----------------------------+
               |   PHANTOM Spatial Engine    |
               +--------------+--------------+
                              |
               +--------------v--------------+
               |    HapticDeviceInterface    |
               +--------------+--------------+
                              |
              +---------------+---------------+
              |                               |
+-------------v---------------+ +-------------v---------------+
|     HapticSimulator         | | UltrasonicPhasedArrayDriver  |
|  - Visual wavefront ripples | |  - Serial handshake probing  |
|  - Sound substitution cue   | |  - Real beamforming delays   |
|  - Radiation pressure math  | |  - Safety watchdog & E-stop  |
|  [ACTIVE BY DEFAULT]        | |  [REQUIRES VERIFIED USB DEV] |
+-----------------------------+ +-----------------------------+
```
