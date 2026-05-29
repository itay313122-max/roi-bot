# FPV Drone Detector — Field Operations Guide

**System:** `drone_detector.py` (acoustic + visual fusion)  
**Field UI:** `ui_field.py` (full-screen terminal)  
**Logger:** `test_logger.py` (CSV data recorder)

---

## Quick Start

```bash
# Terminal 1 — start the detector
python drone_detector.py

# Terminal 2 — launch field UI (full-screen)
python ui_field.py

# Or: calibrate first, then start everything
python drone_detector.py --calibrate
python ui_field.py

# Or: log to CSV from command line (no UI)
python test_logger.py
```

---

## Parameter Reference

### Acoustic Layer

| Parameter | Unit | Meaning |
|-----------|------|---------|
| **f₀** | Hz | Blade-pass frequency — the dominant tone produced by rotating blades. `BPF = (RPM / 60) × num_blades`. This is the primary identifier. |
| **RPM** | rev/min | Estimated motor speed, back-calculated from f₀. Formula: `RPM = f₀ × 60 / num_blades`. Useful for classifying drone type. |
| **SNR** | × (ratio) | Signal-to-noise ratio: peak FFT power at f₀ divided by the median floor across the scan band. SNR ≥ 3× required to process a frame. |
| **Harmonics** | count | Number of integer multiples of f₀ detected (f₀, 2f₀, 3f₀ …). More harmonics = higher confidence. FPV racing drones produce 5–6; commercial drones 3–4. |
| **Confidence** | % | Composite score blending SNR, harmonic depth, and profile match centrality. Values above 35% trigger the alarm hysteresis. |
| **rms_gate** | int16 units | Minimum frame RMS (loudness) required before the FFT is even run. Frames quieter than this threshold are skipped as silence. **Calibrate this per deployment environment.** |
| **EMA** | — | Exponential moving average (α=0.25) applied to raw confidence to smooth noise spikes. Prevents single-frame false positives. |

### Visual Layer (YOLOv8)

| Parameter | Meaning |
|-----------|---------|
| **Confidence %** | YOLOv8 detection confidence for the best bounding box in the current frame. Alert triggers at ≥ 60%. |
| **FPS** | Camera frame rate being processed. |
| **Source** | `LIVE CAMERA` = real webcam active. `SIMULATION` = no webcam found (cloud/test environment). |
| **Model** | `drone-specific (HF)` = `keremberke/yolov8n-drone-detection` (HuggingFace). `YOLOv8n COCO (airplane proxy)` = standard COCO model using the `airplane` class as a proxy. |

### Sensor Fusion

| Status | Meaning | Action |
|--------|---------|--------|
| **CONFIRMED** | Acoustic alarm AND visual alert both active. | Highest confidence. Engage counter-measures. |
| **ACOUSTIC ONLY** | Acoustic alarm active, visual below threshold. | Drone likely present; camera blind spot or beyond visual range. |
| **VISUAL ONLY** | Visual alert active, acoustic below threshold. | Drone may be very quiet (electric, high altitude) or acoustic background too loud. |
| **CLEAR** | Neither sensor triggered. | No threat detected. |

### Doppler / Direction

| Value | Meaning |
|-------|---------|
| **APPROACHING (↑)** | f₀ rising over the last 10 frames — drone is getting closer. Rate shown in Hz/sec. |
| **RECEDING (↓)** | f₀ falling — drone moving away. |
| **STABLE (—)** | f₀ steady — drone hovering or circling at constant distance. |

### Range Estimate (heuristic)

| SNR | Est. Range |
|-----|-----------|
| ≥ 15× | NEAR (<50m) |
| 8–15× | MEDIUM (50–200m) |
| 3–8× | FAR (>200m) |

> **Note:** These thresholds assume a directional microphone in a moderate-noise environment. Calibrate per deployment: use a known-distance test drone to map SNR → actual range.

---

## Drone Frequency Table

| Drone | BPF Range | RPM Range | Threat | Notes |
|-------|-----------|-----------|--------|-------|
| **FPV 5" Racing (Kamikaze)** | 667–1200 Hz | 20k–36k | HIGH | 2306/2207 motor, 4–6S. Most common Ukraine/Lebanon attack drone |
| **FPV 3" Micro (Suicide)** | 833–1500 Hz | 25k–45k | HIGH | Cheap, proliferated. Swarm attack vehicle |
| **FPV 7" Long Range** | 400–733 Hz | 12k–22k | HIGH | Fiber-optic guided; extended strike range |
| **DJI Mavic (Recon/ISR)** | 150–217 Hz | 4.5k–6.5k | MEDIUM | ISR platform — artillery correction, targeting |
| **DJI Matrice (Heavy Lift)** | 100–167 Hz | 3k–5k | MEDIUM | Payload carrier — grenade/IED delivery |
| **Shahed-136 / Geran-2** | 200–283 Hz | 6k–8.5k | CRITICAL | Piston engine; throttle modulation ~1.5 Hz creates non-periodic signature |
| **Lancet Loitering Munition** | 267–500 Hz | 8k–15k | HIGH | Electric motor; relatively quiet; anti-armor |

> BPF formula: `f₀ (Hz) = RPM / 60 × num_blades`  
> Most small FPV props use 2-blade design.

---

## Step-by-Step Test Protocol

### Pre-deployment

1. **Environment setup**
   - Find position with 360° acoustic exposure; avoid enclosed spaces (echoes create false harmonics)
   - Connect microphone (USB recommended; built-in laptop mic is second choice)
   - Note ambient noise sources (generators, traffic, wind)

2. **Install dependencies**
   ```bash
   pip install pyaudio numpy ultralytics opencv-python rich
   apt-get install -y portaudio19-dev   # Linux only, before pip install pyaudio
   ```

3. **Calibrate noise floor** (60 seconds)
   ```bash
   python drone_detector.py --calibrate --cal-duration 60
   ```
   - Keep area clear of drones during this period
   - Note the printed `recommended rms_gate` value
   - Edit `drone_detector.py`: set `RMS_GATE = <value>` at line ~61
   - Or use `[C]` in the UI for a 30s in-session calibration

4. **Verify calibration** — start detector and check the MONITORING baseline
   ```bash
   python drone_detector.py
   python ui_field.py          # in a second terminal
   ```
   - Confidence should stay near 0% with no drone present
   - If it fluctuates above 15% with no drone: increase `RMS_GATE` by 50%

### During Test

5. **Start recording CSV** — press `[L]` in the UI or use test_logger:
   ```bash
   python test_logger.py --no-launch --output test_20260529.csv
   ```

6. **Fly the drone** through different scenarios:
   - Straight approach (APPROACHING → CONFIRMED → RECEDING)
   - Hover overhead (STABLE + high SNR)
   - Fly-by at various altitudes
   - Different drone types if available

7. **Monitor dashboard** — observe:
   - Does the type classification match the actual drone?
   - Is the Doppler trend correct (APPROACHING as drone gets closer)?
   - Note any missed detections (false negatives) or false alarms

8. **Stop recording** — press `[L]` again or Ctrl+C

### Post-Test

9. **Analyse the CSV** — see "Reading the CSV" section below
10. **Adjust thresholds** based on findings — see "Tuning" section

---

## Reading the CSV After Testing

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | ISO8601 | UTC wall-clock time when the row was recorded |
| `acoustic_confidence` | float % | EMA-smoothed acoustic confidence (0–100) |
| `visual_confidence` | float % | YOLOv8 best-box confidence (0–100) |
| `combined_threat` | string | `CONFIRMED` / `ACOUSTIC ONLY` / `VISUAL ONLY` / `CLEAR` |
| `f0_hz` | int Hz | Detected fundamental frequency (0 when no detection) |
| `rpm_estimate` | int | Estimated motor RPM |
| `drone_type` | string | Matched profile label |
| `direction` | string | Doppler trend: `APPROACHING` / `RECEDING` / `STABLE` |
| `distance_phase` | string | Interpreted phase: `INBOUND` / `OUTBOUND` / `HOVER/OVERHEAD` / `N/A` |

### Quick Analysis with Python

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("field_test_20260529.csv", parse_dates=["timestamp"])
df = df.set_index("timestamp")

# Detection timeline
detections = df[df["combined_threat"] != "CLEAR"]
print(f"Total detections: {len(detections)}")
print(detections[["combined_threat", "f0_hz", "drone_type", "direction"]].to_string())

# Confidence over time
df["acoustic_confidence"].plot(title="Acoustic Confidence (%)")
plt.axhline(35, color="r", linestyle="--", label="alarm threshold")
plt.legend(); plt.show()

# Event timeline
confirmed = df[df["combined_threat"] == "CONFIRMED"]
print(f"\nCONFIRMED events: {len(confirmed)}")
print(f"Duration: {confirmed.index.min()} → {confirmed.index.max()}")
```

### Analysis with Excel / LibreOffice

1. Open CSV → Data → Text to Columns (delimiter: comma)
2. Sort by `combined_threat` to group events
3. Insert chart on `acoustic_confidence` column to see the detection timeline
4. Filter `combined_threat == "CONFIRMED"` to isolate high-confidence detections

---

## Handling False Positives

A false positive is when the system reports a drone when none is present.

### Common Causes and Fixes

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Confidence oscillates 10–30% continuously | `rms_gate` too low — ambient noise leaks through | Re-run calibration; increase `RMS_GATE` by 2× |
| Specific frequency always detected | Mechanical interference (fan, engine, HVAC) at a harmonic frequency | Check if f₀ matches HVAC/generator RPM. Add that frequency to a notch filter |
| Alarm triggers on footsteps / vehicle pass | `MIN_SNR` too low | Increase `MIN_SNR` from 3.0 to 4.0 or 5.0 |
| Constant `DJI Mavic` detections (150–217 Hz) | Low-frequency hum (power lines, transformer) | Raise `SCAN_MIN_HZ` from 80 to 120 for non-Mavic deployments |
| High confidence in wind | Wind turbulence creates broadband noise with peaks | Use directional mic with wind shield; calibrate in windy conditions |
| ACOUSTIC ONLY with no drone | CONFIRMED visual simulation artifact | Disable simulation: only occurs in no-camera environments |

### Tuning Parameters

Edit these constants in `drone_detector.py`:

```python
RMS_GATE   = 0.0004   # increase to ignore quiet ambient noise
MIN_SNR    = 3.0      # increase to require stronger signal
ALARM_FRAMES = 2      # increase to require more consecutive frames before alarm
EMA_ALPHA  = 0.25     # decrease (e.g. 0.15) to smooth out spikes
SCAN_MIN_HZ = 80      # raise to exclude low-frequency interference
```

### SNR-Threshold Override Per Profile

Each profile in `DRONE_DATABASE` has an `snr_threshold`. Increase specific ones to reduce false matches:

```python
"DJI_MAVIC_RECON": {
    "snr_threshold": 6.0,   # raised from 4.0 — require stronger Mavic signal
    ...
}
```

---

## Keyboard Shortcuts (Field UI)

| Key | Action |
|-----|--------|
| `C` | Start 30s noise-floor calibration. Keep area drone-free. Result shown in threat panel. |
| `L` | Toggle CSV logging. File named `field_test_YYYYMMDD_HHMMSS.csv` in current directory. |
| `R` | Reset session statistics (frame count, detections, calibration values). |
| `Q` | Quit cleanly. Flushes and closes CSV if open. |

---

## Minimum Terminal Size

The field UI requires at least **80 columns × 30 rows**.  
For narrow terminals: `python ui_field.py --no-guide` hides the Quick Guide panel.

Check terminal size: `echo "cols=$(tput cols) rows=$(tput lines)"`

---

## File Reference

| File | Purpose |
|------|---------|
| `drone_detector.py` | Main detector — acoustic FFT + YOLOv8 + HTTP API (port 8765) |
| `ui_field.py` | Full-screen terminal field UI (rich library) |
| `test_logger.py` | Headless CSV data logger (no UI) |
| `README_FIELD.md` | This file |
| `field_test_*.csv` | Session recordings |

---

*Sources: AUDRON (2024), ScienceDirect propeller acoustics literature, operational drone-warfare reporting.*
