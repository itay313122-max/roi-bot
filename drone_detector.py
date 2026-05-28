#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  DRONE ACOUSTIC DETECTOR — POC v2                        ║
║  BPF-based: 7 drone profiles, field-validated            ║
║                                                          ║
║  PHYSICS:  BPF (Hz) = (RPM / 60) × num_blades           ║
║  FPV 5" racing   20k–36k RPM →  667–1200 Hz             ║
║  FPV 3" micro    25k–45k RPM →  833–1500 Hz             ║
║  FPV 7" LR       12k–22k RPM →  400– 733 Hz             ║
║  DJI Mavic        4.5k–6.5k →  150– 217 Hz              ║
║  DJI Matrice      3k–  5k   →  100– 167 Hz              ║
║  Shahed-136       6k– 8.5k  →  200– 283 Hz              ║
║  Lancet           8k– 15k   →  267– 500 Hz              ║
║                                                          ║
║  INSTALL:  pip install pyaudio numpy                     ║
║  RUN:      python drone_detector.py                      ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import threading
import json
from collections import deque
from datetime import datetime

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠  pyaudio not found — running in SIMULATION MODE")
    print("   Install with: pip install pyaudio numpy\n")

# ─── Core parameters ──────────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_SIZE    = 1024
HZ_PER_BIN    = SAMPLE_RATE / FRAME_SIZE   # 15.625 Hz / bin
SCAN_MIN_HZ   = 80      # DJI Matrice lower bound (100 Hz) minus margin
SCAN_MAX_HZ   = 1600    # FPV 3" upper bound (1500 Hz) plus margin
H2_RATIO_MIN  = 0.07
H3_RATIO_MIN  = 0.035
EMA_ALPHA     = 0.25
ALARM_FRAMES  = 2
RMS_GATE      = 0.0004
MIN_SNR       = 3.0     # global pre-filter = lowest per-profile threshold

# ─── Drone database ───────────────────────────────────────
# BPF = RPM/60 × num_blades — values from field measurements
# Sources: AUDRON (2024), ScienceDirect prop acoustics, Drone-Warfare.com
DRONE_DATABASE = {
    "FPV_5INCH_RACING": {
        "label":         'FPV 5" Racing (Kamikaze)',
        "f0_range":      (667, 1200),    # 20k–36k RPM, 2-blade 5" prop
        "rpm_range":     (20000, 36000),
        "harmonics":     6,
        "snr_threshold": 6.0,            # noisy fast motors require high SNR
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "2306/2207 motor 2300-2500KV 4S-6S — most common Ukraine/Lebanon",
    },
    "FPV_3INCH_MICRO": {
        "label":         'FPV 3" Micro (Suicide)',
        "f0_range":      (833, 1500),    # 25k–45k RPM, 2-blade 3" prop
        "rpm_range":     (25000, 45000),
        "harmonics":     5,
        "snr_threshold": 5.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "cheap proliferated, swarm attacks",
    },
    "FPV_7INCH_LR": {
        "label":         'FPV 7" Long Range',
        "f0_range":      (400, 733),     # 12k–22k RPM, 2-blade 7" prop
        "rpm_range":     (12000, 22000),
        "harmonics":     5,
        "snr_threshold": 5.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "fiber-optic guided extended range strikes",
    },
    "DJI_MAVIC_RECON": {
        "label":         "DJI Mavic (Recon / ISR)",
        "f0_range":      (150, 217),     # 4.5k–6.5k RPM, 2-blade 8.5" prop
        "rpm_range":     (4500, 6500),
        "harmonics":     4,
        "snr_threshold": 4.0,
        "num_blades":    2,
        "threat":        "MEDIUM",
        "notes":         "ISR platform — artillery correction, targeting",
    },
    "DJI_MATRICE_HEAVY": {
        "label":         "DJI Matrice (Heavy Lift)",
        "f0_range":      (100, 167),     # 3k–5k RPM, 2-blade 15" prop
        "rpm_range":     (3000, 5000),
        "harmonics":     4,
        "snr_threshold": 3.0,
        "num_blades":    2,
        "threat":        "MEDIUM",
        "notes":         "payload carrier — grenades / IED delivery",
    },
    "SHAHED_136": {
        "label":         "Shahed-136 / Geran-2",
        "f0_range":      (200, 283),     # 6k–8.5k RPM, MD-550 2-stroke 2-blade
        "rpm_range":     (6000, 8500),
        "harmonics":     4,
        "snr_threshold": 3.0,            # quiet at range; 2-stroke harmonic signature
        "num_blades":    2,
        "threat":        "CRITICAL",
        "notes":         "piston engine — non-periodic modulation ~1.5 Hz from throttle",
    },
    "LANCET_LOITERING": {
        "label":         "Lancet Loitering Munition",
        "f0_range":      (267, 500),     # 8k–15k RPM, 2-blade 12" prop
        "rpm_range":     (8000, 15000),
        "harmonics":     4,
        "snr_threshold": 4.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "electric motor, relatively quiet, anti-armor",
    },
}


def classify_drone(f0: float, snr: float, harmonics_detected: int) -> tuple:
    """
    Match f0 against DRONE_DATABASE.
    Returns (drone_key, confidence_match 0..1, rpm_estimate).
    When profiles overlap, selects highest composite score:
      40% centrality within range + 40% SNR fit + 20% harmonic depth.
    """
    best_key   = None
    best_score = 0.0
    best_rpm   = 0

    for key, p in DRONE_DATABASE.items():
        lo, hi = p["f0_range"]
        if not (lo <= f0 <= hi):
            continue
        if snr < p["snr_threshold"]:
            continue

        center     = (lo + hi) / 2.0
        half_span  = (hi - lo) / 2.0
        centrality = 1.0 - abs(f0 - center) / half_span

        snr_score  = min(snr / (p["snr_threshold"] * 3.0), 1.0)
        h_score    = min(harmonics_detected / p["harmonics"], 1.0)
        score      = centrality * 0.4 + snr_score * 0.4 + h_score * 0.2

        if score > best_score:
            best_score = score
            best_key   = key
            best_rpm   = int(f0 * 60 / p["num_blades"])

    return best_key, best_score, best_rpm


class DroneDetector:
    def __init__(self):
        self.confidence_ema   = 0.0
        self.alarm_count      = 0
        self.alarm_active     = False
        self.detection_log    = []
        self.history          = deque(maxlen=100)
        self.f0_window        = deque(maxlen=10)  # 10 frames = 640 ms drift window
        self.frame_count      = 0
        self.total_detections = 0
        self.lock             = threading.Lock()

        self.dashboard_data = {
            "status":        "MONITORING",
            "confidence":    0.0,
            "alarm":         False,
            "f0_hz":         0,
            "f0_smoothed":   0,
            "snr":           0.0,
            "profile_label": "—",
            "threat":        "LOW",
            "rpm_est":       0,
            "rpm_trend":     "STABLE",
            "rate_hz_per_s": 0.0,
            "history":       [],
            "log":           [],
            "frames":        0,
            "detections":    0,
        }

    def _f0_drift(self) -> tuple:
        """
        Linearly-weighted f₀ average + drift rate (Hz/sec).
        Rate = (mean of recent half − mean of early half) / time_span.
        With 10 frames × 64 ms = 640 ms window, threshold ±15 Hz/sec.
        """
        vals = list(self.f0_window)
        if not vals:
            return 0.0, "STABLE", 0.0

        n       = len(vals)
        weights = list(range(1, n + 1))
        w_sum   = sum(weights)
        smoothed = sum(v * w for v, w in zip(vals, weights)) / w_sum

        if n < 4:
            return smoothed, "STABLE", 0.0

        mid          = n // 2
        early_mean   = sum(vals[:mid]) / mid
        recent_mean  = sum(vals[mid:]) / (n - mid)
        time_span    = mid * (FRAME_SIZE / SAMPLE_RATE)
        rate_hz_per_s = (recent_mean - early_mean) / time_span

        if rate_hz_per_s > 15:
            trend = "APPROACHING"
        elif rate_hz_per_s < -15:
            trend = "RECEDING"
        else:
            trend = "STABLE"

        return smoothed, trend, rate_hz_per_s

    def analyze_frame(self, samples: np.ndarray) -> dict:
        """
        BPF harmonic fingerprint detection:
        1. Hanning + FFT → PSD
        2. Peak scan 80–1600 Hz (covers all BPF fundamentals)
        3. Global SNR pre-filter (3×)
        4. Harmonic verification h2 + h3; extended harmonic count
        5. classify_drone() → profile + RPM estimate
        6. Confidence = 70% acoustic score + 30% profile match
        """
        self.frame_count += 1

        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        if rms < RMS_GATE:
            return {"detected": False, "reason": "silence", "confidence": 0.0}

        x      = samples.astype(np.float32) / 32768.0
        window = np.hanning(len(x))
        fft    = np.fft.rfft(x * window, n=FRAME_SIZE)
        psd    = np.abs(fft) ** 2
        freqs  = np.fft.rfftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)

        scan_mask = (freqs >= SCAN_MIN_HZ) & (freqs <= SCAN_MAX_HZ)
        scan_psd  = psd.copy()
        scan_psd[~scan_mask] = 0.0
        peak_bin  = np.argmax(scan_psd)
        f0        = freqs[peak_bin]
        p_f0      = psd[peak_bin]

        noise_floor = np.median(psd[scan_mask]) + 1e-12
        snr         = p_f0 / noise_floor

        if snr < MIN_SNR:
            return {
                "detected": False, "reason": f"low SNR {snr:.1f}",
                "confidence": 0.0, "f0": f0, "snr": snr,
            }

        # Harmonic search — window ±6% of f₀ (min ±15 Hz) stays proportional
        hw = max(15.0, f0 * 0.06)

        def power_near(hz: float) -> float:
            mask = (freqs >= hz - hw) & (freqs <= hz + hw)
            return float(np.max(psd[mask])) if np.any(mask) else 0.0

        h2 = power_near(f0 * 2) / (p_f0 + 1e-12)
        h3 = power_near(f0 * 3) / (p_f0 + 1e-12)

        if h2 < H2_RATIO_MIN or h3 < H3_RATIO_MIN:
            return {
                "detected": False,
                "reason":   f"harmonics fail h2={h2:.3f} h3={h3:.3f}",
                "confidence": 0.0, "f0": f0, "snr": snr,
            }

        # Count confirmed harmonics beyond h3 (used by classify_drone for profile depth match)
        harmonics_detected = 2  # h2 and h3 already confirmed
        for k in range(4, 9):
            if power_near(f0 * k) / (p_f0 + 1e-12) >= 0.02:
                harmonics_detected += 1

        drone_key, match_score, rpm_est = classify_drone(f0, snr, harmonics_detected)

        if drone_key is None:
            return {
                "detected": False, "reason": "no profile match",
                "confidence": 0.0, "f0": f0, "snr": snr,
            }

        profile = DRONE_DATABASE[drone_key]

        snr_norm  = profile["snr_threshold"] * 3.0
        snr_score = min(snr / snr_norm, 1.0)
        h2_score  = min(h2 / 0.3,  1.0)
        h3_score  = min(h3 / 0.15, 1.0)
        base_conf = snr_score * 0.5 + h2_score * 0.3 + h3_score * 0.2
        confidence = min(base_conf * 0.7 + match_score * 0.3, 1.0)

        return {
            "detected":           True,
            "confidence":         confidence,
            "f0":                 float(f0),
            "snr":                float(snr),
            "h2":                 float(h2),
            "h3":                 float(h3),
            "harmonics_detected": harmonics_detected,
            "drone_key":          drone_key,
            "profile_label":      profile["label"],
            "threat":             profile["threat"],
            "rpm_est":            rpm_est,
        }

    def process(self, result: dict):
        raw_conf = result.get("confidence", 0.0)
        self.confidence_ema = EMA_ALPHA * raw_conf + (1 - EMA_ALPHA) * self.confidence_ema

        if self.confidence_ema > 0.35:
            self.alarm_count += 1
        else:
            self.alarm_count = max(0, self.alarm_count - 1)

        prev_alarm = self.alarm_active
        self.alarm_active = (self.alarm_count >= ALARM_FRAMES)

        if result.get("detected") and result.get("f0"):
            self.f0_window.append(result["f0"])

        f0_smoothed, rpm_trend, rate_hz_per_s = self._f0_drift()

        profile_label = result.get("profile_label", "—") if result.get("detected") else "—"
        threat        = result.get("threat", "LOW")       if result.get("detected") else "LOW"
        rpm_est       = result.get("rpm_est", 0)          if result.get("detected") else 0

        if self.alarm_active and not prev_alarm:
            self.total_detections += 1
            entry = {
                "time":       datetime.now().strftime("%H:%M:%S"),
                "type":       profile_label,
                "threat":     threat,
                "f0":         round(result.get("f0", 0)),
                "snr":        round(result.get("snr", 0), 1),
                "confidence": round(self.confidence_ema, 2),
                "rpm":        rpm_est,
                "trend":      rpm_trend,
            }
            self.detection_log.insert(0, entry)
            self.detection_log = self.detection_log[:50]

        self.history.append(round(self.confidence_ema, 3))

        with self.lock:
            self.dashboard_data = {
                "status":        "🚨 DRONE DETECTED" if self.alarm_active else "👂 MONITORING",
                "confidence":    round(self.confidence_ema * 100, 1),
                "alarm":         self.alarm_active,
                "f0_hz":         round(result.get("f0", 0)) if result.get("detected") else 0,
                "f0_smoothed":   round(f0_smoothed),
                "snr":           round(result.get("snr", 0), 1),
                "profile_label": profile_label,
                "threat":        threat,
                "rpm_est":       rpm_est,
                "rpm_trend":     rpm_trend,
                "rate_hz_per_s": round(rate_hz_per_s, 1),
                "history":       list(self.history),
                "log":           self.detection_log[:10],
                "frames":        self.frame_count,
                "detections":    self.total_detections,
            }

    def get_state(self):
        with self.lock:
            return dict(self.dashboard_data)


# ─── Simulation — 3 drone types, 26-second cycle ──────────
class SimulatedAudio:
    """
    26-second cycle demonstrating 3 profiles from DRONE_DATABASE:

      FPV 5" Racing fly-over:
        approach  2–5s   f₀ 700→950 Hz  (+83 Hz/s  → APPROACHING)
        recede    5–7s   f₀ 950→700 Hz  (−125 Hz/s → RECEDING)
      DJI Mavic ISR hover:
        10–14s   f₀ 183 Hz  stable       (STABLE)
      Shahed-136 cruise:
        17–22s   f₀ 242 Hz  stable       (STABLE, 2-stroke harmonics)

    Harmonic amplitudes from prop acoustic models:
      FPV racing  — distorted waveform  h2 55%  h3 28%  h4 14% …
      DJI Mavic   — smooth efficient    h2 40%  h3 20%  h4 10%
      Shahed-136  — 2-stroke combustion h2 65%  h3 35%  h4 20%
                    (2-stroke boosts even harmonics vs electric)
    """

    _H_AMPS = {
        "FPV_5INCH_RACING": (1.0, 0.55, 0.28, 0.14, 0.07, 0.04),
        "DJI_MAVIC_RECON":  (1.0, 0.40, 0.20, 0.10),
        "SHAHED_136":       (1.0, 0.65, 0.35, 0.20),
    }

    # (t_start, t_end, f0_start, f0_end, base_amp, drone_key)
    _SEGMENTS = (
        ( 2.0,  5.0,  700.0,  950.0, 10000, "FPV_5INCH_RACING"),
        ( 5.0,  7.0,  950.0,  700.0,  8000, "FPV_5INCH_RACING"),
        (10.0, 14.0,  183.0,  183.0,  8500, "DJI_MAVIC_RECON"),
        (17.0, 22.0,  242.0,  242.0,  7000, "SHAHED_136"),
    )
    _CYCLE = 26.0

    def __init__(self):
        self.t = 0

    def read_frame(self):
        t_arr     = np.arange(self.t, self.t + FRAME_SIZE, dtype=np.float64) / SAMPLE_RATE
        self.t   += FRAME_SIZE
        cycle_pos = (self.t / SAMPLE_RATE) % self._CYCLE

        signal = np.random.normal(0, 400, FRAME_SIZE)

        for t_start, t_end, f0_s, f0_e, base_amp, dkey in self._SEGMENTS:
            if t_start < cycle_pos < t_end:
                alpha = (cycle_pos - t_start) / (t_end - t_start)
                f0    = f0_s + alpha * (f0_e - f0_s)
                ramp  = min((cycle_pos - t_start) / 0.3, 1.0, (t_end - cycle_pos) / 0.3)
                amp   = base_amp * ramp
                for k, h in enumerate(self._H_AMPS[dkey], start=1):
                    signal += amp * h * np.sin(2.0 * np.pi * f0 * k * t_arr)
                break

        return signal.astype(np.int16)


# ─── Web dashboard ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drone Acoustic Detector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  :root {
    --bg: #030a06; --panel: #0a1a0f; --border: #1a4025;
    --green: #00ff6a; --green-dim: #005522;
    --red: #ff2020; --yellow: #ffcc00; --text: #b0ffcc; --dim: #3a6a4a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Share Tech Mono', monospace; min-height: 100vh; padding: 20px; }
  .scanline { position: fixed; inset: 0; pointer-events: none; z-index: 100;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,80,0.015) 2px, rgba(0,255,80,0.015) 4px); }
  header { display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px; }
  .logo { font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700; color: var(--green); letter-spacing: 4px; }
  .logo span { color: var(--dim); }
  .uptime { font-size: 11px; color: var(--dim); }
  .grid3 { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .grid2 { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 12px; }
  .panel { background: var(--panel); border: 1px solid var(--border); padding: 16px; position: relative; overflow: hidden; }
  .panel::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--green), transparent); opacity: 0.3; }
  .panel-title { font-size: 10px; color: var(--dim); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 10px; }
  .big-val { font-family: 'Rajdhani', sans-serif; font-size: 52px; font-weight: 700; line-height: 1; color: var(--green); }
  .big-val.alarm { color: var(--red); animation: blink .4s step-end infinite; }
  .big-val.warn  { color: var(--yellow); }
  .unit { font-size: 13px; color: var(--dim); margin-top: 4px; }
  @keyframes blink { 50% { opacity: 0.2; } }
  .status-box { padding: 12px 20px; font-family: 'Rajdhani', sans-serif; font-size: 18px; font-weight: 700;
    letter-spacing: 3px; text-align: center; border: 1px solid var(--border); background: var(--panel); transition: all .3s; }
  .status-box.alarm { border-color: var(--red); background: rgba(255,32,32,0.1); color: var(--red); box-shadow: 0 0 20px rgba(255,32,32,0.3); }
  .status-box.ok { color: var(--green); }
  .bar-wrap { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .bar-track { flex: 1; height: 8px; background: var(--green-dim); }
  .bar-fill { height: 100%; background: var(--green); transition: width .2s; }
  .bar-fill.alarm { background: var(--red); }
  .bar-fill.warn  { background: var(--yellow); }
  .bar-pct { font-size: 13px; color: var(--dim); min-width: 42px; text-align: right; }
  .type-val { font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700; line-height: 1.3; }
  .type-val.threat-critical { color: var(--red); animation: blink .4s step-end infinite; }
  .type-val.threat-high     { color: var(--red); }
  .type-val.threat-medium   { color: var(--yellow); }
  .type-val.unknown         { color: var(--dim); }
  .threat-badge { display: inline-block; margin-top: 6px; padding: 2px 10px;
    font-size: 10px; letter-spacing: 2px; border: 1px solid transparent; }
  .threat-badge.critical { border-color: var(--red); color: var(--red); background: rgba(255,32,32,0.15); }
  .threat-badge.high     { border-color: rgba(255,32,32,0.5); color: var(--red); background: rgba(255,32,32,0.08); }
  .threat-badge.medium   { border-color: rgba(255,204,0,0.5); color: var(--yellow); background: rgba(255,204,0,0.08); }
  .threat-badge.low      { border-color: var(--border); color: var(--dim); }
  .trend-val { font-family: 'Rajdhani', sans-serif; font-size: 20px; font-weight: 700; letter-spacing: 2px; margin-top: 6px; }
  .trend-approaching { color: var(--red); }
  .trend-receding    { color: var(--yellow); }
  .trend-stable      { color: var(--green); }
  canvas { width: 100%; height: 100px; display: block; background: #050f08; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { color: var(--dim); font-weight: normal; padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--border); }
  td { padding: 5px 6px; border-bottom: 1px solid #0d1f12; }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--green);
    margin-left: 6px; animation: pulse 1.5s ease-in-out infinite; }
  .dot.alarm { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
  .mini-stat { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #0d1f12; font-size: 12px; }
  .mini-stat:last-child { border: none; }
  .mini-stat .lbl { color: var(--dim); }
  .mini-stat .val { color: var(--green); }
  .data-sources { margin-top: 12px; padding: 8px 16px; font-size: 9px; color: var(--dim);
    border: 1px solid var(--border); background: var(--panel); letter-spacing: 0.5px; line-height: 1.8; }
</style>
</head>
<body>
<div class="scanline"></div>
<header>
  <div>
    <div class="logo">DRONE<span>::</span>DETECT</div>
    <div class="uptime" id="uptime">INITIALIZING...</div>
  </div>
  <div class="status-box ok" id="status-box"><span class="dot" id="dot"></span> MONITORING</div>
</header>

<!-- Row 1: confidence / f0 / SNR -->
<div class="grid3">
  <div class="panel">
    <div class="panel-title">CONFIDENCE SCORE</div>
    <div class="big-val" id="conf-val">0.0</div>
    <div class="unit">%</div>
    <div class="bar-wrap">
      <div class="bar-track"><div class="bar-fill" id="conf-bar" style="width:0%"></div></div>
      <div class="bar-pct" id="conf-pct">0%</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">FUNDAMENTAL Hz</div>
    <div class="big-val" id="f0-val">—</div>
    <div class="unit">Hz (f₀ raw)</div>
    <div style="margin-top:6px;font-size:11px;color:var(--dim)">
      SMOOTHED: <span id="f0-smooth">—</span> Hz
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">SNR</div>
    <div class="big-val" id="snr-val">—</div>
    <div class="unit">× noise floor</div>
  </div>
</div>

<!-- Row 2: identified type / estimated RPM / direction -->
<div class="grid3">
  <div class="panel">
    <div class="panel-title">IDENTIFIED TYPE</div>
    <div class="type-val unknown" id="type-val">—</div>
    <div><span class="threat-badge" id="threat-badge">&nbsp;</span></div>
  </div>
  <div class="panel">
    <div class="panel-title">EST. RPM</div>
    <div class="big-val" id="rpm-val">—</div>
    <div class="unit">rev / min</div>
  </div>
  <div class="panel">
    <div class="panel-title">DIRECTION</div>
    <div class="trend-val trend-stable" id="trend-val">— STABLE</div>
    <div class="unit" id="rate-val">&nbsp;</div>
  </div>
</div>

<!-- Row 3: chart / stats -->
<div class="grid2">
  <div class="panel">
    <div class="panel-title">CONFIDENCE HISTORY</div>
    <canvas id="chart"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">SYSTEM STATS</div>
    <div class="mini-stat"><span class="lbl">FRAMES PROCESSED</span><span class="val" id="frames">0</span></div>
    <div class="mini-stat"><span class="lbl">TOTAL DETECTIONS</span><span class="val" id="det-count">0</span></div>
    <div class="mini-stat"><span class="lbl">SCAN RANGE</span><span class="val">80–1600 Hz</span></div>
    <div class="mini-stat"><span class="lbl">SNR THRESHOLD</span><span class="val">per profile (3–6×)</span></div>
    <div class="mini-stat"><span class="lbl">PROFILES IN DB</span><span class="val">7 drone types</span></div>
    <div class="mini-stat"><span class="lbl">DRIFT WINDOW</span><span class="val">10 frames / 640 ms</span></div>
    <div class="mini-stat"><span class="lbl">SAMPLE RATE</span><span class="val">16 kHz</span></div>
  </div>
</div>

<!-- Detection log -->
<div class="panel" style="margin-bottom:12px">
  <div class="panel-title">DETECTION LOG</div>
  <table>
    <thead>
      <tr>
        <th>TIME</th><th>TYPE</th><th>f₀ Hz</th>
        <th>EST. RPM</th><th>SNR</th><th>CONF.</th>
      </tr>
    </thead>
    <tbody id="log-body">
      <tr><td colspan="6" style="color:var(--dim);text-align:center;padding:12px">Waiting for detections...</td></tr>
    </tbody>
  </table>
</div>

<!-- Data sources -->
<div class="data-sources">
  DATA SOURCES: AUDRON acoustic drone taxonomy paper (2024) &nbsp;·&nbsp;
  ScienceDirect prop acoustics — BPF = RPM/60 × num_blades &nbsp;·&nbsp;
  Drone-Warfare.com field BPF measurements &nbsp;·&nbsp;
  batear.io open-source BPF detector
</div>

<script>
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
canvas.width = canvas.offsetWidth * 2;
canvas.height = 200;

function drawChart(history) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#050f08'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle = '#0d2010'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = H * i / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  if (history.length < 2) return;
  const step = W / (history.length - 1);
  ctx.beginPath();
  ctx.moveTo(0, H);
  history.forEach((v, i) => ctx.lineTo(i * step, H - v * H));
  ctx.lineTo((history.length - 1) * step, H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,255,106,0.12)'; ctx.fill();
  ctx.beginPath();
  history.forEach((v, i) => { i === 0 ? ctx.moveTo(0, H - v * H) : ctx.lineTo(i * step, H - v * H); });
  ctx.strokeStyle = '#00ff6a'; ctx.lineWidth = 2; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, H * 0.65); ctx.lineTo(W, H * 0.65);
  ctx.strokeStyle = 'rgba(255,204,0,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4,4]); ctx.stroke();
  ctx.setLineDash([]);
}

const THREAT_COLOR = { CRITICAL: 'var(--red)', HIGH: 'var(--red)', MEDIUM: 'var(--yellow)', LOW: 'var(--green)' };

let startTime = Date.now();
function poll() {
  fetch('/api/state').then(r => r.json()).then(d => {
    const sec = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('uptime').textContent =
      'UPTIME ' + String(Math.floor(sec/3600)).padStart(2,'0') + ':' +
      String(Math.floor(sec % 3600 / 60)).padStart(2,'0') + ':' +
      String(sec % 60).padStart(2,'0');

    const alarm = d.alarm;
    const warn  = d.confidence > 30;
    const sb = document.getElementById('status-box');
    sb.innerHTML = '';
    const dotEl = document.createElement('span');
    dotEl.className = 'dot' + (alarm ? ' alarm' : '');
    sb.appendChild(dotEl);
    sb.appendChild(document.createTextNode(' ' + (alarm ? '⚠ DRONE DETECTED' : 'MONITORING')));
    sb.className = 'status-box' + (alarm ? ' alarm' : ' ok');

    const cv = document.getElementById('conf-val');
    cv.textContent = d.confidence.toFixed(1);
    cv.className = 'big-val' + (alarm ? ' alarm' : warn ? ' warn' : '');
    const bar = document.getElementById('conf-bar');
    bar.style.width = d.confidence + '%';
    bar.className = 'bar-fill' + (alarm ? ' alarm' : warn ? ' warn' : '');
    document.getElementById('conf-pct').textContent = d.confidence.toFixed(0) + '%';

    document.getElementById('f0-val').textContent    = d.f0_hz > 0 ? d.f0_hz : '—';
    document.getElementById('f0-smooth').textContent = d.f0_smoothed > 0 ? d.f0_smoothed : '—';
    document.getElementById('snr-val').textContent   = d.snr > 0 ? d.snr.toFixed(1) : '—';

    const label   = d.profile_label && d.profile_label !== '—' ? d.profile_label : null;
    const threat  = (d.threat || 'LOW').toLowerCase();
    const typeEl  = document.getElementById('type-val');
    typeEl.textContent = label || '—';
    typeEl.className   = 'type-val ' + (label ? 'threat-' + threat : 'unknown');
    const badgeEl = document.getElementById('threat-badge');
    if (label) {
      badgeEl.textContent = 'THREAT: ' + (d.threat || 'LOW');
      badgeEl.className   = 'threat-badge ' + threat;
    } else {
      badgeEl.innerHTML  = '&nbsp;';
      badgeEl.className  = 'threat-badge';
    }

    const rpmEl = document.getElementById('rpm-val');
    rpmEl.textContent = d.rpm_est > 0 ? d.rpm_est.toLocaleString() : '—';
    rpmEl.className   = 'big-val' + (alarm ? ' alarm' : '');

    const trend   = d.rpm_trend || 'STABLE';
    const rate    = d.rate_hz_per_s || 0;
    const arrows  = { APPROACHING: '▲', RECEDING: '▼', STABLE: '—' };
    const trendEl = document.getElementById('trend-val');
    trendEl.textContent = (arrows[trend] || '—') + ' ' + trend;
    trendEl.className   = 'trend-val trend-' + trend.toLowerCase();
    document.getElementById('rate-val').textContent =
      Math.abs(rate) > 1 ? Math.abs(rate).toFixed(0) + ' Hz/sec drift' : ' ';

    document.getElementById('frames').textContent    = d.frames.toLocaleString();
    document.getElementById('det-count').textContent = d.detections || 0;

    if (d.history) drawChart(d.history);

    const tbody = document.getElementById('log-body');
    if (d.log && d.log.length > 0) {
      tbody.innerHTML = d.log.map(e => {
        const tc = THREAT_COLOR[e.threat] || 'var(--green)';
        return '<tr>' +
          '<td>' + e.time + '</td>' +
          '<td style="color:' + tc + '">' + (e.type || '—') + '</td>' +
          '<td>' + e.f0 + ' Hz</td>' +
          '<td>' + (e.rpm ? e.rpm.toLocaleString() : '—') + '</td>' +
          '<td>' + e.snr + '×</td>' +
          '<td>' + (e.confidence * 100).toFixed(0) + '%</td>' +
          '</tr>';
      }).join('');
    }
  }).catch(() => {});
}

setInterval(poll, 300);
poll();
</script>
</body>
</html>"""


def start_web_server(detector: DroneDetector, port: int = 8765):
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path == "/api/state":
                data = json.dumps(detector.get_state()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode())

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main():
    print("=" * 58)
    print("  DRONE ACOUSTIC DETECTOR — POC v2")
    print("  7-profile BPF database  |  Doppler drift detection")
    print("=" * 58)

    detector = DroneDetector()
    port = 8765
    start_web_server(detector, port)
    print(f"\n  Dashboard: http://localhost:{port}")

    if AUDIO_AVAILABLE:
        print("  Using real microphone\n")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16, channels=1,
            rate=SAMPLE_RATE, input=True,
            frames_per_buffer=FRAME_SIZE,
        )
        def read_frame():
            raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.int16)
    else:
        print('  Simulation: FPV 5" flyover → DJI Mavic → Shahed-136  (26s cycle)\n')
        sim = SimulatedAudio()
        read_frame = sim.read_frame

    print("  Ctrl+C to stop\n")

    try:
        while True:
            samples = read_frame()
            result  = detector.analyze_frame(samples)
            detector.process(result)

            if detector.frame_count % 10 == 0:
                conf  = detector.confidence_ema
                state = detector.get_state()
                alarm = "ALARM" if detector.alarm_active else "listen"
                bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                label = state["profile_label"]
                label_s = f"  [{label}]" if label != "—" else ""
                rpm_s   = f"  {state['rpm_est']:,} rpm" if state["rpm_est"] else ""
                arrows  = {"APPROACHING": " ↑", "RECEDING": " ↓", "STABLE": ""}
                trend_s = arrows.get(state.get("rpm_trend", ""), "")
                f0_s    = f"  f₀={result.get('f0',0):.0f}Hz" if result.get("detected") else ""
                print(f"\r  [{bar}] {conf*100:5.1f}%  {alarm}{f0_s}{label_s}{rpm_s}{trend_s}  ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
        if AUDIO_AVAILABLE:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    main()
