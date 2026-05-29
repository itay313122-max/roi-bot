#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  DRONE DETECTOR — POC v3                                 ║
║  Acoustic (BPF/FFT) + Visual (YOLOv8) fusion            ║
║                                                          ║
║  ACOUSTIC:  BPF (Hz) = (RPM / 60) × num_blades          ║
║  FPV 5" racing   20k–36k RPM →  667–1200 Hz             ║
║  FPV 3" micro    25k–45k RPM →  833–1500 Hz             ║
║  FPV 7" LR       12k–22k RPM →  400– 733 Hz             ║
║  DJI Mavic        4.5k–6.5k →  150– 217 Hz              ║
║  DJI Matrice      3k–  5k   →  100– 167 Hz              ║
║  Shahed-136       6k– 8.5k  →  200– 283 Hz              ║
║  Lancet           8k– 15k   →  267– 500 Hz              ║
║                                                          ║
║  INSTALL:  pip install pyaudio numpy                     ║
║            pip install ultralytics opencv-python         ║
║  RUN:      python drone_detector.py                      ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import time
import threading
import json
from collections import deque
from datetime import datetime

# ─── Optional dependencies ────────────────────────────────
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠  pyaudio not found — acoustic simulation mode")
    print("   pip install pyaudio numpy\n")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ─── Acoustic parameters ──────────────────────────────────
SAMPLE_RATE   = 16000
FRAME_SIZE    = 1024
HZ_PER_BIN    = SAMPLE_RATE / FRAME_SIZE   # 15.625 Hz / bin
SCAN_MIN_HZ   = 80
SCAN_MAX_HZ   = 1600
H2_RATIO_MIN  = 0.07
H3_RATIO_MIN  = 0.035
EMA_ALPHA     = 0.25
ALARM_FRAMES  = 2
RMS_GATE      = 0.0004
MIN_SNR       = 3.0

# ─── Visual threshold ─────────────────────────────────────
VISUAL_CONF_THRESHOLD = 0.60   # YOLOv8 confidence to trigger visual alert

# ─── Drone database ───────────────────────────────────────
# BPF = RPM/60 × num_blades — field-validated values
# Sources: AUDRON (2024), ScienceDirect prop acoustics, Drone-Warfare.com
DRONE_DATABASE = {
    "FPV_5INCH_RACING": {
        "label":         'FPV 5" Racing (Kamikaze)',
        "f0_range":      (667, 1200),
        "rpm_range":     (20000, 36000),
        "harmonics":     6,
        "snr_threshold": 6.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "2306/2207 motor 2300-2500KV 4S-6S — most common Ukraine/Lebanon",
    },
    "FPV_3INCH_MICRO": {
        "label":         'FPV 3" Micro (Suicide)',
        "f0_range":      (833, 1500),
        "rpm_range":     (25000, 45000),
        "harmonics":     5,
        "snr_threshold": 5.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "cheap proliferated, swarm attacks",
    },
    "FPV_7INCH_LR": {
        "label":         'FPV 7" Long Range',
        "f0_range":      (400, 733),
        "rpm_range":     (12000, 22000),
        "harmonics":     5,
        "snr_threshold": 5.0,
        "num_blades":    2,
        "threat":        "HIGH",
        "notes":         "fiber-optic guided extended range strikes",
    },
    "DJI_MAVIC_RECON": {
        "label":         "DJI Mavic (Recon / ISR)",
        "f0_range":      (150, 217),
        "rpm_range":     (4500, 6500),
        "harmonics":     4,
        "snr_threshold": 4.0,
        "num_blades":    2,
        "threat":        "MEDIUM",
        "notes":         "ISR platform — artillery correction, targeting",
    },
    "DJI_MATRICE_HEAVY": {
        "label":         "DJI Matrice (Heavy Lift)",
        "f0_range":      (100, 167),
        "rpm_range":     (3000, 5000),
        "harmonics":     4,
        "snr_threshold": 3.0,
        "num_blades":    2,
        "threat":        "MEDIUM",
        "notes":         "payload carrier — grenades / IED delivery",
    },
    "SHAHED_136": {
        "label":         "Shahed-136 / Geran-2",
        "f0_range":      (200, 283),
        "rpm_range":     (6000, 8500),
        "harmonics":     4,
        "snr_threshold": 3.0,
        "num_blades":    2,
        "threat":        "CRITICAL",
        "notes":         "piston engine — non-periodic modulation ~1.5 Hz from throttle",
    },
    "LANCET_LOITERING": {
        "label":         "Lancet Loitering Munition",
        "f0_range":      (267, 500),
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


def combined_threat(acoustic_alarm: bool, visual_alert: bool) -> str:
    """Fuse acoustic + visual alerts into a single threat level string."""
    if acoustic_alarm and visual_alert:
        return "CONFIRMED"
    if acoustic_alarm:
        return "ACOUSTIC ONLY"
    if visual_alert:
        return "VISUAL ONLY"
    return "CLEAR"


# ─── Acoustic detector ────────────────────────────────────
class DroneDetector:
    def __init__(self):
        self.confidence_ema   = 0.0
        self.alarm_count      = 0
        self.alarm_active     = False
        self.detection_log    = []
        self.history          = deque(maxlen=100)
        self.f0_window        = deque(maxlen=10)
        self.frame_count      = 0
        self.total_detections = 0
        self.lock             = threading.Lock()

        self.dashboard_data = {
            "status":         "MONITORING",
            "confidence":     0.0,
            "alarm":          False,
            "f0_hz":          0,
            "f0_smoothed":    0,
            "snr":            0.0,
            "profile_label":  "—",
            "threat":         "LOW",
            "rpm_est":        0,
            "rpm_trend":      "STABLE",
            "rate_hz_per_s":  0.0,
            "history":        [],
            "log":            [],
            "frames":         0,
            "detections":     0,
            "combined_threat": "CLEAR",
            "visual":         {},
        }

    def _f0_drift(self) -> tuple:
        vals = list(self.f0_window)
        if not vals:
            return 0.0, "STABLE", 0.0
        n        = len(vals)
        weights  = list(range(1, n + 1))
        w_sum    = sum(weights)
        smoothed = sum(v * w for v, w in zip(vals, weights)) / w_sum
        if n < 4:
            return smoothed, "STABLE", 0.0
        mid           = n // 2
        early_mean    = sum(vals[:mid]) / mid
        recent_mean   = sum(vals[mid:]) / (n - mid)
        time_span     = mid * (FRAME_SIZE / SAMPLE_RATE)
        rate_hz_per_s = (recent_mean - early_mean) / time_span
        if rate_hz_per_s > 15:
            trend = "APPROACHING"
        elif rate_hz_per_s < -15:
            trend = "RECEDING"
        else:
            trend = "STABLE"
        return smoothed, trend, rate_hz_per_s

    def analyze_frame(self, samples: np.ndarray) -> dict:
        """Hanning+FFT → BPF peak → per-profile SNR gate → harmonic check → classify."""
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
            return {"detected": False, "reason": f"low SNR {snr:.1f}",
                    "confidence": 0.0, "f0": f0, "snr": snr}

        hw = max(15.0, f0 * 0.06)

        def power_near(hz: float) -> float:
            mask = (freqs >= hz - hw) & (freqs <= hz + hw)
            return float(np.max(psd[mask])) if np.any(mask) else 0.0

        h2 = power_near(f0 * 2) / (p_f0 + 1e-12)
        h3 = power_near(f0 * 3) / (p_f0 + 1e-12)

        if h2 < H2_RATIO_MIN or h3 < H3_RATIO_MIN:
            return {"detected": False, "reason": f"harmonics fail h2={h2:.3f} h3={h3:.3f}",
                    "confidence": 0.0, "f0": f0, "snr": snr}

        harmonics_detected = 2
        for k in range(4, 9):
            if power_near(f0 * k) / (p_f0 + 1e-12) >= 0.02:
                harmonics_detected += 1

        drone_key, match_score, rpm_est = classify_drone(f0, snr, harmonics_detected)

        if drone_key is None:
            return {"detected": False, "reason": "no profile match",
                    "confidence": 0.0, "f0": f0, "snr": snr}

        profile   = DRONE_DATABASE[drone_key]
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

    def process(self, result: dict, visual_state: dict = None):
        """EMA smoothing + hysteresis alarm + visual fusion."""
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

        # Visual fusion
        v = visual_state or {
            "confidence": 0.0, "alert": False, "bbox": None,
            "fps": 0.0, "model_name": "disabled", "camera_live": False,
        }
        threat_level = combined_threat(self.alarm_active, v.get("alert", False))

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
                "combined":   threat_level,
            }
            self.detection_log.insert(0, entry)
            self.detection_log = self.detection_log[:50]

        self.history.append(round(self.confidence_ema, 3))

        with self.lock:
            self.dashboard_data = {
                "status":          "DRONE DETECTED" if self.alarm_active else "MONITORING",
                "confidence":      round(self.confidence_ema * 100, 1),
                "alarm":           self.alarm_active,
                "f0_hz":           round(result.get("f0", 0)) if result.get("detected") else 0,
                "f0_smoothed":     round(f0_smoothed),
                "snr":             round(result.get("snr", 0), 1),
                "profile_label":   profile_label,
                "threat":          threat,
                "rpm_est":         rpm_est,
                "rpm_trend":       rpm_trend,
                "rate_hz_per_s":   round(rate_hz_per_s, 1),
                "history":         list(self.history),
                "log":             self.detection_log[:10],
                "frames":          self.frame_count,
                "detections":      self.total_detections,
                "combined_threat": threat_level,
                "visual":          v,
            }

    def get_state(self):
        with self.lock:
            return dict(self.dashboard_data)


# ─── Acoustic simulation ──────────────────────────────────
class SimulatedAudio:
    """
    26-second cycle: FPV 5" flyover → DJI Mavic → Shahed-136.
    Harmonic amps from prop acoustic models.
    """
    _H_AMPS = {
        "FPV_5INCH_RACING": (1.0, 0.55, 0.28, 0.14, 0.07, 0.04),
        "DJI_MAVIC_RECON":  (1.0, 0.40, 0.20, 0.10),
        "SHAHED_136":       (1.0, 0.65, 0.35, 0.20),
    }
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
        signal    = np.random.normal(0, 400, FRAME_SIZE)
        for t_start, t_end, f0_s, f0_e, base_amp, dkey in self._SEGMENTS:
            if t_start < cycle_pos < t_end:
                alpha  = (cycle_pos - t_start) / (t_end - t_start)
                f0     = f0_s + alpha * (f0_e - f0_s)
                ramp   = min((cycle_pos - t_start) / 0.3, 1.0, (t_end - cycle_pos) / 0.3)
                amp    = base_amp * ramp
                for k, h in enumerate(self._H_AMPS[dkey], start=1):
                    signal += amp * h * np.sin(2.0 * np.pi * f0 * k * t_arr)
                break
        return signal.astype(np.int16)


# ─── Visual detector ──────────────────────────────────────
class VisualDetector:
    """
    YOLOv8 drone detection on webcam frames.
    Model priority:
      1. keremberke/yolov8n-drone-detection (HuggingFace, drone-specific)
      2. yolov8n.pt (COCO, uses 'airplane' class as proxy)
    Falls back to simulation when camera/model unavailable.
    """

    # COCO class names accepted as drone proxies
    _DRONE_CLASSES = {"drone", "uav", "quadrotor", "airplane", "aircraft"}

    def __init__(self):
        self.confidence  = 0.0
        self.alert       = False
        self.bbox        = None
        self.fps         = 0.0
        self.model_name  = "disabled"
        self.camera_live = False
        self.frame_jpeg  = None          # raw JPEG bytes served by /api/frame
        self.lock        = threading.Lock()
        self.model       = None
        self.cap         = None

        if YOLO_AVAILABLE:
            self._load_model()
        if CV2_AVAILABLE and self.model is not None:
            self._open_camera()

    def _load_model(self):
        # Try drone-specific weights from HuggingFace Hub
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("keremberke/yolov8n-drone-detection", "best.pt")
            self.model      = YOLO(path)
            self.model_name = "drone-specific (HF)"
            return
        except Exception:
            pass
        # Fallback: standard COCO nano model
        try:
            self.model      = YOLO("yolov8n.pt")
            self.model_name = "YOLOv8n COCO (airplane proxy)"
        except Exception:
            self.model      = None
            self.model_name = "load-failed"

    def _open_camera(self):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            self.cap         = cap
            self.camera_live = True
        else:
            cap.release()

    def start(self):
        if not YOLO_AVAILABLE:
            print("  Visual: disabled  (pip install ultralytics)")
            return
        if self.model is None:
            print("  Visual: disabled  (model load failed)")
            return
        if self.camera_live:
            print(f"  Visual: live camera  [{self.model_name}]")
            threading.Thread(target=self._run_live,       daemon=True).start()
        else:
            print(f"  Visual: simulation   [{self.model_name}]")
            threading.Thread(target=self._run_simulation, daemon=True).start()

    # ── Simulation (no webcam) ──────────────────────────────
    def _run_simulation(self):
        """Simulate drone detection every 15 s for 2 s (confidence 0.75)."""
        t_start = time.time()
        while True:
            elapsed = time.time() - t_start
            cycle   = elapsed % 15.0
            if 2.0 < cycle < 4.0:
                conf = 0.75
                bbox = [100, 80, 260, 175]
            else:
                conf = 0.0
                bbox = None
            with self.lock:
                self.confidence = conf
                self.alert      = conf >= VISUAL_CONF_THRESHOLD
                self.bbox       = bbox
                self.fps        = 30.0
            time.sleep(1 / 30)

    # ── Live mode ──────────────────────────────────────────
    def _run_live(self):
        """Process webcam frames through YOLO at full camera FPS."""
        fps_times = deque(maxlen=30)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            results    = self.model(frame, verbose=False)
            best_conf  = 0.0
            best_bbox  = None
            annotated  = frame.copy()

            for r in results:
                names = getattr(r, "names", {})
                for box in r.boxes:
                    cls_name = names.get(int(box.cls[0]), "").lower()
                    det_conf = float(box.conf[0])
                    if not any(c in cls_name for c in self._DRONE_CLASSES):
                        continue
                    xyxy  = [int(x) for x in box.xyxy[0].tolist()]
                    color = (0, 30, 255) if det_conf >= VISUAL_CONF_THRESHOLD else (0, 200, 80)
                    cv2.rectangle(annotated, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(annotated, f"{cls_name.upper()} {det_conf*100:.0f}%",
                                (xyxy[0], xyxy[1] - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    if det_conf > best_conf:
                        best_conf = det_conf
                        best_bbox = xyxy

            # HUD overlay
            hud_color = (0, 30, 255) if best_conf >= VISUAL_CONF_THRESHOLD else (0, 180, 60)
            hud_text  = f"DRONE {best_conf*100:.0f}%" if best_conf > 0.1 else "SCANNING"
            cv2.putText(annotated, hud_text, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])

            fps_times.append(time.time())
            fps = (len(fps_times) - 1) / max(fps_times[-1] - fps_times[0], 0.001) \
                  if len(fps_times) > 1 else 0.0

            with self.lock:
                self.confidence = best_conf
                self.alert      = best_conf >= VISUAL_CONF_THRESHOLD
                self.bbox       = best_bbox
                self.fps        = round(fps, 1)
                self.frame_jpeg = bytes(buf)

    def get_state(self) -> dict:
        with self.lock:
            return {
                "confidence":  round(self.confidence * 100, 1),
                "alert":       self.alert,
                "bbox":        self.bbox,
                "fps":         self.fps,
                "model_name":  self.model_name,
                "camera_live": self.camera_live,
            }

    def get_frame_jpeg(self):
        with self.lock:
            return self.frame_jpeg


# ─── Dashboard HTML ───────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drone Detector v3</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  :root {
    --bg:#030a06; --panel:#0a1a0f; --border:#1a4025;
    --green:#00ff6a; --green-dim:#005522;
    --red:#ff2020; --yellow:#ffcc00; --text:#b0ffcc; --dim:#3a6a4a;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Share Tech Mono',monospace;min-height:100vh;padding:20px}
  .scanline{position:fixed;inset:0;pointer-events:none;z-index:100;
    background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,255,80,.015) 2px,rgba(0,255,80,.015) 4px)}
  header{display:flex;align-items:center;justify-content:space-between;
    border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:16px}
  .logo{font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;color:var(--green);letter-spacing:4px}
  .logo span{color:var(--dim)}
  .uptime{font-size:11px;color:var(--dim)}
  .g3{display:grid;grid-template-columns:2fr 1fr 1fr;gap:12px;margin-bottom:12px}
  .g2{display:grid;grid-template-columns:2fr 1fr;gap:12px;margin-bottom:12px}
  .panel{background:var(--panel);border:1px solid var(--border);padding:16px;position:relative;overflow:hidden}
  .panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,var(--green),transparent);opacity:.3}
  .ptitle{font-size:10px;color:var(--dim);letter-spacing:3px;text-transform:uppercase;margin-bottom:10px}
  .bval{font-family:'Rajdhani',sans-serif;font-size:52px;font-weight:700;line-height:1;color:var(--green)}
  .bval.alarm{color:var(--red);animation:blink .4s step-end infinite}
  .bval.warn{color:var(--yellow)}
  .unit{font-size:13px;color:var(--dim);margin-top:4px}
  @keyframes blink{50%{opacity:.2}}
  .sbox{padding:12px 20px;font-family:'Rajdhani',sans-serif;font-size:18px;font-weight:700;
    letter-spacing:3px;text-align:center;border:1px solid var(--border);background:var(--panel);transition:all .3s}
  .sbox.alarm{border-color:var(--red);background:rgba(255,32,32,.1);color:var(--red);box-shadow:0 0 20px rgba(255,32,32,.3)}
  .sbox.ok{color:var(--green)}
  .bar-wrap{display:flex;align-items:center;gap:10px;margin-top:8px}
  .bar-track{flex:1;height:8px;background:var(--green-dim)}
  .bar-fill{height:100%;background:var(--green);transition:width .2s}
  .bar-fill.alarm{background:var(--red)}.bar-fill.warn{background:var(--yellow)}
  .bar-pct{font-size:13px;color:var(--dim);min-width:42px;text-align:right}
  .tval{font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:700;line-height:1.3}
  .tval.threat-critical{color:var(--red);animation:blink .4s step-end infinite}
  .tval.threat-high{color:var(--red)}
  .tval.threat-medium{color:var(--yellow)}
  .tval.unknown{color:var(--dim)}
  .tbadge{display:inline-block;margin-top:6px;padding:2px 10px;font-size:10px;letter-spacing:2px;border:1px solid transparent}
  .tbadge.critical{border-color:var(--red);color:var(--red);background:rgba(255,32,32,.15)}
  .tbadge.high{border-color:rgba(255,32,32,.5);color:var(--red);background:rgba(255,32,32,.08)}
  .tbadge.medium{border-color:rgba(255,204,0,.5);color:var(--yellow);background:rgba(255,204,0,.08)}
  .tbadge.low{border-color:var(--border);color:var(--dim)}
  .trend-val{font-family:'Rajdhani',sans-serif;font-size:20px;font-weight:700;letter-spacing:2px;margin-top:6px}
  .trend-approaching{color:var(--red)}.trend-receding{color:var(--yellow)}.trend-stable{color:var(--green)}
  /* combined threat panel */
  .ctval{font-family:'Rajdhani',sans-serif;font-size:30px;font-weight:700;letter-spacing:3px;margin-top:8px}
  .ct-confirmed{color:var(--red);animation:blink .4s step-end infinite}
  .ct-acoustic{color:var(--yellow)}.ct-visual{color:var(--yellow)}.ct-clear{color:var(--green)}
  .ind{font-size:11px;margin-top:10px}
  .ind span{margin-right:16px}
  .ind .lbl{color:var(--dim)}
  /* visual panel */
  #cam-feed{width:100%;height:auto;display:none;border:1px solid var(--border)}
  .cam-ph{color:var(--dim);font-size:11px;text-align:center;padding:24px 0;line-height:1.8}
  canvas{width:100%;height:100px;display:block;background:#050f08}
  table{width:100%;border-collapse:collapse;font-size:11px}
  th{color:var(--dim);font-weight:normal;padding:4px 6px;text-align:right;border-bottom:1px solid var(--border)}
  td{padding:5px 6px;border-bottom:1px solid #0d1f12}
  .dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--green);
    margin-left:6px;animation:pulse 1.5s ease-in-out infinite}
  .dot.alarm{background:var(--red)}
  @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.4;transform:scale(.7)}}
  .ms{display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #0d1f12;font-size:12px}
  .ms:last-child{border:none}.ms .lbl{color:var(--dim)}.ms .val{color:var(--green)}
  .footer{margin-top:12px;padding:8px 16px;font-size:9px;color:var(--dim);
    border:1px solid var(--border);background:var(--panel);letter-spacing:.5px;line-height:1.8}
</style>
</head>
<body>
<div class="scanline"></div>
<header>
  <div>
    <div class="logo">DRONE<span>::</span>DETECT <span style="font-size:14px;letter-spacing:2px">v3</span></div>
    <div class="uptime" id="uptime">INITIALIZING...</div>
  </div>
  <div class="sbox ok" id="sbox"><span class="dot" id="dot"></span> MONITORING</div>
</header>

<!-- Row 1: confidence / f0 / SNR -->
<div class="g3">
  <div class="panel">
    <div class="ptitle">ACOUSTIC CONFIDENCE</div>
    <div class="bval" id="conf-val">0.0</div>
    <div class="unit">%</div>
    <div class="bar-wrap">
      <div class="bar-track"><div class="bar-fill" id="conf-bar" style="width:0%"></div></div>
      <div class="bar-pct" id="conf-pct">0%</div>
    </div>
  </div>
  <div class="panel">
    <div class="ptitle">FUNDAMENTAL Hz</div>
    <div class="bval" id="f0-val">—</div>
    <div class="unit">Hz (f₀ raw)</div>
    <div style="margin-top:6px;font-size:11px;color:var(--dim)">SMOOTHED: <span id="f0-sm">—</span> Hz</div>
  </div>
  <div class="panel">
    <div class="ptitle">SNR</div>
    <div class="bval" id="snr-val">—</div>
    <div class="unit">× noise floor</div>
  </div>
</div>

<!-- Row 2: drone type / RPM / direction -->
<div class="g3">
  <div class="panel">
    <div class="ptitle">IDENTIFIED TYPE</div>
    <div class="tval unknown" id="type-val">—</div>
    <div><span class="tbadge" id="threat-badge">&nbsp;</span></div>
  </div>
  <div class="panel">
    <div class="ptitle">EST. RPM</div>
    <div class="bval" id="rpm-val">—</div>
    <div class="unit">rev / min</div>
  </div>
  <div class="panel">
    <div class="ptitle">DIRECTION</div>
    <div class="trend-val trend-stable" id="trend-val">— STABLE</div>
    <div class="unit" id="rate-val">&nbsp;</div>
  </div>
</div>

<!-- Row 3: combined threat / visual detection / camera feed -->
<div class="g3">
  <div class="panel" id="ct-panel">
    <div class="ptitle">COMBINED THREAT LEVEL</div>
    <div class="ctval ct-clear" id="ct-val">CLEAR</div>
    <div class="ind">
      <span><span class="lbl">ACOUSTIC</span> <span id="ac-ind" style="color:var(--green)">CLEAR</span></span>
      <span><span class="lbl">VISUAL</span> <span id="vi-ind" style="color:var(--dim)">OFFLINE</span></span>
    </div>
  </div>
  <div class="panel">
    <div class="ptitle">VISUAL DETECTION (YOLOv8)</div>
    <div class="bval" id="vis-conf">—</div>
    <div class="unit">% confidence</div>
    <div class="ms" style="margin-top:6px"><span class="lbl">MODEL</span><span id="vis-model" style="font-size:9px;color:var(--green);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">—</span></div>
    <div class="ms"><span class="lbl">CAM FPS</span><span class="val" id="vis-fps">—</span></div>
    <div class="ms"><span class="lbl">BBOX</span><span id="vis-bbox" style="font-size:10px;color:var(--text)">—</span></div>
  </div>
  <div class="panel">
    <div class="ptitle">CAMERA FEED</div>
    <img id="cam-feed" src="" alt="">
    <div id="cam-ph" class="cam-ph">
      Simulation / No camera<br>
      <span style="font-size:9px;color:var(--dim)">pip install ultralytics opencv-python</span>
    </div>
  </div>
</div>

<!-- Row 4: chart / stats -->
<div class="g2">
  <div class="panel">
    <div class="ptitle">ACOUSTIC CONFIDENCE HISTORY</div>
    <canvas id="chart"></canvas>
  </div>
  <div class="panel">
    <div class="ptitle">SYSTEM STATS</div>
    <div class="ms"><span class="lbl">FRAMES</span><span class="val" id="frames">0</span></div>
    <div class="ms"><span class="lbl">DETECTIONS</span><span class="val" id="det-cnt">0</span></div>
    <div class="ms"><span class="lbl">ACOUSTIC SCAN</span><span class="val">80–1600 Hz</span></div>
    <div class="ms"><span class="lbl">SNR THRESHOLD</span><span class="val">per profile 3–6×</span></div>
    <div class="ms"><span class="lbl">DRIFT WINDOW</span><span class="val">10 fr / 640 ms</span></div>
    <div class="ms"><span class="lbl">PROFILES</span><span class="val">7 drone types</span></div>
    <div class="ms"><span class="lbl">VISUAL THRESH</span><span class="val">60% YOLO conf</span></div>
  </div>
</div>

<!-- Detection log -->
<div class="panel" style="margin-bottom:12px">
  <div class="ptitle">DETECTION LOG</div>
  <table>
    <thead><tr><th>TIME</th><th>TYPE</th><th>f₀ Hz</th><th>RPM</th><th>SNR</th><th>CONF.</th><th>COMBINED</th></tr></thead>
    <tbody id="log-body">
      <tr><td colspan="7" style="color:var(--dim);text-align:center;padding:12px">Waiting for detections...</td></tr>
    </tbody>
  </table>
</div>

<div class="footer">
  DATA SOURCES: AUDRON acoustic drone taxonomy (2024) &nbsp;·&nbsp;
  ScienceDirect prop acoustics — BPF = RPM/60 × num_blades &nbsp;·&nbsp;
  Drone-Warfare.com field BPF measurements &nbsp;·&nbsp;
  batear.io open-source BPF detector &nbsp;·&nbsp;
  Visual: YOLOv8 / keremberke drone-detection (HuggingFace)
</div>

<script>
const canvas = document.getElementById('chart');
const ctx    = canvas.getContext('2d');
canvas.width  = canvas.offsetWidth * 2;
canvas.height = 200;

function drawChart(history) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#050f08'; ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='#0d2010'; ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=H*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  if(history.length<2) return;
  const step=W/(history.length-1);
  ctx.beginPath(); ctx.moveTo(0,H);
  history.forEach((v,i)=>ctx.lineTo(i*step,H-v*H));
  ctx.lineTo((history.length-1)*step,H); ctx.closePath();
  ctx.fillStyle='rgba(0,255,106,.12)'; ctx.fill();
  ctx.beginPath();
  history.forEach((v,i)=>{i===0?ctx.moveTo(0,H-v*H):ctx.lineTo(i*step,H-v*H)});
  ctx.strokeStyle='#00ff6a'; ctx.lineWidth=2; ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,H*.65); ctx.lineTo(W,H*.65);
  ctx.strokeStyle='rgba(255,204,0,.3)'; ctx.lineWidth=1; ctx.setLineDash([4,4]); ctx.stroke();
  ctx.setLineDash([]);
}

const TC = {CRITICAL:'var(--red)',HIGH:'var(--red)',MEDIUM:'var(--yellow)',LOW:'var(--green)'};
const CT_CLASS = {
  'CONFIRMED':    'ctval ct-confirmed',
  'ACOUSTIC ONLY':'ctval ct-acoustic',
  'VISUAL ONLY':  'ctval ct-visual',
  'CLEAR':        'ctval ct-clear'
};

let camLive = false;
let startTime = Date.now();

function poll() {
  fetch('/api/state').then(r=>r.json()).then(d=>{
    const sec=Math.floor((Date.now()-startTime)/1000);
    document.getElementById('uptime').textContent=
      'UPTIME '+String(Math.floor(sec/3600)).padStart(2,'0')+':'+
      String(Math.floor(sec%3600/60)).padStart(2,'0')+':'+
      String(sec%60).padStart(2,'0');

    const alarm=d.alarm, warn=d.confidence>30;
    const sb=document.getElementById('sbox');
    sb.innerHTML='';
    const dotEl=document.createElement('span');
    dotEl.className='dot'+(alarm?' alarm':'');
    sb.appendChild(dotEl);
    sb.appendChild(document.createTextNode(' '+(alarm?'DRONE DETECTED':'MONITORING')));
    sb.className='sbox'+(alarm?' alarm':' ok');

    const cv=document.getElementById('conf-val');
    cv.textContent=d.confidence.toFixed(1);
    cv.className='bval'+(alarm?' alarm':warn?' warn':'');
    const bar=document.getElementById('conf-bar');
    bar.style.width=d.confidence+'%';
    bar.className='bar-fill'+(alarm?' alarm':warn?' warn':'');
    document.getElementById('conf-pct').textContent=d.confidence.toFixed(0)+'%';

    document.getElementById('f0-val').textContent=d.f0_hz>0?d.f0_hz:'—';
    document.getElementById('f0-sm').textContent=d.f0_smoothed>0?d.f0_smoothed:'—';
    document.getElementById('snr-val').textContent=d.snr>0?d.snr.toFixed(1):'—';

    const label=d.profile_label&&d.profile_label!=='—'?d.profile_label:null;
    const threat=(d.threat||'LOW').toLowerCase();
    const typeEl=document.getElementById('type-val');
    typeEl.textContent=label||'—';
    typeEl.className='tval '+(label?'threat-'+threat:'unknown');
    const badgeEl=document.getElementById('threat-badge');
    if(label){badgeEl.textContent='THREAT: '+(d.threat||'LOW');badgeEl.className='tbadge '+threat;}
    else{badgeEl.innerHTML='&nbsp;';badgeEl.className='tbadge';}

    const rpmEl=document.getElementById('rpm-val');
    rpmEl.textContent=d.rpm_est>0?d.rpm_est.toLocaleString():'—';
    rpmEl.className='bval'+(alarm?' alarm':'');

    const trend=d.rpm_trend||'STABLE';
    const rate=d.rate_hz_per_s||0;
    const arrows={APPROACHING:'▲',RECEDING:'▼',STABLE:'—'};
    const trendEl=document.getElementById('trend-val');
    trendEl.textContent=(arrows[trend]||'—')+' '+trend;
    trendEl.className='trend-val trend-'+trend.toLowerCase();
    document.getElementById('rate-val').textContent=
      Math.abs(rate)>1?Math.abs(rate).toFixed(0)+' Hz/sec drift':' ';

    // Combined threat
    const ct=d.combined_threat||'CLEAR';
    const ctEl=document.getElementById('ct-val');
    ctEl.textContent=ct;
    ctEl.className=CT_CLASS[ct]||'ctval ct-clear';

    // Acoustic indicator
    const acInd=document.getElementById('ac-ind');
    acInd.textContent=alarm?'ALERT':'CLEAR';
    acInd.style.color=alarm?'var(--red)':'var(--green)';

    // Visual state
    const v=d.visual||{};
    camLive=v.camera_live||false;
    const viInd=document.getElementById('vi-ind');
    if(v.model_name==='disabled'){viInd.textContent='OFFLINE';viInd.style.color='var(--dim)'}
    else if(v.alert){viInd.textContent='ALERT';viInd.style.color='var(--red)'}
    else{viInd.textContent='CLEAR';viInd.style.color='var(--green)'}

    const visConf=document.getElementById('vis-conf');
    visConf.textContent=v.confidence!=null?v.confidence.toFixed(1):'—';
    visConf.className='bval'+(v.alert?' alarm':v.confidence>30?' warn':'');
    document.getElementById('vis-model').textContent=v.model_name||'disabled';
    document.getElementById('vis-fps').textContent=v.fps>0?v.fps.toFixed(1):'—';
    document.getElementById('vis-bbox').textContent=v.bbox?'['+v.bbox.join(',')+']':'—';

    // Camera feed visibility
    const camFeedEl=document.getElementById('cam-feed');
    const camPh=document.getElementById('cam-ph');
    if(camLive){camFeedEl.style.display='block';camPh.style.display='none';}
    else{camFeedEl.style.display='none';camPh.style.display='block';}

    document.getElementById('frames').textContent=d.frames.toLocaleString();
    document.getElementById('det-cnt').textContent=d.detections||0;
    if(d.history) drawChart(d.history);

    // Log
    const tbody=document.getElementById('log-body');
    if(d.log&&d.log.length>0){
      tbody.innerHTML=d.log.map(e=>{
        const tc=TC[e.threat]||'var(--green)';
        const ctc={'CONFIRMED':'var(--red)','ACOUSTIC ONLY':'var(--yellow)','VISUAL ONLY':'var(--yellow)','CLEAR':'var(--green)'};
        return '<tr>'+
          '<td>'+e.time+'</td>'+
          '<td style="color:'+tc+'">'+( e.type||'—')+'</td>'+
          '<td>'+e.f0+' Hz</td>'+
          '<td>'+(e.rpm?e.rpm.toLocaleString():'—')+'</td>'+
          '<td>'+e.snr+'×</td>'+
          '<td>'+(e.confidence*100).toFixed(0)+'%</td>'+
          '<td style="color:'+(ctc[e.combined]||'var(--green)')+'">'+( e.combined||'—')+'</td>'+
          '</tr>';
      }).join('');
    }
  }).catch(()=>{});
}

// Camera feed refresh — only when live
setInterval(()=>{
  if(camLive){
    document.getElementById('cam-feed').src='/api/frame?'+Date.now();
  }
},500);

setInterval(poll,300);
poll();
</script>
</body>
</html>"""


# ─── HTTP server ──────────────────────────────────────────
def start_web_server(detector: DroneDetector,
                     visual: VisualDetector,
                     port: int = 8765):
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

            elif self.path.startswith("/api/frame"):
                jpeg = visual.get_frame_jpeg()
                if jpeg:
                    self.send_response(200)
                    self.send_header("Content-Type",  "image/jpeg")
                    self.send_header("Cache-Control", "no-cache, no-store")
                    self.end_headers()
                    self.wfile.write(jpeg)
                else:
                    self.send_response(204)
                    self.end_headers()

            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(DASHBOARD_HTML.encode())

    server = HTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ─── Entry point ──────────────────────────────────────────
def main():
    print("=" * 58)
    print("  DRONE DETECTOR — POC v3")
    print("  Acoustic BPF + YOLOv8 visual fusion")
    print("=" * 58)

    detector = DroneDetector()
    visual   = VisualDetector()
    visual.start()

    port = 8765
    start_web_server(detector, visual, port)
    print(f"\n  Dashboard: http://localhost:{port}\n")

    if AUDIO_AVAILABLE:
        print("  Acoustic: live microphone")
        p      = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                        input=True, frames_per_buffer=FRAME_SIZE)
        def read_frame():
            raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.int16)
    else:
        print('  Acoustic: simulation  (FPV 5" → DJI Mavic → Shahed-136, 26s cycle)')
        sim = SimulatedAudio()
        read_frame = sim.read_frame

    print("  Ctrl+C to stop\n")

    THREAT_ICON = {
        "CONFIRMED":    "[CONFIRMED]",
        "ACOUSTIC ONLY":"[ACOUSTIC] ",
        "VISUAL ONLY":  "[VISUAL]   ",
        "CLEAR":        "           ",
    }

    try:
        while True:
            samples      = read_frame()
            result       = detector.analyze_frame(samples)
            visual_state = visual.get_state()
            detector.process(result, visual_state)

            if detector.frame_count % 10 == 0:
                state  = detector.get_state()
                conf   = detector.confidence_ema
                bar    = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                ct     = state.get("combined_threat", "CLEAR")
                label  = state["profile_label"]
                label_s = f"  [{label}]" if label != "—" else ""
                rpm_s  = f"  {state['rpm_est']:,}rpm" if state["rpm_est"] else ""
                arrows = {"APPROACHING": " ↑", "RECEDING": " ↓", "STABLE": ""}
                trend_s = arrows.get(state.get("rpm_trend", ""), "")
                print(f"\r  [{bar}] {conf*100:5.1f}%  {THREAT_ICON.get(ct,ct)}{label_s}{rpm_s}{trend_s}  ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
        if AUDIO_AVAILABLE:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    main()
