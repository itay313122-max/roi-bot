#!/usr/bin/env python3
"""
web_ui.py — FPV Drone Detector — Web HUD + Field Test Wizard
Completely standalone: all detector logic embedded, no external dependencies
beyond fastapi/uvicorn/numpy.  Works on Linux, macOS, and Windows.

Install: pip install fastapi uvicorn websockets numpy
         pip install pyaudio        # optional: live microphone
         pip install ultralytics    # optional: YOLOv8 visual layer
         pip install opencv-python  # optional: webcam

Usage:
  python web_ui.py                 # auto-detect hardware
  python web_ui.py --simulate      # force simulation
  python web_ui.py --no-camera     # acoustic only
  python web_ui.py --port 8080
  python web_ui.py --no-browser    # don't open browser
"""

import argparse
import asyncio
import csv
import json
import math
import sys
import threading
import time
import urllib.request
import wave
import webbrowser
from datetime import datetime
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
except ImportError:
    print("ERROR: pip install fastapi uvicorn websockets")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: pip install numpy")
    sys.exit(1)

from collections import deque

# ─── Acoustic constants ───────────────────────────────────────────────

SAMPLE_RATE   = 16000
FRAME_SIZE    = 1024
SCAN_MIN_HZ   = 80
SCAN_MAX_HZ   = 1600
H2_RATIO_MIN  = 0.07
H3_RATIO_MIN  = 0.035
EMA_ALPHA     = 0.25
ALARM_FRAMES  = 2
RMS_GATE      = 0.0004      # updated by calibration
MIN_SNR       = 3.0
VISUAL_THRESH = 0.60
VERSION       = "3.0"
CAL_SECS      = 30
SPEC_BINS     = 64                      # spectrogram frequency bins
SPEC_FREQ_MIN = 80.0
SPEC_FREQ_MAX = 4000.0
# log-spaced bin edges 80 → 4000 Hz (computed without numpy)
_SPEC_EDGES   = [SPEC_FREQ_MIN * (SPEC_FREQ_MAX / SPEC_FREQ_MIN) ** (i / SPEC_BINS)
                 for i in range(SPEC_BINS + 1)]
REC_PRE_S     = 5                       # seconds of pre-alarm buffer to save
REC_POST_S    = 5                       # seconds of post-alarm buffer to save
REC_MAX       = 20                      # keep at most N recordings

# ─── Drone database ───────────────────────────────────────────────────

DRONE_DATABASE = {
    "FPV_5INCH_RACING": {
        "label": 'FPV 5" Racing (Kamikaze)', "f0_range": (667, 1200),
        "harmonics": 6, "snr_threshold": 6.0, "num_blades": 2, "threat": "HIGH",
    },
    "FPV_3INCH_MICRO": {
        "label": 'FPV 3" Micro (Suicide)', "f0_range": (833, 1500),
        "harmonics": 5, "snr_threshold": 5.0, "num_blades": 2, "threat": "HIGH",
    },
    "FPV_7INCH_LR": {
        "label": 'FPV 7" Long Range', "f0_range": (400, 733),
        "harmonics": 5, "snr_threshold": 5.0, "num_blades": 2, "threat": "HIGH",
    },
    "DJI_MAVIC_RECON": {
        "label": "DJI Mavic (Recon / ISR)", "f0_range": (150, 217),
        "harmonics": 4, "snr_threshold": 4.0, "num_blades": 2, "threat": "MEDIUM",
    },
    "DJI_MATRICE_HEAVY": {
        "label": "DJI Matrice (Heavy Lift)", "f0_range": (100, 167),
        "harmonics": 4, "snr_threshold": 3.0, "num_blades": 2, "threat": "MEDIUM",
    },
    "SHAHED_136": {
        "label": "Shahed-136 / Geran-2", "f0_range": (200, 283),
        "harmonics": 4, "snr_threshold": 3.0, "num_blades": 2, "threat": "CRITICAL",
    },
    "LANCET_LOITERING": {
        "label": "Lancet Loitering Munition", "f0_range": (267, 500),
        "harmonics": 4, "snr_threshold": 4.0, "num_blades": 2, "threat": "HIGH",
    },
}


def _classify_drone(f0: float, snr: float, harmonics: int) -> tuple:
    best_key, best_score, best_rpm = None, 0.0, 0
    for key, p in DRONE_DATABASE.items():
        lo, hi = p["f0_range"]
        if not (lo <= f0 <= hi) or snr < p["snr_threshold"]:
            continue
        center     = (lo + hi) / 2.0
        centrality = 1.0 - abs(f0 - center) / ((hi - lo) / 2.0)
        snr_score  = min(snr / (p["snr_threshold"] * 3.0), 1.0)
        h_score    = min(harmonics / p["harmonics"], 1.0)
        score      = centrality * 0.4 + snr_score * 0.4 + h_score * 0.2
        if score > best_score:
            best_score, best_key = score, key
            best_rpm = int(f0 * 60 / p["num_blades"])
    return best_key, best_score, best_rpm


def _combined_threat(acoustic: bool, visual: bool) -> str:
    if acoustic and visual: return "CONFIRMED"
    if acoustic:            return "ACOUSTIC ONLY"
    if visual:              return "VISUAL ONLY"
    return "CLEAR"


def _range_est(snr: float, alarm: bool) -> str:
    if not alarm or snr <= 0: return "—"
    if snr >= 15: return "NEAR  (<50m)"
    if snr >= 8:  return "MEDIUM  (50–200m)"
    return "FAR  (>200m)"


_RANGE_METRES = {"NEAR  (<50m)": 50, "MEDIUM  (50–200m)": 150, "FAR  (>200m)": 400}


def _offset_position(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple:
    """Return (lat2, lon2) at dist_m metres from (lat, lon) along bearing_deg."""
    R    = 6371000.0
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    b    = math.radians(bearing_deg)
    lat2 = math.asin(math.sin(lat1) * math.cos(dist_m / R) +
                     math.cos(lat1) * math.sin(dist_m / R) * math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b) * math.sin(dist_m / R) * math.cos(lat1),
                              math.cos(dist_m / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


_CSV_FIELDS = ["timestamp", "acoustic_confidence", "visual_confidence",
               "combined_threat", "f0_hz", "rpm_estimate",
               "drone_type", "direction", "distance_phase"]
_DP = {"APPROACHING": "INBOUND", "RECEDING": "OUTBOUND", "STABLE": "HOVER/OVERHEAD"}


def _csv_row(s: dict) -> dict:
    v     = (s or {}).get("visual") or {}
    trend = s.get("rpm_trend", "STABLE")
    return {"timestamp":           datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
            "acoustic_confidence": round(s.get("confidence", 0.0), 1),
            "visual_confidence":   round(v.get("confidence", 0.0), 1),
            "combined_threat":     s.get("combined_threat", "CLEAR"),
            "f0_hz":               s.get("f0_hz", 0),
            "rpm_estimate":        s.get("rpm_est", 0),
            "drone_type":          s.get("profile_label", "—"),
            "direction":           trend,
            "distance_phase":      _DP.get(trend, "N/A")}


# ─── Acoustic detector ────────────────────────────────────────────────

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
        self._lock            = threading.Lock()
        self._state           = {
            "status": "MONITORING", "confidence": 0.0, "alarm": False,
            "f0_hz": 0, "snr": 0.0, "profile_label": "—", "threat": "LOW",
            "rpm_est": 0, "rpm_trend": "STABLE", "rate_hz_per_s": 0.0,
            "history": [], "log": [], "frames": 0, "detections": 0,
            "combined_threat": "CLEAR", "visual": {}, "fft_bins": [],
        }

    def _f0_drift(self) -> tuple:
        vals = list(self.f0_window)
        if not vals:
            return 0.0, "STABLE", 0.0
        n        = len(vals)
        weights  = list(range(1, n + 1))
        smoothed = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
        if n < 4:
            return smoothed, "STABLE", 0.0
        mid           = n // 2
        rate_hz_per_s = (sum(vals[mid:]) / (n - mid) - sum(vals[:mid]) / mid) \
                        / (mid * (FRAME_SIZE / SAMPLE_RATE))
        trend = ("APPROACHING" if rate_hz_per_s > 15
                 else "RECEDING" if rate_hz_per_s < -15
                 else "STABLE")
        return smoothed, trend, rate_hz_per_s

    def analyze_frame(self, samples: np.ndarray) -> dict:
        self.frame_count += 1

        x     = samples.astype(np.float32) / 32768.0
        fft   = np.fft.rfft(x * np.hanning(len(x)), n=FRAME_SIZE)
        psd   = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)

        # Spectrogram: log-spaced bins, always computed, 0–255 normalised
        spec_power, _ = np.histogram(freqs, bins=_SPEC_EDGES, weights=psd)
        sp_peak       = float(spec_power.max()) + 1e-12
        fft_bins      = [min(255, int(float(v) / sp_peak * 255)) for v in spec_power]

        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms < RMS_GATE:
            return {"detected": False, "confidence": 0.0, "fft_bins": fft_bins}

        mask     = (freqs >= SCAN_MIN_HZ) & (freqs <= SCAN_MAX_HZ)
        scan     = psd.copy(); scan[~mask] = 0.0
        peak_bin = np.argmax(scan)
        f0       = float(freqs[peak_bin])
        p_f0     = float(psd[peak_bin])
        snr      = p_f0 / (float(np.median(psd[mask])) + 1e-12)

        if snr < MIN_SNR:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr,
                    "fft_bins": fft_bins}

        hw = max(SAMPLE_RATE / FRAME_SIZE * 2, f0 * 0.07)

        def _pw(hz):
            m = (freqs >= hz - hw) & (freqs <= hz + hw)
            return float(np.max(psd[m])) if np.any(m) else 0.0

        h2 = _pw(f0 * 2) / (p_f0 + 1e-12)
        h3 = _pw(f0 * 3) / (p_f0 + 1e-12)
        if h2 < H2_RATIO_MIN or h3 < H3_RATIO_MIN:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr,
                    "fft_bins": fft_bins}

        n_harm = 2 + sum(1 for k in range(4, 9)
                         if _pw(f0 * k) / (p_f0 + 1e-12) >= 0.02)
        key, match, rpm = _classify_drone(f0, snr, n_harm)
        if key is None:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr,
                    "fft_bins": fft_bins}

        p    = DRONE_DATABASE[key]
        conf = min((min(snr / (p["snr_threshold"] * 3.0), 1.0) * 0.5
                    + min(h2 / 0.3, 1.0) * 0.3
                    + min(h3 / 0.15, 1.0) * 0.2) * 0.7 + match * 0.3, 1.0)
        return {"detected": True, "confidence": conf, "f0": f0, "snr": snr,
                "drone_key": key, "profile_label": p["label"],
                "threat": p["threat"], "rpm_est": rpm, "fft_bins": fft_bins}

    def process(self, result: dict, visual_state: dict = None):
        raw = result.get("confidence", 0.0)
        self.confidence_ema = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self.confidence_ema
        if self.confidence_ema > 0.35:
            self.alarm_count = min(self.alarm_count + 1, ALARM_FRAMES + 2)
        else:
            self.alarm_count = max(0, self.alarm_count - 1)
        prev              = self.alarm_active
        self.alarm_active = self.alarm_count >= ALARM_FRAMES

        if result.get("detected") and result.get("f0"):
            self.f0_window.append(result["f0"])
        f0s, trend, rate = self._f0_drift()

        label  = result.get("profile_label", "—") if result.get("detected") else "—"
        threat = result.get("threat",         "LOW") if result.get("detected") else "LOW"
        rpm    = result.get("rpm_est",         0)    if result.get("detected") else 0

        v  = visual_state or {"confidence": 0.0, "alert": False, "fps": 0.0,
                               "model_name": "disabled", "camera_live": False}
        ct = _combined_threat(self.alarm_active, v.get("alert", False))

        if self.alarm_active and not prev:
            self.total_detections += 1
            entry = {"time": datetime.now().strftime("%H:%M:%S.%f")[:-3], "type": label,
                     "threat": threat, "f0": round(result.get("f0", 0)),
                     "snr": round(result.get("snr", 0), 1),
                     "confidence": round(self.confidence_ema, 2),
                     "rpm": rpm, "trend": trend, "combined": ct}
            self.detection_log = [entry] + self.detection_log[:49]

        self.history.append(round(self.confidence_ema, 3))
        with self._lock:
            self._state = {
                "status":          "DRONE DETECTED" if self.alarm_active else "MONITORING",
                "confidence":      round(self.confidence_ema * 100, 1),
                "alarm":           self.alarm_active,
                "f0_hz":           round(result.get("f0", 0)) if result.get("detected") else 0,
                "snr":             round(result.get("snr", 0), 1),
                "profile_label":   label, "threat": threat, "rpm_est": rpm,
                "rpm_trend":       trend, "rate_hz_per_s": round(rate, 1),
                "history":         list(self.history),
                "log":             self.detection_log[:10],
                "frames":          self.frame_count,
                "detections":      self.total_detections,
                "combined_threat": ct, "visual": v,
                "fft_bins":        result.get("fft_bins", []),
            }

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def reset(self):
        self.confidence_ema   = 0.0
        self.alarm_count      = 0
        self.alarm_active     = False
        self.detection_log    = []
        self.history.clear()
        self.f0_window.clear()
        self.frame_count      = 0
        self.total_detections = 0


# ─── Simulated audio ──────────────────────────────────────────────────

class SimulatedAudio:
    """26s cycle: FPV 5" flyover → DJI Mavic → Shahed-136."""
    _H = {"FPV_5INCH_RACING": (1.0, 0.55, 0.28, 0.14, 0.07, 0.04),
          "DJI_MAVIC_RECON":  (1.0, 0.40, 0.20, 0.10),
          "SHAHED_136":       (1.0, 0.65, 0.35, 0.20)}
    _SEG = (( 2.0,  5.0,  700.0,  950.0,  120, "FPV_5INCH_RACING"),
            ( 5.0,  7.0,  950.0,  700.0,   80, "FPV_5INCH_RACING"),
            (10.0, 14.0,  183.0,  183.0,   95, "DJI_MAVIC_RECON"),
            (17.0, 22.0,  242.0,  242.0,   85, "SHAHED_136"))
    _CYCLE = 26.0

    def __init__(self):
        self._t = 0

    def read_frame(self):
        t_arr    = np.arange(self._t, self._t + FRAME_SIZE, dtype=np.float64) / SAMPLE_RATE
        self._t += FRAME_SIZE
        pos      = (self._t / SAMPLE_RATE) % self._CYCLE
        for t0, t1, f0s, f0e, amp, dk in self._SEG:
            if t0 < pos < t1:
                f0   = f0s + (pos - t0) / (t1 - t0) * (f0e - f0s)
                ramp = min((pos - t0) / 0.3, 1.0, (t1 - pos) / 0.3)
                sig  = np.random.normal(0, 400, FRAME_SIZE)
                for k, h in enumerate(self._H[dk], 1):
                    sig += amp * ramp * h * np.sin(2 * np.pi * f0 * k * t_arr)
                return sig.astype(np.int16)
        return np.zeros(FRAME_SIZE, dtype=np.int16)


# ─── Recording manager ────────────────────────────────────────────────

class RecordingManager:
    """Saves WAV clips: REC_PRE_S before + REC_POST_S after each alarm onset."""

    def __init__(self):
        _pre_frames        = SAMPLE_RATE * REC_PRE_S // FRAME_SIZE
        self._pre          = deque(maxlen=_pre_frames)
        self._active       = False
        self._post_target  = SAMPLE_RATE * REC_POST_S // FRAME_SIZE
        self._post_count   = 0
        self._buf          = []
        self._last_alarm   = False
        self._lock         = threading.Lock()
        self.recordings: list = []      # file paths, newest last

    def feed(self, samples: np.ndarray, alarm: bool):
        self._pre.append(samples.copy())

        if alarm and not self._last_alarm and not self._active:
            # Alarm onset → start capture
            self._active      = True
            self._post_count  = 0
            self._buf         = list(self._pre)

        if self._active:
            self._post_count += 1
            if self._post_count > 1:        # first frame already in pre-buffer copy
                self._buf.append(samples.copy())
            if self._post_count >= self._post_target:
                self._save()
                self._active = False

        self._last_alarm = alarm

    def _save(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"alarm_{ts}.wav"
        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                for frame in self._buf:
                    wf.writeframes(frame.astype(np.int16).tobytes())
            with self._lock:
                self.recordings.append(path)
                while len(self.recordings) > REC_MAX:
                    old = self.recordings.pop(0)
                    try:
                        Path(old).unlink(missing_ok=True)
                    except Exception:
                        pass
        except Exception:
            pass

    def list(self) -> list:
        with self._lock:
            return [{"name": Path(p).name, "url": f"/{p}"}
                    for p in reversed(self.recordings)]


# ─── Visual detector ──────────────────────────────────────────────────

class VisualDetector:
    _CLASSES = {"drone", "uav", "quadrotor", "airplane", "aircraft"}

    def __init__(self, no_camera: bool = False, force_simulate: bool = False):
        self.confidence  = 0.0
        self.alert       = False
        self.fps         = 0.0
        self.camera_live = False
        self.model_name  = "disabled"
        self._lock       = threading.Lock()
        self._disabled   = no_camera
        self._force_sim  = force_simulate
        self.model       = None
        self.cap         = None

        if no_camera or force_simulate:
            if force_simulate:
                self.model_name = "SIMULATION"
            return

        try:
            from ultralytics import YOLO as _YOLO
            self._load_model(_YOLO)
        except ImportError:
            return
        if self.model is not None:
            self._open_camera()

    def _load_model(self, YOLO):
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("keremberke/yolov8n-drone-detection", "best.pt")
            self.model = YOLO(path); self.model_name = "drone-specific (HF)"; return
        except Exception:
            pass
        try:
            self.model = YOLO("yolov8n.pt"); self.model_name = "YOLOv8n COCO"
        except Exception:
            self.model = None; self.model_name = "load-failed"

    def _open_camera(self):
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS,          30)
                self.cap = cap; self.camera_live = True
            else:
                cap.release()
        except Exception:
            pass

    def start(self):
        if self._disabled:
            return
        if self._force_sim or self.model is None:
            if self.model_name not in ("disabled", "load-failed"):
                threading.Thread(target=self._run_sim, daemon=True).start()
            return
        target = self._run_live if self.camera_live else self._run_sim
        threading.Thread(target=target, daemon=True).start()

    def _run_sim(self):
        t0 = time.time()
        while True:
            cycle = (time.time() - t0) % 15.0
            conf  = 0.75 if 2.0 < cycle < 4.0 else 0.0
            with self._lock:
                self.confidence = conf
                self.alert      = conf >= VISUAL_THRESH
                self.fps        = 30.0
            time.sleep(1 / 30)

    def _run_live(self):
        fps_dq = deque(maxlen=30)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01); continue
            results   = self.model(frame, verbose=False)
            best_conf = 0.0
            for r in results:
                names = getattr(r, "names", {})
                for det in r.boxes:
                    cls_name = names.get(int(det.cls[0]), "").lower()
                    if any(c in cls_name for c in self._CLASSES):
                        best_conf = max(best_conf, float(det.conf[0]))
            fps_dq.append(time.time())
            fps = (len(fps_dq) - 1) / max(fps_dq[-1] - fps_dq[0], 0.001) \
                  if len(fps_dq) > 1 else 0.0
            with self._lock:
                self.confidence = best_conf
                self.alert      = best_conf >= VISUAL_THRESH
                self.fps        = round(fps, 1)

    def get_state(self) -> dict:
        with self._lock:
            return {"confidence":  round(self.confidence * 100, 1),
                    "alert":       self.alert,
                    "fps":         self.fps,
                    "model_name":  self.model_name,
                    "camera_live": self.camera_live}


# ─── Audio thread ─────────────────────────────────────────────────────

class _AudioThread:
    def __init__(self, detector: DroneDetector, visual: VisualDetector,
                 read_frame_fn):
        self._det  = detector
        self._vis  = visual
        self._rf   = read_frame_fn
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                samples = self._rf()
                result  = self._det.analyze_frame(samples)
                self._det.process(result, self._vis.get_state())
                if _recorder is not None:
                    _recorder.feed(samples, self._det.get_state().get("alarm", False))
            except Exception:
                time.sleep(0.01)


# ─── Inline calibration ───────────────────────────────────────────────

_cal_lock  = threading.Lock()
_cal_state: dict = {"running": False, "progress": 0.0, "elapsed": 0.0,
                    "duration": 0.0, "frames": 0,
                    "noise_floor": None, "rms_gate": None}


def _calibrate_terminal(read_frame_fn, duration_s: float = 30.0) -> float:
    print(f"\n{'='*58}")
    print("  CALIBRATION -- Keep area clear of drones")
    print(f"  Measuring noise floor for {duration_s:.0f}s ...")
    print(f"{'='*58}\n")
    rms_values, t0, t_ui = [], time.time(), time.time() - 1.0
    while True:
        elapsed = time.time() - t0
        if elapsed >= duration_s:
            break
        samples = read_frame_fn()
        rms_values.append(float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))))
        if time.time() - t_ui >= 0.2:
            t_ui = time.time()
            pct  = elapsed / duration_s
            bar  = "#" * int(pct * 40) + "." * (40 - int(pct * 40))
            print(f"\r  [{bar}] {elapsed:4.0f}/{duration_s:.0f}s  "
                  f"frames: {len(rms_values)}", end="", flush=True)
    arr  = np.array(rms_values, dtype=np.float64)
    nf   = float(np.median(arr))
    gate = round(max(nf + 3.0 * np.std(arr), nf * 2.5), 6)
    print(f"\n\n  DONE -- noise floor: {nf:.6f}   rms_gate: {gate}\n")
    return gate


def _start_inline_calibration(read_frame_fn, duration_s: float = CAL_SECS):
    with _cal_lock:
        if _cal_state["running"]:
            return
        _cal_state.update({"running": True, "progress": 0.0, "elapsed": 0.0,
                           "duration": duration_s, "frames": 0,
                           "noise_floor": None, "rms_gate": None})

    def _run():
        rms_values, t0 = [], time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= duration_s:
                break
            samples = read_frame_fn()
            rms_values.append(float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))))
            with _cal_lock:
                _cal_state.update({"elapsed": elapsed,
                                   "progress": min(elapsed / duration_s, 1.0),
                                   "frames": len(rms_values)})
        arr  = np.array(rms_values, dtype=np.float64)
        nf   = float(np.median(arr))
        gate = round(max(nf + 3.0 * np.std(arr), nf * 2.5), 6)
        with _cal_lock:
            _cal_state.update({"running": False, "progress": 1.0,
                               "noise_floor": nf, "rms_gate": gate})

    threading.Thread(target=_run, daemon=True).start()


def _get_cal() -> dict:
    with _cal_lock:
        return dict(_cal_state)

# ─── Global state ─────────────────────────────────────────────────────

_detector: DroneDetector = None
_visual:   VisualDetector = None
_audio_th: _AudioThread = None
_read_frame = None
_mic_live   = False
_mode       = "SIMULATION"
_t0         = time.monotonic()

_clients: set = set()

_logging    = False
_csv_fh     = None
_csv_wr     = None
_csv_path   = ""
_csv_rows   = 0

_cal_prev_running = False

_user_lat = 0.0
_user_lon = 0.0

_recorder: RecordingManager = None

_telegram_token   = ""
_telegram_chat_id = ""
_telegram_prev    = "CLEAR"


def _send_telegram(text: str) -> None:
    if not _telegram_token or not _telegram_chat_id:
        return
    url  = f"https://api.telegram.org/bot{_telegram_token}/sendMessage"
    body = json.dumps({"chat_id": _telegram_chat_id, "text": text,
                       "parse_mode": "HTML"}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# ─── Field Test Wizard ────────────────────────────────────────────────

class Wizard:
    STEPS = [
        {"id": 1, "name": "BASELINE",      "title": "Noise Floor",        "duration": 30, "skippable": False},
        {"id": 2, "name": "DRONE UP",       "title": "Initial Detection",  "duration": 60, "skippable": False},
        {"id": 3, "name": "APPROACH",       "title": "Doppler Test",       "duration": 60, "skippable": True},
        {"id": 4, "name": "RANGE 50m",      "title": "Distance Test",      "duration": 15, "skippable": True},
        {"id": 5, "name": "MULTI TYPE",     "title": "Multi-Drone Type",   "duration": 30, "skippable": True},
        {"id": 6, "name": "VISUAL",         "title": "Visual Confirm",     "duration": 30, "skippable": True},
        {"id": 7, "name": "REPORT",         "title": "Final Report",       "duration": 0,  "skippable": False},
    ]
    INSTRUCTIONS = {
        1: "אין רחפן באוויר — לחץ BEGIN למדידת רעש רקע (30 שניות)",
        2: "הפעל את הרחפן — לחץ BEGIN ואז טוס אותו לכיוון המיק",
        3: "טוס את הרחפן ישר לכיוון המיק לאחר BEGIN",
        4: "עמוד עם הרחפן ב-~50 מטר — לחץ BEGIN כשיציב",
        5: "הפעל רחפן שני אם יש — לחץ BEGIN, או לחץ SKIP",
        6: "כוון מצלמה לרחפן — לחץ BEGIN כשמוכן",
        7: "מייצר דוח בדיקה...",
    }

    def __init__(self):
        self.active          = False
        self.current_step    = 0
        self.step_status     = {}   # id -> pending/waiting/running/pass/fail/skip
        self.step_data       = {}   # id -> result dict
        self.step_start_time = None
        self.accumulated     = []
        self._approach_cons  = 0.0
        self._approach_dir   = None
        self._report_path    = ""
        self._overall        = ""

    def start(self):
        self.active       = True
        self.current_step = 1
        self.step_status  = {i: "pending" for i in range(1, 8)}
        self.step_data    = {i: {} for i in range(1, 8)}
        self.step_status[1] = "waiting"
        self._reset_accum()

    def begin(self):
        """User clicks BEGIN for current waiting step."""
        step = self.current_step
        if self.step_status.get(step) == "waiting":
            self.step_status[step] = "running"
            self.step_start_time   = time.monotonic()
            self._reset_accum()

    def skip(self):
        step = self.current_step
        if self.step_status.get(step) in ("waiting", "running", "fail"):
            self.step_status[step] = "skip"
            self.step_data[step]   = {"message": "דולג"}
            self._advance()

    def retry(self):
        step = self.current_step
        if self.step_status.get(step) in ("fail", "running"):
            self.step_status[step] = "waiting"
            self.step_data[step]   = {}
            self._reset_accum()

    def stop(self):
        self.active = False

    def tick(self, s: dict):
        """Called every broadcast cycle with current detector state."""
        step = self.current_step
        if not self.active or self.step_status.get(step) != "running":
            return
        elapsed = time.monotonic() - (self.step_start_time or 0)
        self.accumulated.append(s)

        if step == 1:   self._baseline(elapsed, s)
        elif step == 2: self._drone_up(elapsed, s)
        elif step == 3: self._approach(elapsed, s)
        elif step == 4: self._distance(elapsed, s)
        elif step == 5: self._multi_type(elapsed, s)
        elif step == 6: self._visual(elapsed, s)

    # ── step evaluators ───────────────────────────────────────────────

    def _baseline(self, elapsed, s):
        if elapsed < 30.0:
            return
        confs = [d.get("confidence", 0) for d in self.accumulated]
        max_c = max(confs) if confs else 0
        gate  = RMS_GATE
        if max_c < 20.0:
            self._pass(1, {"max_confidence": round(max_c, 1), "rms_gate": gate,
                           "message": f"רעש רקע תקין — rms_gate: {gate:.6f}"})
        else:
            self._fail(1, {"max_confidence": round(max_c, 1),
                           "message": f"רעש רקע גבוה ({max_c:.0f}%) — הרחק מיק ממאוורר"})

    def _drone_up(self, elapsed, s):
        if s.get("confidence", 0) > 40.0 and s.get("alarm"):
            label = s.get("profile_label", "?")
            f0    = s.get("f0_hz", 0)
            snr   = round(s.get("snr", 0), 1)
            self._pass(2, {"drone_type": label, "f0_hz": f0, "snr": snr,
                           "message": f'רחפן זוהה! {label} @ {f0}Hz SNR:{snr}×'})
        elif elapsed >= 60.0:
            self._fail(2, {"message": "לא זוהה — נסה להתקרב עם הרחפן"})

    def _approach(self, elapsed, s):
        direction = s.get("rpm_trend", "STABLE")
        if direction == "APPROACHING":
            self._approach_cons = (self._approach_cons + 0.2
                                   if self._approach_dir == "APPROACHING" else 0.2)
        else:
            self._approach_cons = 0.0
        self._approach_dir = direction
        rate = s.get("rate_hz_per_s", 0.0)
        if self._approach_cons >= 5.0:
            self._pass(3, {"consecutive_s": round(self._approach_cons, 1),
                           "rate_hz_per_s": round(rate, 1),
                           "message": f"Doppler מזוהה — {rate:+.1f}Hz/sec APPROACHING ✓"})
        elif elapsed >= 60.0:
            self._fail(3, {"message": "לא זוהה התקרבות — נסה מהירות גבוהה יותר"})

    def _distance(self, elapsed, s):
        if elapsed < 15.0:
            return
        confs = [d.get("confidence", 0) for d in self.accumulated]
        avg   = sum(confs) / len(confs) if confs else 0
        if avg >= 30.0:
            self._pass(4, {"avg_confidence": round(avg, 1),
                           "message": f"זיהוי מ-50מ' תקין — confidence ממוצע {avg:.1f}%"})
        else:
            self._fail(4, {"avg_confidence": round(avg, 1),
                           "message": f"לא מזוהה מ-50מ' — confidence {avg:.1f}% < 30%"})

    def _multi_type(self, elapsed, s):
        types = set(d.get("profile_label", "—") for d in self.accumulated if d.get("alarm"))
        if len(types) >= 2:
            self._pass(5, {"types": list(types),
                           "message": f"זוהו {len(types)} סוגים: {', '.join(types)}"})
        elif elapsed >= 30.0:
            self._fail(5, {"types": list(types),
                           "message": "סוג אחד בלבד — skip אם אין רחפן שני"})

    def _visual(self, elapsed, s):
        vis = (s.get("visual") or {})
        if vis.get("alert") and s.get("combined_threat") == "CONFIRMED":
            self._pass(6, {"visual_conf": round(vis.get("confidence", 0), 1),
                           "message": "CONFIRMED — אקוסטי + ויזואלי ✓"})
        elif elapsed >= 30.0:
            self._fail(6, {"message": "רק אקוסטי — מצלמה לא מזהה (בדוק כיוון/מרחק)"})

    # ── state helpers ─────────────────────────────────────────────────

    def _pass(self, sid, data):
        self.step_status[sid] = "pass"
        self.step_data[sid]   = data
        self._advance()

    def _fail(self, sid, data):
        self.step_status[sid] = "fail"
        self.step_data[sid]   = data
        # Don't auto-advance on fail — user must retry or skip

    def _advance(self):
        nxt = self.current_step + 1
        if nxt <= 6:
            self.current_step      = nxt
            self.step_status[nxt]  = "waiting"
            self._reset_accum()
        else:
            self.current_step    = 7
            self.step_status[7]  = "running"
            self._generate_report()

    def _reset_accum(self):
        self.accumulated    = []
        self._approach_cons = 0.0
        self._approach_dir  = None

    def _generate_report(self):
        names = {1:"Baseline",2:"Detection",3:"Doppler",4:"Range 50m",5:"Multi-type",6:"Visual"}
        passed  = sum(1 for i in range(1,7) if self.step_status.get(i) == "pass")
        failed  = sum(1 for i in range(1,7) if self.step_status.get(i) == "fail")
        skipped = sum(1 for i in range(1,7) if self.step_status.get(i) == "skip")
        overall = ("OPERATIONAL ✅" if failed == 0
                   else "DEGRADED ⚠️" if failed <= 2
                   else "FAILED ❌")
        self._overall = overall

        icon_map = {"pass":"✅","fail":"❌","skip":"⏭️","pending":"⏸️","waiting":"⏸️","running":"⏳"}
        rows = "".join(
            f"<tr class='{self.step_status.get(i,'pending')}'>"
            f"<td>{icon_map.get(self.step_status.get(i,''),'⏸️')}</td>"
            f"<td>{names[i]}</td>"
            f"<td>{self.step_data.get(i,{}).get('message','')}</td></tr>"
            for i in range(1, 7)
        )
        drone_type = self.step_data.get(2, {}).get("drone_type", "—")
        today = datetime.now().strftime("%d/%m/%Y %H:%M")

        html = f"""<!DOCTYPE html><html lang="he" dir="rtl">
<head><meta charset="UTF-8"><title>Field Test Report</title>
<style>
body{{font-family:'Courier New',monospace;background:#0a0a0f;color:#00ff41;padding:30px}}
h1{{border-bottom:1px solid #00ff41;padding-bottom:10px;letter-spacing:2px}}
.box{{border:1px solid #1a3a1a;padding:24px;max-width:640px;margin:20px auto;background:#0d1117}}
table{{width:100%;border-collapse:collapse;margin-top:16px}}
td{{padding:9px 14px;border-bottom:1px solid #1a3a1a}}
tr.pass td{{color:#00ff41}} tr.fail td{{color:#ff2a2a}} tr.skip td{{color:#777}}
.overall{{font-size:1.6em;font-weight:bold;margin-top:24px;text-align:center;padding:12px;
          border:1px solid currentColor}}
.meta{{color:#558855;font-size:.85em;margin:4px 0}}
@media print{{body{{background:#fff;color:#000}} .box{{border:1px solid #ccc}} tr.pass td{{color:#006600}} tr.fail td{{color:#cc0000}}}}
</style></head>
<body><div class="box">
<h1>⬡ FIELD TEST REPORT</h1>
<div class="meta">תאריך: {today}</div>
<div class="meta">רחפן מזוהה: {drone_type}</div>
<table><tr style="color:#558855"><td><b>#</b></td><td><b>שלב</b></td><td><b>תוצאה</b></td></tr>
{rows}</table>
<div class="overall">{overall}</div>
<div class="meta" style="text-align:center;margin-top:8px">עבר: {passed} | נכשל: {failed} | דולג: {skipped}</div>
</div></body></html>"""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._report_path = f"field_test_report_{ts}.html"
        Path(self._report_path).write_text(html, encoding="utf-8")

        self.step_status[7] = "pass"
        self.step_data[7]   = {
            "overall": overall, "passed": passed, "failed": failed,
            "skipped": skipped, "report_path": self._report_path,
            "message": f"דוח שמור: {self._report_path}",
        }

    def get_state(self) -> dict:
        if not self.active:
            return {"active": False}
        elapsed  = time.monotonic() - (self.step_start_time or time.monotonic())
        step_inf = next((s for s in self.STEPS if s["id"] == self.current_step), {})
        dur      = step_inf.get("duration", 0)
        progress = min(elapsed / dur, 1.0) if dur > 0 else 1.0
        cur_st   = self.step_status.get(self.current_step, "pending")
        return {
            "active":       True,
            "current_step": self.current_step,
            "step_name":    step_inf.get("name", ""),
            "step_title":   step_inf.get("title", ""),
            "step_status":  cur_st,
            "instruction":  self.INSTRUCTIONS.get(self.current_step, ""),
            "skippable":    step_inf.get("skippable", True),
            "progress":     round(progress, 3),
            "elapsed":      round(elapsed, 1),
            "duration":     dur,
            "results": {
                str(i): {
                    "name":    next((s["name"] for s in self.STEPS if s["id"] == i), ""),
                    "status":  self.step_status.get(i, "pending"),
                    "message": self.step_data.get(i, {}).get("message", ""),
                }
                for i in range(1, 8)
            },
        }


_wizard = Wizard()


# ─── FastAPI ───────────────────────────────────────────────────────────

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def root():
    return _HTML


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _user_lat, _user_lon
    await ws.accept()
    _clients.add(ws)
    try:
        async for raw in ws.iter_text():
            try:
                msg = json.loads(raw)
                if msg.get("type") == "location":
                    _user_lat = float(msg.get("lat", 0.0))
                    _user_lon = float(msg.get("lon", 0.0))
            except Exception:
                pass
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _clients.discard(ws)


@app.post("/calibrate")
async def do_calibrate():
    _start_inline_calibration(_read_frame)
    return {"ok": True}


@app.post("/log/toggle")
async def do_log_toggle():
    global _logging, _csv_fh, _csv_wr, _csv_path, _csv_rows
    if _logging:
        if _csv_fh:
            _csv_fh.close()
        _csv_fh = _csv_wr = None
        _logging = False
        return {"logging": False}
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _csv_path = f"field_test_{ts}.csv"
        _csv_fh   = open(_csv_path, "w", newline="", encoding="utf-8")
        _csv_wr   = csv.DictWriter(_csv_fh, fieldnames=_CSV_FIELDS)
        _csv_wr.writeheader(); _csv_fh.flush()
        _csv_rows = 0; _logging = True
        return {"logging": True, "path": _csv_path}


@app.post("/reset")
async def do_reset():
    _detector.reset()
    return {"ok": True}


@app.post("/wizard/start")
async def wiz_start():
    _wizard.start()
    return {"ok": True}


@app.post("/wizard/begin")
async def wiz_begin():
    _wizard.begin()
    return {"ok": True}


@app.post("/wizard/skip")
async def wiz_skip():
    _wizard.skip()
    return {"ok": True}


@app.post("/wizard/retry")
async def wiz_retry():
    _wizard.retry()
    return {"ok": True}


@app.post("/wizard/stop")
async def wiz_stop():
    _wizard.stop()
    return {"ok": True}


# ─── Broadcast loop ────────────────────────────────────────────────────

async def _broadcast_loop():
    global _csv_rows, _cal_prev_running
    while True:
        await asyncio.sleep(0.2)
        if _detector is None:   # not yet initialised (startup race)
            continue
        s = _detector.get_state()

        # CSV logging
        if _logging and _csv_wr:
            try:
                _csv_wr.writerow(_csv_row(s))
                _csv_fh.flush()
                _csv_rows += 1
            except Exception:
                pass

        # Calibration: apply rms_gate when done
        cal = _get_cal()
        if _cal_prev_running and not cal.get("running") and cal.get("rms_gate"):
            global RMS_GATE
            RMS_GATE = cal["rms_gate"]
        _cal_prev_running = cal.get("running", False)

        # Wizard tick
        _wizard.tick(s)

        # Telegram: fire on CONFIRMED onset
        ct = s.get("combined_threat", "CLEAR")
        global _telegram_prev
        if ct != _telegram_prev:
            if ct == "CONFIRMED":
                _t = (f"🚨 <b>DRONE CONFIRMED</b>\n"
                      f"Type: {s.get('profile_label','—')}\n"
                      f"f0: {s.get('f0_hz',0)} Hz  SNR: {s.get('snr',0):.1f}×\n"
                      f"Dir: {s.get('rpm_trend','—')}  "
                      f"Range: {_range_est(s.get('snr',0), True)}")
                threading.Thread(target=_send_telegram, args=(_t,), daemon=True).start()
            elif ct == "ACOUSTIC ONLY":
                _t = f"⚠️ Acoustic signature detected — {s.get('profile_label','—')}"
                threading.Thread(target=_send_telegram, args=(_t,), daemon=True).start()
            elif _telegram_prev in ("CONFIRMED", "ACOUSTIC ONLY"):
                threading.Thread(target=_send_telegram,
                                 args=("✅ Threat cleared — MONITORING",), daemon=True).start()
        _telegram_prev = ct

        vis = s.get("visual") or {}

        # Drone geo position estimate
        _re      = _range_est(s.get("snr", 0.0), s.get("alarm", False))
        _rad_m   = _RANGE_METRES.get(_re, 150)
        _alarm   = s.get("alarm", False)
        _dir     = s.get("rpm_trend", "STABLE")
        _bearing = {"APPROACHING": 0, "RECEDING": 180, "STABLE": 0}.get(_dir, 0)
        if _alarm and _user_lat != 0.0:
            _dlat, _dlon = _offset_position(_user_lat, _user_lon, _bearing, _rad_m)
        else:
            _dlat, _dlon, _rad_m = 0.0, 0.0, 0

        payload = {
            "acoustic_confidence": s.get("confidence", 0.0) / 100.0,
            "visual_confidence":   vis.get("confidence", 0.0) / 100.0,
            "combined_threat":     s.get("combined_threat", "CLEAR"),
            "f0_hz":               s.get("f0_hz", 0),
            "rpm":                 s.get("rpm_est", 0),
            "drone_type":          s.get("profile_label", "—"),
            "snr":                 s.get("snr", 0.0),
            "direction":           s.get("rpm_trend", "STABLE"),
            "direction_rate":      s.get("rate_hz_per_s", 0.0),
            "range_estimate":      _range_est(s.get("snr", 0.0), s.get("alarm", False)),
            "alarm":               s.get("alarm", False),
            "uptime":              round(time.monotonic() - _t0),
            "mode":                _mode,
            "history":             s.get("history", []),
            "log":                 s.get("log", [])[:20],
            "frames":              s.get("frames", 0),
            "detections":          s.get("detections", 0),
            "logging":             _logging,
            "csv_rows":            _csv_rows,
            "csv_path":            Path(_csv_path).name if _csv_path else "",
            "calibrating":         cal.get("running", False),
            "cal_progress":        round(cal.get("progress", 0.0), 3),
            "visual":              vis,
            "wizard":              _wizard.get_state(),
            "user_lat":            _user_lat,
            "user_lon":            _user_lon,
            "drone_lat":           _dlat,
            "drone_lon":           _dlon,
            "drone_radius_m":      _rad_m,
            "fft_bins":            s.get("fft_bins", []),
            "recordings":          _recorder.list() if _recorder else [],
        }
        msg  = json.dumps(payload)
        dead = set()
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        _clients.difference_update(dead)   # in-place remove, no assignment


@app.on_event("startup")
async def _startup():
    asyncio.create_task(_broadcast_loop())


# ─── HTML / CSS / JS ───────────────────────────────────────────────────

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FPV DETECT v3.0 — Web HUD</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#060a06;color:#00ff41;font-family:'Courier New',monospace;min-height:100vh;overflow-x:hidden}
/* ── Header ── */
#hdr{background:#0a120a;border-bottom:1px solid #1a3a1a;padding:8px 16px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
#hdr-title{font-size:1.3em;font-weight:bold;letter-spacing:3px;color:#00ff41}
#hdr-status{padding:3px 10px;border:1px solid currentColor;font-size:.8em;letter-spacing:1px;transition:color .3s,border-color .3s}
#hdr-uptime,#hdr-mode{font-size:.78em;color:#446644}
#hdr-mode{margin-left:auto;border:1px solid #1a3a1a;padding:3px 10px}
#ws-dot{font-size:.85em}
/* ── Layout ── */
#wrap{padding:10px;display:grid;grid-template-columns:1fr 1fr;gap:10px}
/* cards */
.card{background:#0a120a;border:1px solid #1a3a1a;border-radius:3px;padding:12px}
.card-title{font-size:.7em;letter-spacing:3px;color:#446644;border-bottom:1px solid #1a3a1a;padding-bottom:5px;margin-bottom:10px}
/* progress */
.pbar-wrap{background:#030803;border:1px solid #1a3a1a;height:18px;border-radius:2px;overflow:hidden;margin-bottom:6px}
.pbar{height:100%;transition:width .18s ease,background .3s;background:#00ff41;min-width:2px}
.pbar.warn{background:#ffaa00}.pbar.alert{background:#ff2a2a}
.pval{font-size:.9em;font-weight:bold;color:#00ff41;margin-bottom:8px}
.pval.warn{color:#ffaa00}.pval.alert{color:#ff2a2a}
/* data grid */
.dg{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:.82em}
.dl{color:#446644}.dv{color:#ccffcc;font-weight:bold}
/* ── Threat ── */
#threat{grid-column:1/-1;text-align:center;padding:16px 10px;border:2px solid #1a3a1a;border-radius:3px;transition:background .4s}
#threat.c-CONFIRMED{border-color:#ff0000;background:#120000;animation:pborder .8s infinite}
#threat.c-ACOUSTIC{border-color:#ffaa00;background:#0e0d00}
#threat.c-VISUAL{border-color:#4488ff;background:#00000e}
#threat.c-CLEAR{border-color:#00ff41;background:#000e00}
@keyframes pborder{0%,100%{box-shadow:0 0 18px #ff000055}50%{box-shadow:0 0 36px #ff0000aa}}
#thr-txt{font-size:2.2em;font-weight:bold;letter-spacing:4px;margin-bottom:6px;transition:color .3s}
#thr-txt.c-CONFIRMED{color:#ff2a2a;animation:blink .8s infinite}
#thr-txt.c-ACOUSTIC{color:#ffaa00}
#thr-txt.c-VISUAL{color:#4488ff}
#thr-txt.c-CLEAR{color:#00ff41}
@keyframes blink{0%,49%{opacity:1}50%,100%{opacity:.25}}
@keyframes wiz-blink{0%,100%{opacity:1}50%{opacity:.25}}
.blink-step .ws-icon{animation:wiz-blink .9s infinite}
#thr-dir{font-size:1em;color:#aaa;margin-top:4px}
#thr-range{font-size:.85em;color:#446644;margin-top:3px}
/* ── Canvas row ── */
#canvas-row{grid-column:1/-1;display:grid;grid-template-columns:2fr 1fr;gap:10px}
canvas{display:block;width:100%;border-radius:2px}
/* ── Spectrogram ── */
#spec-block{grid-column:1/-1}
#spec-wrap{position:relative;background:#030803;border-radius:2px;overflow:hidden}
#spec-cv{display:block;width:100%;height:80px;image-rendering:pixelated;image-rendering:crisp-edges}
#spec-labels{position:absolute;top:0;left:0;height:100%;width:28px;display:flex;flex-direction:column;justify-content:space-between;padding:2px 0;pointer-events:none}
#spec-labels span{font-size:.55em;color:#335533;padding-left:2px;line-height:1}
/* ── Recordings ── */
#rec-panel{display:none;position:absolute;bottom:calc(100% + 4px);right:0;
           background:#080f08;border:1px solid #1a3a1a;border-radius:3px;
           padding:8px;min-width:260px;z-index:50;max-height:200px;overflow-y:auto}
#rec-panel a{display:block;font-size:.72em;color:#88aaff;padding:2px 0;
             text-decoration:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#rec-panel a:hover{text-decoration:underline}
#rec-wrap{position:relative}
/* ── Log ── */
#log-block{grid-column:1/-1}
#log-ul{list-style:none;max-height:150px;overflow-y:auto;font-size:.75em}
#log-ul::-webkit-scrollbar{width:3px}#log-ul::-webkit-scrollbar-thumb{background:#1a3a1a}
.le{padding:3px 4px;border-bottom:1px solid #0a120a;display:grid;grid-template-columns:82px 180px 68px 72px 150px auto;gap:3px;border-radius:2px}
.le.confirmed{color:#ff4444;background:rgba(255,0,0,0.15)}.le.acoustic{color:#ffaa00;background:rgba(255,200,0,0.10)}.le.visual{color:#5599ff;background:rgba(0,100,255,0.10)}.le.clear{color:#446644}
/* ── Controls ── */
#ctrls{grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
button{background:#0a120a;border:1px solid #00ff41;color:#00ff41;padding:7px 14px;
       font-family:'Courier New',monospace;font-size:.8em;letter-spacing:1px;cursor:pointer;
       border-radius:2px;transition:background .15s,box-shadow .15s;white-space:nowrap}
button:hover{background:#142a14;box-shadow:0 0 8px #00ff4133}
button.logging{border-color:#ff2a2a;color:#ff6666;background:#120000}
button.wiz-btn{border-color:#88aaff;color:#88aaff}
button.wiz-btn:hover{background:#0a0a20}
#stats-r{margin-left:auto;font-size:.72em;color:#334433}
/* ── Cal bar ── */
#calbar{display:none;flex:1;align-items:center;gap:6px}
#calbar.on{display:flex}
#cal-track{flex:1;background:#030803;border:1px solid #ffaa00;height:12px;border-radius:2px}
#cal-fill{height:100%;background:#ffaa00;transition:width .5s}
#cal-lbl{font-size:.75em;color:#ffaa00;white-space:nowrap}
/* ── Wizard Panel ── */
#wiz-panel{position:fixed;top:0;right:-420px;width:400px;height:100vh;background:#080f08;
           border-left:1px solid #1a3a1a;z-index:100;transition:right .35s ease;
           display:flex;flex-direction:column;overflow:hidden}
#wiz-panel.open{right:0}
#wiz-hdr{background:#0a150a;border-bottom:1px solid #1a3a1a;padding:10px 14px;
          display:flex;align-items:center;gap:10px}
#wiz-title{font-size:1em;font-weight:bold;letter-spacing:2px;flex:1}
#wiz-close{background:none;border:none;color:#446644;font-size:1.2em;cursor:pointer;padding:0 4px}
#wiz-body{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}
#wiz-body::-webkit-scrollbar{width:3px}#wiz-body::-webkit-scrollbar-thumb{background:#1a3a1a}
/* step list */
.wstep{display:flex;align-items:flex-start;gap:8px;padding:8px;border:1px solid #1a3a1a;border-radius:3px;transition:border-color .3s,background .3s}
.wstep.running{border-color:#00ff41;background:#040d04}
.wstep.pass{border-color:#1a3a1a;opacity:.7}
.wstep.fail{border-color:#ff2a2a;background:#0d0000}
.wstep.current{background:#050d05}
.ws-icon{font-size:1.1em;min-width:20px;text-align:center;margin-top:1px}
.ws-info{flex:1}
.ws-name{font-size:.75em;font-weight:bold;letter-spacing:2px}
.ws-msg{font-size:.7em;color:#558855;margin-top:2px;word-break:break-word}
.ws-pbar{height:6px;background:#030803;border-radius:2px;margin-top:5px;overflow:hidden}
.ws-pfill{height:100%;background:#00ff41;transition:width .5s}
/* instruction box */
#wiz-instr{background:#040d04;border:1px solid #00ff4144;border-radius:3px;padding:10px 12px;
           font-size:.82em;line-height:1.5;color:#99ccaa;text-align:right;direction:rtl}
/* wiz controls */
#wiz-ctrls{display:flex;gap:6px;padding:0 0 4px 0;flex-wrap:wrap}
#wiz-ctrls button{font-size:.78em;padding:6px 12px}
#btn-wiz-begin{border-color:#00ff41;color:#00ff41}
#btn-wiz-begin:hover{background:#142a14}
#btn-wiz-skip{border-color:#446644;color:#446644}
#btn-wiz-retry{border-color:#ffaa00;color:#ffaa00}
#btn-wiz-retry:hover{background:#150e00}
/* report link */
#wiz-report{display:none;padding:8px;background:#040d04;border:1px solid #00ff41;border-radius:3px;text-align:center;font-size:.82em}
#wiz-report a{color:#00ff41;text-decoration:none}
#wiz-report a:hover{text-decoration:underline}
/* ws status */
.ws-ok{color:#00ff41}.ws-err{color:#ff2a2a;animation:blink 1s infinite}
/* ── Map panel ── */
#map-block{grid-column:1/-1}
#map-container{width:100%;height:280px;border-radius:2px;background:#0a120a;display:none}
#map-msg{color:#446644;font-size:.8em;padding:8px 0;text-align:center;display:block}
.map-tiles{filter:brightness(.7) invert(1) hue-rotate(170deg) saturate(0.5)}
/* responsive */
@media(max-width:680px){
  #wrap{grid-template-columns:1fr}
  #threat,#canvas-row,#log-block,#map-block,#ctrls{grid-column:1}
  #canvas-row{grid-template-columns:1fr}
  #wiz-panel{width:100%;right:-100%}
}
</style>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
</head>
<body>

<!-- Header -->
<div id="hdr">
  <span id="hdr-title">⬡ FPV DETECT v3.0</span>
  <span id="hdr-status">● MONITORING</span>
  <span id="hdr-uptime">UPTIME 00:00:00</span>
  <span id="ws-dot" class="ws-err">◌ CONNECTING</span>
  <span id="hdr-mode">MODE: —</span>
</div>

<!-- Main grid -->
<div id="wrap">

  <!-- Acoustic card -->
  <div class="card">
    <div class="card-title">ACOUSTIC LAYER</div>
    <div class="pbar-wrap"><div class="pbar" id="ac-bar" style="width:0%"></div></div>
    <div class="pval" id="ac-pct">0.0%</div>
    <div class="dg">
      <span class="dl">f₀</span>      <span class="dv" id="d-f0">— Hz</span>
      <span class="dl">RPM</span>     <span class="dv" id="d-rpm">—</span>
      <span class="dl">Type</span>    <span class="dv" id="d-type">—</span>
      <span class="dl">SNR</span>     <span class="dv" id="d-snr">—</span>
    </div>
  </div>

  <!-- Visual card -->
  <div class="card">
    <div class="card-title">VISUAL LAYER — YOLOv8</div>
    <div class="pbar-wrap"><div class="pbar" id="vis-bar" style="width:0%"></div></div>
    <div class="pval" id="vis-pct">0.0%</div>
    <div class="dg">
      <span class="dl">Alert</span>  <span class="dv" id="d-val">no</span>
      <span class="dl">Source</span> <span class="dv" id="d-vsrc">—</span>
      <span class="dl">Model</span>  <span class="dv" id="d-vmod">—</span>
      <span class="dl">FPS</span>    <span class="dv" id="d-fps">—</span>
    </div>
  </div>

  <!-- Threat block -->
  <div id="threat" class="c-CLEAR">
    <div id="thr-txt" class="c-CLEAR">✓  CLEAR</div>
    <div id="thr-dir">Direction: —</div>
    <div id="thr-range">Range: —</div>
  </div>

  <!-- Canvas row -->
  <div id="canvas-row">
    <div class="card" id="hist-card">
      <div class="card-title">CONFIDENCE HISTORY — 60s</div>
      <canvas id="hist-cv" height="130"></canvas>
    </div>
    <div class="card" id="radar-card">
      <div class="card-title">DIRECTION</div>
      <canvas id="radar-cv" height="130"></canvas>
    </div>
  </div>

  <!-- Spectrogram -->
  <div class="card" id="spec-block">
    <div class="card-title">ACOUSTIC SPECTROGRAM — 80 Hz → 4 kHz  ◀ time</div>
    <div id="spec-wrap">
      <canvas id="spec-cv" width="300" height="64"></canvas>
      <div id="spec-labels">
        <span>4k</span><span>2k</span><span>500</span><span>80</span>
      </div>
    </div>
  </div>

  <!-- Map panel -->
  <div class="card" id="map-block">
    <div class="card-title">LIVE POSITION MAP</div>
    <div id="map-container"></div>
    <div id="map-msg">📍 ENABLE LOCATION TO SEE MAP</div>
  </div>

  <!-- Event log -->
  <div class="card" id="log-block">
    <div class="card-title">DETECTION LOG</div>
    <ul id="log-ul"></ul>
  </div>

  <!-- Controls -->
  <div id="ctrls">
    <button onclick="doCalibrate()">⊙ CALIBRATE</button>
    <div id="calbar">
      <div id="cal-track"><div id="cal-fill"></div></div>
      <span id="cal-lbl">calibrating…</span>
    </div>
    <button id="btn-log" onclick="doLog()">● START LOG</button>
    <button onclick="doReset()">↺ RESET</button>
    <button class="wiz-btn" onclick="openWizard()">🧪 FIELD TEST</button>
    <div id="rec-wrap">
      <button id="btn-rec" onclick="toggleRecPanel()" style="border-color:#446644;color:#446644">
        🎙 RECORDINGS (0)</button>
      <div id="rec-panel"><em style="color:#446644;font-size:.75em">No recordings yet</em></div>
    </div>
    <span id="stats-r">frames: — | detections: —</span>
  </div>

</div><!-- /wrap -->

<!-- ── Wizard Panel ── -->
<div id="wiz-panel">
  <div id="wiz-hdr">
    <span id="wiz-title">🧪 FIELD TEST WIZARD</span>
    <button id="wiz-close" onclick="closeWizard()" title="Close">✕</button>
  </div>
  <div id="wiz-body">
    <div id="wiz-not-started" style="color:#446644;font-size:.85em;text-align:right;direction:rtl;padding:10px 12px;border-bottom:1px solid #1a3a1a;margin-bottom:4px">
      לחץ ▶▶ BEGIN WIZARD להתחלת בדיקת שדה מודרכת
    </div>
    <div id="wiz-steps" style="display:none"></div>
    <div id="wiz-instr" style="display:none"></div>
    <div id="wiz-ctrls" style="display:none">
      <button id="btn-wiz-begin" onclick="wizBegin()">▶ BEGIN STEP</button>
      <button id="btn-wiz-skip"  onclick="wizSkip()">⏭ SKIP</button>
      <button id="btn-wiz-retry" onclick="wizRetry()">↺ RETRY</button>
    </div>
    <div id="wiz-report"></div>
    <button id="btn-wiz-start" onclick="wizStart()" style="margin-top:auto">▶▶ BEGIN WIZARD</button>
  </div>
</div>

<script>
// ── Audio alerts (Web Audio API) ─────────────────────────────────────
let _audioCtx = null, _prevThreat = 'CLEAR';
function _initAudio() {
  if (!_audioCtx) _audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (_audioCtx.state === 'suspended') _audioCtx.resume();
}
function _beep(freq, dur, vol) {
  try {
    _initAudio();
    const o = _audioCtx.createOscillator(), g = _audioCtx.createGain();
    o.connect(g); g.connect(_audioCtx.destination);
    o.frequency.value = freq;
    g.gain.setValueAtTime(vol, _audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.001, _audioCtx.currentTime + dur);
    o.start(); o.stop(_audioCtx.currentTime + dur);
  } catch(e) {}
}
document.addEventListener('click', _initAudio);

// ── Radar RAF state ───────────────────────────────────────────────────
let _lastRadar = {direction:'STABLE', rate:0, range:'', alarm:false};

// ── WebSocket ────────────────────────────────────────────────────────
let ws, reconnTimer, lastLogJSON = "";

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    clearTimeout(reconnTimer);
    document.getElementById('ws-dot').className = 'ws-ok';
    document.getElementById('ws-dot').textContent = '● LIVE';
  };
  ws.onmessage = e => update(JSON.parse(e.data));
  ws.onclose = ws.onerror = () => {
    document.getElementById('ws-dot').className = 'ws-err';
    document.getElementById('ws-dot').textContent = '◌ RECONNECTING';
    reconnTimer = setTimeout(connect, 2000);
  };
}

// ── Main update ──────────────────────────────────────────────────────
function fmtUp(s) {
  const h=Math.floor(s/3600), m=Math.floor((s%3600)/60), ss=s%60;
  return `UPTIME ${pad(h)}:${pad(m)}:${pad(ss)}`;
}
function pad(n){return String(n).padStart(2,'0')}

const CT={
  'CONFIRMED':    {cls:'CONFIRMED',icon:'‼  CONFIRMED'},
  'ACOUSTIC ONLY':{cls:'ACOUSTIC', icon:'⚠  ACOUSTIC ONLY'},
  'VISUAL ONLY':  {cls:'VISUAL',   icon:'●  VISUAL ONLY'},
  'CLEAR':        {cls:'CLEAR',    icon:'✓  CLEAR'},
};
const DIR={'APPROACHING':'↑  APPROACHING','RECEDING':'↓  RECEDING','STABLE':'—  STABLE'};
const LC={'CONFIRMED':'confirmed','ACOUSTIC ONLY':'acoustic','VISUAL ONLY':'visual','CLEAR':'clear'};

function update(d) {
  // header
  document.getElementById('hdr-uptime').textContent = fmtUp(d.uptime);
  document.getElementById('hdr-mode').textContent = `MODE: ${d.mode}`;

  const ct = CT[d.combined_threat] || CT['CLEAR'];
  const sb = document.getElementById('hdr-status');
  sb.textContent = d.alarm ? '● DRONE DETECTED' : '● MONITORING';
  sb.style.color = d.alarm ? '#ff2a2a' : '#00ff41';
  sb.style.borderColor = d.alarm ? '#ff2a2a' : '#00ff41';

  // acoustic
  const acPct = Math.round(d.acoustic_confidence * 100 * 10) / 10;
  const acCls = acPct >= 35 ? (acPct >= 70 ? 'alert' : 'warn') : '';
  setBar('ac-bar', 'ac-pct', acPct, acCls);
  document.getElementById('d-f0').textContent   = d.f0_hz > 0 ? `${d.f0_hz} Hz` : '—';
  document.getElementById('d-rpm').textContent   = d.rpm > 0 ? `~${d.rpm.toLocaleString()}` : '—';
  document.getElementById('d-type').textContent  = d.drone_type || '—';
  document.getElementById('d-snr').textContent   = d.snr > 0 ? `${d.snr.toFixed(1)}×` : '—';

  // visual
  const vis = d.visual || {};
  const visPct = Math.round(d.visual_confidence * 100 * 10) / 10;
  const visCls = visPct >= 60 ? 'alert' : '';
  setBar('vis-bar', 'vis-pct', visPct, visCls);
  const va = document.getElementById('d-val');
  va.textContent = d.visual_confidence >= 0.6 ? 'YES ◉' : 'no';
  va.style.color = d.visual_confidence >= 0.6 ? '#ff2a2a' : '#00ff41';
  document.getElementById('d-vsrc').textContent = vis.camera_live ? 'LIVE CAMERA' : 'SIMULATION';
  document.getElementById('d-vmod').textContent = vis.model_name || '—';
  document.getElementById('d-fps').textContent  = vis.fps != null ? `${vis.fps} fps` : '—';

  // threat
  const tb = document.getElementById('threat');
  const tt = document.getElementById('thr-txt');
  tb.className = `c-${ct.cls}`; tt.className = `c-${ct.cls}`;
  tt.textContent = ct.icon;
  const rate = Math.abs(d.direction_rate) > 0.5 ?
    ` (${d.direction_rate > 0 ? '+' : ''}${d.direction_rate.toFixed(1)} Hz/s)` : '';
  document.getElementById('thr-dir').textContent  = `Direction: ${DIR[d.direction] || d.direction}${rate}`;
  document.getElementById('thr-range').textContent = `Est. Range: ${d.range_estimate || '—'}`;

  // sound alert on threat change
  if (d.combined_threat !== _prevThreat) {
    if (d.combined_threat === 'CONFIRMED')     _beep(880, 0.4, 0.3);
    if (d.combined_threat === 'ACOUSTIC ONLY') _beep(440, 0.25, 0.2);
    _prevThreat = d.combined_threat;
  }

  // stats
  document.getElementById('stats-r').textContent =
    `frames: ${d.frames.toLocaleString()} | detections: ${d.detections}`;

  // log toggle
  const bl = document.getElementById('btn-log');
  bl.textContent = d.logging ? `■ STOP LOG  (${d.csv_rows} rows)` : '● START LOG';
  bl.className   = d.logging ? 'logging' : '';

  // calibration bar
  const cb = document.getElementById('calbar');
  if (d.calibrating) {
    cb.className = 'on';
    document.getElementById('cal-fill').style.width = (d.cal_progress * 100) + '%';
    document.getElementById('cal-lbl').textContent  = `CALIBRATING ${Math.round(d.cal_progress * 100)}%`;
  } else {
    cb.className = '';
  }

  // event log
  const logJSON = JSON.stringify(d.log);
  if (logJSON !== lastLogJSON) {
    lastLogJSON = logJSON;
    const ul = document.getElementById('log-ul');
    ul.innerHTML = '';
    for (const e of (d.log || [])) {
      const li = document.createElement('li');
      li.className = `le ${LC[e.combined] || ''}`;
      li.innerHTML = `<span>${e.time}</span><span>${(e.type||'—').slice(0,20)}</span>`+
        `<span>${e.f0||0} Hz</span><span>${(e.snr||0).toFixed(1)}×</span>`+
        `<span>${e.combined||'—'}</span><span>${e.trend||'—'}</span>`;
      ul.appendChild(li);
    }
  }

  // canvases
  drawHistory(d.history || [], d.acoustic_confidence, d.alarm);
  _lastRadar = {direction:d.direction, rate:d.direction_rate, range:d.range_estimate, alarm:d.alarm};

  // wizard
  if (d.wizard) updateWizard(d.wizard);

  // map
  _updateMap(d);

  // spectrogram
  if (d.fft_bins && d.fft_bins.length) _drawSpectrogram(d.fft_bins);

  // recordings
  _updateRecordings(d.recordings || []);
}

function setBar(barId, lblId, pct, cls) {
  const bar = document.getElementById(barId);
  const lbl = document.getElementById(lblId);
  bar.style.width = Math.min(pct, 100) + '%';
  bar.className = `pbar ${cls}`;
  lbl.className = `pval ${cls}`;
  lbl.textContent = pct.toFixed(1) + '%';
}

// ── Confidence history canvas ────────────────────────────────────────
let histW = 0;
function drawHistory(hist, curConf, alarm) {
  const cv = document.getElementById('hist-cv');
  const card = document.getElementById('hist-card');
  const W = card.clientWidth - 24;
  if (W <= 0) return;
  if (cv.width !== W) cv.width = W;
  const H = cv.height, ctx = cv.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#030803'; ctx.fillRect(0,0,W,H);

  // grid
  ctx.strokeStyle = '#0a180a'; ctx.lineWidth = 1;
  for (let i=1;i<4;i++){const y=H*i/4;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}
  for (let i=1;i<6;i++){const x=W*i/6;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}

  // y-labels
  ctx.fillStyle='#224422';ctx.font='9px Courier New';ctx.textAlign='left';
  ['100','75','50','25','0'].forEach((l,i)=>ctx.fillText(l+'%',2,H*i/4+10));

  // 35% threshold
  const ty = H*(1-0.35);
  ctx.setLineDash([5,4]); ctx.strokeStyle='#552222'; ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,ty);ctx.lineTo(W,ty);ctx.stroke();
  ctx.setLineDash([]); ctx.fillStyle='#552222';
  ctx.textAlign='right'; ctx.fillText('35%',W-2,ty-3);

  if (hist.length < 2) return;
  const pts = hist.slice(-300);
  const step = W / Math.max(pts.length-1,1);

  // fill
  const lineColor = alarm ? '#ff4444' : '#00ff41';
  const grad = ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0, alarm ? 'rgba(255,40,40,.28)' : 'rgba(0,255,65,.28)');
  grad.addColorStop(1, 'rgba(0,0,0,.02)');
  ctx.fillStyle = grad;
  ctx.beginPath(); ctx.moveTo(0,H);
  pts.forEach((v,i) => ctx.lineTo(i*step, H*(1-v)));
  ctx.lineTo((pts.length-1)*step, H); ctx.closePath(); ctx.fill();

  // line
  ctx.strokeStyle = lineColor; ctx.lineWidth = 1.5;
  ctx.beginPath();
  pts.forEach((v,i) => { const x=i*step,y=H*(1-v); i?ctx.lineTo(x,y):ctx.moveTo(x,y); });
  ctx.stroke();

  // dot
  const lx=(pts.length-1)*step, ly=H*(1-pts[pts.length-1]);
  ctx.fillStyle=lineColor; ctx.beginPath();ctx.arc(lx,ly,3,0,2*Math.PI);ctx.fill();
}

// ── Direction radar canvas ───────────────────────────────────────────
function drawRadar(direction, rate, range, alarm) {
  const cv = document.getElementById('radar-cv');
  const card = document.getElementById('radar-card');
  const W = card.clientWidth - 24;
  if (W <= 0) return;
  if (cv.width !== W) cv.width = W;
  const H = cv.height, ctx = cv.getContext('2d');
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#030803'; ctx.fillRect(0,0,W,H);
  const cx=W/2, cy=H/2-6, r=Math.min(cx,cy)-8;

  // rings
  [1,2,3].forEach(i => {
    ctx.beginPath(); ctx.arc(cx,cy,r*i/3,0,2*Math.PI);
    ctx.strokeStyle = i===3?'#1a3a1a':'#0a180a'; ctx.lineWidth=1; ctx.stroke();
  });
  ctx.strokeStyle='#0a180a'; ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(cx-r,cy);ctx.lineTo(cx+r,cy);ctx.stroke();
  ctx.beginPath();ctx.moveTo(cx,cy-r);ctx.lineTo(cx,cy+r);ctx.stroke();

  // labels
  ctx.fillStyle='#224422'; ctx.font='10px Courier New';
  ctx.textAlign='center'; ctx.fillText('N',cx,cy-r+12);
  ctx.fillText('S',cx,cy+r-2);
  ctx.textAlign='left';  ctx.fillText('E',cx+r-12,cy+4);
  ctx.textAlign='right'; ctx.fillText('W',cx-r+12,cy+4);

  // range label
  const rc={'NEAR  (<50m)':'#ff5500','MEDIUM  (50–200m)':'#ffaa00','FAR  (>200m)':'#446644'}[range]||'#224422';
  ctx.fillStyle=rc; ctx.font='bold 9px Courier New'; ctx.textAlign='center';
  ctx.fillText(range||'—',cx,H-4);

  if (direction === 'STABLE') {
    ctx.beginPath(); ctx.arc(cx,cy,r*.18,0,2*Math.PI);
    ctx.strokeStyle = alarm ? '#00ff41' : '#224422'; ctx.lineWidth=2; ctx.stroke();
    ctx.fillStyle = alarm ? '#00ff4122' : '#0a180a'; ctx.fill();
    ctx.fillStyle = alarm ? '#00ff41' : '#224422';
    ctx.font='bold 8px Courier New'; ctx.textAlign='center';
    ctx.fillText('STABLE',cx,cy+4); return;
  }

  const isHot   = alarm && direction === 'APPROACHING';
  const isWarm  = alarm && direction === 'RECEDING';
  const baseClr = direction === 'APPROACHING' ? '#ff2a2a' : '#ffaa00';
  const dimClr  = direction === 'APPROACHING' ? 'rgba(255,42,42,0.22)' : 'rgba(255,170,0,0.22)';
  const color   = (isHot || isWarm) ? baseClr : dimClr;
  const lineW   = (isHot || isWarm) ? 2.5 : 1.2;

  let shadowBlur = 0;
  if (isHot) {
    const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
    shadowBlur = 8 + 20 * pulse;
  } else if (isWarm) {
    shadowBlur = 6;
  }

  const ang  = direction === 'APPROACHING' ? -Math.PI/2 : Math.PI/2;
  const alen = r * (direction === 'APPROACHING' ? .82 : .72);
  const tx=cx+Math.cos(ang)*alen, ty=cy+Math.sin(ang)*alen;

  ctx.shadowColor=baseClr; ctx.shadowBlur=shadowBlur;
  ctx.strokeStyle=color; ctx.lineWidth=lineW;
  ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(tx,ty); ctx.stroke();
  ctx.shadowBlur=0;

  const hl=11, ha=Math.PI/5.5;
  ctx.fillStyle=color;
  ctx.beginPath();
  ctx.moveTo(tx,ty);
  ctx.lineTo(tx-hl*Math.cos(ang-ha), ty-hl*Math.sin(ang-ha));
  ctx.lineTo(tx-hl*Math.cos(ang+ha), ty-hl*Math.sin(ang+ha));
  ctx.closePath(); ctx.fill();

  ctx.fillStyle=color; ctx.font='bold 9px Courier New'; ctx.textAlign='center';
  ctx.fillText(direction, cx, cy+r-12);
}

function startRadarLoop() {
  (function loop() {
    drawRadar(_lastRadar.direction, _lastRadar.rate, _lastRadar.range, _lastRadar.alarm);
    requestAnimationFrame(loop);
  })();
}

// ── Controls ─────────────────────────────────────────────────────────
function doCalibrate() { fetch('/calibrate',{method:'POST'}); }
function doLog()       { fetch('/log/toggle',{method:'POST'}); }
function doReset()     { fetch('/reset',{method:'POST'}); }

// ── Wizard ───────────────────────────────────────────────────────────
const STEP_ICONS = {pending:'⬜',waiting:'⬜',running:'🔵',pass:'✅',fail:'❌',skip:'⏭️'};
const STEP_DNAMES = ['BASELINE','DETECTION','DOPPLER','RANGE 50m','MULTI-TYPE','VISUAL','REPORT'];
const STEP_COLORS = {pending:'#224422',waiting:'#446644',running:'#00ff41',pass:'#00aa33',fail:'#ff2a2a',skip:'#445544'};

function openWizard()  { document.getElementById('wiz-panel').classList.add('open'); }
function closeWizard() { document.getElementById('wiz-panel').classList.remove('open'); }
function wizStart()    { fetch('/wizard/start',{method:'POST'}); }
function wizBegin()    { fetch('/wizard/begin',{method:'POST'}); }
function wizSkip()     { fetch('/wizard/skip', {method:'POST'}); }
function wizRetry()    { fetch('/wizard/retry',{method:'POST'}); }

function updateWizard(wiz) {
  const notStarted = document.getElementById('wiz-not-started');
  const stepsDiv   = document.getElementById('wiz-steps');
  const instrDiv   = document.getElementById('wiz-instr');
  const ctrlsDiv   = document.getElementById('wiz-ctrls');
  const reportDiv  = document.getElementById('wiz-report');
  const btnStart   = document.getElementById('btn-wiz-start');

  // Always render step list
  const results = wiz.active ? (wiz.results || {}) : {};
  let html = '';
  for (let i = 1; i <= 7; i++) {
    const r          = results[i] || {};
    const st         = r.status || 'pending';
    const isCurrent  = wiz.active && i === wiz.current_step;
    const isCurAct   = isCurrent && (st === 'waiting' || st === 'running');
    const icon       = isCurAct ? '🔵' : (STEP_ICONS[st] || '⬜');
    const blinkCls   = isCurAct ? ' blink-step' : '';
    const color      = STEP_COLORS[st] || '#224422';
    const nameColor  = isCurrent ? color : '#446644';
    const isRunning  = st === 'running' && isCurrent;
    const progress   = (isRunning && wiz.duration > 0) ? wiz.progress * 100
                       : (st === 'pass' || st === 'skip') ? 100 : 0;
    const bdrStyle   = isCurrent ? `border-color:${color}` : '';
    html += `<div class="wstep ${st}${blinkCls}" style="${bdrStyle}">
      <div class="ws-icon" style="color:${color}">${icon}</div>
      <div class="ws-info">
        <div class="ws-name" style="color:${nameColor}">${STEP_DNAMES[i-1]}</div>
        ${r.message ? `<div class="ws-msg">${r.message}</div>` : ''}
        ${isRunning && wiz.duration > 0 ? `<div class="ws-pbar"><div class="ws-pfill" style="width:${progress.toFixed(1)}%"></div></div>` : ''}
      </div>
    </div>`;
  }
  stepsDiv.innerHTML = html;
  stepsDiv.style.display = '';

  if (!wiz.active) {
    notStarted.style.display = '';
    instrDiv.style.display   = 'none';
    ctrlsDiv.style.display   = 'none';
    reportDiv.style.display  = 'none';
    btnStart.style.display   = '';
    return;
  }

  notStarted.style.display = 'none';
  btnStart.style.display   = 'none';

  // Instruction
  if (wiz.instruction && wiz.current_step < 7) {
    instrDiv.style.display = '';
    instrDiv.textContent   = wiz.instruction;
  } else {
    instrDiv.style.display = 'none';
  }

  // Controls
  const curStatus = (results[wiz.current_step] || {}).status || 'pending';
  if (wiz.current_step <= 6) {
    ctrlsDiv.style.display = 'flex';
    document.getElementById('btn-wiz-begin').style.display = curStatus === 'waiting' ? '' : 'none';
    document.getElementById('btn-wiz-skip').style.display  = wiz.skippable ? '' : 'none';
    document.getElementById('btn-wiz-retry').style.display = curStatus === 'fail' ? '' : 'none';
  } else {
    ctrlsDiv.style.display = 'none';
  }

  // Report
  const repData = (results[7] || {});
  if (repData.status === 'pass' && repData.message) {
    reportDiv.style.display = '';
    const path = repData.message.replace('דוח שמור: ', '');
    reportDiv.innerHTML = `<div>${repData.overall || ''}</div>
      <a href="/${path}" target="_blank">📄 פתח דוח מלא</a>`;
  } else {
    reportDiv.style.display = 'none';
  }
}

// ── Spectrogram ──────────────────────────────────────────────────────
let _specCtx = null, _specImg = null, _specW = 300, _specH = 64;

function _initSpec() {
  const cv = document.getElementById('spec-cv');
  _specCtx  = cv.getContext('2d');
  _specW    = cv.width; _specH = cv.height;
  _specImg  = _specCtx.createImageData(_specW, _specH);
  // fill dark green background
  for (let i = 0; i < _specImg.data.length; i += 4) {
    _specImg.data[i]=6; _specImg.data[i+1]=10; _specImg.data[i+2]=6; _specImg.data[i+3]=255;
  }
}

function _drawSpectrogram(bins) {
  if (!_specCtx) _initSpec();
  const w = _specW, h = _specH, d = _specImg.data;
  // Shift image left by 1px
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w - 1; x++) {
      const si = (y * w + x + 1) * 4, di = (y * w + x) * 4;
      d[di] = d[si]; d[di+1] = d[si+1]; d[di+2] = d[si+2]; d[di+3] = 255;
    }
  }
  // Draw new column at right; y=0 = highest freq, y=h-1 = lowest
  const n = bins.length;
  for (let y = 0; y < h; y++) {
    const binIdx = Math.floor((h - 1 - y) * n / h);  // flip: low freq at bottom
    const v = (bins[binIdx] || 0) / 255;              // 0-1
    let r = 6, g = 10, b = 6;
    if (v > 0.02) {
      if (v < 0.45) {
        g = Math.round(v / 0.45 * 200);
        r = 6; b = 6;
      } else if (v < 0.75) {
        const t = (v - 0.45) / 0.30;
        r = Math.round(t * 255); g = 200; b = 6;
      } else {
        const t = (v - 0.75) / 0.25;
        r = 255; g = Math.round((1 - t) * 200); b = 6;
      }
    }
    const di = (y * w + w - 1) * 4;
    d[di] = r; d[di+1] = g; d[di+2] = b; d[di+3] = 255;
  }
  _specCtx.putImageData(_specImg, 0, 0);
}

// ── Recordings ────────────────────────────────────────────────────────
let _recPanelOpen = false;

function toggleRecPanel() {
  _recPanelOpen = !_recPanelOpen;
  document.getElementById('rec-panel').style.display = _recPanelOpen ? 'block' : 'none';
}

function _updateRecordings(recs) {
  const btn = document.getElementById('btn-rec');
  btn.textContent = `🎙 RECORDINGS (${recs.length})`;
  btn.style.color       = recs.length > 0 ? '#88aaff' : '#446644';
  btn.style.borderColor = recs.length > 0 ? '#88aaff' : '#446644';
  if (!_recPanelOpen) return;
  const panel = document.getElementById('rec-panel');
  if (!recs.length) {
    panel.innerHTML = '<em style="color:#446644;font-size:.75em">No recordings yet</em>';
    return;
  }
  panel.innerHTML = recs.map(r =>
    `<a href="${r.url}" download="${r.name}">⬇ ${r.name}</a>`
  ).join('');
}

// ── Map ──────────────────────────────────────────────────────────────
let _map = null, _userMarker = null, _droneMarker = null;
let _droneCircle = null, _droneTrail = null, _trailPts = [];

function _initMap(lat, lon) {
  if (_map) return;
  const el = document.getElementById('map-container');
  el.style.display = 'block';
  document.getElementById('map-msg').style.display = 'none';

  _map = L.map('map-container', {zoomControl: true, attributionControl: false}).setView([lat, lon], 16);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, className: 'map-tiles'
  }).addTo(_map);

  const userIcon = L.divIcon({html:'📍', className:'', iconSize:[24,24], iconAnchor:[12,24]});
  _userMarker = L.marker([lat, lon], {icon: userIcon, zIndexOffset: 1000})
    .bindTooltip('YOU', {permanent: false, direction:'top'})
    .addTo(_map);

  const droneIcon = L.divIcon({html:'🚁', className:'', iconSize:[24,24], iconAnchor:[12,24]});
  _droneMarker = L.marker([lat, lon], {icon: droneIcon, opacity: 0}).addTo(_map);

  _droneCircle = L.circle([lat, lon], {
    radius: 0, color: '#ff2a2a', fillColor: '#ff2a2a',
    fillOpacity: 0.10, weight: 2, opacity: 0
  }).addTo(_map);

  _droneTrail = L.polyline([], {
    color: '#ff4444', weight: 2, opacity: 0.7, dashArray: '4 6'
  }).addTo(_map);
}

function _updateMap(d) {
  const uLat = d.user_lat || 0, uLon = d.user_lon || 0;
  const dLat = d.drone_lat || 0, dLon = d.drone_lon || 0;
  const dRad = d.drone_radius_m || 0;

  if (!uLat && !uLon) return;  // no geolocation yet

  if (!_map) _initMap(uLat, uLon);

  // update user marker
  if (_userMarker) _userMarker.setLatLng([uLat, uLon]);

  // drone visible?
  if (dLat !== 0 && dLon !== 0 && dRad > 0) {
    if (_droneMarker) {
      _droneMarker.setLatLng([dLat, dLon]);
      _droneMarker.setOpacity(1);
    }
    if (_droneCircle) {
      _droneCircle.setLatLng([dLat, dLon]);
      _droneCircle.setRadius(dRad);
      _droneCircle.setStyle({opacity: 0.9});
    }
    // trail
    _trailPts.push([dLat, dLon]);
    if (_trailPts.length > 10) _trailPts.shift();
    if (_droneTrail) _droneTrail.setLatLngs(_trailPts);
  } else {
    // CLEAR — hide drone marker
    if (_droneMarker) _droneMarker.setOpacity(0);
    if (_droneCircle) _droneCircle.setStyle({opacity: 0});
    if (_droneTrail) _droneTrail.setLatLngs([]);
    _trailPts = [];
  }
}

function _startGeo() {
  const msg = document.getElementById('map-msg');
  if (!navigator.geolocation) {
    msg.style.display = 'block';
    msg.textContent = '📍 GEOLOCATION NOT SUPPORTED';
    return;
  }
  msg.style.display = 'block';
  msg.textContent = '📍 ENABLE LOCATION TO SEE MAP';
  navigator.geolocation.watchPosition(
    pos => {
      const lat = pos.coords.latitude, lon = pos.coords.longitude;
      if (ws && ws.readyState === 1) {
        ws.send(JSON.stringify({type: 'location', lat, lon}));
      }
      msg.style.display = 'none';
    },
    err => {
      msg.style.display = 'block';
      msg.textContent = '📍 LOCATION DENIED — enable in browser';
    },
    {enableHighAccuracy: true, maximumAge: 5000, timeout: 10000}
  );
}

window.addEventListener('resize', () => {
  drawHistory([], 0, false);
  if (_map) _map.invalidateSize();
});
startRadarLoop();
_startGeo();
connect();
</script>
</body>
</html>"""


# ─── Serve report files ───────────────────────────────────────────────

from fastapi.responses import FileResponse

@app.get("/api/recordings")
async def api_recordings():
    return _recorder.list() if _recorder else []


@app.get("/{filename:path}")
async def serve_file(filename: str):
    from fastapi.responses import Response
    p = Path(filename)
    if p.is_absolute() or not p.exists():
        return Response(status_code=404)
    if p.suffix == ".html":
        return FileResponse(str(p), media_type="text/html")
    if p.suffix == ".wav":
        return FileResponse(str(p), media_type="audio/wav",
                            headers={"Content-Disposition":
                                     f'attachment; filename="{p.name}"'})
    return Response(status_code=404)


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    global _detector, _visual, _audio_th, _read_frame, _mic_live, _mode, _t0
    global _recorder, _telegram_token, _telegram_chat_id

    ap = argparse.ArgumentParser(description="FPV Detect Web UI",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--simulate",       action="store_true")
    ap.add_argument("--no-camera",      action="store_true")
    ap.add_argument("--port",           type=int, default=8080)
    ap.add_argument("--no-browser",     action="store_true")
    ap.add_argument("--calibrate",      action="store_true")
    ap.add_argument("--cal-duration",   type=float, default=30.0)
    ap.add_argument("--telegram-token", type=str, default="",
                    help="Telegram Bot API token for alarm alerts")
    ap.add_argument("--telegram-chat",  type=str, default="",
                    help="Telegram chat_id to send alerts to")
    args = ap.parse_args()

    _telegram_token   = args.telegram_token
    _telegram_chat_id = args.telegram_chat

    # ── Audio ─────────────────────────────────────────────────────────
    try:
        import pyaudio
        _PYAUDIO_OK = True
    except ImportError:
        _PYAUDIO_OK = False

    stream = pa = None

    if args.simulate:
        sim = SimulatedAudio(); _read_frame = sim.read_frame
        print("  Acoustic: SIMULATION")
    elif _PYAUDIO_OK:
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                             input=True, frames_per_buffer=FRAME_SIZE)
            _mic_live   = True
            _read_frame = lambda: np.frombuffer(
                stream.read(FRAME_SIZE, exception_on_overflow=False), dtype=np.int16)
            print("  Acoustic: live microphone")
        except OSError as e:
            print(f"  Acoustic: no mic ({e}) — simulation")
            if pa: pa.terminate()
            pa = stream = None
            sim = SimulatedAudio(); _read_frame = sim.read_frame
    else:
        print("  Acoustic: simulation (pyaudio not installed)")
        sim = SimulatedAudio(); _read_frame = sim.read_frame

    # ── Calibration ───────────────────────────────────────────────────
    if args.calibrate:
        gate = _calibrate_terminal(_read_frame, args.cal_duration)
        global RMS_GATE
        RMS_GATE = gate
        print(f"  RMS_GATE applied: {gate}")

    # ── Visual ────────────────────────────────────────────────────────
    _visual = VisualDetector(no_camera=args.no_camera, force_simulate=args.simulate)
    _visual.start()

    # ── Detector + Recorder ───────────────────────────────────────────
    _detector = DroneDetector()
    _recorder = RecordingManager()
    _audio_th = _AudioThread(_detector, _visual, _read_frame)
    _audio_th.start()
    print(f"  Recorder: saving alarms to alarm_YYYYMMDD_HHMMSS.wav")

    # ── Mode ─────────────────────────────────────────────────────────
    _mode = ("FULL"          if _mic_live and _visual.camera_live
             else "ACOUSTIC ONLY" if _mic_live and args.no_camera
             else "ACOUSTIC ONLY" if _mic_live
             else "VISUAL ONLY"   if _visual.camera_live
             else "SIMULATION")

    _t0 = time.monotonic()
    url = f"http://localhost:{args.port}"
    print(f"  Mode    : {_mode}")
    print(f"  Server  : {url}")
    if _telegram_token:
        print(f"  Telegram: alerts → chat {_telegram_chat_id}")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
    finally:
        _audio_th.stop()
        if stream: stream.stop_stream(); stream.close()
        if pa:     pa.terminate()
        if _csv_fh: _csv_fh.close()


if __name__ == "__main__":
    main()
