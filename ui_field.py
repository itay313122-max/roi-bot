#!/usr/bin/env python3
"""
ui_field.py — Standalone drone detector + field terminal UI

All-in-one: acoustic FFT + YOLOv8 + rich terminal.  No separate server.

Install:
  pip install numpy rich
  pip install pyaudio          # Linux: apt install portaudio19-dev first
  pip install ultralytics      # YOLOv8 visual layer
  pip install opencv-python    # webcam support

Usage:
  python ui_field.py                # auto-detect all hardware
  python ui_field.py --simulate     # full simulation (no mic / no camera)
  python ui_field.py --no-camera    # acoustic only, no visual
  python ui_field.py --calibrate    # 30s noise-floor calibration before start
  python ui_field.py --no-guide     # hide guide panel (narrow terminal)

Keys: C Calibrate · L Log CSV · R Reset · Q Quit
"""

import argparse
import csv
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

# ─── Optional dependencies ────────────────────────────────────────────

try:
    import pyaudio
    _PYAUDIO_OK = True
except ImportError:
    _PYAUDIO_OK = False

try:
    from ultralytics import YOLO
    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ─── Acoustic constants ───────────────────────────────────────────────

SAMPLE_RATE    = 16000
FRAME_SIZE     = 1024
SCAN_MIN_HZ    = 80
SCAN_MAX_HZ    = 1600
H2_RATIO_MIN   = 0.07
H3_RATIO_MIN   = 0.035
EMA_ALPHA      = 0.25
ALARM_FRAMES   = 2
RMS_GATE       = 0.0004      # updated by calibration
MIN_SNR        = 3.0
VISUAL_THRESH  = 0.60

VERSION        = "3.0"
CAL_SECS       = 30

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
            "combined_threat": "CLEAR", "visual": {},
        }

    def _f0_drift(self) -> tuple:
        vals = list(self.f0_window)
        if not vals:
            return 0.0, "STABLE", 0.0
        n       = len(vals)
        weights = list(range(1, n + 1))
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
        rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))
        if rms < RMS_GATE:
            return {"detected": False, "confidence": 0.0}

        x     = samples.astype(np.float32) / 32768.0
        fft   = np.fft.rfft(x * np.hanning(len(x)), n=FRAME_SIZE)
        psd   = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)

        mask     = (freqs >= SCAN_MIN_HZ) & (freqs <= SCAN_MAX_HZ)
        scan     = psd.copy(); scan[~mask] = 0.0
        peak_bin = np.argmax(scan)
        f0       = float(freqs[peak_bin])
        p_f0     = float(psd[peak_bin])
        snr      = p_f0 / (float(np.median(psd[mask])) + 1e-12)

        if snr < MIN_SNR:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr}

        hw = max(SAMPLE_RATE / FRAME_SIZE * 2, f0 * 0.07)

        def _pw(hz):
            m = (freqs >= hz - hw) & (freqs <= hz + hw)
            return float(np.max(psd[m])) if np.any(m) else 0.0

        h2 = _pw(f0 * 2) / (p_f0 + 1e-12)
        h3 = _pw(f0 * 3) / (p_f0 + 1e-12)
        if h2 < H2_RATIO_MIN or h3 < H3_RATIO_MIN:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr}

        n_harm = 2 + sum(1 for k in range(4, 9)
                         if _pw(f0 * k) / (p_f0 + 1e-12) >= 0.02)
        key, match, rpm = _classify_drone(f0, snr, n_harm)
        if key is None:
            return {"detected": False, "confidence": 0.0, "f0": f0, "snr": snr}

        p     = DRONE_DATABASE[key]
        conf  = min((min(snr / (p["snr_threshold"] * 3.0), 1.0) * 0.5
                     + min(h2 / 0.3, 1.0) * 0.3
                     + min(h3 / 0.15, 1.0) * 0.2) * 0.7 + match * 0.3, 1.0)
        return {"detected": True, "confidence": conf, "f0": f0, "snr": snr,
                "drone_key": key, "profile_label": p["label"],
                "threat": p["threat"], "rpm_est": rpm}

    def process(self, result: dict, visual_state: dict = None):
        raw = result.get("confidence", 0.0)
        self.confidence_ema = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self.confidence_ema
        if self.confidence_ema > 0.35:
            self.alarm_count = min(self.alarm_count + 1, ALARM_FRAMES + 2)
        else:
            self.alarm_count = max(0, self.alarm_count - 1)
        prev          = self.alarm_active
        self.alarm_active = self.alarm_count >= ALARM_FRAMES

        if result.get("detected") and result.get("f0"):
            self.f0_window.append(result["f0"])
        f0s, trend, rate = self._f0_drift()

        label  = result.get("profile_label", "—") if result.get("detected") else "—"
        threat = result.get("threat",         "LOW") if result.get("detected") else "LOW"
        rpm    = result.get("rpm_est",         0)    if result.get("detected") else 0

        v = visual_state or {"confidence": 0.0, "alert": False, "fps": 0.0,
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
                "status":         "DRONE DETECTED" if self.alarm_active else "MONITORING",
                "confidence":     round(self.confidence_ema * 100, 1),
                "alarm":          self.alarm_active,
                "f0_hz":          round(result.get("f0", 0)) if result.get("detected") else 0,
                "snr":            round(result.get("snr", 0), 1),
                "profile_label":  label, "threat": threat, "rpm_est": rpm,
                "rpm_trend":      trend, "rate_hz_per_s": round(rate, 1),
                "history":        list(self.history),
                "log":            self.detection_log[:10],
                "frames":         self.frame_count,
                "detections":     self.total_detections,
                "combined_threat": ct, "visual": v,
            }

    def get_state(self) -> dict:
        with self._lock:
            return dict(self._state)

    def reset(self):
        self.confidence_ema = 0.0
        self.alarm_count    = 0
        self.alarm_active   = False
        self.detection_log  = []
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
        return np.zeros(FRAME_SIZE, dtype=np.int16)  # silence between segments


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

        if no_camera:
            return

        if force_simulate:
            self.model_name = "SIMULATION"
            return

        if _YOLO_OK:
            self._load_model()
        if _CV2_OK and self.model is not None:
            self._open_camera()

    def _load_model(self):
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
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS,          30)
            self.cap = cap; self.camera_live = True
        else:
            cap.release()

    def start(self):
        if self._disabled:
            return
        if self._force_sim or (not _YOLO_OK) or self.model is None:
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
                for box in r.boxes:
                    if any(c in names.get(int(box.cls[0]), "").lower()
                           for c in self._CLASSES):
                        best_conf = max(best_conf, float(box.conf[0]))
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
        self._det    = detector
        self._vis    = visual
        self._rf     = read_frame_fn
        self._stop   = threading.Event()

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
            except Exception:
                time.sleep(0.01)


# ─── Inline calibration ───────────────────────────────────────────────

_cal_lock  = threading.Lock()
_cal_state: dict = {"running": False, "progress": 0.0, "elapsed": 0.0,
                    "duration": 0.0, "frames": 0,
                    "noise_floor": None, "rms_gate": None}


def _calibrate_terminal(read_frame_fn, duration_s: float = 30.0) -> float:
    """Blocking terminal calibration (run before rich Live starts)."""
    print(f"\n{'='*58}")
    print("  CALIBRATION — Keep area clear of drones")
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
            t_ui  = time.time()
            pct   = elapsed / duration_s
            bar   = "█" * int(pct * 40) + "░" * (40 - int(pct * 40))
            print(f"\r  [{bar}] {elapsed:4.0f}/{duration_s:.0f}s  "
                  f"frames: {len(rms_values)}", end="", flush=True)
    arr  = np.array(rms_values, dtype=np.float64)
    nf   = float(np.median(arr))
    gate = round(max(nf + 3.0 * np.std(arr), nf * 2.5), 6)
    print(f"\n\n  DONE — noise floor: {nf:.6f}   recommended rms_gate: {gate}\n")
    return gate


def _start_inline_calibration(read_frame_fn, duration_s: float = CAL_SECS):
    """Non-blocking background calibration (called from within rich Live)."""
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


# ─── Keyboard reader ──────────────────────────────────────────────────

class _KeyReader:
    def __init__(self):
        self._key  = None
        self._lock = threading.Lock()
        self._fd   = sys.stdin.fileno()
        self._old  = termios.tcgetattr(self._fd)

    def start(self):
        tty.setraw(self._fd)
        threading.Thread(target=self._loop, daemon=True).start()

    def restore(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        except Exception:
            pass

    def _loop(self):
        while True:
            try:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    ch = os.read(self._fd, 1)
                    if ch:
                        with self._lock:
                            self._key = ch.decode("utf-8", errors="replace").upper()
            except Exception:
                break

    def pop(self) -> str | None:
        with self._lock:
            k, self._key = self._key, None
        return k


# ─── Rich helpers ─────────────────────────────────────────────────────

def _bar(v: float, width: int = 22, color: str = "green") -> Text:
    v = max(0.0, min(1.0, v))
    n = int(v * width)
    t = Text()
    t.append("█" * n,           style=f"bold {color}")
    t.append("░" * (width - n), style="dim")
    t.append(f"  {v * 100:.0f}%", style=f"bold {color}")
    return t


def _uptime(secs: float) -> str:
    h, r = divmod(int(secs), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _range_est(snr: float, alarm: bool) -> str:
    if not alarm or snr <= 0: return "—"
    if snr >= 15: return "NEAR  (<50m)"
    if snr >= 8:  return "MEDIUM  (50–200m)"
    return "FAR  (>200m)"


_CT_COLOR = {"CONFIRMED": "bold red", "ACOUSTIC ONLY": "bold yellow",
             "VISUAL ONLY": "bold blue", "CLEAR": "bold green"}
_CT_ICON  = {"CONFIRMED": "‼  CONFIRMED", "ACOUSTIC ONLY": "⚠  ACOUSTIC ONLY",
             "VISUAL ONLY": "●  VISUAL ONLY", "CLEAR": "✓  CLEAR"}
_TR_ICON  = {"APPROACHING": "↑  APPROACHING", "RECEDING": "↓  RECEDING",
             "STABLE": "—  STABLE"}

_MODE_COLOR = {"FULL": "bold bright_green", "ACOUSTIC ONLY": "bold yellow",
               "VISUAL ONLY": "bold blue", "SIMULATION": "bold magenta"}


# ─── Panel builders ───────────────────────────────────────────────────

def _header(uptime, logging, csv_name, csv_rows) -> Panel:
    t = Text()
    t.append(f"  FPV DETECT v{VERSION} — FIELD MODE", style="bold bright_green")
    t.append("   ")
    if logging:
        t.append(f"● {csv_name}  ({csv_rows} rows)", style="bold red")
    else:
        t.append("○ log off", style="dim green")
    t.append(f"          UPTIME {_uptime(uptime)}", style="dim green")
    return Panel(t, style="green", padding=(0, 0))


def _acoustic_panel(s: dict) -> Panel:
    alarm = s.get("alarm", False)
    conf  = s.get("confidence", 0.0) / 100.0
    col   = "red" if alarm else "green"
    t     = Text()
    t.append("  "); t.append(_bar(conf, 22, col))
    t.append(f"\n  f₀   {s.get('f0_hz', 0):>6} Hz\n",    style="white")
    t.append(f"  RPM  ~{s.get('rpm_est', 0):>8,}\n",      style="white")
    t.append(f"  Type  {s.get('profile_label','—')[:20]}\n", style="bold white")
    t.append(f"  SNR   {s.get('snr', 0.0):.1f}×",         style="white")
    return Panel(t, title=" ACOUSTIC ", border_style="bold red" if alarm else "green",
                 padding=(0, 0))


def _visual_panel(s: dict) -> Panel:
    v     = (s or {}).get("visual") or {}
    alert = v.get("alert", False)
    conf  = v.get("confidence", 0.0) / 100.0
    col   = "red" if alert else "blue"
    t     = Text()
    t.append("  "); t.append(_bar(conf, 22, col))
    t.append(f"\n  Model  {v.get('model_name','—')[:24]}\n", style="white")
    t.append(f"  FPS    {v.get('fps', 0.0):.1f}\n",          style="white")
    cam_s = "LIVE CAMERA" if v.get("camera_live") else "SIMULATION"
    t.append(f"  Source {cam_s}\n",                           style="bold white")
    t.append(f"  Alert  {'YES' if alert else 'no'}",
             style=f"bold {'red' if alert else 'green'}")
    return Panel(t, title=" VISUAL (YOLOv8) ",
                 border_style="bold red" if alert else "blue", padding=(0, 0))


def _threat_panel(s: dict, cal: dict | None, cal_done_until: float, now: float) -> Panel:
    if cal and cal.get("running"):
        dur  = cal.get("duration", CAL_SECS)
        elap = cal.get("elapsed", 0.0)
        frm  = cal.get("frames", 0)
        t    = Text(justify="center")
        t.append("\n  CALIBRATING — KEEP AREA CLEAR\n\n", style="bold yellow")
        t.append("  "); t.append(_bar(cal.get("progress", 0.0), 32, "yellow"))
        t.append(f"\n\n  {elap:.0f}/{dur:.0f}s   frames: {frm:,}", style="yellow")
        return Panel(Align.center(t, vertical="middle"),
                     title=" CALIBRATION ", border_style="bold yellow")

    if (cal and not cal.get("running")
            and cal.get("noise_floor") is not None
            and now < cal_done_until):
        nf, gate = cal["noise_floor"], cal["rms_gate"]
        t = Text(justify="center")
        t.append("\n  CALIBRATION COMPLETE\n\n", style="bold bright_green")
        t.append(f"  noise floor : {nf:.6f}\n",    style="white")
        t.append(f"  rms_gate    : {gate:.6f}\n\n", style="bold cyan")
        t.append("Applied to current session", style="dim")
        return Panel(Align.center(t, vertical="middle"),
                     title=" DONE ", border_style="bold bright_green")

    ct    = s.get("combined_threat", "CLEAR")
    alarm = s.get("alarm", False)
    col   = _CT_COLOR.get(ct, "bold green")
    t     = Text(justify="center")
    icon  = _CT_ICON.get(ct, ct)
    if ct == "CONFIRMED":
        t.append(f"\n  {icon}  \n", style="bold red blink")
    else:
        t.append(f"\n  {icon}  \n", style=col)
    rate  = s.get("rate_hz_per_s", 0.0)
    ts    = _TR_ICON.get(s.get("rpm_trend", "STABLE"), "STABLE")
    rts   = f"  ({rate:+.1f} Hz/sec)" if abs(rate) > 0.5 else ""
    t.append(f"\n  Direction : {ts}{rts}\n", style="white")
    t.append(f"  Est. Range: {_range_est(s.get('snr', 0.0), alarm)}", style="cyan")
    return Panel(Align.center(t, vertical="middle"),
                 title=" COMBINED THREAT ", border_style=col)


def _stats_line(s: dict, nf, rg) -> Text:
    t = Text()
    t.append(f"  noise floor: {f'{nf:.6f}' if nf else '—'}   ", style="dim")
    t.append(f"rms_gate: {f'{rg:.6f}' if rg else '—'}   ",       style="dim cyan")
    t.append(f"frames: {s.get('frames', 0):,}   ",               style="dim")
    dets = s.get("detections", 0)
    t.append(f"detections: {dets}", style="bold yellow" if dets else "dim")
    return t


_LC = {"CONFIRMED": "bold red", "ACOUSTIC ONLY": "yellow",
       "VISUAL ONLY": "blue", "CLEAR": "dim"}


def _log_table(entries: list) -> Table:
    tbl = Table(box=box.SIMPLE, show_header=True,
                header_style="bold dim green", expand=True, padding=(0, 1))
    for col, w, sty in [("TIME", 8, "dim"), ("TYPE", 22, None), ("f₀", 9, "cyan"),
                        ("SNR", 6, "cyan"), ("THREAT", 16, None), ("DIR", 11, "dim")]:
        tbl.add_column(col, width=w, style=sty)
    for e in (entries or [])[:6]:
        ct = e.get("combined", "—")
        tbl.add_row(e.get("time", "—"), e.get("type", "—")[:20],
                    f"{e.get('f0', 0)} Hz", f"{e.get('snr', 0):.1f}×",
                    Text(ct, style=_LC.get(ct, "dim")), e.get("trend", "—"))
    return tbl


_GUIDE = """\
[bold bright_green]QUICK GUIDE[/]

[bold]f₀  (Hz)[/]
  BPF = RPM/60 × blades

[bold]SNR  (×)[/]
  Peak ÷ noise floor

[bold]APPROACHING[/]
  f₀ rising → closer

[bold]RECEDING[/]
  f₀ falling → farther

[bold]CONFIRMED[/]
  Acoustic [bold red]AND[/] visual

[bold]ACOUSTIC ONLY[/]
  Sound — no camera

[bold]rms_gate[/]
  Silence threshold
  Press [bold yellow]C[/] to calibrate

[dim]──────────────────[/]
[dim]BPF ranges:[/]

[bold]FPV 5"[/]   667–1200 Hz
[bold]FPV 3"[/]   833–1500 Hz
[bold]FPV 7"[/]   400– 733 Hz
[bold]Mavic[/]    150– 217 Hz
[bold]Matrice[/]  100– 167 Hz
[bold]Shahed[/]   200– 283 Hz
[bold]Lancet[/]   267– 500 Hz
"""


def _guide_panel() -> Panel:
    return Panel(_GUIDE, title=" QUICK GUIDE ",
                 border_style="dim green", padding=(0, 1))


def _controls_panel(logging, csv_name, csv_rows, mode) -> Panel:
    mc = _MODE_COLOR.get(mode, "dim")
    t  = Text()
    t.append("  [", style="dim"); t.append("C", style="bold yellow")
    t.append("] Calibrate    [", style="dim"); t.append("L", style="bold yellow")
    t.append(f"] {'Stop Log  ' if logging else 'Start Log'}  [", style="dim")
    t.append("R", style="bold yellow"); t.append("] Reset    [", style="dim")
    t.append("Q", style="bold yellow"); t.append("] Quit", style="dim")
    t.append(f"\n  MODE: ", style="dim")
    t.append(mode, style=mc)
    if logging:
        t.append(f"   ● {csv_name}  ({csv_rows} rows)", style="bold red")
    return Panel(t, style="dim green", padding=(0, 0))


# ─── CSV ──────────────────────────────────────────────────────────────

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


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    global RMS_GATE

    ap = argparse.ArgumentParser(
        description="FPV Detect v3 — standalone field terminal",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--simulate",    action="store_true",
                    help="Force full simulation (no mic, no camera)")
    ap.add_argument("--no-camera",   action="store_true",
                    help="Disable visual layer entirely")
    ap.add_argument("--calibrate",   action="store_true",
                    help="Run 30s noise-floor calibration before starting UI")
    ap.add_argument("--cal-duration", type=float, default=30.0,
                    help="Calibration duration in seconds (default: 30)")
    ap.add_argument("--no-guide",    action="store_true",
                    help="Hide quick-guide panel (narrow terminal)")
    args = ap.parse_args()

    # ── Audio setup ───────────────────────────────────────
    mic_live      = False
    stream        = None
    pa            = None

    if args.simulate:
        sim         = SimulatedAudio()
        read_frame  = sim.read_frame
        print("  Acoustic: SIMULATION")
    elif _PYAUDIO_OK:
        try:
            pa     = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                             input=True, frames_per_buffer=FRAME_SIZE)
            mic_live   = True
            read_frame = lambda: np.frombuffer(
                stream.read(FRAME_SIZE, exception_on_overflow=False), dtype=np.int16)
            print("  Acoustic: live microphone")
        except OSError as e:
            print(f"  Acoustic: no mic hardware ({e}) — simulation")
            if pa: pa.terminate()
            pa = stream = None
            sim = SimulatedAudio(); read_frame = sim.read_frame
    else:
        print("  Acoustic: simulation (pyaudio not installed)")
        sim = SimulatedAudio(); read_frame = sim.read_frame

    # ── Pre-run calibration ───────────────────────────────
    if args.calibrate:
        gate = _calibrate_terminal(read_frame, args.cal_duration)
        RMS_GATE = gate
        print(f"  RMS_GATE applied: {gate}\n")

    # ── Visual setup ──────────────────────────────────────
    visual = VisualDetector(
        no_camera    = args.no_camera,
        force_simulate = args.simulate,
    )
    visual.start()

    # ── Detector + audio thread ───────────────────────────
    detector = DroneDetector()
    audio_th = _AudioThread(detector, visual, read_frame)
    audio_th.start()

    # ── Determine mode label ──────────────────────────────
    if mic_live and visual.camera_live:
        mode = "FULL"
    elif mic_live and not visual.camera_live and not args.no_camera:
        mode = "ACOUSTIC ONLY"  # cam present but not live
    elif mic_live and args.no_camera:
        mode = "ACOUSTIC ONLY"
    elif not mic_live and visual.camera_live:
        mode = "VISUAL ONLY"
    else:
        mode = "SIMULATION"

    # ── Layout ────────────────────────────────────────────
    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="body"),
        Layout(name="controls", size=5),
    )
    if not args.no_guide:
        layout["body"].split_row(
            Layout(name="main_col",  ratio=7),
            Layout(name="guide_col", ratio=3),
        )
        layout["guide_col"].update(_guide_panel())
    else:
        layout["body"].split_row(Layout(name="main_col"))

    layout["main_col"].split_column(
        Layout(name="sensors", size=9),
        Layout(name="threat",  size=7),
        Layout(name="stats",   size=1),
        Layout(name="log"),
    )
    layout["sensors"].split_row(
        Layout(name="acoustic"),
        Layout(name="visual"),
    )

    # ── Session state ─────────────────────────────────────
    cons           = Console()
    keys           = _KeyReader()
    t0             = time.monotonic()
    csv_fh         = None
    csv_wr         = None
    csv_path       = ""
    csv_rows       = 0
    logging_on     = False
    cal_snap       = None
    cal_done_until = 0.0
    noise_floor    = None
    rms_gate_disp  = None

    def toggle_csv():
        nonlocal csv_fh, csv_wr, csv_path, csv_rows, logging_on
        if logging_on:
            if csv_fh: csv_fh.close()
            csv_fh = csv_wr = None; logging_on = False
        else:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"field_test_{ts}.csv"
            csv_fh   = open(csv_path, "w", newline="", encoding="utf-8")
            csv_wr   = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS)
            csv_wr.writeheader(); csv_fh.flush()
            csv_rows = 0; logging_on = True

    def write_row(s):
        nonlocal csv_rows
        if not logging_on or csv_wr is None: return
        csv_wr.writerow(_csv_row(s)); csv_fh.flush(); csv_rows += 1

    def redraw(now):
        up    = now - t0
        cname = Path(csv_path).name if csv_path else ""
        show_cal = cal_snap and (
            cal_snap.get("running")
            or (not cal_snap.get("running")
                and cal_snap.get("noise_floor") is not None
                and now < cal_done_until))
        layout["header"].update(_header(up, logging_on, cname, csv_rows))
        layout["controls"].update(_controls_panel(logging_on, cname, csv_rows, mode))
        s = detector.get_state()
        write_row(s)
        layout["acoustic"].update(_acoustic_panel(s))
        layout["visual"].update(_visual_panel(s))
        layout["threat"].update(
            _threat_panel(s, cal_snap if show_cal else None, cal_done_until, now))
        layout["stats"].update(_stats_line(s, noise_floor, rms_gate_disp))
        layout["log"].update(
            Panel(_log_table(s.get("log", [])),
                  title=" DETECTION LOG ", border_style="dim green"))

    # ── Run ───────────────────────────────────────────────
    keys.start()
    try:
        with Live(layout, console=cons, refresh_per_second=6, screen=True):
            while True:
                now = time.monotonic()
                key = keys.pop()

                if key == "Q":
                    break
                elif key == "L":
                    toggle_csv()
                elif key == "R":
                    detector.reset()
                    cal_snap = None; noise_floor = rms_gate_disp = None
                elif key == "C":
                    _start_inline_calibration(read_frame, CAL_SECS)
                    cal_snap = _get_cal()
                    cal_done_until = 0.0

                # Poll calibration state when running
                if cal_snap and cal_snap.get("running"):
                    cal_snap = _get_cal()
                    if not cal_snap.get("running") \
                            and cal_snap.get("noise_floor") is not None:
                        noise_floor    = cal_snap["noise_floor"]
                        rms_gate_disp  = cal_snap["rms_gate"]
                        RMS_GATE       = cal_snap["rms_gate"]
                        cal_done_until = now + 12.0

                redraw(now)
                time.sleep(0.05)

    finally:
        audio_th.stop()
        keys.restore()
        if stream:
            stream.stop_stream(); stream.close()
        if pa:
            pa.terminate()
        if csv_fh:
            csv_fh.close()
        cons.clear()
        cons.print(f"\n[bright_green]  FPV Detect v{VERSION} — session ended.[/]")
        if csv_path:
            cons.print(f"  CSV: {csv_path}  ({csv_rows} rows)")
        cons.print()


if __name__ == "__main__":
    main()
