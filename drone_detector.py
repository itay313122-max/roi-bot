#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║  FPV DRONE ACOUSTIC DETECTOR — POC                       ║
║  Based on Batear open-source logic (batear.io)           ║
║  Uses microphone → FFT → harmonic analysis               ║
║                                                          ║
║  HOW IT WORKS:                                           ║
║  FPV drone rotors spin at 100–800Hz (depending on size)  ║
║  They produce a fundamental frequency f₀ + harmonics     ║
║  2f₀, 3f₀ etc. This is their "fingerprint".             ║
║                                                          ║
║  INSTALL:  pip install pyaudio numpy scipy               ║
║  RUN:      python drone_detector.py                      ║
╚══════════════════════════════════════════════════════════╝
"""

import numpy as np
import sys
import time
import threading
import json
from collections import deque
from datetime import datetime

# ─── Try importing audio ───────────────────────────────────
try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠  pyaudio not found — running in SIMULATION MODE")
    print("   Install with: pip install pyaudio numpy scipy\n")

# ─── Config (mirrors Batear v2 parameters) ────────────────
SAMPLE_RATE     = 16000   # Hz
FRAME_SIZE      = 1024    # samples per frame
HZ_PER_BIN      = SAMPLE_RATE / FRAME_SIZE   # 15.625 Hz/bin
SCAN_MIN_HZ     = 180
SCAN_MAX_HZ     = 2400
SNR_THRESHOLD   = 4.0     # peak must be 4x noise floor
H2_RATIO_MIN    = 0.07    # 2nd harmonic / fundamental
H3_RATIO_MIN    = 0.035   # 3rd harmonic / fundamental
EMA_ALPHA       = 0.25    # smoothing factor
ALARM_FRAMES    = 2       # consecutive frames before alarm
RMS_GATE        = 0.0004  # ignore silence


class DroneDetector:
    def __init__(self):
        self.confidence_ema    = 0.0
        self.alarm_count       = 0
        self.alarm_active      = False
        self.detection_log     = []
        self.history           = deque(maxlen=100)   # confidence history
        self.f0_history        = deque(maxlen=20)    # detected frequencies
        self.frame_count       = 0
        self.total_detections  = 0
        self.lock              = threading.Lock()

        # Web dashboard state
        self.dashboard_data = {
            "status": "MONITORING",
            "confidence": 0.0,
            "alarm": False,
            "f0_hz": 0,
            "snr": 0.0,
            "history": [],
            "log": [],
            "frames": 0,
        }

    def analyze_frame(self, samples: np.ndarray) -> dict:
        """
        Core detection algorithm — identical logic to Batear v2:
        1. Hanning window
        2. FFT → Power Spectral Density
        3. Find strongest peak in 180–2400 Hz
        4. Check SNR vs noise floor
        5. Verify harmonics 2f₀ and 3f₀
        6. Compute confidence score
        """
        self.frame_count += 1

        # RMS gate — ignore silence / very quiet frames
        rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        if rms < RMS_GATE:
            return {"detected": False, "reason": "silence", "confidence": 0.0}

        # Normalize
        x = samples.astype(np.float32) / 32768.0

        # Hanning window + FFT
        window = np.hanning(len(x))
        fft    = np.fft.rfft(x * window, n=FRAME_SIZE)
        psd    = np.abs(fft) ** 2

        # Frequency bins
        freqs  = np.fft.rfftfreq(FRAME_SIZE, 1.0 / SAMPLE_RATE)

        # Scan 180–2400 Hz for strongest peak (f₀)
        scan_mask = (freqs >= SCAN_MIN_HZ) & (freqs <= SCAN_MAX_HZ)
        scan_psd  = psd.copy()
        scan_psd[~scan_mask] = 0.0
        peak_bin  = np.argmax(scan_psd)
        f0        = freqs[peak_bin]
        p_f0      = psd[peak_bin]

        # Noise floor = median of scan region (robust to single spikes)
        noise_floor = np.median(psd[scan_mask]) + 1e-12

        # SNR check
        snr = p_f0 / noise_floor
        if snr < SNR_THRESHOLD:
            return {"detected": False, "reason": f"low SNR {snr:.1f}", "confidence": 0.0, "f0": f0, "snr": snr}

        # Harmonic check
        def power_near(target_hz, window_hz=30):
            mask = (freqs >= target_hz - window_hz) & (freqs <= target_hz + window_hz)
            return np.max(psd[mask]) if np.any(mask) else 0.0

        p_2f0 = power_near(f0 * 2)
        p_3f0 = power_near(f0 * 3)

        h2 = p_2f0 / (p_f0 + 1e-12)
        h3 = p_3f0 / (p_f0 + 1e-12)

        harmonics_ok = (h2 >= H2_RATIO_MIN) and (h3 >= H3_RATIO_MIN)

        if not harmonics_ok:
            return {
                "detected": False,
                "reason": f"harmonics fail h2={h2:.3f} h3={h3:.3f}",
                "confidence": 0.0,
                "f0": f0,
                "snr": snr,
            }

        # Confidence score [0..1]
        snr_score = min(snr / 20.0, 1.0)
        h2_score  = min(h2 / 0.3,   1.0)
        h3_score  = min(h3 / 0.15,  1.0)
        confidence = (snr_score * 0.5 + h2_score * 0.3 + h3_score * 0.2)

        return {
            "detected": True,
            "confidence": confidence,
            "f0": float(f0),
            "snr": float(snr),
            "h2": float(h2),
            "h3": float(h3),
        }

    def process(self, result: dict):
        """EMA smoothing + hysteresis alarm logic"""
        raw_conf = result.get("confidence", 0.0)

        # EMA smooth
        self.confidence_ema = EMA_ALPHA * raw_conf + (1 - EMA_ALPHA) * self.confidence_ema

        # Hysteresis
        if self.confidence_ema > 0.35:
            self.alarm_count += 1
        else:
            self.alarm_count = max(0, self.alarm_count - 1)

        prev_alarm = self.alarm_active
        self.alarm_active = (self.alarm_count >= ALARM_FRAMES)

        # Log new detections
        if self.alarm_active and not prev_alarm:
            self.total_detections += 1
            entry = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "f0": round(result.get("f0", 0)),
                "snr": round(result.get("snr", 0), 1),
                "confidence": round(self.confidence_ema, 2),
            }
            self.detection_log.insert(0, entry)
            self.detection_log = self.detection_log[:50]  # keep last 50

        # Update history
        self.history.append(round(self.confidence_ema, 3))
        if result.get("detected") and result.get("f0"):
            self.f0_history.append(result["f0"])

        # Update dashboard data (thread-safe)
        with self.lock:
            self.dashboard_data = {
                "status":     "🚨 DRONE DETECTED" if self.alarm_active else "👂 MONITORING",
                "confidence": round(self.confidence_ema * 100, 1),
                "alarm":      self.alarm_active,
                "f0_hz":      round(result.get("f0", 0)) if result.get("detected") else 0,
                "snr":        round(result.get("snr", 0), 1),
                "history":    list(self.history),
                "log":        self.detection_log[:10],
                "frames":     self.frame_count,
                "detections": self.total_detections,
            }

    def get_state(self):
        with self.lock:
            return dict(self.dashboard_data)


# ─── Simulation mode (no mic needed) ──────────────────────
class SimulatedAudio:
    """Generates fake audio: background noise + occasional drone burst"""
    def __init__(self):
        self.t = 0
        self.drone_active = False
        self.drone_timer  = 0

    def read_frame(self):
        t = np.arange(self.t, self.t + FRAME_SIZE) / SAMPLE_RATE
        self.t += FRAME_SIZE

        # Background noise
        signal = np.random.normal(0, 500, FRAME_SIZE).astype(np.float32)

        # Simulate drone every ~8 seconds for 3 seconds
        cycle = (self.t / SAMPLE_RATE) % 11
        if 2.0 < cycle < 5.0:
            f0 = 280.0   # typical FPV fundamental ~280Hz
            amp = 8000
            signal += amp     * np.sin(2 * np.pi * f0       * t)
            signal += amp*0.4 * np.sin(2 * np.pi * f0 * 2   * t)
            signal += amp*0.2 * np.sin(2 * np.pi * f0 * 3   * t)
            signal += amp*0.1 * np.sin(2 * np.pi * f0 * 4   * t)

        return signal.astype(np.int16)


# ─── Web dashboard ─────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FPV Drone Detector</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
  :root {
    --bg: #030a06;
    --panel: #0a1a0f;
    --border: #1a4025;
    --green: #00ff6a;
    --green-dim: #005522;
    --red: #ff2020;
    --yellow: #ffcc00;
    --text: #b0ffcc;
    --dim: #3a6a4a;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Share Tech Mono', monospace;
    min-height: 100vh;
    padding: 20px;
  }
  .scanline {
    position: fixed; inset: 0; pointer-events: none; z-index: 100;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,80,0.015) 2px, rgba(0,255,80,0.015) 4px);
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px; margin-bottom: 20px;
  }
  .logo { font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 700; color: var(--green); letter-spacing: 4px; }
  .logo span { color: var(--dim); }
  .uptime { font-size: 11px; color: var(--dim); }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
  .grid2 { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 16px;
    position: relative;
    overflow: hidden;
  }
  .panel::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, var(--green), transparent);
    opacity: 0.3;
  }
  .panel-title {
    font-size: 10px; color: var(--dim); letter-spacing: 3px;
    text-transform: uppercase; margin-bottom: 10px;
  }
  .big-val {
    font-family: 'Rajdhani', sans-serif;
    font-size: 52px; font-weight: 700;
    line-height: 1; color: var(--green);
  }
  .big-val.alarm { color: var(--red); animation: blink .4s step-end infinite; }
  .big-val.warn  { color: var(--yellow); }
  .unit { font-size: 14px; color: var(--dim); margin-top: 4px; }
  @keyframes blink { 50% { opacity: 0.2; } }
  .status-box {
    padding: 12px 20px;
    font-family: 'Rajdhani', sans-serif;
    font-size: 18px; font-weight: 700; letter-spacing: 3px;
    text-align: center;
    border: 1px solid var(--border);
    background: var(--panel);
    transition: all .3s;
  }
  .status-box.alarm {
    border-color: var(--red);
    background: rgba(255,32,32,0.1);
    color: var(--red);
    box-shadow: 0 0 20px rgba(255,32,32,0.3);
  }
  .status-box.ok { color: var(--green); }
  canvas { width: 100%; height: 100px; display: block; background: #050f08; }
  .bar-wrap { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
  .bar-track { flex: 1; height: 8px; background: var(--green-dim); }
  .bar-fill { height: 100%; background: var(--green); transition: width .2s; }
  .bar-fill.alarm { background: var(--red); }
  .bar-fill.warn  { background: var(--yellow); }
  .bar-pct { font-size: 13px; color: var(--dim); min-width: 42px; text-align: right; }
  table { width: 100%; border-collapse: collapse; font-size: 11px; }
  th { color: var(--dim); font-weight: normal; padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--border); }
  td { padding: 5px 6px; border-bottom: 1px solid #0d1f12; }
  td.alarm-row { color: var(--red); }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--green); margin-left: 6px; animation: pulse 1.5s ease-in-out infinite; }
  .dot.alarm { background: var(--red); }
  @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
  .mini-stat { display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid #0d1f12; font-size: 12px; }
  .mini-stat:last-child { border: none; }
  .mini-stat .lbl { color: var(--dim); }
  .mini-stat .val { color: var(--green); }
</style>
</head>
<body>
<div class="scanline"></div>
<header>
  <div>
    <div class="logo">FPV<span>::</span>DETECT</div>
    <div class="uptime" id="uptime">INITIALIZING...</div>
  </div>
  <div class="status-box ok" id="status-box">
    <span class="dot" id="dot"></span> MONITORING
  </div>
</header>

<div class="grid" style="grid-template-columns:2fr 1fr 1fr">
  <div class="panel">
    <div class="panel-title">CONFIDENCE SCORE</div>
    <div class="big-val" id="conf-val">0.0</div>
    <div class="unit">%</div>
    <div class="bar-wrap">
      <div class="bar-track" style="flex:1"><div class="bar-fill" id="conf-bar" style="width:0%"></div></div>
      <div class="bar-pct" id="conf-pct">0%</div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">FUNDAMENTAL Hz</div>
    <div class="big-val" id="f0-val">—</div>
    <div class="unit">Hz (f₀)</div>
  </div>
  <div class="panel">
    <div class="panel-title">SNR</div>
    <div class="big-val" id="snr-val">—</div>
    <div class="unit">× noise floor</div>
  </div>
</div>

<div class="grid2">
  <div class="panel">
    <div class="panel-title">CONFIDENCE HISTORY</div>
    <canvas id="chart"></canvas>
  </div>
  <div class="panel">
    <div class="panel-title">SYSTEM STATS</div>
    <div class="mini-stat"><span class="lbl">FRAMES PROCESSED</span><span class="val" id="frames">0</span></div>
    <div class="mini-stat"><span class="lbl">TOTAL DETECTIONS</span><span class="val" id="det-count">0</span></div>
    <div class="mini-stat"><span class="lbl">SCAN RANGE</span><span class="val">180–2400 Hz</span></div>
    <div class="mini-stat"><span class="lbl">SNR THRESHOLD</span><span class="val">4.0×</span></div>
    <div class="mini-stat"><span class="lbl">EMA α</span><span class="val">0.25</span></div>
    <div class="mini-stat"><span class="lbl">SAMPLE RATE</span><span class="val">16 kHz</span></div>
  </div>
</div>

<div class="panel" style="margin-top:12px">
  <div class="panel-title">DETECTION LOG</div>
  <table>
    <thead><tr><th>TIME</th><th>f₀ Hz</th><th>SNR</th><th>CONFIDENCE</th></tr></thead>
    <tbody id="log-body">
      <tr><td colspan="4" style="color:var(--dim);text-align:center;padding:12px">Waiting for detections...</td></tr>
    </tbody>
  </table>
</div>

<script>
const canvas = document.getElementById('chart');
const ctx = canvas.getContext('2d');
canvas.width = canvas.offsetWidth * 2;
canvas.height = 200;

function drawChart(history) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#050f08';
  ctx.fillRect(0,0,W,H);
  // Grid
  ctx.strokeStyle = '#0d2010'; ctx.lineWidth = 1;
  for(let i=0;i<=4;i++){
    const y = H * i/4;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke();
  }
  if(history.length < 2) return;
  const step = W / (history.length - 1);
  // Fill
  ctx.beginPath();
  ctx.moveTo(0, H);
  history.forEach((v,i) => ctx.lineTo(i*step, H - v*H));
  ctx.lineTo((history.length-1)*step, H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(0,255,106,0.12)';
  ctx.fill();
  // Line
  ctx.beginPath();
  history.forEach((v,i) => { i===0 ? ctx.moveTo(0,H-v*H) : ctx.lineTo(i*step,H-v*H); });
  ctx.strokeStyle = '#00ff6a'; ctx.lineWidth = 2; ctx.stroke();
  // Threshold line
  ctx.beginPath(); ctx.moveTo(0,H*0.65); ctx.lineTo(W,H*0.65);
  ctx.strokeStyle='rgba(255,204,0,0.3)'; ctx.lineWidth=1; ctx.setLineDash([4,4]); ctx.stroke();
  ctx.setLineDash([]);
}

let startTime = Date.now();
function poll() {
  fetch('/api/state').then(r=>r.json()).then(d => {
    // Uptime
    const sec = Math.floor((Date.now()-startTime)/1000);
    document.getElementById('uptime').textContent = `UPTIME ${String(Math.floor(sec/3600)).padStart(2,'0')}:${String(Math.floor(sec%3600/60)).padStart(2,'0')}:${String(sec%60).padStart(2,'0')}`;
    // Alarm state
    const alarm = d.alarm;
    const warn  = d.confidence > 30;
    const sb = document.getElementById('status-box');
    const dot = document.getElementById('dot');
    sb.textContent = '';
    const dotEl = document.createElement('span');
    dotEl.className = 'dot' + (alarm?' alarm':'');
    sb.appendChild(dotEl);
    sb.appendChild(document.createTextNode(' ' + (alarm ? '⚠ DRONE DETECTED' : 'MONITORING')));
    sb.className = 'status-box' + (alarm?' alarm':' ok');
    // Confidence
    const cv = document.getElementById('conf-val');
    cv.textContent = d.confidence.toFixed(1);
    cv.className = 'big-val' + (alarm?' alarm': warn?' warn':'');
    const bar = document.getElementById('conf-bar');
    bar.style.width = d.confidence + '%';
    bar.className = 'bar-fill' + (alarm?' alarm': warn?' warn':'');
    document.getElementById('conf-pct').textContent = d.confidence.toFixed(0) + '%';
    // f0 + snr
    document.getElementById('f0-val').textContent = d.f0_hz > 0 ? d.f0_hz : '—';
    document.getElementById('snr-val').textContent = d.snr > 0 ? d.snr.toFixed(1) : '—';
    // Stats
    document.getElementById('frames').textContent = d.frames.toLocaleString();
    document.getElementById('det-count').textContent = d.detections || 0;
    // Chart
    if(d.history) drawChart(d.history);
    // Log
    const tbody = document.getElementById('log-body');
    if(d.log && d.log.length > 0) {
      tbody.innerHTML = d.log.map(e=>`
        <tr class="alarm-row">
          <td>${e.time}</td>
          <td>${e.f0} Hz</td>
          <td>${e.snr}×</td>
          <td>${(e.confidence*100).toFixed(0)}%</td>
        </tr>`).join('');
    }
  }).catch(()=>{});
}

setInterval(poll, 300);
poll();
</script>
</body>
</html>"""


def start_web_server(detector: DroneDetector, port=8765):
    """Minimal HTTP server — no dependencies needed"""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass   # silence access log

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
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def main():
    print("=" * 56)
    print("  FPV DRONE ACOUSTIC DETECTOR — POC")
    print("  Based on Batear open-source algorithm")
    print("=" * 56)

    detector = DroneDetector()

    # Start web dashboard
    port = 8765
    start_web_server(detector, port)
    print(f"\n  ✅ Dashboard: http://localhost:{port}")

    if AUDIO_AVAILABLE:
        print("  🎤 Using real microphone\n")
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAME_SIZE,
        )
        def read_frame():
            raw = stream.read(FRAME_SIZE, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.int16)
    else:
        print("  🤖 Simulation mode — drone appears every ~8 sec\n")
        sim = SimulatedAudio()
        read_frame = sim.read_frame

    print("  Press Ctrl+C to stop\n")

    try:
        while True:
            samples = read_frame()
            result  = detector.analyze_frame(samples)
            detector.process(result)

            # Console output every 10 frames
            if detector.frame_count % 10 == 0:
                conf = detector.confidence_ema
                alarm = "🚨 ALARM" if detector.alarm_active else "👂 listen"
                bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                f0_str = f"  f₀={result.get('f0',0):.0f}Hz" if result.get("detected") else ""
                print(f"\r  [{bar}] {conf*100:5.1f}% {alarm}{f0_str}  ", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n  Stopped.")
        if AUDIO_AVAILABLE:
            stream.stop_stream()
            stream.close()
            p.terminate()


if __name__ == "__main__":
    main()
