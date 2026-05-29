#!/usr/bin/env python3
"""
ui_field.py — Full-screen terminal field UI for drone_detector.py

Requires:
  pip install rich

Usage:
  python ui_field.py [--port 8765] [--no-guide]
  python ui_field.py --port 8765 --no-guide    # narrow terminal

Keys:
  C  Run 30s noise-floor calibration
  L  Toggle CSV logging
  R  Reset session statistics
  Q  Quit
"""

import argparse
import csv
import json
import os
import select
import sys
import termios
import threading
import time
import tty
import urllib.request
from datetime import datetime
from pathlib import Path

from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

VERSION  = "2.0"
POLL_INT = 0.5    # seconds between API polls
CAL_SECS = 30     # calibration duration


# ─── HTTP helpers ─────────────────────────────────────────────────────

def _get(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=0.4) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _post(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=0.4) as r:
            return json.loads(r.read())
    except Exception:
        return None


# ─── Keyboard reader ──────────────────────────────────────────────────

class _KeyReader:
    """Non-blocking raw key reader (Linux / macOS)."""

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

def _bar(value: float, width: int = 22, color: str = "green") -> Text:
    """Render a [████░░░] progress bar as rich Text (value 0.0–1.0)."""
    v = max(0.0, min(1.0, value))
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
    """Heuristic range estimate from SNR — calibrate per deployment."""
    if not alarm or snr <= 0:
        return "—"
    if snr >= 15:
        return "NEAR  (<50m)"
    if snr >= 8:
        return "MEDIUM  (50–200m)"
    return "FAR  (>200m)"


_CT_COLOR = {
    "CONFIRMED":    "bold red",
    "ACOUSTIC ONLY":"bold yellow",
    "VISUAL ONLY":  "bold blue",
    "CLEAR":        "bold green",
}
_CT_ICON = {
    "CONFIRMED":    "‼  CONFIRMED",
    "ACOUSTIC ONLY":"⚠  ACOUSTIC ONLY",
    "VISUAL ONLY":  "●  VISUAL ONLY",
    "CLEAR":        "✓  CLEAR",
}
_TREND_ICON = {
    "APPROACHING": "↑  APPROACHING",
    "RECEDING":    "↓  RECEDING",
    "STABLE":      "—  STABLE",
}


# ─── Panel builders ───────────────────────────────────────────────────

def _header(uptime: float, logging: bool, csv_name: str, csv_rows: int) -> Panel:
    t = Text()
    t.append(f"  FPV DETECT v{VERSION} — FIELD MODE", style="bold bright_green")
    t.append("   ")
    if logging:
        t.append(f"● {csv_name}  ({csv_rows} rows)", style="bold red")
    else:
        t.append("○ log off", style="dim green")
    t.append(f"          UPTIME {_uptime(uptime)}", style="dim green")
    return Panel(t, style="green", padding=(0, 0))


def _acoustic(s: dict) -> Panel:
    alarm = s.get("alarm", False)
    conf  = s.get("confidence", 0.0) / 100.0
    f0    = s.get("f0_hz",   0)
    snr   = s.get("snr",     0.0)
    rpm   = s.get("rpm_est", 0)
    label = s.get("profile_label", "—")
    col   = "red" if alarm else "green"
    t = Text()
    t.append("  ")
    t.append(_bar(conf, width=22, color=col))
    t.append(f"\n  f₀   {f0:>6} Hz\n",       style="white")
    t.append(f"  RPM  ~{rpm:>8,}\n",           style="white")
    t.append(f"  Type  {label[:20]}\n",        style="bold white")
    t.append(f"  SNR   {snr:.1f}×",            style="white")
    border = "bold red" if alarm else "green"
    return Panel(t, title=" ACOUSTIC ", border_style=border, padding=(0, 0))


def _visual(s: dict) -> Panel:
    v     = (s or {}).get("visual") or {}
    alert = v.get("alert",       False)
    conf  = v.get("confidence",  0.0) / 100.0
    model = v.get("model_name",  "—")[:24]
    fps   = v.get("fps",         0.0)
    live  = v.get("camera_live", False)
    col   = "red" if alert else "blue"
    cam_s = "LIVE CAMERA" if live else "SIMULATION"
    t = Text()
    t.append("  ")
    t.append(_bar(conf, width=22, color=col))
    t.append(f"\n  Model  {model}\n",           style="white")
    t.append(f"  FPS    {fps:.1f}\n",           style="white")
    t.append(f"  Source {cam_s}\n",             style="bold white")
    t.append(f"  Alert  {'YES' if alert else 'no'}",
             style=f"bold {'red' if alert else 'green'}")
    border = "bold red" if alert else "blue"
    return Panel(t, title=" VISUAL (YOLOv8) ", border_style=border, padding=(0, 0))


def _threat(s: dict, cal: dict | None, cal_done_until: float, now: float) -> Panel:
    # ── Calibration overlay ───────────────────────────────
    if cal and cal.get("running"):
        dur  = cal.get("duration", CAL_SECS)
        elap = cal.get("elapsed",  0.0)
        frm  = cal.get("frames",   0)
        prog = cal.get("progress", 0.0)
        t = Text(justify="center")
        t.append("\n  CALIBRATING — KEEP AREA CLEAR OF DRONES  \n\n", style="bold yellow")
        t.append("  ")
        t.append(_bar(prog, width=32, color="yellow"))
        t.append(f"\n\n  {elap:.0f}/{dur:.0f}s   frames sampled: {frm:,}", style="yellow")
        return Panel(Align.center(t, vertical="middle"),
                     title=" CALIBRATION IN PROGRESS ", border_style="bold yellow")

    if cal and not cal.get("running") \
            and cal.get("noise_floor") is not None \
            and now < cal_done_until:
        nf   = cal["noise_floor"]
        gate = cal["rms_gate"]
        t = Text(justify="center")
        t.append("\n  CALIBRATION COMPLETE\n\n", style="bold bright_green")
        t.append(f"  noise floor  : {nf:.6f}\n",    style="white")
        t.append(f"  rms_gate     : {gate:.6f}\n\n", style="bold cyan")
        t.append("  Set  ", style="dim")
        t.append(f"RMS_GATE = {gate}", style="bold cyan")
        t.append("  in drone_detector.py", style="dim")
        return Panel(Align.center(t, vertical="middle"),
                     title=" CALIBRATION DONE ", border_style="bold bright_green")

    # ── Normal threat panel ───────────────────────────────
    ct    = s.get("combined_threat", "CLEAR")
    alarm = s.get("alarm", False)
    trend = s.get("rpm_trend",     "STABLE")
    rate  = s.get("rate_hz_per_s",  0.0)
    snr   = s.get("snr",            0.0)
    col   = _CT_COLOR.get(ct, "bold green")
    icon  = _CT_ICON.get(ct, ct)
    ts    = _TREND_ICON.get(trend, trend)
    rts   = f"  ({rate:+.1f} Hz/sec)" if abs(rate) > 0.5 else ""
    rng   = _range_est(snr, alarm)

    t = Text(justify="center")
    if ct == "CONFIRMED":
        t.append(f"\n  {icon}  \n", style="bold red blink")
    else:
        t.append(f"\n  {icon}  \n", style=col)
    t.append(f"\n  Direction : {ts}{rts}\n", style="white")
    t.append(f"  Est. Range: {rng}",          style="cyan")
    return Panel(Align.center(t, vertical="middle"),
                 title=" COMBINED THREAT ", border_style=col)


def _offline_panel(retry_in: float) -> Panel:
    t = Text(justify="center")
    t.append("\n  DETECTOR OFFLINE\n\n", style="bold red blink")
    t.append(f"  Retrying in {retry_in:.0f}s ...\n", style="yellow")
    t.append("\n  Run: python drone_detector.py", style="dim")
    return Panel(Align.center(t, vertical="middle"),
                 title=" NO CONNECTION ", border_style="bold red")


def _stats_line(s: dict, nf: float | None, rg: float | None) -> Text:
    frames = (s or {}).get("frames",     0)
    dets   = (s or {}).get("detections", 0)
    nf_s   = f"{nf:.6f}" if nf is not None else "—"
    rg_s   = f"{rg:.6f}" if rg is not None else "—"
    t = Text()
    t.append(f"  noise floor: {nf_s}   ", style="dim")
    t.append(f"rms_gate: {rg_s}   ",       style="dim cyan")
    t.append(f"frames: {frames:,}   ",     style="dim")
    t.append(f"detections: {dets}",
             style="bold yellow" if dets > 0 else "dim")
    return t


_LOG_CT_STYLE = {
    "CONFIRMED":    "bold red",
    "ACOUSTIC ONLY":"yellow",
    "VISUAL ONLY":  "blue",
    "CLEAR":        "dim",
}


def _log_table(entries: list) -> Table:
    tbl = Table(box=box.SIMPLE, show_header=True,
                header_style="bold dim green", expand=True, padding=(0, 1))
    tbl.add_column("TIME",    width=8,  style="dim")
    tbl.add_column("TYPE",    width=22)
    tbl.add_column("f₀",     width=9,  style="cyan")
    tbl.add_column("SNR",     width=6,  style="cyan")
    tbl.add_column("THREAT",  width=16)
    tbl.add_column("DIR",     width=11, style="dim")
    for e in (entries or [])[:6]:
        ct  = e.get("combined", "—")
        sty = _LOG_CT_STYLE.get(ct, "dim")
        tbl.add_row(
            e.get("time",  "—"),
            e.get("type",  "—")[:20],
            f"{e.get('f0', 0)} Hz",
            f"{e.get('snr', 0):.1f}×",
            Text(ct, style=sty),
            e.get("trend", "—"),
        )
    return tbl


_GUIDE_MARKUP = """\
[bold bright_green]QUICK GUIDE[/]

[bold]f₀  (Hz)[/]
  Blade-pass frequency
  BPF = RPM/60 × blades

[bold]RPM[/]
  Motor rev/min
  Estimated from f₀

[bold]SNR  (×)[/]
  Peak ÷ noise floor
  Higher = stronger sig.

[bold]APPROACHING[/]
  f₀ rising → closer

[bold]RECEDING[/]
  f₀ falling → farther

[bold]CONFIRMED[/]
  Acoustic [bold red]AND[/] visual
  both triggered

[bold]ACOUSTIC ONLY[/]
  Sound — no camera hit

[bold]VISUAL ONLY[/]
  Seen — too quiet

[bold]rms_gate[/]
  Silence threshold
  Below → frame ignored
  Press [bold yellow]C[/] to calibrate

[dim]──────────────────[/]
[dim]BPF freq ranges:[/]

[bold]FPV 5"[/]   667–1200 Hz
[bold]FPV 3"[/]   833–1500 Hz
[bold]FPV 7"[/]   400– 733 Hz
[bold]Mavic[/]    150– 217 Hz
[bold]Matrice[/]  100– 167 Hz
[bold]Shahed[/]   200– 283 Hz
[bold]Lancet[/]   267– 500 Hz

[dim]──────────────────[/]
[dim]Sensor fusion:[/]

CONFIRMED = acoustic+visual
ACOUSTIC  = sound only
VISUAL    = camera only
CLEAR     = nothing
"""


def _guide_panel() -> Panel:
    return Panel(_GUIDE_MARKUP, title=" QUICK GUIDE ",
                 border_style="dim green", padding=(0, 1))


def _controls_panel(logging: bool, csv_name: str, csv_rows: int) -> Panel:
    t = Text()
    t.append("  [", style="dim"); t.append("C", style="bold yellow")
    t.append("] Calibrate    [", style="dim"); t.append("L", style="bold yellow")
    t.append(f"] {'Stop Log  ' if logging else 'Start Log'}  [", style="dim")
    t.append("R", style="bold yellow"); t.append("] Reset    [", style="dim")
    t.append("Q", style="bold yellow"); t.append("] Quit", style="dim")
    if logging:
        t.append(f"\n  ● RECORDING → {csv_name}  ({csv_rows} rows)", style="bold red")
    return Panel(t, style="dim green", padding=(0, 0))


# ─── CSV ──────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "timestamp", "acoustic_confidence", "visual_confidence",
    "combined_threat", "f0_hz", "rpm_estimate",
    "drone_type", "direction", "distance_phase",
]
_DP = {"APPROACHING": "INBOUND", "RECEDING": "OUTBOUND", "STABLE": "HOVER/OVERHEAD"}


def _csv_row(s: dict) -> dict:
    v     = (s or {}).get("visual") or {}
    trend = s.get("rpm_trend", "STABLE")
    return {
        "timestamp":           datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
        "acoustic_confidence": round(s.get("confidence",   0.0), 1),
        "visual_confidence":   round(v.get("confidence",   0.0), 1),
        "combined_threat":     s.get("combined_threat",  "CLEAR"),
        "f0_hz":               s.get("f0_hz",                 0),
        "rpm_estimate":        s.get("rpm_est",               0),
        "drone_type":          s.get("profile_label",        "—"),
        "direction":           trend,
        "distance_phase":      _DP.get(trend, "N/A"),
    }


# ─── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="FPV Detect v2 — full-screen field terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--port",     type=int, default=8765,
                    help="drone_detector.py HTTP port (default: 8765)")
    ap.add_argument("--no-guide", action="store_true",
                    help="Hide the Quick Guide panel (use on narrow terminals)")
    args = ap.parse_args()

    base = f"http://localhost:{args.port}"
    cons = Console()
    keys = _KeyReader()
    t0   = time.monotonic()

    # Live state
    last_state = {}
    offline    = True
    retry_t    = 0.0
    last_poll  = 0.0

    # CSV
    csv_fh     = None
    csv_wr     = None
    csv_path   = ""
    csv_rows   = 0
    logging_on = False

    # Calibration
    cal_state      = None
    cal_done_until = 0.0
    noise_floor    = None
    rms_gate       = None

    # ── Layout ────────────────────────────────────────────
    layout = Layout()
    layout.split_column(
        Layout(name="header",   size=3),
        Layout(name="body"),
        Layout(name="controls", size=4),
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

    # ── Helpers ───────────────────────────────────────────
    def toggle_csv():
        nonlocal csv_fh, csv_wr, csv_path, csv_rows, logging_on
        if logging_on:
            if csv_fh:
                csv_fh.close()
            csv_fh = csv_wr = None
            logging_on = False
        else:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = f"field_test_{ts}.csv"
            csv_fh   = open(csv_path, "w", newline="", encoding="utf-8")
            csv_wr   = csv.DictWriter(csv_fh, fieldnames=_CSV_FIELDS)
            csv_wr.writeheader()
            csv_fh.flush()
            csv_rows   = 0
            logging_on = True

    def write_row(s: dict):
        nonlocal csv_rows
        if not logging_on or csv_wr is None:
            return
        csv_wr.writerow(_csv_row(s))
        csv_fh.flush()
        csv_rows += 1

    def redraw(now: float):
        up    = now - t0
        cname = Path(csv_path).name if csv_path else ""
        layout["header"].update(_header(up, logging_on, cname, csv_rows))
        layout["controls"].update(_controls_panel(logging_on, cname, csv_rows))

        if offline:
            ri = max(0.0, retry_t - now)
            layout["acoustic"].update(Panel("", border_style="dim"))
            layout["visual"].update(Panel("", border_style="dim"))
            layout["threat"].update(_offline_panel(ri))
            layout["stats"].update(Text(""))
            layout["log"].update(Panel("", border_style="dim"))
            return

        s = last_state
        show_cal = cal_state and (
            cal_state.get("running")
            or (not cal_state.get("running")
                and cal_state.get("noise_floor") is not None
                and now < cal_done_until)
        )
        layout["acoustic"].update(_acoustic(s))
        layout["visual"].update(_visual(s))
        layout["threat"].update(
            _threat(s, cal_state if show_cal else None, cal_done_until, now))
        layout["stats"].update(_stats_line(s, noise_floor, rms_gate))
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

                # Key handling
                if key == "Q":
                    break
                elif key == "L":
                    toggle_csv()
                elif key == "R":
                    last_state  = {}
                    cal_state   = None
                    noise_floor = rms_gate = None
                elif key == "C":
                    _post(f"{base}/api/calibrate?duration={CAL_SECS}")
                    cal_state = {
                        "running": True, "progress": 0.0, "elapsed": 0.0,
                        "duration": CAL_SECS, "frames": 0,
                        "noise_floor": None, "rms_gate": None,
                    }
                    cal_done_until = 0.0

                # API polling
                if now - last_poll >= POLL_INT:
                    last_poll = now
                    s = _get(f"{base}/api/state")
                    if s is None:
                        offline = True
                        retry_t = now + 2.0
                    else:
                        offline    = False
                        last_state = s
                        write_row(s)

                    if cal_state and cal_state.get("running"):
                        cs = _get(f"{base}/api/calibrate")
                        if cs:
                            cal_state = cs
                            if not cs.get("running") \
                                    and cs.get("noise_floor") is not None:
                                noise_floor    = cs["noise_floor"]
                                rms_gate       = cs["rms_gate"]
                                cal_done_until = now + 12.0

                redraw(now)
                time.sleep(0.05)   # ~20 fps render loop

    finally:
        keys.restore()
        if csv_fh:
            csv_fh.close()
        cons.clear()
        cons.print(f"\n[bright_green]  FPV Detect v{VERSION} — session ended.[/]")
        if csv_path:
            cons.print(f"  CSV saved: {csv_path}  ({csv_rows} rows)")
        cons.print()


if __name__ == "__main__":
    main()
