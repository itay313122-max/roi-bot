#!/usr/bin/env python3
"""
test_logger.py — field data recorder for drone_detector.py

Launches drone_detector.py in the background (or attaches to a running
instance), polls /api/state every 500 ms, and writes one CSV row per poll.

CSV columns:
  timestamp, acoustic_confidence, visual_confidence, combined_threat,
  f0_hz, rpm_estimate, drone_type, direction, distance_phase

Usage:
  python test_logger.py                       # launch detector + log
  python test_logger.py --calibrate           # calibrate first, then log
  python test_logger.py --no-launch           # attach to already-running detector
  python test_logger.py --output my_test.csv  # custom output file
  python test_logger.py --port 8765           # detector port
"""

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


# ─── Helpers ──────────────────────────────────────────────
def _distance_phase(trend: str) -> str:
    return {
        "APPROACHING": "INBOUND",
        "RECEDING":    "OUTBOUND",
        "STABLE":      "HOVER/OVERHEAD",
    }.get(trend, "N/A")


def _poll(port: int) -> dict | None:
    try:
        url = f"http://localhost:{port}/api/state"
        with urllib.request.urlopen(url, timeout=0.4) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def _wait_for_api(port: int, timeout_s: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _poll(port) is not None:
            return True
        time.sleep(0.5)
    return False


# ─── Main ─────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Field data logger for drone_detector.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--port",        type=int,   default=8765,  help="Detector HTTP port (default: 8765)")
    ap.add_argument("--output",      default=None,              help="CSV output path (default: session_TIMESTAMP.csv)")
    ap.add_argument("--interval",    type=float, default=0.5,   help="Poll interval in seconds (default: 0.5)")
    ap.add_argument("--no-launch",   action="store_true",       help="Attach to already-running detector")
    ap.add_argument("--calibrate",   action="store_true",       help="Pass --calibrate to drone_detector.py")
    ap.add_argument("--cal-duration",type=float, default=60.0,  help="Calibration duration in seconds (default: 60)")
    args = ap.parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else Path(f"session_{ts}.csv")

    # ── Launch detector ────────────────────────────────────
    proc = None
    if not args.no_launch:
        cmd = [sys.executable, "drone_detector.py", "--port", str(args.port)]
        if args.calibrate:
            cmd += ["--calibrate", "--cal-duration", str(args.cal_duration)]
        print(f"  Starting: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        print(f"  PID: {proc.pid}")

        # During calibration the detector doesn't accept API calls, so
        # extend the startup wait accordingly.
        wait_s = args.cal_duration + 20.0 if args.calibrate else 20.0
        print(f"  Waiting for API on port {args.port} (up to {wait_s:.0f}s) ...")
        if not _wait_for_api(args.port, timeout_s=wait_s):
            print("  ERROR: detector API did not respond in time — aborting.")
            if proc:
                proc.terminate()
            sys.exit(1)
    else:
        if _poll(args.port) is None:
            print(f"  ERROR: no detector responding on port {args.port}.")
            sys.exit(1)

    # ── CSV setup ─────────────────────────────────────────
    FIELDS = [
        "timestamp",
        "acoustic_confidence",
        "visual_confidence",
        "combined_threat",
        "f0_hz",
        "rpm_estimate",
        "drone_type",
        "direction",
        "distance_phase",
    ]

    rows = 0

    def _stop(sig, frame):
        nonlocal rows
        print(f"\n\n  Stopped. {rows} rows written → {out_path}")
        if proc and proc.poll() is None:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    print(f"\n  Logging to  : {out_path}")
    print(f"  Interval    : {args.interval * 1000:.0f} ms")
    print(f"  Ctrl+C to stop\n")

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        fh.flush()

        while True:
            t0    = time.monotonic()
            state = _poll(args.port)

            if state is not None:
                visual = state.get("visual") or {}
                trend  = state.get("rpm_trend", "STABLE")
                writer.writerow({
                    "timestamp":           datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                    "acoustic_confidence": round(state.get("confidence",        0.0), 1),
                    "visual_confidence":   round(visual.get("confidence",        0.0), 1),
                    "combined_threat":     state.get("combined_threat",        "CLEAR"),
                    "f0_hz":               state.get("f0_hz",                      0),
                    "rpm_estimate":        state.get("rpm_est",                    0),
                    "drone_type":          state.get("profile_label",            "—"),
                    "direction":           trend,
                    "distance_phase":      _distance_phase(trend),
                })
                fh.flush()
                rows += 1

                # Console status line
                ct    = state.get("combined_threat", "CLEAR")
                conf  = state.get("confidence",       0.0)
                label = state.get("profile_label",    "—")
                ICONS = {
                    "CONFIRMED":    "[!!CONFIRMED!!]",
                    "ACOUSTIC ONLY":"[ACOUSTIC]     ",
                    "VISUAL ONLY":  "[VISUAL]       ",
                    "CLEAR":        "[CLEAR]        ",
                }
                print(
                    f"\r  row {rows:6d} | {ICONS.get(ct, ct)} | "
                    f"ac:{conf:5.1f}% | {label[:28]:<28s}  ",
                    end="", flush=True,
                )

            # Sleep for the rest of the interval
            elapsed = time.monotonic() - t0
            sleep_t = args.interval - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
