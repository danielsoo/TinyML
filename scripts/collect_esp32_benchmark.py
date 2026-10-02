#!/usr/bin/env python3
"""
Collect the ESP32 on-device benchmark (esp32_tflite_project) from Serial into JSON.

Expected Serial lines (see esp32_tflite_project/src/main.cpp):
  DEVICE chip=ESP32-D0WD-V3 cores=2 cpu_mhz=240 flash_bytes=4194304 sdk=v4.4.7
  MODEL name=compressed bytes=66600 input_type=float32 output_type=float32
  PARITY_SUMMARY model=compressed max_abs_diff=0.003906 label_agree=8/8
  BENCHMARK model=compressed latency_us=1234 arena_used=5678 input_dim=78 run=1
  BENCHMARK_DONE
Older firmware lines without model=... are grouped under "default".

Usage:
  python scripts/collect_esp32_benchmark.py --port COM3            # Windows
  python scripts/collect_esp32_benchmark.py --port /dev/ttyUSB0    # Linux
  python scripts/collect_esp32_benchmark.py --port /dev/cu.usbserial-0001   # macOS
  python scripts/collect_esp32_benchmark.py --log-file esp32_serial.log
  python scripts/collect_esp32_benchmark.py --list-ports
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
DONE_MARKER = "BENCHMARK_DONE"
KV = re.compile(r"(\w+)=(\S+)")


def _kv(line: str) -> Dict[str, str]:
    return dict(KV.findall(line))


def _num(v: str):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def parse_log(text: str) -> dict:
    device: Dict[str, object] = {}
    models: Dict[str, dict] = {}
    errors: List[str] = []

    def model(name: str) -> dict:
        return models.setdefault(name, {"samples": []})

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("DEVICE"):
            device = {k: _num(v) for k, v in _kv(line).items()}
        elif line.startswith("MODEL"):
            kv = _kv(line)
            model(kv.get("name", "default")).update(
                {k: _num(v) for k, v in kv.items() if k != "name"}
            )
        elif line.startswith("PARITY_SUMMARY"):
            kv = _kv(line)
            model(kv.get("model", "default"))["parity"] = {
                "max_abs_diff": float(kv["max_abs_diff"]),
                "label_agree": kv.get("label_agree"),
            }
        elif line.startswith("BENCHMARK") and "latency_us=" in line:
            kv = _kv(line)
            sample = {"latency_us": float(kv["latency_us"])}
            for key in ("arena_used", "input_dim", "run"):
                if key in kv:
                    sample[key] = int(kv[key])
            model(kv.get("model", "default"))["samples"].append(sample)
        elif line.startswith("ERROR"):
            errors.append(line)
    return {"device": device, "models": models, "errors": errors}


def summarize(samples: List[dict]) -> dict:
    if not samples:
        return {"count": 0}
    lat = sorted(s["latency_us"] for s in samples)
    p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
    return {
        "count": len(lat),
        "latency_us_mean": mean(lat),
        "latency_us_median": median(lat),
        "latency_us_std": pstdev(lat),
        "latency_us_p95": p95,
        "latency_us_min": lat[0],
        "latency_us_max": lat[-1],
        "latency_ms_mean": mean(lat) / 1000.0,
        "arena_used": samples[-1].get("arena_used"),
        "input_dim": samples[-1].get("input_dim"),
    }


def read_serial(port: str, timeout_s: float) -> str:
    try:
        import serial
    except ImportError as e:
        raise ImportError("pip install pyserial") from e

    buf = []
    # Opening the port resets most ESP32 dev boards, so the run starts from boot.
    with serial.Serial(port, 115200, timeout=1) as ser:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            buf.append(line)
            if not line.startswith(("BENCHMARK model", "PARITY model")):
                print(line)
            if line == DONE_MARKER:
                break
        else:
            print(f"⚠️ Timed out after {timeout_s:.0f}s without {DONE_MARKER}; "
                  "press the board's EN/RST button and retry, or raise --timeout.")
    return "\n".join(buf)


def main():
    parser = argparse.ArgumentParser(description="Collect ESP32 benchmark from Serial")
    parser.add_argument("--port", default=None, help="e.g. COM3, /dev/ttyUSB0")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--list-ports", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="data/processed/ablation/esp32_benchmark.json",
    )
    args = parser.parse_args()

    if args.list_ports:
        from serial.tools import list_ports

        for p in list_ports.comports():
            print(f"{p.device}\t{p.description}")
        return 0

    if args.log_file:
        text = Path(args.log_file).read_text(encoding="utf-8")
    elif args.port:
        text = read_serial(args.port, args.timeout)
    else:
        print("Provide --port, --log-file or --list-ports", file=sys.stderr)
        return 1

    parsed = parse_log(text)
    for info in parsed["models"].values():
        info["summary"] = summarize(info["samples"])

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
    out.with_suffix(".log").write_text(text + "\n", encoding="utf-8")
    print(f"\n✅ ESP32 benchmark saved: {out} (raw log: {out.with_suffix('.log').name})")

    dev = parsed["device"]
    if dev:
        print(f"   Device: {dev.get('chip')} @ {dev.get('cpu_mhz')} MHz, SDK {dev.get('sdk')}")
    for name, info in parsed["models"].items():
        s = info["summary"]
        if not s.get("count"):
            continue
        parity = info.get("parity", {})
        print(f"   {name:<11} {info.get('bytes', '?')} B  "
              f"mean {s['latency_ms_mean']:.3f} ms  median {s['latency_us_median'] / 1000:.3f} ms  "
              f"p95 {s['latency_us_p95'] / 1000:.3f} ms  (n={s['count']}, arena {s['arena_used']} B)  "
              f"parity max|Δ|={parity.get('max_abs_diff', 'n/a')} labels {parity.get('label_agree', 'n/a')}")
    for err in parsed["errors"]:
        print(f"   ❌ {err}")
    return 0 if parsed["models"] and not parsed["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
