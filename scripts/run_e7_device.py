#!/usr/bin/env python3
"""
E7 on the boards, unattended (run with Windows Python from the PC worker through WSL interop,
or directly: py scripts/run_e7_device.py --spec ... --output-dir ...).

For every board in the spec whose USB serial port is present (matched by USB vendor id), it
builds and flashes esp32_agreement with PlatformIO, then runs device_agreement.py run against the
bundle. Boards that are not plugged in are reported and skipped.

Spec (YAML):
  bundle: data/processed/revision/2026-10-29_z4_device_bundle/device_bundle
  boards:
    - {name: esp32, env: esp32dev, vid: 0x10C4}               # CP2102 bridge (classic ESP32)
    - {name: esp32s3, env: esp32-s3-native-usb, vid: 0x303A}  # ESP32-S3 native USB
  limit: null      # inputs per model (null = all)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


def ensure(pkg: str, module: str) -> None:
    try:
        __import__(module)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", pkg], check=True)


def find_port(vid: int):
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == vid:
            return p.device
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    ensure("pyyaml", "yaml")
    ensure("pyserial", "serial")
    ensure("numpy", "numpy")
    import yaml
    from serial.tools import list_ports

    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("python:", sys.executable, sys.version.split()[0])
    ports = [{"device": p.device, "vid": p.vid, "pid": p.pid, "description": p.description}
             for p in list_ports.comports()]
    print("serial ports:", ports)
    summary = {"ports": ports, "boards": {}}
    from scripts.device_agreement import run as run_agreement

    for b in spec["boards"]:
        name, env, vid = b["name"], b["env"], int(b["vid"])
        port = find_port(vid)
        if not port:
            print(f"⏭  {name}: no port with USB vendor id {vid:#06x} (board not connected)")
            summary["boards"][name] = {"status": "not connected"}
            continue
        print(f"▶ {name}: build + upload ({env}) on {port}", flush=True)
        log = out / f"{name}_platformio.log"
        with open(log, "w", encoding="utf-8", errors="replace") as fh:
            rc = subprocess.run([sys.executable, "-m", "platformio", "run", "-d", str(ROOT / "esp32_agreement"),
                                 "-e", env, "-t", "upload", "--upload-port", port],
                                stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        if rc != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print(f"❌ {name}: platformio failed (rc={rc})\n" + "\n".join(tail))
            summary["boards"][name] = {"status": "build/upload failed", "log_tail": tail}
            continue
        time.sleep(6)  # native USB re-enumerates after the reset
        port = find_port(vid) or port
        print(f"▶ {name}: agreement run on {port}", flush=True)
        try:
            rc = run_agreement(str(ROOT / spec["bundle"]), port, str(out / name), spec.get("limit"),
                               int(b.get("baud", 921600)))
            summary["boards"][name] = {"status": "ok" if rc == 0 else f"rc={rc}", "port": port}
        except Exception as err:  # keep going with the next board
            print(f"❌ {name}: {err!r}")
            summary["boards"][name] = {"status": f"error: {err!r}", "port": port}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = ["# E7 device agreement", ""]
    for name, s in summary["boards"].items():
        md.append(f"## {name}: {s['status']}")
        f = out / name / "device_agreement.md"
        if f.exists():
            md += ["", f.read_text(encoding="utf-8")]
        md.append("")
    (out / "device_e7.md").write_text("\n".join(md), encoding="utf-8")
    ok = [s for s in summary["boards"].values() if s["status"] == "ok"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
