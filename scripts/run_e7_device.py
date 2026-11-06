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
    - {name: esp32, env: esp32dev, vids: [0x10C4, 0x1A86, 0x0403],   # USB-UART bridge (classic ESP32)
       alt: {name: esp32s3, env: esp32-s3-devkitc-1, marker: ESP32-S3}}  # ... or an ESP32-S3 on its UART port
    - {name: esp32s3, env: esp32-s3-native-usb, vids: [0x303A]}      # ESP32-S3 native USB
    # or {..., port: COM3} to pick the port explicitly
  limit: null      # inputs per model (null = all)
  skip_done_in: <summary.json of an earlier device_e7 run>   # optional: boards already done there are skipped
  models: [ton_QC80, ...]   # optional: only these models
Exit code 3 (with --probe, or when none of the remaining boards is connected): nothing to do yet;
the PC worker retries the job at its next poll, so boards can be plugged in one at a time.
"""
from __future__ import annotations

import argparse
import json
import os
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
        # a user site-packages directory created just now is not on sys.path yet
        import importlib
        import site
        site.addsitedir(site.getusersitepackages())
        importlib.invalidate_caches()
        __import__(module)


# USB vendor ids of the serial bridges on ESP32 dev boards
BRIDGE_VIDS = {0x10C4: "Silicon Labs CP210x", 0x1A86: "WCH CH340/CH343", 0x0403: "FTDI FT232"}
ESPRESSIF_VID = 0x303A  # ESP32-S3 native USB


def find_port(b: dict):
    """Port for a board spec: explicit `port`, else the first port whose vendor id is in `vids`."""
    from serial.tools import list_ports
    ports = list_ports.comports()
    if b.get("port"):
        return b["port"] if any(p.device == b["port"] for p in ports) else None
    vids = [int(v) for v in (b.get("vids") or [b["vid"]])]
    for p in ports:
        if p.vid in vids:
            return p.device
    return None


def ensure_idf_python() -> None:
    """espressif32@6.5.0's ESP-IDF builder imports pkg_resources, which Python 3.12 no longer ships
    (setuptools is not preinstalled, and setuptools >= 81 drops pkg_resources). Install an older
    setuptools for the PlatformIO interpreter and for any ESP-IDF virtualenv PlatformIO created."""
    pys = [sys.executable]
    penv = Path.home() / ".platformio" / "penv"
    for d in penv.glob(".espidf-*"):
        for exe in (d / "Scripts" / "python.exe", d / "bin" / "python"):
            if exe.exists():
                pys.append(str(exe))
    for py in pys:
        ok = subprocess.run([py, "-c", "import pkg_resources"], capture_output=True).returncode == 0
        if not ok:
            extra = ["--user"] if py == sys.executable else []
            print(f"  installing setuptools<81 (pkg_resources) for {py}", flush=True)
            subprocess.run([py, "-m", "pip", "install", *extra, "setuptools<81"], stdin=subprocess.DEVNULL)
    import importlib
    import site
    site.addsitedir(site.getusersitepackages())
    importlib.invalidate_caches()


def save_build_meta(project: Path, env: str, dest: Path) -> None:
    """Runtime provenance: library / component versions and the effective build configuration."""
    import shutil
    dest.mkdir(parents=True, exist_ok=True)
    pats = [f".pio/libdeps/{env}/TensorFlowLite_ESP32/library.*", f".pio/libdeps/{env}/TensorFlowLite_ESP32/.piopm",
            "dependencies.lock", f"sdkconfig.{env}", "managed_components/espressif__esp-tflite-micro/idf_component.yml",
            "managed_components/espressif__esp-tflite-micro/.component_hash", "managed_components/espressif__esp-nn/idf_component.yml",
            "platformio.ini", "sdkconfig.defaults"]
    for pat in pats:
        for f in project.glob(pat):
            shutil.copyfile(f, dest / (project.name + "__" + f.parent.name + "__" + f.name.lstrip(".")))
    try:
        import platformio
        (dest / "platformio_version.txt").write_text(platformio.__version__, encoding="utf-8")
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--probe", action="store_true", help="only check whether a board to test is connected")
    a = ap.parse_args()
    ensure("pyyaml", "yaml")
    ensure("pyserial", "serial")
    ensure("numpy", "numpy")
    import yaml
    from serial.tools import list_ports

    spec = yaml.safe_load(open(a.spec, encoding="utf-8"))
    done = set()
    for f in [spec.get("skip_done_in")] + [str(Path(a.output_dir) / "summary.json")]:
        if f and (ROOT / f).exists():
            prev = json.loads((ROOT / f).read_text(encoding="utf-8"))
            done |= {n for n, s in prev.get("boards", {}).items() if s.get("status") == "ok"}
    todo = [b for b in spec["boards"] if b["name"] not in done]
    if not todo:
        print("every board in the spec is already done")
        return 0
    from serial.tools import list_ports as _lp
    for p in _lp.comports():
        print(f"  port {p.device}: vid={p.vid if p.vid is None else hex(p.vid)} pid={p.pid if p.pid is None else hex(p.pid)} "
              f"{p.description} | {p.hwid}")
    present = [b["name"] for b in todo if find_port(b)]
    print(f"boards done: {sorted(done)}; to do: {[b['name'] for b in todo]}; connected: {present}")
    if not present:
        print("no board to test is connected yet (exit 3: retry later)")
        return 3
    if a.probe:
        return 0
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("python:", sys.executable, sys.version.split()[0])
    ports = [{"device": p.device, "vid": p.vid, "pid": p.pid, "description": p.description}
             for p in list_ports.comports()]
    print("serial ports:", ports)
    old = out / "summary.json"
    summary = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {"boards": {}}
    summary["ports"] = ports
    from scripts.device_agreement import run as run_agreement

    for b in todo:
        name, env = b["name"], b["env"]
        port = find_port(b)
        if not port:
            print(f"⏭  {name}: not connected")
            summary["boards"][name] = {"status": "not connected"}
            continue
        def upload(name, env):
            print(f"▶ {name}: build + upload ({env}) on {port}", flush=True)
            log = out / f"{name}_platformio.log"
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                rc = subprocess.run([sys.executable, "-m", "platformio", "run", "-d", str(ROOT / project),
                                     "-e", env, "-t", "upload", "--upload-port", port],
                                    stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
            return rc, log

        project = b.get("project", "esp32_agreement")  # esp32_agreement_idf: current esp-tflite-micro runtime
        is_idf = project.endswith("_idf")
        if is_idf:
            ensure_idf_python()
        rc, log = upload(name, env)
        if rc != 0 and is_idf and "pkg_resources" in log.read_text(encoding="utf-8", errors="replace"):
            # first build creates the ESP-IDF virtualenv; give it setuptools as well and retry once
            ensure_idf_python()
            os.replace(log, log.with_suffix(".attempt1.log"))
            rc, log = upload(name, env)
        alt = b.get("alt")
        if rc != 0 and alt and alt["marker"] in log.read_text(encoding="utf-8", errors="replace"):
            # a bridge port can belong to either chip (e.g. an ESP32-S3 on its UART "COM" port):
            # esptool names the chip it found, so flash the matching firmware instead
            print(f"  {port} is an {alt['marker']}: switching to {alt['env']}")
            name, env = alt["name"], alt["env"]
            if name in done:
                print(f"⏭  {name}: already done")
                continue
            rc, log = upload(name, env)
        if rc != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            print(f"❌ {name}: platformio failed (rc={rc})\n" + "\n".join(tail))
            summary["boards"][name] = {"status": "build/upload failed", "log_tail": tail}
            continue
        time.sleep(6)  # native USB re-enumerates after the reset
        port = find_port(b) or port
        print(f"▶ {name}: agreement run on {port}", flush=True)
        try:
            save_build_meta(ROOT / project, env, out / name / "build_meta")
            rc = run_agreement(str(ROOT / spec["bundle"]), port, str(out / name), spec.get("limit"),
                               int(b.get("baud", 921600)), spec.get("models"))
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
