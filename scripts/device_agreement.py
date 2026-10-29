#!/usr/bin/env python3
"""
E7 device agreement (Dr. Saha's guide): do the deployed INT8 models make the same decisions on
the microcontroller as on the host, on thousands of inputs that stress the conversion?

  prepare (PC worker, needs the data and TensorFlow): builds a bundle with the models, the input
          vectors and the host TFLite outputs. Inputs per dataset: random test records, test
          records whose host score lies within two INT8 output steps of a validation-chosen
          threshold (per model and FAR budget), test records with a feature beyond the clipped
          input range, and synthetic stress vectors (one feature set to +-50 / +-1000 SD, all zeros).
  run     (Windows / any PC with the board; needs only numpy + pyserial): streams the inputs to
          esp32_agreement firmware, compares the device's float32 output bits with the host
          outputs, and reports exact matches, the largest difference in INT8 output steps,
          decision agreement at both thresholds, and per-stage latency / arena use.

Usage:
  python scripts/device_agreement.py prepare --spec <spec.yaml> --output-dir <bundle dir>
  py scripts/device_agreement.py run --bundle <bundle dir> --port COM3 [--out <dir>] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEP = 1.0 / 256.0  # TFLite sigmoid output step


# --------------------------------------------------------------------------- prepare (PC)
def prepare(spec_path: str, out_dir: str) -> int:
    import os
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    from scripts.ablation_utils import load_yaml
    from scripts.compare_arms import load_data, threshold_at_far
    from scripts.recall_priority import tflite_probs

    spec = load_yaml(spec_path)
    out = Path(out_dir)
    (out / "models").mkdir(parents=True, exist_ok=True)
    fars = [float(b) for b in spec.get("far_budgets", [0.01, 0.05])]
    clip = float(spec.get("clip", 5.0))
    bundle = {"models": [], "datasets": {}, "far_budgets": fars}
    for ds in spec["datasets"]:
        name = ds["name"]
        fd = load_data({"dataset_config": ds["dataset_config"]})
        val = fd.pooled_val()
        xv, yv = val.x, val.y
        xt, yt, fam = fd.test.x, fd.test.y, np.asarray(fd.test.attack).astype(str)
        rng = np.random.default_rng(int(ds.get("seed", 1)))
        models = []
        for m in ds["models"]:
            src = ROOT / m["path"]
            dst = out / "models" / f"{m['name']}.tflite"
            shutil.copyfile(src, dst)
            pv, pt = tflite_probs(dst, xv), tflite_probs(dst, xt)
            thr = {str(b): threshold_at_far(pv, yv, b) for b in fars}
            models.append((m["name"], dst, pt, thr))
        # input selection (indices into the test split; synthetic rows appended afterwards)
        sel, cat = [], []
        pick = lambda idx, k: rng.permutation(idx)[:k] if len(idx) else idx
        r = pick(np.arange(len(yt)), int(ds.get("n_random", 1500)))
        sel += list(r); cat += ["random"] * len(r)
        for mname, _, pt, thr in models:
            for b, t in thr.items():
                near = np.where(np.abs(pt - t) <= 2 * STEP)[0]
                near = pick(near, int(ds.get("n_near", 300)))
                sel += list(near); cat += [f"near_threshold:{mname}:{b}"] * len(near)
        oor = np.where((np.abs(xt) > clip).any(axis=1))[0]
        oor = pick(oor, int(ds.get("n_out_of_range", 800)))
        sel += list(oor); cat += ["out_of_range"] * len(oor)
        X = xt[np.asarray(sel, dtype=np.int64)].astype(np.float32)
        Y = yt[np.asarray(sel, dtype=np.int64)].astype(np.int8)
        F = fam[np.asarray(sel, dtype=np.int64)]
        n_syn = int(ds.get("n_synthetic", 400))
        base = xt[pick(np.arange(len(yt)), n_syn)].astype(np.float32)
        feat = rng.integers(0, xt.shape[1], size=len(base))
        mag = rng.choice([-1000.0, -50.0, 50.0, 1000.0], size=len(base)).astype(np.float32)
        base[np.arange(len(base)), feat] = mag
        syn = np.vstack([base, np.zeros((10, xt.shape[1]), np.float32)])
        X = np.vstack([X, syn])
        Y = np.concatenate([Y, -np.ones(len(syn), np.int8)])
        F = np.concatenate([F, np.array(["synthetic"] * len(syn))])
        cat += ["synthetic_extreme"] * len(base) + ["zeros"] * 10
        np.save(out / f"inputs_{name}.npy", X)
        np.save(out / f"labels_{name}.npy", Y)
        np.save(out / f"families_{name}.npy", F)
        np.save(out / f"categories_{name}.npy", np.array(cat))
        for mname, dst, _, thr in models:
            host = tflite_probs(dst, X).astype(np.float32)
            np.save(out / f"host_{mname}.npy", host)
            bundle["models"].append({"name": mname, "file": f"models/{mname}.tflite", "dataset": name,
                                     "dim": int(X.shape[1]), "bytes": dst.stat().st_size, "thresholds": thr})
        bundle["datasets"][name] = {"n_inputs": int(len(X)),
                                    "categories": {c: cat.count(c) for c in sorted(set(cat))}}
        print(f"{name}: {len(X)} inputs, {len(models)} models")
    (out / "bundle.json").write_text(json.dumps(bundle, indent=2))
    return 0


# --------------------------------------------------------------------------- run (board)
def _readline(ser, timeout: float = 10.0) -> str:
    end = time.time() + timeout
    buf = b""
    while time.time() < end:
        c = ser.readline()
        if c:
            buf += c
            if buf.endswith(b"\n"):
                return buf.decode(errors="replace").strip()
    raise TimeoutError(f"no reply from device (partial: {buf!r})")


def _kv(line: str) -> dict:
    return dict(p.split("=", 1) for p in line.split()[1:] if "=" in p)


def run(bundle_dir: str, port: str, out_dir: str | None, limit: int | None, baud: int) -> int:
    import serial  # pip install pyserial

    bdir = Path(bundle_dir)
    bundle = json.loads((bdir / "bundle.json").read_text())
    out = Path(out_dir or bdir / "device_run")
    out.mkdir(parents=True, exist_ok=True)
    ser = serial.Serial(port, baud, timeout=0.5)
    time.sleep(2.0)  # opening the port resets most dev boards
    device, dev_models = {}, {}

    def banner(seconds: float) -> bool:
        t_end = time.time() + seconds
        while time.time() < t_end:
            line = ser.readline().decode(errors="replace").strip()
            if line.startswith("DEVICE"):
                device.update(_kv(line))
            elif line.startswith("MODEL"):
                kv = _kv(line)
                dev_models[kv["name"]] = int(kv["idx"])
            elif line.startswith("READY"):
                return True
        return False

    if not banner(8.0):  # no reset on open (e.g. native USB): ask for the model list
        ser.write(b"I")
        banner(5.0)
    if not dev_models:
        print("❌ no model list from the device (wrong port or firmware?)")
        return 1
    print("device:", device)
    report = {"device": device, "port": port, "models": {}}
    for m in bundle["models"]:
        name, ds = m["name"], m["dataset"]
        if name not in dev_models:
            print(f"⏭  {name} not in firmware")
            continue
        X = np.load(bdir / f"inputs_{ds}.npy")
        host = np.load(bdir / f"host_{name}.npy")
        cats = np.load(bdir / f"categories_{ds}.npy")
        if limit:
            X, host, cats = X[:limit], host[:limit], cats[:limit]
        ser.reset_input_buffer()
        ser.write(b"S" + bytes([dev_models[name]]))
        line = _readline(ser)
        if not line.startswith("SELECTED"):
            print(f"❌ {name}: {line}")
            report["models"][name] = {"error": line}
            continue
        sel = _kv(line)
        dev = np.full(len(X), np.nan, np.float32)
        tq, ti, td = [], [], []
        t0 = time.time()
        for i, x in enumerate(X):
            ser.write(b"X" + x.astype("<f4").tobytes())
            line = _readline(ser)
            if not line.startswith("Y "):
                print(f"❌ {name} vec {i}: {line}")
                break
            _, bits, q, inv, dq = line.split()
            dev[i] = struct.unpack("<f", struct.pack("<I", int(bits, 16)))[0]
            tq.append(int(q)); ti.append(int(inv)); td.append(int(dq))
            if i % 500 == 0:
                print(f"  {name}: {i}/{len(X)}", flush=True)
        ok = ~np.isnan(dev)
        diff = np.abs(dev[ok] - host[ok])
        res = {"n": int(ok.sum()), "exact_match": int(np.sum(dev[ok] == host[ok])),
               "max_abs_diff": float(diff.max()) if diff.size else None,
               "max_diff_output_steps": float(diff.max() / STEP) if diff.size else None,
               "selected": sel, "flash_bytes": m["bytes"], "seconds": round(time.time() - t0, 1),
               "latency_us": {k: {"median": statistics.median(v), "p95": float(np.percentile(v, 95)),
                                  "max": max(v)} for k, v in (("quantize_input", tq), ("invoke", ti),
                                                             ("dequantize_output", td)) if v},
               "decision_agreement": {}, "by_category": {}}
        for b, t in m["thresholds"].items():
            agree = (dev[ok] >= t) == (host[ok] >= t)
            res["decision_agreement"][b] = {"agree": int(agree.sum()), "n": int(ok.sum()), "threshold": t}
        for c in sorted(set(cats[ok])):
            mk = cats[ok] == c
            res["by_category"][c] = {"n": int(mk.sum()), "exact_match": int(np.sum(dev[ok][mk] == host[ok][mk]))}
        report["models"][name] = res
        np.save(out / f"device_{name}.npy", dev)
        print(f"✅ {name}: exact {res['exact_match']}/{res['n']}, max diff {res['max_diff_output_steps']} steps, "
              f"decisions " + ", ".join(f"FAR{b}: {v['agree']}/{v['n']}" for b, v in res["decision_agreement"].items())
              + f", invoke median {res['latency_us'].get('invoke', {}).get('median')} us")
        (out / "device_agreement.json").write_text(json.dumps(report, indent=2))
    lines = [f"# Device agreement ({device.get('chip', '?')}, {device.get('cpu_mhz', '?')} MHz)", "",
             "| model | inputs | exact output match | max diff (INT8 steps) | decisions FAR 1% | decisions FAR 5% | flash B | arena B | quantize µs | invoke µs (median / p95) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in report["models"].items():
        if "error" in r:
            lines.append(f"| {name} | – | {r['error']} | | | | | | | |")
            continue
        da = r["decision_agreement"]
        lat = r["latency_us"]
        lines.append(f"| {name} | {r['n']} | {r['exact_match']} | {r['max_diff_output_steps']:.2f} | "
                     f"{da.get('0.01', {}).get('agree')}/{r['n']} | {da.get('0.05', {}).get('agree')}/{r['n']} | "
                     f"{r['flash_bytes']} | {r['selected'].get('arena_used')} | {lat['quantize_input']['median']} | "
                     f"{lat['invoke']['median']} / {lat['invoke']['p95']:.0f} |")
    (out / "device_agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {out / 'device_agreement.md'}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--spec", required=True)
    p.add_argument("--output-dir", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--port", required=True)
    r.add_argument("--out")
    r.add_argument("--limit", type=int)
    r.add_argument("--baud", type=int, default=921600)
    a = ap.parse_args()
    if a.cmd == "prepare":
        return prepare(a.spec, a.output_dir)
    return run(a.bundle, a.port, a.out, a.limit, a.baud)


if __name__ == "__main__":
    raise SystemExit(main())
