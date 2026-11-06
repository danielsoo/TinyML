#!/usr/bin/env python3
"""
Runtime comparison (Dr. Saha, 2026-11-05): the same model files and inputs on the same classic ESP32
through the old Arduino TensorFlowLite_ESP32 1.0.0 runtime (z10) and the current Espressif port
esp-tflite-micro (z10d: esp-nn optimized kernels; z10e: reference ANSI C kernels). Compares every
device output with the host TFLite interpreter output of the same file and with the other runtimes,
and splits every decision change into extra misses / extra false alarms / fewer false alarms / extra
detections at the frozen thresholds of the bundle.

  python scripts/runtime_compare.py --out docs/runtime_compare
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "data/processed/revision"
BUNDLE = REV / "2026-11-05_z7_device_bundle/device_bundle"
RUNS = {  # label -> device output folder
    "TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels":
        REV / "2026-11-06_z10_runtime_compare/device_e7/esp32_runtime_arduino_tflm_1_0_0",
    "esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels":
        REV / "2026-11-06_z10d_runtime_compare_idf55/device_e7/esp32_runtime_esp_tflite_micro",
    "esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels":
        REV / "2026-11-06_z10e_runtime_kernels_ansi_c/device_e7/esp32_runtime_esp_tflite_micro_ansi_c",
}
MODELS = ["ton_S80", "ton_S80_pt", "ton_PQ80_int16"]


def op_summary(path: Path) -> dict:
    try:
        import tflite  # pip install tflite
        from tflite.BuiltinOperator import BuiltinOperator
        from tflite.TensorType import TensorType
    except ImportError:
        return {}
    opn = {v: k for k, v in BuiltinOperator.__dict__.items() if not k.startswith("_")}
    tn = {v: k for k, v in TensorType.__dict__.items() if not k.startswith("_")}
    buf = path.read_bytes()
    m = tflite.Model.GetRootAsModel(buf, 0)
    g = m.Subgraphs(0)

    def code(oc):
        return opn[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]
    ops = sorted({f"{code(m.OperatorCodes(i))} v{m.OperatorCodes(i).Version()}" for i in range(m.OperatorCodesLength())})
    graph = []
    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        ins = [tn[g.Tensors(op.Inputs(j)).Type()] for j in range(op.InputsLength()) if op.Inputs(j) >= 0]
        graph.append(f"{code(m.OperatorCodes(op.OpcodeIndex()))}({','.join(ins)})")
    per_channel = sum(1 for i in range(g.TensorsLength())
                      if g.Tensors(i).Quantization() and g.Tensors(i).Quantization().ScaleLength() > 1)
    return {"ops": ops, "graph": " -> ".join(graph), "per_channel_tensors": per_channel}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/runtime_compare"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((BUNDLE / "bundle.json").read_text())
    thr = {m["name"]: m["thresholds"] for m in bundle["models"]}
    files = {m["name"]: m["file"] for m in bundle["models"]}
    y = np.load(BUNDLE / "labels_ton.npy")  # 1 attack, 0 benign, -1 synthetic probe (no label)
    runs = {k: v for k, v in RUNS.items() if (v / "device_agreement.json").exists()}
    rows, md = [], ["# Runtime comparison on the classic ESP32 (same files, same inputs)", ""]
    md += [f"Inputs: {len(y)} ({int((y == 1).sum())} attacks, {int((y == 0).sum())} benign, "
           f"{int((y == -1).sum())} synthetic probes without a label). Reference: host TFLite interpreter on the same file. "
           "Decision changes at the frozen thresholds of the bundle; misses / false alarms count labelled records only.", ""]
    md += ["## Runtimes", "", "| runtime | device banner | build |", "|---|---|---|"]
    for label, d in runs.items():
        dev = json.loads((d / "device_agreement.json").read_text())["device"]
        meta = sorted(p.name.split("__")[-1] for p in (d / "build_meta").glob("*")) if (d / "build_meta").exists() else []
        md.append(f"| {label} | {', '.join(f'{k}={v}' for k, v in dev.items())} | {', '.join(meta)} |")
    md += ["", "## Models", "", "| model | sha256 (first 16) | bytes | ops (versions) | graph | per-channel tensors |",
           "|---|---|---|---|---|---|"]
    for m in MODELS:
        p = BUNDLE / files[m]
        s = op_summary(p)
        md.append(f"| {m} | {hashlib.sha256(p.read_bytes()).hexdigest()[:16]} | {p.stat().st_size} | "
                  f"{', '.join(s.get('ops', []))} | {s.get('graph', '')} | {s.get('per_channel_tensors', '')} |")
    md += ["", "## Outputs and decisions vs host", "",
           "| model | runtime | exact output = host | max abs diff | output std (host / device) | arena used B | free heap B | invoke µs median | "
           "FAR | decisions = host | extra misses | extra false alarms | fewer false alarms | extra detections |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    dev_out = {}
    for m in MODELS:
        h = np.load(BUNDLE / f"host_{m}.npy").astype(np.float32)
        for label, d in runs.items():
            o = np.load(d / f"device_{m}.npy").astype(np.float32)
            dev_out[(m, label)] = o
            j = json.loads((d / "device_agreement.json").read_text())["models"][m]
            for far, t in thr[m].items():
                hd, dd = h >= t, o >= t
                r = {"model": m, "runtime": label, "far": far, "n": len(h),
                     "exact": int((o == h).sum()), "max_abs_diff": float(np.nanmax(np.abs(o - h))),
                     "std_host": float(h.std()), "std_device": float(np.nanstd(o)),
                     "arena_used": j["selected"].get("arena_used"), "free_heap": j["selected"].get("free_heap"),
                     "invoke_us_median": j["latency_us"]["invoke"]["median"],
                     "agree": int((hd == dd).sum()),
                     "extra_miss": int((hd & ~dd & (y == 1)).sum()), "extra_fa": int((~hd & dd & (y == 0)).sum()),
                     "fewer_fa": int((hd & ~dd & (y == 0)).sum()), "extra_detect": int((~hd & dd & (y == 1)).sum())}
                rows.append(r)
                md.append(f"| {m} | {label} | {r['exact']}/{r['n']} | {r['max_abs_diff']:.4f} | {r['std_host']:.3f} / {r['std_device']:.3f} | "
                          f"{r['arena_used']} | {r['free_heap']} | {r['invoke_us_median']:.0f} | {far} | {r['agree']}/{r['n']} | "
                          f"{r['extra_miss']} | {r['extra_fa']} | {r['fewer_fa']} | {r['extra_detect']} |")
    md += ["", "## Device vs device (bit-identical outputs)", "", "| model | runtime A | runtime B | identical outputs |", "|---|---|---|---|"]
    labels = list(runs)
    for m in MODELS:
        for i in range(len(labels)):
            for k in range(i + 1, len(labels)):
                a_, b_ = dev_out[(m, labels[i])], dev_out[(m, labels[k])]
                md.append(f"| {m} | {labels[i]} | {labels[k]} | {int((a_ == b_).sum())}/{len(a_)} |")
    (out / "runtime_compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    import csv
    with open(out / "runtime_compare.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
