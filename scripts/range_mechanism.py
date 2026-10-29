#!/usr/bin/env python3
"""
Mechanism tests for INT8 failures (Dr. Saha's guide E3, E4), on the v13.1 checkpoints.

E3  input / hidden range, 2 x 2, with every other factor fixed (same FP32 model, same calibration
    draw, per-tensor fake-quant export):
      A  input min/max,      hidden min/max
      B  input clipped +-c,  hidden min/max      (hidden ranges measured on clipped inputs)
      C  input min/max,      hidden 99.99th pct
      D  input clipped +-c,  hidden 99.99th pct
    crossed with calibration draws: random draws, the same draws with the client's most extreme
    records swapped in, and draws without records beyond `extreme_z` SD. For each cell: recall
    and FAR at the validation-chosen FAR budgets, attacks lost / recovered against the FP32 model,
    the input quantization step (in SD), and the share of test inputs / hidden units saturated.

E4  output discretization: the TFLite sigmoid output has 256 levels (step 1/256), so at a low FAR
    the threshold falls inside a tie group. Rows: FP32; FP32 with only its output rounded to the
    INT8 grid; FP32 with inputs clipped to +-5 / +-10 SD; INT8 (cell D); INT8 with a linear (logit)
    output, thresholded on the logit; 16-bit activations with INT8 weights (TFLite int16x8). Tie
    statistics: distinct output values on validation, and benign validation records sharing the
    threshold's output value.

Models per teacher seed: teacher F, compact FP32 after pruning + fine-tuning (PQ_<budget>.h5 saved
by compare_arms.py) and the from-scratch compact models S<budget>.

Spec (YAML):
  dataset_config: config/v13/ton_fl.yaml
  train_dir: data/processed/revision/2026-10-29_z1_train_ton/train_fl
  compare_dir: data/processed/revision/2026-10-29_z1d_compare_ton_fed/compare_arms
  seeds: [1, 2, 3]
  models: [F, PQ_80, S80]            # F, PQ_<budget>, R_<budget>, S80, S128
  far_budgets: [0.01, 0.05]
  calib: {n: 2000, client: 0, draws: 3, clip: 5.0, hidden_pct: pct9999, extreme_z: 10.0, n_extreme: 10}

Usage:
  python scripts/range_mechanism.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import collections
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tensorflow import keras

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown
from scripts.compare_arms import check_manifest, keras_probs, load_data, metrics, threshold_at_far
from scripts.compression_ablation import _load_fl_model
from scripts.quant_distill_ablation import convert
from scripts.quant_lit_methods import Ranges, dense_layers, estimate, fq_export, tensors
from scripts.recall_priority import tflite_probs

CELLS = {"A": ("max", "max"), "B": ("clip", "max"), "C": ("max", "pct"), "D": ("clip", "pct")}


def cell_ranges(model, xcal: np.ndarray, input_mode: str, hidden_mode: str, clip: float, pct: str) -> Ranges:
    xin = np.clip(xcal, -clip, clip) if input_mode == "clip" else xcal
    t = tensors(model, xin)
    method = "max" if hidden_mode == "max" else pct
    r = {k: estimate(v, method) for k, v in t.items() if k != "input"}
    r["input"] = estimate(t["input"], "max")
    return r


def saturation(model, x: np.ndarray, r: Ranges) -> Dict[str, float]:
    """Share of test inputs with any feature outside the input range, and share of hidden units
    (over records x units) above their range, with inputs saturated as on the device."""
    lo, hi = r["input"]
    out = {"input_rows_saturated": float(np.mean(((x < lo) | (x > hi)).any(axis=1)))}
    t = tensors(model, np.clip(x, lo, hi))
    hid = [k for k in t if k != "input"][:-1]
    out["hidden_values_saturated"] = float(np.mean(np.concatenate(
        [(t[k] > r[k][1]).ravel() for k in hid]))) if hid else 0.0
    return out


def linear_output(model) -> keras.Model:
    """Same weights, last Dense without the sigmoid: the INT8 output is the logit."""
    dense = dense_layers(model)
    inp = keras.Input(shape=(dense[0].get_weights()[0].shape[0],))
    h = inp
    for i, layer in enumerate(dense):
        last = i == len(dense) - 1
        h = keras.layers.Dense(layer.units, activation=None if last else "relu", name=layer.name)(h)
    m = keras.Model(inp, h)
    for src, dst in zip(dense, dense_layers(m)):
        dst.set_weights(src.get_weights())
    return m


def draws(ctx_x: np.ndarray, cal: dict) -> List[Tuple[str, np.ndarray]]:
    n, rowmax = int(cal["n"]), np.abs(ctx_x).max(axis=1)
    order = np.argsort(-rowmax)
    ext = order[: int(cal["n_extreme"])]
    calm = np.where(rowmax <= float(cal["extreme_z"]))[0]
    out = []
    for d in range(int(cal["draws"])):
        rng = np.random.default_rng(500 + d)
        idx = rng.permutation(len(ctx_x))[:n]
        out.append((f"random{d}", idx))
        out.append((f"random{d}+extremes", np.concatenate([idx[: n - len(ext)], ext])))
        out.append((f"no_extremes{d}", rng.permutation(calm)[:n]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    spec = load_yaml(args.spec)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cal = {"n": 2000, "client": 0, "draws": 3, "clip": 5.0, "hidden_pct": "pct9999", "extreme_z": 10.0,
           "n_extreme": 10, **spec.get("calib", {})}
    fars = [float(b) for b in spec.get("far_budgets", [0.01, 0.05])]
    rel = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    tdir, cdir = rel(spec["train_dir"]), rel(spec["compare_dir"])
    fd = load_data(spec)
    val = fd.pooled_val()
    xv, yv = val.x, val.y.astype(np.float32)
    xt, yt, fam = fd.test.x, fd.test.y.astype(np.float32), np.asarray(fd.test.attack).astype(str)
    xc_all = fd.clients[int(cal["client"])].train.x
    cal_sets = draws(xc_all, cal)
    att = yt == 1
    e3, e4 = [], []

    def evaluate(pv, pt, ref) -> List[Dict[str, Any]]:
        rows = []
        for b in fars:
            t = threshold_at_far(pv, yv, b)
            vm, tm = metrics(pv, yv, t), metrics(pt, yt, t)
            d = pt >= t
            r0 = ref[b]
            lost, rec = att & r0 & ~d, att & ~r0 & d
            rows.append({"far_budget": b, "threshold": t, "val_far": vm["far"], "val_recall": vm["recall"],
                         "test_recall": tm["recall"], "test_far": tm["far"], "test_fn": tm["fn"],
                         "attacks_lost_vs_fp32": int(lost.sum()), "attacks_recovered_vs_fp32": int(rec.sum()),
                         "lost_by_family": json.dumps(dict(collections.Counter(fam[lost]).most_common())),
                         "val_distinct_outputs": int(len(np.unique(pv))),
                         "benign_val_tied_at_threshold": int(np.sum(pv[yv == 0] == np.max(pv[(yv == 0) & (pv < t)], initial=-np.inf)))})
        return rows

    for seed in spec.get("seeds", [1, 2, 3]):
        sdir = out / f"s{seed}"
        sdir.mkdir(exist_ok=True)
        check_manifest(fd, tdir / "models" / f"F_s{seed}_manifest.json")
        for mname in spec.get("models", ["F", "PQ_80", "S80"]):
            try:
                if mname == "F" or mname.startswith("S"):
                    path = tdir / "models" / f"{mname}_s{seed}.h5"
                else:
                    path = cdir / f"s{seed}" / f"{mname}.h5"
                model = _load_fl_model(str(path))
                fv, ft = keras_probs(model, xv), keras_probs(model, xt)
                ref = {b: ft >= threshold_at_far(fv, yv, b) for b in fars}
                base = {"seed": seed, "model": mname}
                for r in evaluate(fv, ft, ref):
                    e4.append({**base, "variant": "fp32", **r})
                g = lambda p: np.clip(np.round(p * 256.0), 0, 255) / 256.0  # TFLite logistic output grid
                for r in evaluate(g(fv), g(ft), ref):
                    e4.append({**base, "variant": "fp32, output on INT8 grid", **r})
                for c in (5.0, 10.0):  # does input saturation alone explain INT8 beating FP32?
                    pv_c, pt_c = keras_probs(model, np.clip(xv, -c, c)), keras_probs(model, np.clip(xt, -c, c))
                    for r in evaluate(pv_c, pt_c, ref):
                        e4.append({**base, "variant": f"fp32, inputs clipped +-{c:g} SD", **r})
                # E3: 2 x 2 x calibration draws
                for dname, idx in cal_sets:
                    xcal = xc_all[idx]
                    for cell, (im, hm) in CELLS.items():
                        r = cell_ranges(model, xcal, im, hm, float(cal["clip"]), str(cal["hidden_pct"]))
                        p = sdir / f"{mname}_{cell}_{dname}.tflite"
                        fq_export(model, r, p)
                        pv, pt = tflite_probs(p, xv), tflite_probs(p, xt)
                        lo, hi = r["input"]
                        info = {**base, "cell": cell, "input": im, "hidden": hm, "draw": dname,
                                "calib_max_abs_z": round(float(np.abs(xcal).max()), 1),
                                "input_range": f"[{lo:.2f}, {hi:.2f}]", "input_step_sd": (hi - lo) / 255.0,
                                **saturation(model, xt[:20000], r)}
                        for row in evaluate(pv, pt, ref):
                            e3.append({**info, **row})
                        if dname == "random0" and cell == "D":
                            for row in evaluate(pv, pt, ref):
                                e4.append({**base, "variant": "INT8 (cell D)", **row})
                            lm = linear_output(model)
                            rl = cell_ranges(lm, xcal, im, hm, float(cal["clip"]), str(cal["hidden_pct"]))
                            pl = sdir / f"{mname}_D_logit.tflite"
                            fq_export(lm, rl, pl)
                            for row in evaluate(tflite_probs(pl, xv), tflite_probs(pl, xt), ref):
                                e4.append({**base, "variant": "INT8 (cell D), logit output", **row})
                            p16 = sdir / f"{mname}_int16x8.tflite"
                            convert(model, "int16x8", np.clip(xcal, -float(cal["clip"]), float(cal["clip"])), p16)
                            for row in evaluate(tflite_probs(p16, xv), tflite_probs(p16, xt), ref):
                                e4.append({**base, "variant": "16-bit activations, INT8 weights (clipped calib.)", **row})
                        if dname != "random0":
                            p.unlink(missing_ok=True)
                    print(f"  s{seed} {mname} {dname}: " + " ".join(
                        f"{x['cell']}:{x['test_recall']:.4f}" for x in e3[-2 * len(CELLS):] if x["far_budget"] == fars[0]), flush=True)
            except Exception as err:
                traceback.print_exc()
                e3.append({"seed": seed, "model": mname, "error": str(err)[:200]})
            rows_to_csv(e3, out / "e3_ranges_rows.csv")
            rows_to_csv(e4, out / "e4_output_rows.csv")

    # summaries: mean over seeds (and draws) per condition
    def summ(rows, keys):
        g = collections.defaultdict(list)
        for r in rows:
            if "error" not in r:
                g[tuple(r[k] for k in keys)].append(r)
        res = []
        for k, rs in sorted(g.items(), key=lambda kv: [str(v) for v in kv[0]]):
            m = lambda f, s=100.0: f"{statistics.mean(float(r[f]) * s for r in rs):.2f}"
            res.append({**dict(zip(keys, k)), "n": len(rs), "test_recall_%": m("test_recall"),
                        "test_far_%": m("test_far"), "val_far_%": m("val_far"),
                        "attacks_lost": m("attacks_lost_vs_fp32", 1.0), "attacks_recovered": m("attacks_recovered_vs_fp32", 1.0),
                        **({"input_step_sd": m("input_step_sd", 1.0), "input_rows_sat_%": m("input_rows_saturated"),
                            "hidden_sat_%": m("hidden_values_saturated")} if "input_step_sd" in rs[0] else
                           {"val_distinct_outputs": m("val_distinct_outputs", 1.0),
                            "benign_tied": m("benign_val_tied_at_threshold", 1.0)})})
        return res
    for r in e3:
        if "draw" in r:
            r["draw_type"] = "with extremes" if "extremes" in r["draw"] and "no_" not in r["draw"] else (
                "no extremes" if r["draw"].startswith("no_") else "random")
    s3 = summ(e3, ["model", "far_budget", "draw_type", "cell"])
    s4 = summ(e4, ["model", "far_budget", "variant"])
    rows_to_csv(s3, out / "e3_summary.csv")
    rows_to_csv(s4, out / "e4_summary.csv")
    rows_to_markdown(s3, out / "e3_ranges.md", "E3: input / hidden range 2x2 x calibration draws (mean over seeds and draws)")
    rows_to_markdown(s4, out / "e4_output.md", "E4: output discretization (mean over seeds)")
    md = (out / "e3_ranges.md").read_text() + "\n" + (out / "e4_output.md").read_text()
    (out / "range_mechanism.md").write_text(md)
    return 1 if any("error" in r for r in e3) else 0


if __name__ == "__main__":
    raise SystemExit(main())
