#!/usr/bin/env python3
"""
Follow-up to ptq_calibration_check.py (paper Section 5.9).

Part A — is QAT fine-tuning stable where full-integer PTQ is not?
  The deployed recipe (prune 50% -> fine-tune 3 ep -> QAT fine-tune 2 ep -> INT8) is repeated
  over several random 10,000-sample fine-tuning draws of client 0; each draw is also exported
  with full-integer PTQ calibrated on 500 samples of the same fine-tuning set.

Part B — mechanism and a cheap fix for PTQ instability.
  The federated model is PTQ-converted with exactly the n-sample calibration draws of
  ptq_calibration_check.py (same seeds), unclipped and with calibration inputs clipped to
  [-c, c]. For each set the float model's per-layer max |activation| over the calibration
  inputs is recorded, to test whether collapsed conversions coincide with stretched ranges.

Spec (YAML):
  ft_samples: 10000
  ft_draws: 5
  calib_n: 2000
  calib_draws: 3
  clip_values: [null, 3, 5, 10]
  models:
    - {name: ton_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>, ft_client: 0}

Usage:
  python scripts/qat_stability_check.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import (
    PRUNE_FT_EPOCHS,
    PRUNE_RATIO,
    _clone,
    _fit,
    _load_fl_model,
    _qat,
)
from scripts.ptq_calibration_check import convert_int8
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import export_tflite_qat

METRICS = ["accuracy", "precision", "attack_recall", "f1", "false_alarm_rate"]


def dense_act_max(model: keras.Model, x: np.ndarray) -> List[float]:
    """Max |output| of each Dense layer of the float model over x."""
    dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    probe = keras.Model(model.inputs, [l.output for l in dense])
    outs = probe.predict(x, batch_size=2048, verbose=0)
    return [round(float(np.abs(o).max()), 2) for o in outs]


def run_model(entry: dict, spec: dict, out_dir: Path) -> List[Dict[str, Any]]:
    name = entry["name"]
    cfg = load_yaml(ROOT / entry["config"])
    data_cfg = dict(cfg.get("data", {}))
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_train, y_train, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)
    parts = partition_data(
        x_train, y_train, int(data_cfg.get("num_clients", 4)),
        strategy=data_cfg.get("partition_strategy", "label_balanced"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
    )
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
    base = _load_fl_model(str(ROOT / entry["model"]))
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def evaluate(path: Path, row: dict):
        m = evaluate_tflite_model(path, x_test, y_test, threshold=threshold, latency_runs=0)
        row.update({k: m[k] for k in METRICS})
        rows.append(row)
        print("  " + " ".join(f"{k}={v}" for k, v in row.items() if k not in {"model"} and k not in METRICS)
              + f" f1={row['f1']:.4f} far={row['false_alarm_rate']:.4f}")
        path.unlink()

    # ---- Part A: deployed recipe over fine-tuning draws ----
    k = int(entry.get("ft_client", 0))
    cx, cy = parts[k]["x"], parts[k]["y"].astype(np.float32)
    n_ft = int(spec.get("ft_samples", 10000))
    for draw in range(int(spec.get("ft_draws", 5))):
        tf.keras.utils.set_random_seed(100 + draw)
        idx = np.random.default_rng(100 + draw).permutation(len(cy))[:n_ft]
        xf, yf = cx[idx], cy[idx]
        pr = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
        _fit(pr, xf, yf, PRUNE_FT_EPOCHS)
        p = tdir / f"A_d{draw}_ptq.tflite"
        convert_int8(pr, xf[:500], "builtins_int8", p)
        evaluate(p, {"model": name, "part": "A_ft_draw", "draw": draw, "variant": "prune_ft_ptq"})
        q = _qat(pr, xf, yf)
        p = tdir / f"A_d{draw}_qat.tflite"
        export_tflite_qat(q, str(p))
        evaluate(p, {"model": name, "part": "A_ft_draw", "draw": draw, "variant": "prune_ft_qat"})

    # ---- Part B: same calibration draws as ptq_calibration_check.py, clipped and unclipped ----
    n = int(spec.get("calib_n", 2000))
    for draw in range(int(spec.get("calib_draws", 3))):
        rng = np.random.default_rng(1000 + draw)  # same order of draws as ptq_calibration_check.py
        sets = [("pooled_rand", x_train[rng.choice(len(y_train), size=n, replace=False)])]
        for cid, part in enumerate(parts):
            sets.append((f"client{cid}", part["x"][rng.choice(len(part["y"]), size=min(n, len(part["y"])), replace=False)]))
        for source, xc in sets:
            try:
                act = dense_act_max(base, xc)
            except Exception as err:  # diagnostic only; never block the conversions
                print(f"  ⚠️ activation probe failed: {err}")
                act = []
            for clip in spec.get("clip_values", [None, 3, 5, 10]):
                xcal = xc if clip is None else np.clip(xc, -float(clip), float(clip))
                tag = "none" if clip is None else str(clip)
                p = tdir / f"B_{source}_d{draw}_clip{tag}.tflite"
                convert_int8(base, xcal, "builtins_int8", p)
                evaluate(p, {"model": name, "part": "B_calib", "source": source, "draw": draw, "n": n,
                             "clip": tag, "input_absmax": round(float(np.abs(xc).max()), 1),
                             "act_max": "/".join(str(a) for a in act)})
    return rows


def main():
    parser = argparse.ArgumentParser(description="QAT stability and PTQ calibration clipping")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for entry in spec["models"]:
        rows.extend(run_model(entry, spec, out_dir))
        rows_to_csv(rows, out_dir / "qat_stability.csv")
    rows_to_markdown(rows, out_dir / "qat_stability.md", "QAT stability and PTQ calibration clipping")
    save_json(rows, out_dir / "qat_stability.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
