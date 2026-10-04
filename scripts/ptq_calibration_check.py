#!/usr/bin/env python3
"""
INT8 PTQ calibration sensitivity: does full-integer PTQ of a float FL model depend on
which samples calibrate the activation ranges?

Table 3 calibrates on the first 500 pooled training samples; the quantization-method
comparison calibrates on 500 samples of client 0. On TON_IoT these gave F1 98.38 vs.
91.24. This script isolates the cause by converting the same federated model with
  - calibration source: pooled (first N, as compression.py), pooled (random N), each client (random N)
  - calibration size N
  - op set: TFLITE_BUILTINS (export_tflite default) vs. TFLITE_BUILTINS_INT8 (strict)
and evaluating every variant on the test split. Several random draws per source show
how much of the effect is draw-to-draw noise.

Spec (YAML):
  sizes: [100, 500, 2000]
  draws: 3
  models:
    - {name: ton_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>}

Usage:
  python scripts/ptq_calibration_check.py --spec <spec.yaml> --output-dir <dir>
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

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import _load_fl_model
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.tinyml.export_tflite import _strip_bn_dropout_for_tflite

OPSETS = {"builtins": [tf.lite.OpsSet.TFLITE_BUILTINS],
          "builtins_int8": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]}


def convert_int8(model, rep: np.ndarray, opset: str, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(_strip_bn_dropout_for_tflite(model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(rep)):
            yield [rep[i:i + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = OPSETS[opset]
    out_path.write_bytes(converter.convert())


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
    model = _load_fl_model(str(ROOT / entry["model"]))
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def add(source: str, n: int, draw: int, opset: str, x_cal: np.ndarray, y_cal: np.ndarray):
        p = tdir / f"{source}_n{n}_d{draw}_{opset}.tflite"
        convert_int8(model, x_cal, opset, p)
        m = evaluate_tflite_model(p, x_test, y_test, threshold=threshold, latency_runs=0)
        row = {"model": name, "source": source, "n": n, "draw": draw, "opset": opset,
               "cal_attack_share": round(float(np.mean(y_cal)), 4),
               "cal_absmax_p99": round(float(np.percentile(np.abs(x_cal).max(axis=1), 99)), 2)}
        row.update({k: m[k] for k in ["accuracy", "precision", "attack_recall", "f1", "false_alarm_rate"]})
        rows.append(row)
        print(f"  {source:10s} n={n:5d} d={draw} {opset:13s} attack={row['cal_attack_share']:.2f} "
              f"f1={row['f1']:.4f} far={row['false_alarm_rate']:.4f}")
        p.unlink()  # keep the output small; only metrics matter here

    # Exactly the two settings that disagree in the paper (Table 3 vs. Table 6)
    add("pooled_head", 500, 0, "builtins", x_train[:500], y_train[:500])
    add("pooled_head", 500, 0, "builtins_int8", x_train[:500], y_train[:500])
    perm0 = np.random.default_rng(0).permutation(len(parts[0]["y"]))[:500]
    add("client0_qd", 500, 0, "builtins", parts[0]["x"][perm0], parts[0]["y"][perm0])
    add("client0_qd", 500, 0, "builtins_int8", parts[0]["x"][perm0], parts[0]["y"][perm0])

    for n in spec.get("sizes", [100, 500, 2000]):
        for draw in range(int(spec.get("draws", 3))):
            rng = np.random.default_rng(1000 + draw)
            idx = rng.choice(len(y_train), size=n, replace=False)
            add("pooled_rand", n, draw, "builtins_int8", x_train[idx], y_train[idx])
            for cid, part in enumerate(parts):
                idx = rng.choice(len(part["y"]), size=min(n, len(part["y"])), replace=False)
                add(f"client{cid}", n, draw, "builtins_int8", part["x"][idx], part["y"][idx])
    return rows


def main():
    parser = argparse.ArgumentParser(description="INT8 PTQ calibration sensitivity")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(int(spec.get("seed", 42)))
    rows: List[Dict[str, Any]] = []
    for entry in spec["models"]:
        rows.extend(run_model(entry, spec, out_dir))
        rows_to_csv(rows, out_dir / "ptq_calibration.csv")
    rows_to_markdown(rows, out_dir / "ptq_calibration.md", "INT8 PTQ calibration sensitivity")
    save_json(rows, out_dir / "ptq_calibration.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
