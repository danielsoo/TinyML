#!/usr/bin/env python3
"""
Quantization-method comparison and client-local knowledge distillation for float FL models.

Part 1 — quantization methods, applied to (a) the federated model itself and (b) the
pruned (50%) + client-fine-tuned model:
  fp32, dynamic_range (int8 weights, float activations), float16, int8 (full-integer PTQ),
  int16x8 (int8 weights, int16 activations). Calibration uses the fine-tuning client's data.
  Only fp32 and full-integer int8 are known to run on TensorFlow Lite Micro (ESP32); the
  'mcu' column records this.

Part 2 — knowledge distillation without pooled data: a narrower MLP (width ratio r of
512-256-128) is trained on one client's local data, either from scratch (hard labels) or
distilled from the federated model (targets = a*y + (1-a)*sigmoid(logit_teacher / T)), then
exported as int8 PTQ and after 2 epochs of QAT fine-tuning.

Spec (YAML):
  ft_samples: 10000      # pruning fine-tune set (client-local)
  kd_samples: 50000      # distillation / scratch training set (client-local)
  kd_epochs: 10
  kd_temperature: 2.0
  kd_alpha: 0.5
  student_ratios: [0.5, 0.25, 0.125]
  quant_methods: [fp32, dynamic_range, float16, int8, int16x8]
  models:
    - {name: cic_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>, ft_client: 0}

Usage:
  python scripts/quant_distill_ablation.py --spec <spec.yaml> --output-dir <dir>
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
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_tflite, export_tflite_qat

MCU_SUPPORT = {"fp32": "yes", "int8": "yes", "int16x8": "untested",
               "dynamic_range": "no", "float16": "no"}


def convert(model: keras.Model, method: str, rep: np.ndarray, out_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(_strip_bn_dropout_for_tflite(model))

    def representative_dataset():
        for i in range(min(500, len(rep))):
            yield [rep[i:i + 1].astype(np.float32)]

    if method == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif method == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif method == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif method == "int16x8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    elif method != "fp32":
        raise ValueError(f"unknown quantization method {method}")
    out_path.write_bytes(converter.convert())


def make_student(input_dim: int, ratio: float) -> keras.Model:
    widths = [max(4, int(round(w * ratio))) for w in (512, 256, 128)]
    model = keras.Sequential(
        [keras.Input(shape=(input_dim,))]
        + [keras.layers.Dense(w, activation="relu") for w in widths]
        + [keras.layers.Dense(1, activation="sigmoid")]
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def kd_targets(teacher: keras.Model, x: np.ndarray, y: np.ndarray, temperature: float, alpha: float):
    p = np.clip(teacher.predict(x, batch_size=2048, verbose=0).ravel(), 1e-6, 1 - 1e-6)
    soft = 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / temperature))
    return (alpha * y + (1.0 - alpha) * soft).astype(np.float32)


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
    k = int(entry.get("ft_client", 0))
    cx, cy = parts[k]["x"], parts[k]["y"].astype(np.float32)
    perm = np.random.default_rng(0).permutation(len(cy))
    ft = (cx[perm[: int(spec.get("ft_samples", 10000))]], cy[perm[: int(spec.get("ft_samples", 10000))]])
    kd = (cx[perm[: int(spec.get("kd_samples", 50000))]], cy[perm[: int(spec.get("kd_samples", 50000))]])
    print(f"[{name}] client {k}: {len(cy):,} samples; FT {len(ft[1]):,} (attack {ft[1].mean():.1%}); "
          f"KD {len(kd[1]):,}")

    threshold = float(spec.get("threshold", cfg.get("evaluation", {}).get("prediction_threshold", 0.3)))
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def add(part: str, variant: str, method: str, path: Path, params: int):
        m = evaluate_tflite_model(path, x_test, y_test, threshold=threshold, latency_runs=50)
        row = {"model": name, "part": part, "variant": variant, "method": method,
               "mcu": MCU_SUPPORT.get(method, "yes"), "params": params,
               "size_kb": round(path.stat().st_size / 1024, 2),
               "latency_ms": round(float(m.get("latency_ms", 0.0)), 4)}
        row.update({key: m[key] for key in ["accuracy", "precision", "attack_recall", "f1",
                                            "false_alarm_rate", "fp", "fn", "threshold"]})
        rows.append(row)
        print(f"  {part:6s} {variant:24s} {method:14s} {row['size_kb']:8.1f} KB  f1={row['f1']:.4f} "
              f"far={row['false_alarm_rate']:.4f}")

    # ---- Part 1: quantization methods ----
    base = _load_fl_model(str(ROOT / entry["model"]))
    pruned = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
    _fit(pruned, ft[0], ft[1], PRUNE_FT_EPOCHS)
    for variant, model in (("federated", base), ("pruned50_clientft", pruned)):
        for method in spec.get("quant_methods", ["fp32", "dynamic_range", "float16", "int8", "int16x8"]):
            p = tdir / f"{variant}_{method}.tflite"
            try:
                convert(model, method, ft[0], p)
                add("quant", variant, method, p, model.count_params())
            except Exception as err:  # e.g. an op without int16x8 kernel
                print(f"  ⚠️ {variant}/{method} failed: {err}")
                rows.append({"model": name, "part": "quant", "variant": variant, "method": method,
                             "error": str(err)[:200]})

    # ---- Part 2: client-local distillation vs. scratch ----
    T, a = float(spec.get("kd_temperature", 2.0)), float(spec.get("kd_alpha", 0.5))
    kd_y = kd_targets(base, kd[0], kd[1], T, a)
    epochs = int(spec.get("kd_epochs", 10))
    for ratio in spec.get("student_ratios", [0.5, 0.25, 0.125]):
        rt = f"r{int(round(float(ratio) * 1000)):03d}"
        for mode, targets in (("kd", kd_y), ("scratch", kd[1])):
            student = make_student(kd[0].shape[1], float(ratio))
            student.fit(kd[0], targets, epochs=epochs, batch_size=256, validation_split=0.1, verbose=0)
            variant = f"student_{rt}_{mode}"
            p = tdir / f"{variant}_fp32.tflite"
            convert(student, "fp32", kd[0], p)
            add("distill", variant, "fp32", p, student.count_params())
            p = tdir / f"{variant}_int8.tflite"
            convert(student, "int8", kd[0], p)
            add("distill", variant, "int8", p, student.count_params())
            q = _qat(student, kd[0], targets)
            p = tdir / f"{variant}_qat.tflite"
            export_tflite_qat(q, str(p))
            add("distill", variant, "int8_qat", p, student.count_params())
    return rows


def main():
    parser = argparse.ArgumentParser(description="Quantization methods + client-local distillation")
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
        rows_to_csv(rows, out_dir / "quant_distill.csv")
    rows_to_markdown(rows, out_dir / "quant_distill.md", "Quantization methods and client-local distillation")
    save_json(rows, out_dir / "quant_distill.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
