#!/usr/bin/env python3
"""
Compression ablation: where does the post-compression gain come from?

compression.py fine-tunes the pruned model (3 epochs) and the QAT model (2 epochs) on
the first 10,000 samples of the *pooled* training set, i.e. data a federated server
would not hold. This script re-compresses an already-trained FL model with the
fine-tuning step varied, so the paper can separate:
  - pruning itself vs. the fine-tuning that follows it, and
  - pooled (server-side) fine-tuning vs. fine-tuning on one client's local data.

Spec file (YAML):
  threshold: 0.3          # optional; default evaluation.prediction_threshold of each config
  ft_samples: 10000
  models:
    - name: near_iid
      model: data/processed/revision/<run>/baseline/models/b_fl.h5
      config: data/processed/revision/<run>/baseline/configs/b_fl.yaml
      ft_client: 0        # client whose local partition is used for "client" variants
  prune_ratios: [0.3, 0.5, 0.7, 0.85]   # optional: compression-strength sweep instead of the
  ft_sources: [client]                  # default variant set (pooled and/or client fine-tuning)

Usage:
  python scripts/compression_ablation.py --spec config/<dir>/compression_ablation.yaml \
      --output-dir data/processed/revision/<id>/compression_ablation
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # same Keras as compression.py / tfmot

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

from compression import has_qat_layers, strip_qat_layers
from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_qat, export_tflite, export_tflite_qat

import tensorflow_model_optimization as tfmot

PRUNE_RATIO = 0.5  # compression.py default preset
PRUNE_FT_EPOCHS = 3
QAT_FT_EPOCHS = 2


def _load_fl_model(path: str) -> keras.Model:
    with tfmot.quantization.keras.quantize_scope():
        model = keras.models.load_model(path, compile=False)
    if has_qat_layers(model):
        model = strip_qat_layers(model)
    if any("BatchNormalization" in type(l).__name__ for l in model.layers):
        model = _strip_bn_dropout_for_qat(model)  # exact BN folding; pruner would reset BN stats
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _clone(model: keras.Model) -> keras.Model:
    m = keras.models.clone_model(model)
    m.set_weights(model.get_weights())
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def _fit(model: keras.Model, x, y, epochs: int) -> None:
    # Same settings as compression.py's fine-tuning
    model.fit(x, y, epochs=epochs, batch_size=128, validation_split=0.1, verbose=0)


def _qat(pruned: keras.Model, x, y) -> keras.Model:
    q = tfmot.quantization.keras.quantize_model(_strip_bn_dropout_for_qat(pruned))
    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    _fit(q, x, y, QAT_FT_EPOCHS)
    return q


def run_model(entry: dict, spec: dict, out_dir: Path) -> List[Dict[str, Any]]:
    name = entry["name"]
    cfg = load_yaml(ROOT / entry["config"])
    data_cfg = dict(cfg.get("data", {}))
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_train, y_train, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)

    n_ft = int(spec.get("ft_samples", 10000))
    pooled = (x_train[:n_ft], y_train[:n_ft])  # exactly what compression.py uses
    parts = partition_data(
        x_train, y_train, int(data_cfg.get("num_clients", 4)),
        strategy=data_cfg.get("partition_strategy", "label_balanced"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
    )
    k = int(entry.get("ft_client", 0))
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(parts[k]["y"]))[:n_ft]
    client = (parts[k]["x"][idx], parts[k]["y"][idx])
    print(f"[{name}] pooled FT set: {len(pooled[1])} (attack {pooled[1].mean():.1%}); "
          f"client {k} FT set: {len(client[1])} (attack {client[1].mean():.1%})")

    threshold = float(spec.get("threshold", cfg.get("evaluation", {}).get("prediction_threshold", 0.3)))
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    base = _load_fl_model(str(ROOT / entry["model"]))

    variants = []

    def add(vid: str, desc: str, path: Path, ft_data: str):
        m = evaluate_tflite_model(path, x_test, y_test, threshold=threshold, latency_runs=50)
        row = {"model": name, "variant": vid, "description": desc, "ft_data": ft_data,
               "size_kb": round(path.stat().st_size / 1024, 2)}
        row.update({k: m[k] for k in ["accuracy", "precision", "attack_recall", "f1",
                                      "false_alarm_rate", "fp", "fn", "threshold"]})
        variants.append(row)
        print(f"  {vid:<22} acc={row['accuracy']:.4f} f1={row['f1']:.4f} "
              f"recall={row['attack_recall']:.4f} far={row['false_alarm_rate']:.4f}")

    p = tdir / "fp32.tflite"
    export_tflite(base, str(p), quantize=False)
    add("fp32", "FL model, no compression", p, "-")

    p = tdir / "ptq_only.tflite"
    export_tflite(base, str(p), quantize=True, representative_data=pooled[0])
    add("ptq_only", "INT8 PTQ only (no pruning, no fine-tune)", p, "-")

    ratios = spec.get("prune_ratios")
    if ratios:  # compression-strength sweep
        sources = {"pooled": pooled, "client": client}
        for r in ratios:
            rt = f"r{int(round(float(r) * 100))}"
            pr = apply_structured_pruning(_clone(base), pruning_ratio=float(r), skip_last_layer=True, verbose=False)
            p = tdir / f"prune_noft_ptq_{rt}.tflite"
            export_tflite(pr, str(p), quantize=True, representative_data=pooled[0])
            add(f"prune_noft_ptq_{rt}", f"prune {float(r):.0%} -> PTQ (no fine-tune)", p, "-")
            for tag in spec.get("ft_sources", ["client"]):
                xf, yf = sources[tag]
                pr = apply_structured_pruning(_clone(base), pruning_ratio=float(r), skip_last_layer=True, verbose=False)
                _fit(pr, xf, yf, PRUNE_FT_EPOCHS)
                p = tdir / f"prune_ft_{tag}_ptq_{rt}.tflite"
                export_tflite(pr, str(p), quantize=True, representative_data=xf)
                add(f"prune_ft_{tag}_ptq_{rt}", f"prune {float(r):.0%} -> fine-tune -> PTQ", p, tag)
                q = _qat(pr, xf, yf)
                p = tdir / f"prune_ft_{tag}_qat_{rt}.tflite"
                export_tflite_qat(q, str(p))
                add(f"prune_ft_{tag}_qat_{rt}", f"prune {float(r):.0%} -> fine-tune -> QAT", p, tag)
        return variants

    pruned = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
    p = tdir / "prune_noft_ptq.tflite"
    export_tflite(pruned, str(p), quantize=True, representative_data=pooled[0])
    add("prune_noft_ptq", "prune 50% -> PTQ (no fine-tune)", p, "-")

    for tag, (xf, yf) in (("pooled", pooled), ("client", client)):
        ft_only = _clone(base)
        _fit(ft_only, xf, yf, PRUNE_FT_EPOCHS)
        p = tdir / f"ftonly_{tag}_ptq.tflite"
        export_tflite(ft_only, str(p), quantize=True, representative_data=xf)
        add(f"ftonly_{tag}_ptq", f"fine-tune {PRUNE_FT_EPOCHS} ep (no pruning) -> PTQ", p, tag)

        pr = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
        _fit(pr, xf, yf, PRUNE_FT_EPOCHS)
        p = tdir / f"prune_ft_{tag}_ptq.tflite"
        export_tflite(pr, str(p), quantize=True, representative_data=xf)
        add(f"prune_ft_{tag}_ptq", "prune 50% -> fine-tune -> PTQ", p, tag)

        q = _qat(pr, xf, yf)
        p = tdir / f"prune_ft_{tag}_qat.tflite"
        export_tflite_qat(q, str(p))
        add(f"prune_ft_{tag}_qat", f"prune -> fine-tune -> QAT {QAT_FT_EPOCHS} ep (deploy recipe)", p, tag)
    return variants


def main():
    parser = argparse.ArgumentParser(description="Compression fine-tuning ablation")
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
        rows_to_csv(rows, out_dir / "compression_ablation.csv")  # keep partial results
    rows_to_markdown(rows, out_dir / "compression_ablation.md", "Compression fine-tuning ablation")
    save_json(rows, out_dir / "compression_ablation.json")
    print(f"✅ Compression ablation saved to {out_dir}")


if __name__ == "__main__":
    main()
