#!/usr/bin/env python3
"""
Local-only baseline: what would each client get by training alone on its own partition?

For each config (near-IID and Dirichlet), the training split is partitioned exactly as in
federated training; one model per client is trained on that client's data only (same MLP,
loss and optimizer as the federated recipe, hard labels) and evaluated on the shared test
split. This is the comparison that tells whether federation buys anything over local
training — e.g. for Dirichlet clients that hold almost no attacks.

Spec (YAML):
  max_samples_per_client: 50000
  epochs: 10
  entries:
    - {name: cic_dirichlet, config: <fl_dirichlet_c4.yaml>}

Usage:
  python scripts/local_only_baseline.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_keras_model
from src.models.nets import get_model


def main():
    parser = argparse.ArgumentParser(description="Local-only (no federation) per-client baseline")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(int(spec.get("seed", 42)))

    rows = []
    for entry in spec["entries"]:
        cfg = load_yaml(ROOT / entry["config"])
        data_cfg, fed_cfg = dict(cfg["data"]), cfg.get("federated", {})
        kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
        if "path" in kwargs:
            kwargs["data_path"] = kwargs.pop("path")
        x_train, y_train, x_test, y_test = load_dataset(data_cfg["name"], **kwargs)
        parts = partition_data(
            x_train, y_train, int(data_cfg.get("num_clients", 4)),
            strategy=data_cfg.get("partition_strategy", "label_balanced"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
        )
        threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
        cap = int(spec.get("max_samples_per_client", 50000))
        for cid, part in enumerate(parts):
            x, y = part["x"], part["y"]
            idx = np.random.default_rng(cid).permutation(len(y))[:cap]
            x, y = x[idx], y[idx]
            row = {"entry": entry["name"], "client": cid, "client_samples": int(len(part["y"])),
                   "train_samples": int(len(y)), "attack_share": round(float(np.mean(y)), 4)}
            if len(np.unique(y)) < 2:
                row["note"] = "single class; model would predict a constant"
            model = get_model(
                cfg.get("model", {}).get("name", "mlp"), (x.shape[1],), 2,
                float(fed_cfg.get("learning_rate", 1e-3)),
                use_focal_loss=bool(fed_cfg.get("use_focal_loss", True)),
                focal_loss_alpha=float(fed_cfg.get("focal_loss_alpha", 0.35)),
            )
            model.fit(x, y, epochs=int(spec.get("epochs", 10)),
                      batch_size=int(fed_cfg.get("batch_size", 128)), verbose=0)
            m = evaluate_keras_model(model, x_test, y_test, threshold=threshold)
            row.update({k: m[k] for k in ["accuracy", "precision", "attack_recall", "f1",
                                          "false_alarm_rate", "fn", "fp"]})
            rows.append(row)
            print(f"[{entry['name']}] client {cid}: n={len(y):,} attack={row['attack_share']:.2%} "
                  f"f1={row['f1']:.4f} recall={row['attack_recall']:.4f} far={row['false_alarm_rate']:.4f}")
            rows_to_csv(rows, out_dir / "local_only.csv")
    rows_to_markdown(rows, out_dir / "local_only.md", "Local-only per-client baseline (no federation)")
    save_json(rows, out_dir / "local_only.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
