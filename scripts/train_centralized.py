#!/usr/bin/env python3
"""
Centralized Training script.

Trains on full data at once without FedAvg/FL.
Used as baseline for FL vs Centralized performance comparison.

Usage:
    python scripts/train_centralized.py --config config/federated_scratch.yaml
    python run.py --config config/federated_scratch.yaml --centralized
"""
import argparse
import os
import sys
from pathlib import Path

# Project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# PSU server scratch space (avoid home disk quota)
if "TMPDIR" not in os.environ and os.path.exists("/scratch"):
    user = os.environ.get("USER", "")
    if user:
        scratch_tmp = f"/scratch/{user}/tmp"
        if not os.path.exists(scratch_tmp):
            try:
                os.makedirs(scratch_tmp, exist_ok=True)
            except OSError:
                pass
        if os.path.exists(scratch_tmp):
            os.environ["TMPDIR"] = scratch_tmp
            os.environ["TEMP"] = scratch_tmp
            os.environ["TMP"] = scratch_tmp

import numpy as np
import yaml

from src.data.loader import load_dataset
from src.models.nets import get_model


def main():
    parser = argparse.ArgumentParser(
        description="Centralized training (no FedAvg) for baseline comparison"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated_local.yaml",
        help="Config file (same as FL run for fair comparison)",
    )
    parser.add_argument(
        "--save-model",
        type=str,
        default="src/models/global_model.h5",
        help="Output model path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Default: num_rounds * local_epochs (roughly matches FL effort)",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fed_cfg = cfg.get("federated", {})

    dataset_name = data_cfg.get("name", "bot_iot")
    dataset_kwargs = {
        k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}
    }
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    print("\n" + "=" * 60)
    print("  📊 CENTRALIZED TRAINING (No FedAvg)")
    print("=" * 60)
    print(f"\nConfig: {args.config}")
    print(f"Dataset: {dataset_name}")

    print("\n📂 Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

    if x_train.ndim == 2:
        input_shape = (x_train.shape[1],)
    else:
        input_shape = x_train.shape[1:]

    print(f"\n📊 Data:")
    print(f"  - Train: {len(x_train):,} | Test: {len(x_test):,}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Classes: {num_classes}")
    print(f"  - Attack (train): {np.sum(y_train == 1):,} | Normal: {np.sum(y_train == 0):,}")

    use_focal_loss = fed_cfg.get("use_focal_loss", False)
    learning_rate = fed_cfg.get("learning_rate", 0.001)
    batch_size = fed_cfg.get("batch_size", 128)
    local_epochs = fed_cfg.get("local_epochs", 15)
    num_rounds = fed_cfg.get("num_rounds", 25)

    epochs = args.epochs
    if epochs is None:
        # Roughly match FL effort: local_epochs per round, num_rounds iterations
        # Centralized uses single model, so total epochs should be comparable
        epochs = min(100, num_rounds * 2)
        print(f"\n  Auto epochs: {epochs} (from num_rounds={num_rounds})")

    model_name = model_cfg.get("name", "mlp")
    print(f"\n🏗️  Building {model_name.upper()} (focal_loss={use_focal_loss})...")
    model = get_model(model_name, input_shape, num_classes, learning_rate, use_focal_loss)

    print(f"\n🚀 Training...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n📈 Final Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    out_path = Path(args.save_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    print(f"\n✅ Model saved: {out_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
