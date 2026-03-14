#!/usr/bin/env python3
"""
Adversarial fine-tuning after FL: load global model, mix FGSM adversarial examples
with clean data, train for a few epochs, save back. Used between Step 1 (Training)
and Step 2 (Compression) when adversarial_training.enabled is true.

Usage:
  python scripts/run_at_after_training.py --config config/federated_local_sky.yaml
  python scripts/run_at_after_training.py --config config/federated.yaml --model models/global_model.h5 --epochs 3
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import tensorflow as tf
from src.data.loader import load_dataset
from src.adversarial.fgsm_hook import (
    generate_adversarial_dataset,
    generate_adversarial_dataset_pgd,
)


def load_model(path: str) -> tf.keras.Model:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        import tensorflow_model_optimization as tfmot
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(path, compile=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Adversarial fine-tuning after FL")
    parser.add_argument("--config", type=str, default="config/federated_local_sky.yaml")
    parser.add_argument("--model", type=str, default="models/global_model.h5")
    parser.add_argument("--output", type=str, default=None, help="Default: overwrite --model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--adv-ratio", type=float, default=None,
                        help="Fraction of training batch that is adversarial (0.0-1.0)")
    parser.add_argument("--max-train-samples", type=int, default=50000,
                        help="Cap train size for AT (faster)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    at_cfg = cfg.get("adversarial_training", {})
    enabled = at_cfg.get("enabled", False)
    if not enabled:
        print("adversarial_training.enabled is false; skipping AT.")
        return 0

    epochs = args.epochs if args.epochs is not None else at_cfg.get("epochs", 3)
    epsilon = args.epsilon if args.epsilon is not None else at_cfg.get("epsilon", 0.05)
    adv_ratio = args.adv_ratio if args.adv_ratio is not None else at_cfg.get("adv_ratio", 0.5)
    attack = (at_cfg.get("attack") or "fgsm").strip().lower()
    pgd_steps = at_cfg.get("pgd_steps", 10)
    pgd_alpha = at_cfg.get("pgd_alpha")  # None → 2.5*eps/steps
    out_path = args.output or args.model

    print(f"\n{'='*60}")
    print("  Adversarial fine-tuning (after FL, before compression)")
    print(f"  Model: {args.model}  →  Output: {out_path}")
    print(f"  Attack: {attack.upper()}, Epochs: {epochs}, epsilon: {epsilon}, adv_ratio: {adv_ratio}")
    if attack == "pgd":
        print(f"  PGD steps: {pgd_steps}, alpha: {pgd_alpha or ('2.5*eps/steps')}")
    print(f"{'='*60}\n")

    model = load_model(args.model)
    last = model.layers[-1]
    num_classes = getattr(last, "units", 2)
    loss = "binary_crossentropy" if num_classes <= 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    data_cfg = cfg.get("data", {})
    name = data_cfg.get("name", "cicids2017")
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    x_train, y_train, x_test, y_test = load_dataset(name, **kwargs)
    if len(x_train) > args.max_train_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_train), size=args.max_train_samples, replace=False)
        x_train, y_train = x_train[idx], y_train[idx]
    print(f"  Train samples: {len(x_train):,}\n")

    # Binary: ensure (batch, 1) for loss
    if num_classes <= 2 and y_train.ndim == 1:
        y_train = np.reshape(y_train, (-1, 1))

    # Generate adversarial subset (FGSM or PGD)
    n_adv = max(1, int(len(x_train) * adv_ratio))
    if attack == "pgd":
        x_adv, y_adv = generate_adversarial_dataset_pgd(
            model, x_train[:n_adv], y_train[:n_adv],
            eps=epsilon, steps=pgd_steps, alpha=pgd_alpha,
        )
    else:
        x_adv, y_adv = generate_adversarial_dataset(
            model, x_train[:n_adv], y_train[:n_adv], eps=epsilon,
        )
    if y_adv.ndim == 1:
        y_adv = np.reshape(y_adv, (-1, 1))
    x_comb = np.concatenate([x_train, x_adv], axis=0)
    y_comb = np.concatenate([y_train, y_adv], axis=0)
    shuffle_idx = np.random.permutation(len(x_comb))
    x_comb = x_comb[shuffle_idx]
    y_comb = y_comb[shuffle_idx]
    print(f"  Combined: {len(x_comb):,} (clean {len(x_train):,} + adv {len(x_adv):,})\n")

    model.fit(
        x_comb, y_comb,
        epochs=epochs,
        batch_size=cfg.get("federated", {}).get("batch_size", 128),
        validation_split=0.1,
        verbose=1,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    print(f"\n  Saved: {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
