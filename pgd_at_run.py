"""
PGD Adversarial Training (AT) — Post-compression fine-tune on the 4 selected models.

For each model:
  1. Evaluate clean + PGD-adversarial metrics BEFORE AT
  2. Fine-tune with PGD-AT (mixed clean/adversarial batches)
  3. Evaluate clean + PGD-adversarial metrics AFTER AT
  4. Save hardened model as final_model_at.h5 alongside original

Results saved to pgd_at_results.csv.
"""
from __future__ import annotations

import copy
import glob
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.adversarial.pgd_adversarial_training import (
    pgd_attack,
    load_model_for_at,
    compute_class_weights,
)
from src.data.loader import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH  = "config/federated_local.yaml"
EPSILON      = 0.01       # same as fgsm_run.py
PGD_ALPHA    = 0.001      # step size = epsilon / 10
PGD_STEPS    = 10
ADV_RATIO    = 0.5        # fraction of each batch that is adversarial
EPOCHS       = 10         # AT fine-tuning epochs
BATCH_SIZE   = 128
LR           = 0.0001     # low LR to avoid catastrophic forgetting
EVAL_SAMPLES = 5000       # test subset size
THRESHOLD    = 0.5

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
data_cfg = cfg.get("data", {})
fed_cfg  = cfg.get("federated", {})


# ── helpers ───────────────────────────────────────────────────────────────────

def _select_models(sweep_csv: str) -> Dict[str, dict]:
    df = pd.read_csv(sweep_csv)
    sweep_dir = str(Path(sweep_csv).parent)

    most_compressed = df.nsmallest(1, "tflite_size_kb").iloc[0]
    most_accurate   = df.nlargest(1, "final_f1").iloc[0]

    ptq_df = df[df["ptq"] == True].copy()
    ptq_df["rank_f1"]   = ptq_df["final_f1"].rank()
    ptq_df["rank_size"] = ptq_df["tflite_size_kb"].rank()
    ptq_df["bal"]       = (ptq_df["rank_f1"] + ptq_df["rank_size"]) / 2
    most_balanced_ptq = ptq_df.loc[(ptq_df["bal"] - ptq_df["bal"].median()).abs().idxmin()]

    no_ptq_df = df[df["ptq"] == False].copy()
    no_ptq_df["rank_f1"]   = no_ptq_df["final_f1"].rank()
    no_ptq_df["rank_size"] = no_ptq_df["tflite_size_kb"].rank()
    no_ptq_df["bal"]       = (no_ptq_df["rank_f1"] + no_ptq_df["rank_size"]) / 2
    most_balanced_no_ptq = no_ptq_df.loc[(no_ptq_df["bal"] - no_ptq_df["bal"].median()).abs().idxmin()]

    result = {}
    for label, row in [
        ("most_compressed",      most_compressed),
        ("most_accurate",        most_accurate),
        ("most_balanced_ptq",    most_balanced_ptq),
        ("most_balanced_no_ptq", most_balanced_no_ptq),
    ]:
        tag        = row["tag"]
        model_path = os.path.join(sweep_dir, tag, "final_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        result[label] = {"tag": tag, "model_path": model_path}
    return result


def _load_data():
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_tr, y_tr, x_te, y_te = load_dataset(data_cfg["name"], **kwargs)
    return (x_tr.astype("float32"), y_tr.astype("float32"),
            x_te.astype("float32"), y_te.astype("float32"))


def _subsample(x, y, n, seed=42):
    if n and len(x) > n:
        idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
        return x[idx], y[idx]
    return x, y


def _f1_acc(model, x, y) -> Tuple[float, float]:
    preds = model.predict(x, verbose=0).flatten()
    pl = (preds >= THRESHOLD).astype(int)
    yi = y.flatten().astype(int)
    tp = int(((pl == 1) & (yi == 1)).sum())
    fp = int(((pl == 1) & (yi == 0)).sum())
    fn = int(((pl == 0) & (yi == 1)).sum())
    tn = int(((pl == 0) & (yi == 0)).sum())
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(f1), float((tp + tn) / len(yi))


def _pgd_eval(model, x, y) -> Tuple[float, float]:
    """Evaluate model on PGD adversarial examples."""
    x_adv = pgd_attack(model, x, y,
                       epsilon=EPSILON, alpha=PGD_ALPHA, steps=PGD_STEPS,
                       loss_fn=model.loss)
    return _f1_acc(model, x_adv, y)


def _at_finetune(model, x_train, y_train) -> float:
    """
    PGD adversarial training fine-tune (single model, no FL aggregation).

    Each batch is split into clean and adversarial portions by ADV_RATIO.
    PGD adversarial examples are generated fresh every batch.
    Returns total wall-clock time in seconds.
    """
    import tensorflow as tf
    from tensorflow import keras

    # set fine-tune LR
    model.optimizer.learning_rate.assign(LR)
    class_weights = compute_class_weights(y_train)
    loss_fn = model.loss

    n = len(x_train)
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        perm    = np.random.default_rng(epoch).permutation(n)
        x_shuf  = x_train[perm]
        y_shuf  = y_train[perm]
        n_batch = (n + BATCH_SIZE - 1) // BATCH_SIZE
        epoch_loss = 0.0

        for b in range(n_batch):
            xb = x_shuf[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            yb = y_shuf[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]

            n_adv   = max(1, int(len(xb) * ADV_RATIO))
            n_clean = len(xb) - n_adv

            x_adv = pgd_attack(model, xb[n_clean:], yb[n_clean:],
                               epsilon=EPSILON, alpha=PGD_ALPHA, steps=PGD_STEPS,
                               loss_fn=loss_fn)

            if n_clean > 0:
                x_combined = np.concatenate([xb[:n_clean], x_adv], axis=0)
                y_combined = np.concatenate([yb[:n_clean], yb[n_clean:]], axis=0)
            else:
                x_combined, y_combined = x_adv, yb

            bl = model.train_on_batch(x_combined, y_combined,
                                      class_weight=class_weights)
            epoch_loss += float(bl[0] if isinstance(bl, (list, tuple)) else bl)

        print(f"    Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/n_batch:.4f}")

    return time.perf_counter() - t0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    sweep_files = sorted(glob.glob("data/processed/sweep/*/sweep_results.csv"))
    if not sweep_files:
        raise FileNotFoundError("No sweep_results.csv found.")
    sweep_csv = sweep_files[-1]
    print(f"Sweep CSV: {sweep_csv}\n")

    print("Selecting models...")
    models_info = _select_models(sweep_csv)
    for label, info in models_info.items():
        print(f"  [{label}]  {info['tag']}")
    print()

    print(f"Loading data (test up to {EVAL_SAMPLES} samples)...")
    x_train, y_train, x_test_full, y_test_full = _load_data()
    x_test, y_test = _subsample(x_test_full, y_test_full, EVAL_SAMPLES)
    print(f"  Train: {x_train.shape}  Test: {x_test.shape}\n")

    rows = []

    for model_label, info in models_info.items():
        print(f"{'='*60}")
        print(f"Model: {model_label}  ({info['tag']})")
        print(f"{'='*60}")

        model = load_model_for_at(info["model_path"], fed_cfg)
        model.optimizer.learning_rate.assign(LR)

        # ── before AT ────────────────────────────────────────────────────────
        print("  Evaluating BEFORE AT...")
        clean_f1_before,  clean_acc_before  = _f1_acc(model, x_test, y_test)
        adv_f1_before,    adv_acc_before    = _pgd_eval(model, x_test, y_test)
        print(f"    Clean  F1={clean_f1_before:.4f}  Acc={clean_acc_before:.4f}")
        print(f"    Adv    F1={adv_f1_before:.4f}  Acc={adv_acc_before:.4f}")

        # ── AT fine-tune ─────────────────────────────────────────────────────
        print(f"  AT fine-tuning ({EPOCHS} epochs, adv_ratio={ADV_RATIO})...")
        at_time = _at_finetune(model, x_train, y_train)
        print(f"  AT done in {at_time:.1f}s")

        # ── after AT ─────────────────────────────────────────────────────────
        print("  Evaluating AFTER AT...")
        clean_f1_after,   clean_acc_after   = _f1_acc(model, x_test, y_test)
        adv_f1_after,     adv_acc_after     = _pgd_eval(model, x_test, y_test)
        print(f"    Clean  F1={clean_f1_after:.4f}  Acc={clean_acc_after:.4f}")
        print(f"    Adv    F1={adv_f1_after:.4f}  Acc={adv_acc_after:.4f}")

        # ── save hardened model ───────────────────────────────────────────────
        at_path = str(Path(info["model_path"]).parent / "final_model_at.h5")
        model.save(at_path)
        print(f"  Saved: {at_path}\n")

        def pct(after, before):
            return (after - before) / before * 100 if before != 0 else float("nan")

        rows.append({
            "model":              model_label,
            "tag":                info["tag"],
            "clean_f1_before":    clean_f1_before,
            "clean_acc_before":   clean_acc_before,
            "adv_f1_before":      adv_f1_before,
            "adv_acc_before":     adv_acc_before,
            "clean_f1_after":     clean_f1_after,
            "clean_acc_after":    clean_acc_after,
            "adv_f1_after":       adv_f1_after,
            "adv_acc_after":      adv_acc_after,
            "delta_clean_f1_pct": pct(clean_f1_after,  clean_f1_before),
            "delta_clean_acc_pct":pct(clean_acc_after, clean_acc_before),
            "delta_adv_f1_pct":   pct(adv_f1_after,    adv_f1_before),
            "delta_adv_acc_pct":  pct(adv_acc_after,   adv_acc_before),
            "at_time_s":          at_time,
        })

    out = "pgd_at_results.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
