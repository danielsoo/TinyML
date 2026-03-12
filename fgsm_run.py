"""
FGSM Robustness Evaluation across 4 selected models and 4 attack methods.

Models selected from the latest sweep_results.csv:
  1. Most Compressed   – smallest tflite_size_kb
  2. Most Accurate     – highest final_f1
  3. Most Balanced w/  PTQ – ptq=True,  mid-rank on (f1, tflite_size_kb)
  4. Most Balanced w/o PTQ – ptq=False, mid-rank on (f1, tflite_size_kb)

Attack methods:
  A. FGI  (Prior-Guided FGSM)  – GradientPrior seeded on test batch then attacked
  B. FGSM (Standard)           – vanilla FGSM, timing baseline = 100%
  C. GA   (Gradient-Aligned)   – local == global (same model), alignment mask applied
  D. PGD  (PGD-10)             – 10-step iterative attack

Results saved to fgsm_results.csv with:
  - clean_f1, clean_acc
  - adv_f1,   adv_acc
  - delta_f1_pct, delta_acc_pct   (change vs clean, %)
  - time_s                        (wall-clock seconds)
  - time_pct                      (relative to standard FGSM per model = 100%)
"""
from __future__ import annotations

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

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.adversarial.fgsm_hook import compute_gradients, generate_fgsm_attack
from src.adversarial.fgsm_standard import standard_fgsm
from src.adversarial.fgsm_prior_guided import GradientPrior, prior_guided_fgsm
from src.adversarial.fgsm_gradient_aligned import gradient_aligned_fgsm_single_model
from src.adversarial.pgd_adversarial_training import pgd_attack, load_model_for_at
from src.data.loader import load_dataset

# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = "config/federated_local.yaml"
# Data is StandardScaler-normalised (std≈0.85, range≈[-1, 50+]).
# eps=0.1 equals ~12% of a std-dev and completely collapses all models.
# eps=0.01 (~1% std-dev) produces meaningful, non-trivial degradation.
EPSILON             = 0.01
PGD_ALPHA           = 0.001   # alpha = epsilon/10 (standard PGD ratio)
PGD_STEPS           = 10
EVAL_SAMPLES        = 5000    # test subset size
FGI_PRIOR_BATCHES   = 20      # number of training batches to build the EMA prior
FGI_PRIOR_BATCH_SIZE = 512    # samples per prior update batch
GA_GLOBAL_SAMPLES   = 10000   # training samples to approximate global gradient
THRESHOLD           = 0.5

# ── load FL config ────────────────────────────────────────────────────────────
with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
data_cfg = cfg.get("data", {})
fed_cfg  = cfg.get("federated", {})


# ── helpers ───────────────────────────────────────────────────────────────────

def _select_models(sweep_csv: str) -> Dict[str, dict]:
    """Return the 4 selected model rows as {label: row_dict}."""
    df = pd.read_csv(sweep_csv)
    sweep_dir = str(Path(sweep_csv).parent)

    # 1. Most compressed
    most_compressed = df.nsmallest(1, "tflite_size_kb").iloc[0]

    # 2. Most accurate (final_f1)
    most_accurate = df.nlargest(1, "final_f1").iloc[0]

    # 3. Most balanced WITH ptq
    ptq_df = df[df["ptq"] == True].copy()
    ptq_df["rank_f1"]   = ptq_df["final_f1"].rank()
    ptq_df["rank_size"] = ptq_df["tflite_size_kb"].rank()
    ptq_df["bal"]       = (ptq_df["rank_f1"] + ptq_df["rank_size"]) / 2
    med = ptq_df["bal"].median()
    most_balanced_ptq = ptq_df.loc[(ptq_df["bal"] - med).abs().idxmin()]

    # 4. Most balanced WITHOUT ptq
    no_ptq_df = df[df["ptq"] == False].copy()
    no_ptq_df["rank_f1"]   = no_ptq_df["final_f1"].rank()
    no_ptq_df["rank_size"] = no_ptq_df["tflite_size_kb"].rank()
    no_ptq_df["bal"]       = (no_ptq_df["rank_f1"] + no_ptq_df["rank_size"]) / 2
    med2 = no_ptq_df["bal"].median()
    most_balanced_no_ptq = no_ptq_df.loc[(no_ptq_df["bal"] - med2).abs().idxmin()]

    selections = {
        "most_compressed":     most_compressed,
        "most_accurate":       most_accurate,
        "most_balanced_ptq":   most_balanced_ptq,
        "most_balanced_no_ptq": most_balanced_no_ptq,
    }

    result = {}
    for label, row in selections.items():
        tag        = row["tag"]
        model_path = os.path.join(sweep_dir, tag, "final_model.h5")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        result[label] = {
            "tag":        tag,
            "model_path": model_path,
            "clean_f1":   float(row["final_f1"]),
            "clean_acc":  float(row["final_acc"]),
            "tflite_kb":  float(row["tflite_size_kb"]),
        }
    return result


def _load_model(model_path: str):
    """Load a .h5 model, handling focal loss recompile."""
    return load_model_for_at(model_path, fed_cfg)


def _load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train + test splits from CIC-IDS2017 (same preprocessing as FL)."""
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_train, y_train, x_test, y_test = load_dataset(data_cfg["name"], **kwargs)
    return (x_train.astype(np.float32), y_train.astype(np.float32),
            x_test.astype(np.float32),  y_test.astype(np.float32))


def _subsample(x, y, n, seed=42):
    if n and len(x) > n:
        idx = np.random.default_rng(seed).choice(len(x), n, replace=False)
        return x[idx], y[idx]
    return x, y


def _f1_acc(model, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Compute binary F1 and accuracy from model predictions."""
    preds = model.predict(x, verbose=0).flatten()
    pred_labels = (preds >= THRESHOLD).astype(int)
    y_int = y.flatten().astype(int)

    tp = int(((pred_labels == 1) & (y_int == 1)).sum())
    fp = int(((pred_labels == 1) & (y_int == 0)).sum())
    fn = int(((pred_labels == 0) & (y_int == 1)).sum())
    tn = int(((pred_labels == 0) & (y_int == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc       = (tp + tn) / len(y_int) if len(y_int) > 0 else 0.0
    return float(f1), float(acc)


# ── attack runners ────────────────────────────────────────────────────────────
# Data is StandardScaler-normalised — range is NOT [0,1].
# clip_min/clip_max must match the actual feature range so we don't
# destroy the input distribution.  We set them per-batch from the data.

def _data_bounds(x: np.ndarray):
    return float(x.min()), float(x.max())


def run_fgsm(model, x, y) -> Tuple[np.ndarray, float]:
    clip_min, clip_max = _data_bounds(x)
    t0 = time.perf_counter()
    x_adv, _ = standard_fgsm(model, x, y, epsilon=EPSILON,
                              clip_min=clip_min, clip_max=clip_max)
    return x_adv, time.perf_counter() - t0


def run_fgi(model, x, y, x_train, y_train) -> Tuple[np.ndarray, float]:
    """Prior-Guided FGSM.

    Simulates FL round history by calling prior.update() on multiple
    random batches drawn from the training set (FGI_PRIOR_BATCHES batches
    of FGI_PRIOR_BATCH_SIZE samples each).  This builds a genuine EMA of
    gradient directions across the training distribution before the attack.
    The prior is then blended 50/50 with the current test-batch gradient.
    """
    clip_min, clip_max = _data_bounds(x)
    t0 = time.perf_counter()
    prior = GradientPrior(decay=0.9)
    rng = np.random.default_rng(0)
    for _ in range(FGI_PRIOR_BATCHES):
        idx = rng.choice(len(x_train), FGI_PRIOR_BATCH_SIZE, replace=False)
        prior.update(model, x_train[idx], y_train[idx])
    x_adv, _ = prior_guided_fgsm(
        model, x, y,
        epsilon=EPSILON,
        prior=prior,
        prior_weight=0.5,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return x_adv, time.perf_counter() - t0


def run_gradient_aligned(model, x, y, x_train, y_train) -> Tuple[np.ndarray, float]:
    """Gradient-Aligned FGSM.

    Computes the 'global' gradient as the mean gradient over a large
    random sample of the full training set (GA_GLOBAL_SAMPLES samples),
    approximating the true global model gradient direction.  Only features
    where the local gradient (on the test batch) and global gradient agree
    in sign are perturbed — sparser, more targeted attack.
    """
    clip_min, clip_max = _data_bounds(x)
    t0 = time.perf_counter()
    x_global, y_global = _subsample(x_train, y_train, GA_GLOBAL_SAMPLES, seed=1)
    # Mean over samples → (78,) vector representing the global gradient direction
    global_grads = compute_gradients(model, x_global, y_global).mean(axis=0)
    x_adv, _, _ = gradient_aligned_fgsm_single_model(
        model, x, y,
        global_gradients=global_grads,
        epsilon=EPSILON,
        alignment_threshold=0.0,
        clip_min=clip_min,
        clip_max=clip_max,
    )
    return x_adv, time.perf_counter() - t0


def run_pgd(model, x, y) -> Tuple[np.ndarray, float]:
    # PGD already clips to epsilon-ball around original — no global clip needed.
    t0 = time.perf_counter()
    x_adv = pgd_attack(
        model, x, y,
        epsilon=EPSILON,
        alpha=PGD_ALPHA,
        steps=PGD_STEPS,
        loss_fn=model.loss,
    )
    return x_adv, time.perf_counter() - t0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Find latest sweep CSV
    sweep_files = sorted(glob.glob("data/processed/sweep/*/sweep_results.csv"))
    if not sweep_files:
        raise FileNotFoundError("No sweep_results.csv found.")
    sweep_csv = sweep_files[-1]
    print(f"Sweep CSV: {sweep_csv}\n")

    # Select models
    print("Selecting models...")
    models_info = _select_models(sweep_csv)
    for label, info in models_info.items():
        print(f"  [{label}]  tag={info['tag']}")
        print(f"    clean_f1={info['clean_f1']:.4f}  clean_acc={info['clean_acc']:.4f}  tflite={info['tflite_kb']:.2f} KB")
    print()

    # Load train + test data once
    print(f"Loading data (test up to {EVAL_SAMPLES} samples)...")
    x_train, y_train, x_test_full, y_test_full = _load_data()
    x_test, y_test = _subsample(x_test_full, y_test_full, EVAL_SAMPLES)
    print(f"  Train shape: {x_train.shape}  Test shape: {x_test.shape}\n")

    # attack dispatch — fgsm/pgd only need test data; fgi/ga also need train data
    def dispatch(name, model, x, y):
        if name == "fgsm":
            return run_fgsm(model, x, y)
        elif name == "fgi":
            return run_fgi(model, x, y, x_train, y_train)
        elif name == "ga":
            return run_gradient_aligned(model, x, y, x_train, y_train)
        else:
            return run_pgd(model, x, y)

    rows = []

    for model_label, info in models_info.items():
        print(f"{'='*60}")
        print(f"Model: {model_label}  ({info['tag']})")
        print(f"{'='*60}")

        model = _load_model(info["model_path"])

        clean_f1, clean_acc = _f1_acc(model, x_test, y_test)
        print(f"  Clean F1={clean_f1:.4f}  Acc={clean_acc:.4f}")

        attack_times: Dict[str, float] = {}
        attack_results: Dict[str, Tuple[float, float]] = {}

        for attack_name in ("fgsm", "fgi", "ga", "pgd"):
            print(f"  Running {attack_name.upper()}...", end=" ", flush=True)
            x_adv, elapsed = dispatch(attack_name, model, x_test, y_test)
            adv_f1, adv_acc = _f1_acc(model, x_adv, y_test)
            attack_times[attack_name]   = elapsed
            attack_results[attack_name] = (adv_f1, adv_acc)
            print(f"done  ({elapsed:.1f}s)  adv_f1={adv_f1:.4f}  adv_acc={adv_acc:.4f}")

        fgsm_time = attack_times["fgsm"]

        for attack_name in ("fgsm", "fgi", "ga", "pgd"):
            adv_f1, adv_acc = attack_results[attack_name]
            elapsed = attack_times[attack_name]
            rows.append({
                "model":         model_label,
                "tag":           info["tag"],
                "attack":        attack_name,
                "clean_f1":      clean_f1,
                "clean_acc":     clean_acc,
                "adv_f1":        adv_f1,
                "adv_acc":       adv_acc,
                "delta_f1_pct":  (adv_f1  - clean_f1)  / clean_f1  * 100 if clean_f1  > 0 else float("nan"),
                "delta_acc_pct": (adv_acc - clean_acc) / clean_acc * 100 if clean_acc > 0 else float("nan"),
                "time_s":        elapsed,
                "time_pct":      elapsed / fgsm_time * 100 if fgsm_time > 0 else float("nan"),
            })

        print()

    # Save results
    results_df = pd.DataFrame(rows)
    out_path = "fgsm_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
