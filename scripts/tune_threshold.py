#!/usr/bin/env python3
"""
Recommend threshold for a given test-set ratio (normal% : attack%).
Sweeps thresholds and reports metrics; recommends threshold by F1 or attack_recall.

Usage:
  python scripts/tune_threshold.py --config config/federated_scratch.yaml --model models/tflite/saved_model_original.tflite --ratio 90
  python scripts/tune_threshold.py --ratio 90 --metric attack_recall   # maximize attack recall
  python scripts/tune_threshold.py --ratio 90 --metric f1 --out threshold_90.csv
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser(description="Recommend threshold for a given normal:attack ratio")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="models/tflite/saved_model_original.tflite", help="Model path (.h5 or .tflite)")
    parser.add_argument("--ratio", type=int, default=90, help="Normal%% (e.g. 90 = 90:10 normal:attack)")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "attack_recall", "normal_recall", "balanced"],
                        help="Metric to maximize for recommendation (balanced = 0.5*F1 + 0.5*attack_recall)")
    parser.add_argument("--thresholds", type=str, default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                        help="Comma-separated thresholds to try")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reuse loader from evaluate_ratio_sweep (same dir)
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    import evaluate_ratio_sweep as ers
    load_config_and_data = ers.load_config_and_data
    subsample_by_ratio = ers.subsample_by_ratio
    _predict_proba = ers._predict_proba

    class Args:
        pass
    a = Args()
    a.config = args.config
    a.model = args.model
    x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, model, _ = load_config_and_data(a)
    rng = np.random.default_rng(args.seed)
    eval_idx = subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, args.ratio, rng)
    x_eval = x_test[eval_idx]
    y_eval = y_test[eval_idx]
    n_normal = int(np.sum(y_eval == 0))
    n_attack = int(np.sum(y_eval == 1))
    print(f"\nTest set: Normal {args.ratio}% : Attack {100-args.ratio}%  (n={len(y_eval):,}, N={n_normal:,}, A={n_attack:,})")
    print("Getting predictions...")
    y_prob = _predict_proba(model, x_eval)

    th_list = [float(t.strip()) for t in args.thresholds.split(",")]
    rows = []
    for th in th_list:
        y_pred = (y_prob >= th).astype(np.int32)
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, average="binary", zero_division=0)
        rec = recall_score(y_eval, y_pred, average="binary", zero_division=0)
        f1 = f1_score(y_eval, y_pred, average="binary", zero_division=0)
        prec_per = precision_score(y_eval, y_pred, average=None, zero_division=0)
        rec_per = recall_score(y_eval, y_pred, average=None, zero_division=0)
        normal_recall = float(rec_per[0])
        normal_precision = float(prec_per[0])
        attack_recall = float(rec_per[1]) if len(rec_per) > 1 else 0.0
        attack_precision = float(prec_per[1]) if len(prec_per) > 1 else 0.0
        rows.append({
            "threshold": th,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "normal_recall": normal_recall,
            "normal_precision": normal_precision,
            "attack_recall": attack_recall,
            "attack_precision": attack_precision,
        })

    # Recommend by metric
    if args.metric == "f1":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["f1_score"])
    elif args.metric == "attack_recall":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["attack_recall"])
    elif args.metric == "normal_recall":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["normal_recall"])
    else:  # balanced
        best_idx = max(range(len(rows)), key=lambda i: 0.5 * rows[i]["f1_score"] + 0.5 * rows[i]["attack_recall"])
    best = rows[best_idx]
    best_th = best["threshold"]

    print(f"\n--- Threshold sweep (Normal {args.ratio}% : Attack {100-args.ratio}%) ---")
    print(f"{'Threshold':<10} {'Accuracy':<8} {'F1':<8} {'NormRec':<8} {'NormPrec':<8} {'AttackRec':<8} {'AttackPrec':<8}")
    print("-" * 74)
    for r in rows:
        mark = " <--" if r["threshold"] == best_th else ""
        print(f"{r['threshold']:<10.2f} {r['accuracy']:<8.4f} {r['f1_score']:<8.4f} {r['normal_recall']:<8.4f} {r['normal_precision']:<8.4f} {r['attack_recall']:<8.4f} {r['attack_precision']:<8.4f}{mark}")
    print("-" * 74)
    print(f"\nRecommended threshold (by {args.metric}): {best_th}")
    print(f"  At threshold {best_th}: Accuracy={best['accuracy']:.4f}, F1={best['f1_score']:.4f}, Normal Recall={best['normal_recall']:.4f}, Normal Precision={best['normal_precision']:.4f}, Attack Recall={best['attack_recall']:.4f}, Attack Precision={best['attack_precision']:.4f}")

    if args.out:
        import csv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {args.out}")
    print()


if __name__ == "__main__":
    main()
