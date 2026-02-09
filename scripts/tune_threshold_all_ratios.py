#!/usr/bin/env python3
"""
Run threshold tuning for all ratios (100, 90, ..., 0) and append results to ratio_sweep_report.md.
Called by run.py after evaluate_ratio_sweep.py.

Usage:
  python scripts/tune_threshold_all_ratios.py --config config/federated_scratch.yaml --model models/tflite/saved_model_original.tflite --append-to data/processed/runs/v12/2026-02-06_04-06-24/eval/ratio_sweep_report.md
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_threshold_sweep_for_ratio(
    model, x_eval, y_eval, ratio_normal, th_list, metric="f1"
):
    """Run threshold sweep for one ratio; return best row and all rows."""
    y_prob = _predict_proba(model, x_eval)
    rows = []
    for th in th_list:
        y_pred = (y_prob >= th).astype(np.int32)
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, average="binary", pos_label=1, zero_division=0)
        rec = recall_score(y_eval, y_pred, average="binary", pos_label=1, zero_division=0)
        f1 = f1_score(y_eval, y_pred, average="binary", pos_label=1, zero_division=0)
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
    if metric == "f1":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["f1_score"])
    elif metric == "attack_recall":
        best_idx = max(range(len(rows)), key=lambda i: rows[i]["attack_recall"])
    else:
        best_idx = max(range(len(rows)), key=lambda i: 0.5 * rows[i]["f1_score"] + 0.5 * rows[i]["attack_recall"])
    return rows[best_idx], rows


def main():
    parser = argparse.ArgumentParser(description="Threshold tuning for all ratios; append to ratio_sweep_report.md")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="models/tflite/saved_model_original.tflite", help="Model path (.h5 or .tflite)")
    parser.add_argument("--append-to", required=True, help="Path to ratio_sweep_report.md to append to")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "attack_recall", "balanced"],
                        help="Metric to maximize for best threshold")
    parser.add_argument("--ratios", type=str, default="100,90,80,70,60,50,40,30,20,10,0",
                        help="Comma-separated normal%% (default: 100,90,...,0)")
    parser.add_argument("--thresholds", type=str, default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                        help="Comma-separated thresholds to try")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import evaluate_ratio_sweep as ers
    load_config_and_data = ers.load_config_and_data
    subsample_by_ratio = ers.subsample_by_ratio
    global _predict_proba
    _predict_proba = ers._predict_proba

    class LoadArgs:
        pass
    load_args = LoadArgs()
    load_args.config = args.config
    load_args.model = args.model
    x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, model, _ = load_config_and_data(load_args)
    rng = np.random.default_rng(args.seed)

    ratios = [int(x.strip()) for x in args.ratios.split(",")]
    th_list = [float(t.strip()) for t in args.thresholds.split(",")]

    results = []
    for normal_pct in ratios:
        eval_idx = subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, normal_pct, rng)
        if len(eval_idx) == 0:
            continue
        x_eval = x_test[eval_idx]
        y_eval = y_test[eval_idx]
        best_row, _ = run_threshold_sweep_for_ratio(
            model, x_eval, y_eval, normal_pct, th_list, metric=args.metric
        )
        results.append({
            "normal_pct": normal_pct,
            "attack_pct": 100 - normal_pct,
            "best_threshold": best_row["threshold"],
            "accuracy": best_row["accuracy"],
            "f1_score": best_row["f1_score"],
            "normal_recall": best_row["normal_recall"],
            "normal_precision": best_row["normal_precision"],
            "attack_recall": best_row["attack_recall"],
            "attack_precision": best_row["attack_precision"],
        })
        print(f"  Normal {normal_pct:3d}% : Attack {100-normal_pct:3d}%  |  Best th={best_row['threshold']:.2f}  Acc={best_row['accuracy']:.4f}  F1={best_row['f1_score']:.4f}")

    append_path = Path(args.append_to)
    if not append_path.exists():
        print(f"⚠️  Report not found: {append_path}. Skipping append.")
        return 0

    md = []
    md.append("\n\n## Threshold Tuning (best per ratio)\n")
    md.append(f"Metric used for recommendation: **{args.metric}**\n\n")
    md.append("| Normal% | Attack% | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |\n")
    md.append("|---------|---------|----------------|----------|----------|---------------|------------------|---------------|------------------|\n")
    for r in results:
        md.append(
            f"| {r['normal_pct']} | {r['attack_pct']} | {r['best_threshold']:.2f} | "
            f"{r['accuracy']:.4f} | {r['f1_score']:.4f} | {r['normal_recall']:.4f} | {r['normal_precision']:.4f} | "
            f"{r['attack_recall']:.4f} | {r['attack_precision']:.4f} |\n"
        )
    md.append("\n")

    with open(append_path, "a", encoding="utf-8") as f:
        f.writelines(md)
    print(f"\n✅ Appended threshold tuning to {append_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
