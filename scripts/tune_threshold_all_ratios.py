#!/usr/bin/env python3
"""
Tune threshold for all ratios from 90% normal down to 0% in 10% steps.
For each ratio, sweeps thresholds and recommends best by chosen metric (F1, attack_recall, etc.).

Usage:
  python scripts/tune_threshold_all_ratios.py --config config/federated_local.yaml --model models/tflite/saved_model_original.tflite
  python scripts/tune_threshold_all_ratios.py --metric attack_recall --out threshold_all.csv --report threshold_all_report.md
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    parser = argparse.ArgumentParser(description="Tune threshold for all ratios (90 down to 0 in steps of 10)")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="models/tflite/saved_model_original.tflite", help="Model path (.h5 or .tflite)")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "attack_recall", "normal_recall", "balanced"],
                        help="Metric to maximize for recommendation")
    parser.add_argument("--thresholds", type=str,
                        default="0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8",
                        help="Comma-separated thresholds to try")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path (one row per ratio)")
    parser.add_argument("--report", type=str, default="", help="Optional Markdown report path")
    parser.add_argument("--append-to", type=str, default="", help="Append threshold section to this file (e.g. ratio_sweep_report.md)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    th_list = [float(t.strip()) for t in args.thresholds.split(",")]

    # Ratios: 90, 80, 70, ..., 0
    ratios = list(range(90, -1, -10))

    summary_rows = []
    all_ratio_sweeps = []  # (normal_pct, sweep_rows, best_row) per ratio

    print(f"\nTune threshold for ratios: Normal% = {ratios}  (metric={args.metric})")
    print("=" * 100)

    for normal_pct in ratios:
        eval_idx = subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, normal_pct, rng)
        x_eval = x_test[eval_idx]
        y_eval = y_test[eval_idx]
        n_normal = int(np.sum(y_eval == 0))
        n_attack = int(np.sum(y_eval == 1))
        y_prob = _predict_proba(model, x_eval)

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

        if args.metric == "f1":
            best_idx = max(range(len(rows)), key=lambda i: rows[i]["f1_score"])
        elif args.metric == "attack_recall":
            best_idx = max(range(len(rows)), key=lambda i: rows[i]["attack_recall"])
        elif args.metric == "normal_recall":
            best_idx = max(range(len(rows)), key=lambda i: rows[i]["normal_recall"])
        else:
            best_idx = max(range(len(rows)), key=lambda i: 0.5 * rows[i]["f1_score"] + 0.5 * rows[i]["attack_recall"])
        best = rows[best_idx]
        best_th = best["threshold"]
        all_ratio_sweeps.append((normal_pct, rows, best))

        summary_rows.append({
            "normal_pct": normal_pct,
            "attack_pct": 100 - normal_pct,
            "n_normal": n_normal,
            "n_attack": n_attack,
            "best_threshold": best_th,
            "accuracy": best["accuracy"],
            "f1_score": best["f1_score"],
            "normal_recall": best["normal_recall"],
            "normal_precision": best["normal_precision"],
            "attack_recall": best["attack_recall"],
            "attack_precision": best["attack_precision"],
        })

        # One-line summary per ratio
        print(f"  Normal {normal_pct:3d}% : Attack {100-normal_pct:3d}%  |  best_th={best_th:.2f}  |  "
              f"Acc={best['accuracy']:.4f}  F1={best['f1_score']:.4f}  |  "
              f"NormRec={best['normal_recall']:.4f}  NormPrec={best['normal_precision']:.4f}  |  "
              f"AttackRec={best['attack_recall']:.4f}  AttackPrec={best['attack_precision']:.4f}")

    # Full sweep table per ratio (console)
    print("\n" + "=" * 100)
    print("Full threshold sweep per ratio")
    print("=" * 100)
    hdr = f"{'Threshold':<10} {'Accuracy':<8} {'F1':<8} {'NormRec':<8} {'NormPrec':<8} {'AttackRec':<8} {'AttackPrec':<8}"
    for normal_pct, sweep_rows, best_row in all_ratio_sweeps:
        best_th = best_row["threshold"]
        print(f"\n--- Normal {normal_pct}% : Attack {100-normal_pct}% ---")
        print(hdr)
        print("-" * 74)
        for r in sweep_rows:
            mark = "  <--" if r["threshold"] == best_th else ""
            print(f"{r['threshold']:<10.2f} {r['accuracy']:<8.4f} {r['f1_score']:<8.4f} {r['normal_recall']:<8.4f} {r['normal_precision']:<8.4f} {r['attack_recall']:<8.4f} {r['attack_precision']:<8.4f}{mark}")
        print(f"Recommended threshold (by {args.metric}): {best_th}")
        print(f"  At threshold {best_th}: Accuracy={best_row['accuracy']:.4f}, F1={best_row['f1_score']:.4f}, Normal Recall={best_row['normal_recall']:.4f}, Normal Precision={best_row['normal_precision']:.4f}, Attack Recall={best_row['attack_recall']:.4f}, Attack Precision={best_row['attack_precision']:.4f}")

    print("\n" + "=" * 100)
    print(f"Done. Recommended thresholds (by {args.metric}) for each ratio.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print(f"Saved CSV: {out_path}")

    threshold_section = _format_threshold_section(
        all_ratio_sweeps, summary_rows, args.model, args.config, args.metric
    )
    if args.append_to:
        append_path = Path(args.append_to)
        append_path.parent.mkdir(parents=True, exist_ok=True)
        existing = append_path.read_text(encoding="utf-8").rstrip() if append_path.exists() else ""
        append_path.write_text(existing + "\n\n" + threshold_section + "\n", encoding="utf-8")
        print(f"Appended threshold section to: {append_path}")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(threshold_section + "\n", encoding="utf-8")
        print(f"Saved report: {report_path}")
    print()


def _format_threshold_section(all_ratio_sweeps, summary_rows, model_path: str, config_path: str, metric: str) -> str:
    """Format threshold tuning section as Markdown (for append or standalone report)."""
    lines = [
        "## Threshold Tuning (All Ratios)",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| **Model** | `{model_path}` |",
        f"| **Config** | `{config_path}` |",
        f"| **Metric** | `{metric}` |",
        f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        "",
        "Best threshold per ratio (90% normal down to 0% in steps of 10).",
        "",
        "### Summary (Best Threshold per Ratio)",
        "",
        "| Normal% | Attack% | n_Normal | n_Attack | Best Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision |",
        "|---------|----------|----------|----------|-----------------|----------|----------|---------------|-------------------|---------------|-------------------|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['normal_pct']} | {r['attack_pct']} | {r['n_normal']:,} | {r['n_attack']:,} | "
            f"{r['best_threshold']:.2f} | {r['accuracy']:.4f} | {r['f1_score']:.4f} | "
            f"{r['normal_recall']:.4f} | {r['normal_precision']:.4f} | "
            f"{r['attack_recall']:.4f} | {r['attack_precision']:.4f} |"
        )
    lines.extend(["", "### Full Threshold Sweep per Ratio", ""])

    for normal_pct, sweep_rows, best_row in all_ratio_sweeps:
        best_th = best_row["threshold"]
        attack_pct = 100 - normal_pct
        title = f"Normal {normal_pct}% : Attack {attack_pct}%"
        lines.extend([f"#### {title}", ""])
        lines.append("| Threshold | Accuracy | F1-Score | Normal Recall | Normal Precision | Attack Recall | Attack Precision | Best |")
        lines.append("|-----------|----------|----------|---------------|-------------------|---------------|-------------------|------|")
        for r in sweep_rows:
            best_mark = "✓" if r["threshold"] == best_th else ""
            lines.append(
                f"| {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['f1_score']:.4f} | "
                f"{r['normal_recall']:.4f} | {r['normal_precision']:.4f} | "
                f"{r['attack_recall']:.4f} | {r['attack_precision']:.4f} | {best_mark} |"
            )
        lines.extend([
            "",
            f"**Recommended threshold (by {metric}):** {best_th}",
            f"- At threshold {best_th}: Accuracy={best_row['accuracy']:.4f}, F1={best_row['f1_score']:.4f}, "
            f"Normal Recall={best_row['normal_recall']:.4f}, Normal Precision={best_row['normal_precision']:.4f}, "
            f"Attack Recall={best_row['attack_recall']:.4f}, Attack Precision={best_row['attack_precision']:.4f}",
            "",
        ])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
