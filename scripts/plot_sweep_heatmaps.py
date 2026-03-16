#!/usr/bin/env python3
"""
Plot 6x8 heatmaps from sweep_compression_grid_with_pgd.csv.
Rows = (fl_qat, distillation), Cols = (pruning, ptq). One heatmap per metric:
final_f1, pgd_adv_acc, final_size_kb.

Usage:
  python scripts/plot_sweep_heatmaps.py --run-dir data/processed/runs/sweep_pgd/2026-03-14_12-00-00
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Row order: (fl_qat, distillation) -> 6 labels
ROW_ORDER = [
    ("no_qat", "none"),
    ("no_qat", "direct"),
    ("no_qat", "progressive"),
    ("yes_qat", "none"),
    ("yes_qat", "direct"),
    ("yes_qat", "progressive"),
]
# Col order: (pruning, ptq) -> 8 labels
COL_ORDER = [
    ("prune_none", False),
    ("prune_none", True),
    ("prune_10x5", False),
    ("prune_10x5", True),
    ("prune_5x5", False),
    ("prune_5x5", True),
    ("prune_3x3", False),
    ("prune_3x3", True),
]


def _norm_qat(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return "yes_qat" if v else "no_qat"
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return "yes_qat"
    if s in ("false", "0", "no"):
        return "no_qat"
    return "yes_qat" if "yes" in s or "true" in s else "no_qat"


def _norm_ptq(v):
    if v is None or v == "":
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return "yes" in s or "true" in s


def _norm_distillation(v):
    if v is None or v == "":
        return "none"
    s = str(v).strip().lower()
    if s in ("none", "direct", "progressive"):
        return s
    return "none"


def _norm_pruning(v):
    if v is None or v == "":
        return "prune_none"
    s = str(v).strip()
    for name in ("prune_none", "prune_10x5", "prune_5x5", "prune_3x3"):
        if name in s or s == name.replace("prune_", ""):
            return name
    return s if s.startswith("prune_") else f"prune_{s}"


def _safe_float(val, default=np.nan):
    if val is None or val == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def build_matrix(rows: list[dict], metric: str) -> tuple[np.ndarray, list[str], list[str]]:
    """Build 6x8 matrix for metric. Returns (matrix, row_labels, col_labels)."""
    row_labels = [f"{fq}/{d}" for fq, d in ROW_ORDER]
    col_labels = [f"{p}/ptq_{'yes' if ptq else 'no'}" for p, ptq in COL_ORDER]

    data = np.full((len(ROW_ORDER), len(COL_ORDER)), np.nan)
    for r in rows:
        fq = _norm_qat(r.get("fl_qat"))
        d = _norm_distillation(r.get("distillation"))
        p = _norm_pruning(r.get("pruning"))
        ptq = _norm_ptq(r.get("ptq"))
        if fq is None or ptq is None:
            continue
        try:
            ri = ROW_ORDER.index((fq, d))
        except ValueError:
            continue
        try:
            ci = COL_ORDER.index((p, ptq))
        except ValueError:
            continue
        val = _safe_float(r.get(metric))
        if not np.isnan(val):
            data[ri, ci] = val
    return data, row_labels, col_labels


def main():
    parser = argparse.ArgumentParser(description="Plot sweep heatmaps (final_f1, pgd_adv_acc, final_size_kb)")
    parser.add_argument("--run-dir", required=True, help="Run directory containing sweep_compression_grid_with_pgd.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "sweep_compression_grid_with_pgd.csv"
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("CSV has no data rows.", file=sys.stderr)
        return 1

    metrics = [
        ("final_f1", "final_f1", "Final F1 (higher is better)"),
        ("pgd_adv_acc", "pgd_adv_acc", "PGD adversarial accuracy (higher is better)"),
        ("final_size_kb", "final_size_kb", "Final size (KB, lower is better)"),
    ]

    for key, col_name, title in metrics:
        data, row_labels, col_labels = build_matrix(rows, col_name)
        if np.all(np.isnan(data)):
            print(f"  Skip {key}: no numeric values.", file=sys.stderr)
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = "viridis" if "size" not in key else "viridis_r"  # size: lower better -> reversed
        im = ax.imshow(data, aspect="auto", cmap=cmap)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticklabels(row_labels)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_title(title)
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                v = data[i, j]
                if not np.isnan(v):
                    text = f"{v:.2f}" if abs(v) < 1000 else f"{v:.0f}"
                    ax.text(j, i, text, ha="center", va="center", color="white" if 0.3 < im.norm(v) < 0.7 else "black", fontsize=7)
        plt.tight_layout()
        out = run_dir / f"sweep_heatmap_{key}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   📊 Saved {out}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
