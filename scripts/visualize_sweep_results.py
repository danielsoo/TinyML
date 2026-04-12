#!/usr/bin/env python3
"""
Visualize sweep_compression_grid_with_pgd.csv.

Produces:
  1. Bar charts — F1, Accuracy, Size, PGD Adv Acc grouped by distillation/pruning/ptq
  2. Heatmaps   — full 6×8 grid for each metric
  3. Scatter    — F1 vs Size, coloured by distillation
  4. Summary PNG with top-5 / bottom-5 tables

Usage:
  python scripts/visualize_sweep_results.py \
      --csv data/processed/runs/sweep_pgd/2026-03-16_19-19-15/sweep_compression_grid_with_pgd.csv \
      --out data/processed/runs/sweep_pgd/2026-03-16_19-19-15/viz
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── colour palette ────────────────────────────────────────────────────────────
DIST_COLORS   = {"none": "#4c72b0", "direct": "#dd8452", "progressive": "#55a868"}
PRUNE_MARKERS = {"prune_none": "o", "prune_10x5": "s", "prune_10x2": "^", "prune_5x10": "D"}
QAT_HATCH     = {False: "", True: "//"}

METRICS = [
    ("final_f1",     "Final F1",           "viridis",   False),
    ("final_acc",    "Final Accuracy",      "viridis",   False),
    ("final_size_kb","Final Size (KB)",     "viridis_r", False),   # lower=better → reverse
    ("pgd_adv_acc",  "PGD Adv Accuracy",    "plasma",    False),
    ("pgd_success_rate", "PGD Attack Success Rate", "Reds", False),
]

# Heatmap axes
ROW_ORDER = [
    ("no_qat",  "none"),  ("no_qat",  "direct"),  ("no_qat",  "progressive"),
    ("yes_qat", "none"),  ("yes_qat", "direct"),  ("yes_qat", "progressive"),
]
COL_ORDER = [
    ("prune_none", False), ("prune_none", True),
    ("prune_10x5", False), ("prune_10x5", True),
    ("prune_10x2", False), ("prune_10x2", True),
    ("prune_5x10", False), ("prune_5x10", True),
]
ROW_LABELS = [f"{fq}/{d}" for fq, d in ROW_ORDER]
COL_LABELS = [f"{p}/{'ptq' if ptq else 'no-ptq'}" for p, ptq in COL_ORDER]


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(val, default=np.nan):
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _norm_qat(v):
    s = str(v).strip().lower()
    return "yes_qat" if s in ("true", "1", "yes") else "no_qat"


def _norm_ptq(v):
    s = str(v).strip().lower()
    return s in ("true", "1", "yes")


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_heatmap(rows, metric):
    data = np.full((len(ROW_ORDER), len(COL_ORDER)), np.nan)
    for r in rows:
        fq  = _norm_qat(r.get("fl_qat", ""))
        d   = r.get("distillation", "none").strip().lower() or "none"
        p   = r.get("pruning", "prune_none").strip() or "prune_none"
        ptq = _norm_ptq(r.get("ptq", ""))
        val = _f(r.get(metric))
        if np.isnan(val):
            continue
        try:
            ri = ROW_ORDER.index((fq, d))
            ci = COL_ORDER.index((p, ptq))
        except ValueError:
            continue
        data[ri, ci] = val
    return data


# ── 1. Heatmaps ───────────────────────────────────────────────────────────────

def plot_heatmaps(rows, out_dir):
    for key, title, cmap, _ in METRICS:
        data = build_heatmap(rows, key)
        if np.all(np.isnan(data)):
            print(f"  skip heatmap {key}: no data")
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data, aspect="auto", cmap=cmap,
                       vmin=np.nanmin(data), vmax=np.nanmax(data))
        ax.set_xticks(range(len(COL_LABELS)))
        ax.set_yticks(range(len(ROW_LABELS)))
        ax.set_xticklabels(COL_LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(ROW_LABELS, fontsize=9)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(title, fontsize=9)
        ax.set_title(f"Heatmap — {title}", fontsize=11, fontweight="bold")

        for ri in range(len(ROW_LABELS)):
            for ci in range(len(COL_LABELS)):
                v = data[ri, ci]
                if not np.isnan(v):
                    txt = f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}"
                    norm_v = (v - np.nanmin(data)) / max(np.nanmax(data) - np.nanmin(data), 1e-9)
                    color = "white" if (0.25 < norm_v < 0.75) else "black"
                    ax.text(ci, ri, txt, ha="center", va="center", fontsize=7, color=color)

        plt.tight_layout()
        p = out_dir / f"heatmap_{key}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {p.name}")


# ── 2. Bar charts ─────────────────────────────────────────────────────────────

def plot_bars(rows, out_dir):
    """One figure per metric: bars grouped by (fl_qat, distillation), split by pruning."""

    groups = [
        ("no_qat",  "none"),  ("no_qat",  "direct"),  ("no_qat",  "progressive"),
        ("yes_qat", "none"),  ("yes_qat", "direct"),  ("yes_qat", "progressive"),
    ]
    prunings = ["prune_none", "prune_10x5", "prune_10x2", "prune_5x10"]
    ptqs     = [False, True]

    # Index rows by (fl_qat_str, dist, prune, ptq)
    idx = {}
    for r in rows:
        key = (_norm_qat(r["fl_qat"]), r["distillation"].strip(), r["pruning"].strip(), _norm_ptq(r["ptq"]))
        idx[key] = r

    for metric, title, _, _ in METRICS:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)

        for ax_i, ptq in enumerate(ptqs):
            ax = axes[ax_i]
            n_groups  = len(groups)
            n_pruning = len(prunings)
            bar_w     = 0.8 / n_pruning
            x_pos     = np.arange(n_groups)

            prune_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

            for pi, prune in enumerate(prunings):
                vals = []
                for fq, d in groups:
                    r = idx.get((fq, d, prune, ptq))
                    vals.append(_f(r.get(metric)) if r else np.nan)

                offset = (pi - n_pruning / 2 + 0.5) * bar_w
                bars = ax.bar(x_pos + offset, vals, width=bar_w * 0.9,
                              color=prune_colors[pi], label=prune, alpha=0.85)

                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f"{v:.3f}" if abs(v) < 10 else f"{v:.0f}",
                                ha="center", va="bottom", fontsize=5.5, rotation=90)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"{fq}\n{d}" for fq, d in groups], fontsize=7)
            ax.set_title(f"PTQ={'Yes' if ptq else 'No'}", fontsize=9)
            ax.set_ylabel(title, fontsize=9)
            ax.legend(fontsize=7, title="Pruning", title_fontsize=7)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle(f"Bar Chart — {title}  (left: no PTQ, right: PTQ)", fontsize=11, fontweight="bold")
        plt.tight_layout()
        p = out_dir / f"bar_{metric}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {p.name}")


# ── 3. Scatter: F1 vs Size, coloured by distillation ─────────────────────────

def plot_scatter(rows, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_i, (fq_filter, title) in enumerate([("no_qat", "No-QAT"), ("yes_qat", "Yes-QAT")]):
        ax = axes[ax_i]
        subset = [r for r in rows if _norm_qat(r["fl_qat"]) == fq_filter]

        for dist, color in DIST_COLORS.items():
            dr = [r for r in subset if r["distillation"].strip() == dist]
            for r in dr:
                x = _f(r.get("final_size_kb"))
                y = _f(r.get("final_f1"))
                a = _f(r.get("pgd_adv_acc"))
                ptq = _norm_ptq(r["ptq"])
                prune = r.get("pruning", "prune_none").strip()
                marker = PRUNE_MARKERS.get(prune, "o")
                ms = 60 if not ptq else 120
                ax.scatter(x, y, c=color, marker=marker, s=ms,
                           edgecolors="black" if ptq else "none",
                           linewidths=0.8, alpha=0.85, zorder=3)
                if not np.isnan(a):
                    ax.annotate(f"{a:.2f}", (x, y), textcoords="offset points",
                                xytext=(3, 3), fontsize=5.5, color="gray")

        ax.set_xlabel("Final Size (KB)", fontsize=9)
        ax.set_ylabel("Final F1", fontsize=9)
        ax.set_title(f"F1 vs Size — {title}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.3)

        # Legend: distillation colour
        dist_patches = [mpatches.Patch(color=c, label=d) for d, c in DIST_COLORS.items()]
        # Legend: pruning marker
        prune_handles = [
            plt.scatter([], [], marker=m, c="gray", s=50, label=p)
            for p, m in PRUNE_MARKERS.items()
        ]
        # PTQ indicator
        ptq_handles = [
            plt.scatter([], [], c="gray", s=60, edgecolors="none", label="no PTQ"),
            plt.scatter([], [], c="gray", s=120, edgecolors="black", linewidths=0.8, label="PTQ"),
        ]
        leg1 = ax.legend(handles=dist_patches,  loc="lower right", fontsize=7,
                         title="Distillation", title_fontsize=7)
        leg2 = ax.legend(handles=prune_handles, loc="upper right",  fontsize=7,
                         title="Pruning",       title_fontsize=7)
        ax.add_artist(leg1)
        leg3 = ax.legend(handles=ptq_handles,   loc="lower left",   fontsize=7,
                         title="PTQ",           title_fontsize=7)
        ax.add_artist(leg2)

    plt.tight_layout()
    p = out_dir / "scatter_f1_vs_size.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")


# ── 4. PGD adv acc vs Final F1 scatter ───────────────────────────────────────

def plot_robustness_scatter(rows, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for r in rows:
        fq   = _norm_qat(r["fl_qat"])
        dist = r["distillation"].strip()
        color = DIST_COLORS.get(dist, "gray")
        marker = "o" if fq == "no_qat" else "s"
        x = _f(r.get("final_f1"))
        y = _f(r.get("pgd_adv_acc"))
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, c=color, marker=marker, s=60,
                   edgecolors="black" if fq == "yes_qat" else "none",
                   linewidths=0.7, alpha=0.85, zorder=3)

    ax.set_xlabel("Final F1 (clean)", fontsize=10)
    ax.set_ylabel("PGD Adversarial Accuracy", fontsize=10)
    ax.set_title("Robustness vs Clean Performance", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3)

    dist_patches  = [mpatches.Patch(color=c, label=f"dist={d}") for d, c in DIST_COLORS.items()]
    qat_handles   = [
        plt.scatter([], [], c="gray", marker="o", s=50, edgecolors="none", label="no-QAT"),
        plt.scatter([], [], c="gray", marker="s", s=50, edgecolors="black", linewidths=0.7, label="yes-QAT"),
    ]
    ax.legend(handles=dist_patches + qat_handles, fontsize=8, loc="best")

    plt.tight_layout()
    p = out_dir / "scatter_robustness_vs_f1.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")


# ── 5. Summary table figure ───────────────────────────────────────────────────

def plot_summary_table(rows, out_dir):
    def _topn(rows, metric, n=5, reverse=True):
        scored = [(r, _f(r.get(metric))) for r in rows]
        scored = [(r, v) for r, v in scored if not np.isnan(v)]
        scored.sort(key=lambda x: x[1], reverse=reverse)
        return scored[:n]

    top_f1    = _topn(rows, "final_f1",      5, reverse=True)
    top_small = _topn(rows, "final_size_kb", 5, reverse=False)
    top_pgd   = _topn(rows, "pgd_adv_acc",   5, reverse=True)

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    tables = [
        (top_f1,    "Top 5 by Final F1",       "final_f1"),
        (top_small, "Top 5 Smallest (Size KB)", "final_size_kb"),
        (top_pgd,   "Top 5 PGD Adv Acc",       "pgd_adv_acc"),
    ]
    col_headers = ["Tag (short)", "F1", "Acc", "Size KB", "PGD Acc"]

    for ax, (data, title, _) in zip(axes, tables):
        ax.axis("off")
        table_data = []
        for r, _ in data:
            short_tag = r["tag"].replace("__", "\n")
            table_data.append([
                short_tag,
                f"{_f(r.get('final_f1')):.3f}",
                f"{_f(r.get('final_acc')):.3f}",
                f"{_f(r.get('final_size_kb')):.1f}",
                f"{_f(r.get('pgd_adv_acc')):.3f}",
            ])
        tbl = ax.table(cellText=table_data, colLabels=col_headers,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.8)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2d6a9f")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f0f4f8")
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    plt.suptitle("Sweep Summary Tables", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    p = out_dir / "summary_tables.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")


# ── 6. Metric distribution boxplot by distillation ───────────────────────────

def plot_boxplots(rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_metrics = [
        ("final_f1",     "Final F1"),
        ("final_acc",    "Final Accuracy"),
        ("final_size_kb","Final Size (KB)"),
        ("pgd_adv_acc",  "PGD Adv Accuracy"),
    ]

    for ax, (metric, label) in zip(axes.flat, plot_metrics):
        groups = {}
        for r in rows:
            fq   = _norm_qat(r["fl_qat"])
            dist = r["distillation"].strip() or "none"
            key  = f"{fq}\n{dist}"
            v    = _f(r.get(metric))
            if not np.isnan(v):
                groups.setdefault(key, []).append(v)

        keys = sorted(groups)
        data = [groups[k] for k in keys]
        colors = []
        for k in keys:
            dist_part = k.split("\n")[1]
            colors.append(DIST_COLORS.get(dist_part, "#888888"))

        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(keys) + 1))
        ax.set_xticklabels(keys, fontsize=7)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"Distribution of {label}", fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    dist_patches = [mpatches.Patch(color=c, label=d, alpha=0.7) for d, c in DIST_COLORS.items()]
    fig.legend(handles=dist_patches, loc="lower center", ncol=3, fontsize=8,
               title="Distillation", title_fontsize=8, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle("Metric Distributions by (QAT, Distillation)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = out_dir / "boxplots_by_distillation.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to sweep_compression_grid_with_pgd.csv")
    parser.add_argument("--out", default=None, help="Output directory (default: same dir as CSV)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out) if args.out else csv_path.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    rows = load_csv(csv_path)
    print(f"  {len(rows)} rows loaded\n")

    print("Plotting heatmaps ...")
    plot_heatmaps(rows, out_dir)

    print("\nPlotting bar charts ...")
    plot_bars(rows, out_dir)

    print("\nPlotting scatter F1 vs Size ...")
    plot_scatter(rows, out_dir)

    print("\nPlotting robustness scatter ...")
    plot_robustness_scatter(rows, out_dir)

    print("\nPlotting summary tables ...")
    plot_summary_table(rows, out_dir)

    print("\nPlotting boxplots ...")
    plot_boxplots(rows, out_dir)

    print(f"\nAll plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
