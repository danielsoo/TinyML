#!/usr/bin/env python3
"""
Visualize QAT vs no-QAT comparison from sweep_results_3_4_2026.csv.
Focuses on: does QAT help when pruning? Does no-QAT win without pruning?

Usage:
  python scripts/visualize_qat_vs_noqat.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

CSV_PATH = Path("sweep_results_3_4_2026 - sweep_results.csv")
OUT_DIR  = Path("data/processed/runs/sweep_pgd/2026-03-16_19-19-15/viz")


def load(path):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def is_pruned(row):
    return row["pruning"].strip() != "prune_none"


def is_qat(row):
    return str(row["fl_qat"]).strip().lower() in ("true", "1", "yes")


# ── colours ──────────────────────────────────────────────────────────────────
C_QAT    = "#e05c2a"   # orange-red
C_NOQAT  = "#2a7be0"   # blue
ALPHA    = 0.75

PRUNE_LABELS = {
    "prune_none": "No pruning",
    "prune_10x5": "10%×5",
    "prune_10x2": "10%×2",
    "prune_5x10": "5%×10",
}
DIST_MARKERS = {"none": "o", "direct": "s", "progressive": "^"}


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load(CSV_PATH)

    # ── 1. Strip-plot: F1 grouped by pruning × QAT ───────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    prune_keys = ["prune_none", "prune_10x2", "prune_10x5", "prune_5x10"]
    x_base = np.arange(len(prune_keys))
    jitter = 0.07

    for row in rows:
        prune = row["pruning"].strip()
        if prune not in prune_keys:
            continue
        xi = prune_keys.index(prune)
        dist = row["distillation"].strip()
        marker = DIST_MARKERS.get(dist, "o")
        color  = C_QAT if is_qat(row) else C_NOQAT
        xpos   = xi + (0.15 if is_qat(row) else -0.15) + np.random.uniform(-jitter, jitter)
        yval   = f(row["final_f1"])
        if np.isnan(yval):
            continue
        ax.scatter(xpos, yval, c=color, marker=marker, s=70,
                   alpha=ALPHA, edgecolors="white", linewidths=0.4, zorder=3)

    # Mean lines per (prune, qat)
    for pi, prune in enumerate(prune_keys):
        for qat, offset, color in [(True, 0.15, C_QAT), (False, -0.15, C_NOQAT)]:
            vals = [f(r["final_f1"]) for r in rows
                    if r["pruning"].strip() == prune and is_qat(r) == qat
                    and not np.isnan(f(r["final_f1"]))]
            if vals:
                ax.plot([pi + offset - 0.12, pi + offset + 0.12],
                        [np.mean(vals)] * 2, color=color, lw=2.5, zorder=4)

    ax.set_xticks(x_base)
    ax.set_xticklabels([PRUNE_LABELS[p] for p in prune_keys], fontsize=10)
    ax.set_ylabel("Final F1", fontsize=11)
    ax.set_title("F1 by Pruning Intensity — QAT vs No-QAT\n(horizontal bar = group mean)", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    qat_patch   = mpatches.Patch(color=C_QAT,   label="QAT")
    noqat_patch = mpatches.Patch(color=C_NOQAT, label="No-QAT")
    dist_handles = [plt.scatter([], [], marker=m, c="gray", s=50, label=f"dist={d}")
                    for d, m in DIST_MARKERS.items()]
    ax.legend(handles=[qat_patch, noqat_patch] + dist_handles,
              fontsize=8, loc="lower right")

    # Annotate the no-pruning gap
    none_qat   = np.mean([f(r["final_f1"]) for r in rows
                          if r["pruning"].strip() == "prune_none" and is_qat(r)
                          and not np.isnan(f(r["final_f1"]))])
    none_noqat = np.mean([f(r["final_f1"]) for r in rows
                          if r["pruning"].strip() == "prune_none" and not is_qat(r)
                          and not np.isnan(f(r["final_f1"]))])
    ax.annotate(f"Gap: {none_noqat - none_qat:.2f}",
                xy=(0, (none_qat + none_noqat) / 2),
                xytext=(0.35, (none_qat + none_noqat) / 2),
                fontsize=8, color="black",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))

    plt.tight_layout()
    p = OUT_DIR / "qat_vs_noqat_stripplot.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # ── 2. Grouped bar: mean F1 per (prune × QAT), split by distillation ─────
    dists = ["none", "direct", "progressive"]
    fig, axes = plt.subplots(1, len(dists), figsize=(15, 5), sharey=True)

    bar_w = 0.35
    x = np.arange(len(prune_keys))

    for ax, dist in zip(axes, dists):
        qat_means, noqat_means = [], []
        qat_std,   noqat_std   = [], []
        for prune in prune_keys:
            qv = [f(r["final_f1"]) for r in rows
                  if r["pruning"].strip() == prune and is_qat(r)
                  and r["distillation"].strip() == dist
                  and not np.isnan(f(r["final_f1"]))]
            nv = [f(r["final_f1"]) for r in rows
                  if r["pruning"].strip() == prune and not is_qat(r)
                  and r["distillation"].strip() == dist
                  and not np.isnan(f(r["final_f1"]))]
            qat_means.append(np.mean(qv) if qv else np.nan)
            noqat_means.append(np.mean(nv) if nv else np.nan)
            qat_std.append(np.std(qv) if len(qv) > 1 else 0)
            noqat_std.append(np.std(nv) if len(nv) > 1 else 0)

        ax.bar(x - bar_w/2, qat_means,   bar_w, color=C_QAT,   alpha=0.85,
               yerr=qat_std,   capsize=3, label="QAT",    error_kw={"lw": 1})
        ax.bar(x + bar_w/2, noqat_means, bar_w, color=C_NOQAT, alpha=0.85,
               yerr=noqat_std, capsize=3, label="No-QAT", error_kw={"lw": 1})

        for xi, (qm, nm) in enumerate(zip(qat_means, noqat_means)):
            if not np.isnan(qm):
                ax.text(xi - bar_w/2, qm + 0.01, f"{qm:.2f}", ha="center", fontsize=7)
            if not np.isnan(nm):
                ax.text(xi + bar_w/2, nm + 0.01, f"{nm:.2f}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([PRUNE_LABELS[p] for p in prune_keys], fontsize=8, rotation=15)
        ax.set_title(f"Distillation: {dist}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Final F1", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("QAT vs No-QAT Mean F1 — by Distillation × Pruning", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = OUT_DIR / "qat_vs_noqat_grouped_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # ── 3. Line chart: F1 vs pruning intensity, QAT vs no-QAT per distillation
    fig, axes = plt.subplots(1, len(dists), figsize=(15, 4), sharey=True)
    prune_x = [0, 1, 2, 3]  # none, 10x2, 10x5, 5x10

    for ax, dist in zip(axes, dists):
        for qat, color, label in [(True, C_QAT, "QAT"), (False, C_NOQAT, "No-QAT")]:
            means = []
            for prune in prune_keys:
                vals = [f(r["final_f1"]) for r in rows
                        if r["pruning"].strip() == prune and is_qat(r) == qat
                        and r["distillation"].strip() == dist
                        and not np.isnan(f(r["final_f1"]))]
                means.append(np.mean(vals) if vals else np.nan)

            # Fill where QAT > no-QAT (shade the crossover)
            ax.plot(prune_x, means, color=color, marker="o", lw=2,
                    label=label, markersize=7)
            for xi, m in zip(prune_x, means):
                if not np.isnan(m):
                    ax.text(xi, m + 0.015, f"{m:.3f}", ha="center", fontsize=7, color=color)

        # Shade region where QAT wins
        qat_m   = [np.mean([f(r["final_f1"]) for r in rows
                            if r["pruning"].strip() == p and is_qat(r)
                            and r["distillation"].strip() == dist
                            and not np.isnan(f(r["final_f1"]))]) or np.nan
                   for p in prune_keys]
        noqat_m = [np.mean([f(r["final_f1"]) for r in rows
                            if r["pruning"].strip() == p and not is_qat(r)
                            and r["distillation"].strip() == dist
                            and not np.isnan(f(r["final_f1"]))]) or np.nan
                   for p in prune_keys]
        ax.fill_between(prune_x, qat_m, noqat_m,
                        where=[not np.isnan(q) and not np.isnan(n) and q > n
                               for q, n in zip(qat_m, noqat_m)],
                        alpha=0.15, color=C_QAT, label="QAT wins")
        ax.fill_between(prune_x, qat_m, noqat_m,
                        where=[not np.isnan(q) and not np.isnan(n) and n > q
                               for q, n in zip(qat_m, noqat_m)],
                        alpha=0.15, color=C_NOQAT, label="No-QAT wins")

        ax.set_xticks(prune_x)
        ax.set_xticklabels([PRUNE_LABELS[p] for p in prune_keys], fontsize=8)
        ax.set_title(f"Distillation: {dist}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Final F1", fontsize=9)
        ax.set_ylim(0.2, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("F1 Trajectory: QAT vs No-QAT across Pruning Levels", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = OUT_DIR / "qat_vs_noqat_lines.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # ── 4. Heatmap: F1 difference (no-QAT minus QAT) ─────────────────────────
    fig, axes = plt.subplots(1, len(dists), figsize=(15, 4))

    ptqs = [False, True]
    ptq_labels = ["no PTQ", "PTQ"]

    for ax, dist in zip(axes, dists):
        data = np.full((len(prune_keys), len(ptqs)), np.nan)
        for pi, prune in enumerate(prune_keys):
            for qi, ptq in enumerate(ptqs):
                qat_vals = [f(r["final_f1"]) for r in rows
                            if r["pruning"].strip() == prune
                            and is_qat(r)
                            and r["distillation"].strip() == dist
                            and _norm_ptq(r["ptq"]) == ptq
                            and not np.isnan(f(r["final_f1"]))]
                noqat_vals = [f(r["final_f1"]) for r in rows
                              if r["pruning"].strip() == prune
                              and not is_qat(r)
                              and r["distillation"].strip() == dist
                              and _norm_ptq(r["ptq"]) == ptq
                              and not np.isnan(f(r["final_f1"]))]
                if qat_vals and noqat_vals:
                    data[pi, qi] = np.mean(noqat_vals) - np.mean(qat_vals)

        vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 1
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(ptq_labels)))
        ax.set_xticklabels(ptq_labels, fontsize=9)
        ax.set_yticks(range(len(prune_keys)))
        ax.set_yticklabels([PRUNE_LABELS[p] for p in prune_keys], fontsize=8)
        ax.set_title(f"Distillation: {dist}", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, label="no-QAT F1 − QAT F1", shrink=0.8)

        for pi in range(len(prune_keys)):
            for qi in range(len(ptqs)):
                v = data[pi, qi]
                if not np.isnan(v):
                    sign = "no-QAT +" if v > 0 else "QAT +"
                    ax.text(qi, pi, f"{sign}{abs(v):.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > vmax * 0.5 else "black")

    fig.suptitle("Heatmap: no-QAT F1 − QAT F1  (blue=QAT wins, red=no-QAT wins)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = OUT_DIR / "qat_vs_noqat_diff_heatmap.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # ── 5. Size vs F1 scatter: QAT vs no-QAT ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))

    for row in rows:
        x = f(row.get("tflite_size_kb"))
        y = f(row["final_f1"])
        if np.isnan(x) or np.isnan(y):
            continue
        dist   = row["distillation"].strip()
        marker = DIST_MARKERS.get(dist, "o")
        color  = C_QAT if is_qat(row) else C_NOQAT
        pruned = is_pruned(row)
        ax.scatter(x, y, c=color, marker=marker, s=80 if pruned else 120,
                   alpha=ALPHA, edgecolors="black" if pruned else "none",
                   linewidths=0.6, zorder=3)

    ax.set_xlabel("TFLite Size (KB)", fontsize=10)
    ax.set_ylabel("Final F1", fontsize=10)
    ax.set_title("F1 vs Model Size — QAT (red) vs No-QAT (blue)\n"
                 "(marker=distillation, border=pruned, size=no-prune)",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)

    dist_handles  = [plt.scatter([], [], marker=m, c="gray", s=60, label=f"dist={d}")
                     for d, m in DIST_MARKERS.items()]
    color_handles = [mpatches.Patch(color=C_QAT,   label="QAT"),
                     mpatches.Patch(color=C_NOQAT, label="No-QAT")]
    prune_handles = [plt.scatter([], [], c="gray", s=80, edgecolors="black",
                                 linewidths=0.6, label="Pruned"),
                     plt.scatter([], [], c="gray", s=120, edgecolors="none",
                                 label="Not pruned")]
    ax.legend(handles=color_handles + dist_handles + prune_handles,
              fontsize=8, loc="lower right")

    plt.tight_layout()
    p = OUT_DIR / "qat_vs_noqat_size_f1.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")


def _norm_ptq(v):
    return str(v).strip().lower() in ("true", "1", "yes")


if __name__ == "__main__":
    main()
