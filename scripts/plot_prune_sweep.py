#!/usr/bin/env python3
"""Paper figure: F1 vs deployed size for the pruning-ratio sweep (CIC: job 2026-10-04_g_prune_sweep;
TON_IoT: improved model, job 2026-10-05_w_toniot_v2_sweep_robustness)."""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SRCS = [ROOT / "data/processed/revision/2026-10-04_g_prune_sweep/compression_ablation/compression_ablation.json",
        ROOT / "data/processed/revision/2026-10-05_w_toniot_v2_sweep_robustness/compression_ablation/compression_ablation.json"]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "paper/figures/prune_sweep.png"

# Validated categorical slots 1-3 (dataviz reference palette); aqua needs relief -> markers + direct labels
SERIES = [
    ("prune_ft_client_qat", "client FT + QAT", "#2a78d6", "o"),
    ("prune_ft_client_ptq", "client FT + PTQ", "#eb6834", "s"),
    ("prune_noft_ptq", "no fine-tuning", "#1baf7a", "^"),
]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PANELS = [("cic_near_iid", "CIC-IDS2017", -12), ("ton2_near_iid", "TON_IoT", -12)]  # ratio-label offset

rows = [r for src in SRCS for r in json.loads(src.read_text())]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=False)
for ax, (model, title, label_dy) in zip(axes, PANELS):
    mrows = [r for r in rows if r["model"] == model]
    fp32 = next(r for r in mrows if r["variant"] == "fp32")
    ax.axhline(fp32["f1"] * 100, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(0.02, fp32["f1"] * 100, f"FL FP32 reference ({fp32['size_kb']:.0f} KB)",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=7.5, color=MUTED)
    for prefix, label, color, marker in SERIES:
        pts = []
        for r in mrows:
            m = re.fullmatch(prefix + r"_r(\d+)", r["variant"])
            if m:
                pts.append((r["size_kb"], r["f1"] * 100, int(m.group(1))))
        pts.sort()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=5, mec="white", mew=1, label=label, zorder=3)
        if prefix == "prune_ft_client_qat":
            for x, y, ratio in pts:
                ax.annotate(f"{ratio}%", (x, y), textcoords="offset points", xytext=(0, label_dy),
                            ha="center", fontsize=7, color=MUTED)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10, color=TEXT)
    ax.set_xlabel("Deployed model size (KB, log scale)")
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("F1 (%)")
axes[0].legend(loc="lower right", fontsize=7.5, frameon=False)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(f"saved {OUT}")
