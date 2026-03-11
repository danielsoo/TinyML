import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# Load the most recent sweep_results.csv
sweep_files = sorted(glob.glob("data/processed/sweep/*/sweep_results.csv"))
if not sweep_files:
    raise FileNotFoundError("No sweep_results.csv found")
csv_path = sweep_files[-1]
print(f"Loading: {csv_path}")

df = pd.read_csv(csv_path)

# tflite_size_kb is the actual deployed model size for all rows (pruned or not,
# quantized or not) and gives the correct ordering. final_size_kb accumulates
# pipeline artifacts and inflates pruned-non-PTQ model sizes incorrectly.
METRICS = {
    "f1":   "final_f1",
    "acc":  "final_acc",
    "size": "tflite_size_kb",
}

# df_no_dist: isolates distillation=none rows for pruning size comparison
df_no_dist = df[df["distillation"] == "none"]

# Per-category baselines (each "no compression" group = 100%)
BASE_DIST_KB  = df[df["distillation"] == "none"]["tflite_size_kb"].mean()
BASE_PRUN_KB  = df_no_dist[df_no_dist["pruning"] == "prune_none"]["tflite_size_kb"].mean()
BASE_QUANT_KB = df[~df["fl_qat"] & ~df["ptq"]]["tflite_size_kb"].mean()
print(f"Baseline dist  (no distillation): {BASE_DIST_KB:.2f} KB")
print(f"Baseline prun  (no pruning):      {BASE_PRUN_KB:.2f} KB")
print(f"Baseline quant (no quantization): {BASE_QUANT_KB:.2f} KB")

# ── helper ────────────────────────────────────────────────────────────────────
def mean_table(df, group_col, label_map):
    rows = []
    for key, label in label_map.items():
        sub = df[df[group_col] == key]
        rows.append({
            "group":    label,
            "mean_f1":   sub[METRICS["f1"]].mean(),
            "mean_acc":  sub[METRICS["acc"]].mean(),
            "mean_size_kb": sub[METRICS["size"]].mean(),
        })
    return pd.DataFrame(rows)


def bar_chart(table, group_col, metric_col, ylabel, title, color, out_path, pct=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(table))
    bars = ax.bar(x, table[metric_col], color=color, edgecolor="black", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(table[group_col], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fmt = "{:.1f}%" if pct else "{:.4f}"
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h,
                fmt.format(h), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── 1. Distillation ───────────────────────────────────────────────────────────
dist_map = {
    "direct":      "Direct",
    "progressive": "Progressive",
    "none":        "No Distillation",
}
dist_table = mean_table(df, "distillation", dist_map)
dist_table["size_pct"] = dist_table["mean_size_kb"] / BASE_DIST_KB * 100
dist_table.to_csv("dist.csv", index=False)
print("Saved: dist.csv")

bar_chart(dist_table, "group", "mean_f1",  "Mean F1",       "Distillation – F1",       "#4C72B0", "dist_f1.png")
bar_chart(dist_table, "group", "mean_acc", "Mean Accuracy", "Distillation – Accuracy", "#55A868", "dist_acc.png")
bar_chart(dist_table, "group", "size_pct", "Size (% of baseline)", "Distillation – Model Size", "#C44E52", "dist_size.png", pct=True)

# ── 2. Pruning ────────────────────────────────────────────────────────────────
# F1/acc use all rows; size is computed on distillation=none rows only so that
# distillation's effect on model architecture doesn't confound the size comparison.
prun_map = {
    "prune_10x5": "10x5",
    "prune_10x2": "10x2",
    "prune_5x10": "5x10",
    "prune_none": "No Pruning",
}

prun_rows = []
for key, label in prun_map.items():
    sub_all    = df[df["pruning"] == key]
    sub_nodist = df_no_dist[df_no_dist["pruning"] == key]
    prun_rows.append({
        "group":        label,
        "mean_f1":      sub_all[METRICS["f1"]].mean(),
        "mean_acc":     sub_all[METRICS["acc"]].mean(),
        "mean_size_kb": sub_nodist[METRICS["size"]].mean(),
    })
prun_table = pd.DataFrame(prun_rows)
prun_table["size_pct"] = prun_table["mean_size_kb"] / BASE_PRUN_KB * 100
prun_table.to_csv("prun.csv", index=False)
print("Saved: prun.csv")

bar_chart(prun_table, "group", "mean_f1",  "Mean F1",       "Pruning – F1",       "#4C72B0", "prun_f1.png")
bar_chart(prun_table, "group", "mean_acc", "Mean Accuracy", "Pruning – Accuracy", "#55A868", "prun_acc.png")
bar_chart(prun_table, "group", "size_pct", "Size (% of baseline)", "Pruning – Model Size", "#C44E52", "prun_size.png", pct=True)

# ── 3. Quantization ──────────────────────────────────────────────────────────
# Each group is filtered by the exact combination of fl_qat / ptq flags.
# Rows where the relevant flag is False are excluded from that group's mean.
quant_subsets = {
    "QAT":            df[ df["fl_qat"] &  ~df["ptq"]],
    "PTQ":            df[~df["fl_qat"] &   df["ptq"]],
    "QAT+PTQ":        df[ df["fl_qat"] &   df["ptq"]],
    "No Quantization":df[~df["fl_qat"] &  ~df["ptq"]],
}

quant_rows = []
for label, sub in quant_subsets.items():
    quant_rows.append({
        "group":        label,
        "mean_f1":      sub[METRICS["f1"]].mean(),
        "mean_acc":     sub[METRICS["acc"]].mean(),
        "mean_size_kb": sub[METRICS["size"]].mean(),
    })
quant_table = pd.DataFrame(quant_rows)
quant_table["size_pct"] = quant_table["mean_size_kb"] / BASE_QUANT_KB * 100
quant_table.to_csv("quant.csv", index=False)
print("Saved: quant.csv")

bar_chart(quant_table, "group", "mean_f1",  "Mean F1",       "Quantization – F1",       "#4C72B0", "quant_f1.png")
bar_chart(quant_table, "group", "mean_acc", "Mean Accuracy", "Quantization – Accuracy", "#55A868", "quant_acc.png")
bar_chart(quant_table, "group", "size_pct", "Size (% of baseline)", "Quantization – Model Size", "#C44E52", "quant_size.png", pct=True)

print("\nDone. All CSVs and bar charts saved.")
