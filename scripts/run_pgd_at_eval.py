#!/usr/bin/env python3
"""
PGD Adversarial Training (AT) + PTQ Evaluation.

Evaluates global_model.h5 directly (no TFLite for AT phases) so AT actually
transfers — TFLite models can't be fine-tuned, so comparing pre/post AT
on frozen TFLite is meaningless.

Pipeline:
  Phase A: h5 pre-AT        — evaluate global_model.h5 with PGD (eps sweep)
  Phase B: h5 post-AT       — same h5 after PGD adversarial training
  Phase C: float TFLite     — export h5 pre-AT to float32 TFLite, attack via
                               transfer from AT'd h5 (white-box on surrogate)
  Phase D: PTQ TFLite       — export h5 pre-AT with PTQ, attack via transfer
                               from AT'd h5

This answers:
  A→B : does AT improve h5 robustness?
  A→C : does float32 TFLite conversion degrade robustness under transfer attack?
  A→D : does PTQ further degrade robustness under transfer attack?
  B vs D : does PTQ-before-AT hurt compared to AT'd h5?

Usage:
  python scripts/run_pgd_at_eval.py --config config/federated_local_sky.yaml
"""
import argparse
import copy
import csv
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import tensorflow as tf

from src.data.loader import load_dataset
from src.adversarial.fgsm_hook import generate_pgd_attack
from src.tinyml.export_tflite import export_tflite

ATTACKER_H5 = project_root / "models" / "global_model.h5"
OUT_DIR     = project_root / "data" / "processed" / "pgd_at"

EPSILONS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]


def _f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_h5(path: Path):
    try:
        import tensorflow_model_optimization as tfmot
        with tfmot.quantization.keras.quantize_scope():
            m = tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        m = tf.keras.models.load_model(str(path), compile=False)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def eval_h5_metrics(model, x_orig, x_adv, y_true, threshold=0.5) -> dict:
    p_orig = model.predict(x_orig, verbose=0).ravel()
    p_adv  = model.predict(x_adv,  verbose=0).ravel()
    return _metrics(p_orig, p_adv, y_true, threshold)


def predict_tflite(tflite_path: Path, x: np.ndarray) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    preds = []
    for i in range(len(x)):
        interp.set_tensor(in_d["index"], x[i:i+1].astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        preds.append(float(out.ravel()[-1]))
    return np.array(preds)


def eval_tflite_metrics(tflite_path: Path, x_orig, x_adv, y_true, threshold=0.5) -> dict:
    p_orig = predict_tflite(tflite_path, x_orig)
    p_adv  = predict_tflite(tflite_path, x_adv)
    return _metrics(p_orig, p_adv, y_true, threshold)


def _metrics(p_orig, p_adv, y_true, threshold=0.5) -> dict:
    y = np.asarray(y_true).ravel().astype(int)
    orig_acc = float(((p_orig >= threshold).astype(int) == y).mean())
    adv_acc  = float(((p_adv  >= threshold).astype(int) == y).mean())
    tp = int(np.sum((p_adv >= threshold) & (y == 1)))
    fp = int(np.sum((p_adv >= threshold) & (y == 0)))
    fn = int(np.sum((p_adv <  threshold) & (y == 1)))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "orig_acc":     round(orig_acc, 4),
        "adv_acc":      round(adv_acc,  4),
        "adv_f1":       round(f1,       4),
        "adv_rec":      round(rec,      4),
        "success_rate": round(orig_acc - adv_acc, 4),
    }


def pgd_finetune(model, x_train, y_train, eps=0.1, steps=10,
                 epochs=3, batch_size=128):
    """Fine-tune model with PGD adversarial training (50/50 clean+adv mix)."""
    print(f"  PGD AT: eps={eps}, steps={steps}, epochs={epochs}, n={len(x_train)}")
    for ep in range(epochs):
        idx = np.random.permutation(len(x_train))
        for start in range(0, len(x_train), batch_size):
            bi = idx[start:start + batch_size]
            xb = x_train[bi].astype(np.float32)
            yb = y_train[bi].astype(np.float32)
            x_adv_b, _ = generate_pgd_attack(model, xb, yb, eps=eps, steps=steps)
            x_mix = np.concatenate([xb, x_adv_b], axis=0)
            y_mix = np.concatenate([yb, yb], axis=0)
            model.train_on_batch(x_mix, y_mix)
        p = model.predict(x_train[:500], verbose=0).ravel()
        acc = ((p >= 0.5).astype(int) == y_train[:500].astype(int)).mean()
        print(f"    epoch {ep+1}/{epochs}  train_acc={acc:.3f}")
    return model


def gen_pgd_adv(model, x, y, eps, steps, batch_size=256):
    """Generate PGD adversarial examples in batches."""
    parts = []
    for start in range(0, len(x), batch_size):
        xb_adv, _ = generate_pgd_attack(
            model, x[start:start+batch_size], y[start:start+batch_size],
            eps=eps, steps=steps)
        parts.append(xb_adv)
    return np.concatenate(parts, axis=0)


def export_model(model, out_path: Path, quantize: bool, rep_data: np.ndarray) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_tflite(model, str(out_path),
                      quantize=quantize,
                      representative_data=rep_data.astype(np.float32) if quantize else None)
        print(f"  Exported {'PTQ' if quantize else 'float'} TFLite: {out_path.name} "
              f"({out_path.stat().st_size//1024}KB)")
        return True
    except Exception as e:
        print(f"  Export failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/federated_local_sky.yaml")
    parser.add_argument("--epsilons",   nargs="+", type=float, default=EPSILONS)
    parser.add_argument("--pgd-steps",  type=int,   default=10)
    parser.add_argument("--at-epsilon", type=float, default=0.1)
    parser.add_argument("--at-epochs",  type=int,   default=3)
    parser.add_argument("--n-eval",     type=int,   default=5000)
    parser.add_argument("--n-train",    type=int,   default=20000)
    parser.add_argument("--threshold",  type=float, default=0.5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tflite_dir = OUT_DIR / "exported_tflite"
    tflite_dir.mkdir(exist_ok=True)
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    # Load config + data
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    print("Loading dataset ...")
    x_tr_raw, y_tr_raw, x_te, y_te = load_dataset(
        data_cfg.get("name", "cicids2017"), **kwargs)
    x_train = np.asarray(x_tr_raw, dtype=np.float32)[:args.n_train]
    y_train = np.asarray(y_tr_raw, dtype=np.float32)[:args.n_train]
    x_test  = np.asarray(x_te,     dtype=np.float32)[:args.n_eval]
    y_test  = np.asarray(y_te,     dtype=np.float32)[:args.n_eval]
    print(f"  Train: {len(x_train)}  Test: {len(x_test)}")

    # ── Step 1: Export float32 + PTQ TFLite from clean h5 ─────────────────────
    float_tflite = tflite_dir / "global_model_float.tflite"
    ptq_tflite   = tflite_dir / "global_model_ptq.tflite"

    print(f"\nLoading h5: {ATTACKER_H5.name}")
    model_preat = load_h5(ATTACKER_H5)

    if not float_tflite.exists():
        print("\nExporting float32 TFLite (pre-AT) ...")
        export_model(model_preat, float_tflite, quantize=False, rep_data=x_train[:500])
    else:
        print(f"  [Exists] {float_tflite.name}")

    if not ptq_tflite.exists():
        print("\nExporting PTQ TFLite (pre-AT) ...")
        export_model(model_preat, ptq_tflite, quantize=True, rep_data=x_train[:500])
    else:
        print(f"  [Exists] {ptq_tflite.name}")

    # ── Step 2: PGD AT on h5 ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PGD Adversarial Training on h5 ...")
    print(f"{'='*60}")
    model_postat = load_h5(ATTACKER_H5)
    model_postat = pgd_finetune(
        model_postat, x_train, y_train,
        eps=args.at_epsilon, steps=args.pgd_steps, epochs=args.at_epochs,
    )

    # ── Step 3: Evaluate all phases at each epsilon ────────────────────────────
    all_results = []
    phases = [
        ("pre_AT_h5",    "h5 before AT",             "h5",    model_preat),
        ("post_AT_h5",   "h5 after AT",               "h5",    model_postat),
        ("float_tflite", "float TFLite (transfer)",   "tflite_float", model_postat),
        ("ptq_tflite",   "PTQ TFLite (transfer)",     "tflite_ptq",   model_postat),
    ]

    for phase_name, phase_label, eval_mode, attack_model in phases:
        print(f"\n{'='*60}")
        print(f"Phase: {phase_name} — {phase_label}")
        print(f"{'='*60}")

        for eps in args.epsilons:
            # Generate adversarial examples from the attack model
            x_adv = gen_pgd_adv(attack_model, x_test, y_test,
                                 eps=eps, steps=args.pgd_steps)

            try:
                if eval_mode == "h5":
                    m = eval_h5_metrics(attack_model, x_test, x_adv, y_test, args.threshold)
                elif eval_mode == "tflite_float":
                    m = eval_tflite_metrics(float_tflite, x_test, x_adv, y_test, args.threshold)
                elif eval_mode == "tflite_ptq":
                    m = eval_tflite_metrics(ptq_tflite, x_test, x_adv, y_test, args.threshold)

                print(f"  eps={eps:.2f}  orig={m['orig_acc']:.3f}  "
                      f"adv_acc={m['adv_acc']:.3f}  f1={m['adv_f1']:.3f}  "
                      f"rec={m['adv_rec']:.3f}  success={m['success_rate']:+.3f}")
                all_results.append({
                    "phase": phase_name, "phase_label": phase_label,
                    "epsilon": eps, **m,
                })
            except Exception as e:
                print(f"  eps={eps:.2f}  ERROR: {e}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    result_fields = ["phase", "phase_label", "epsilon",
                     "orig_acc", "adv_acc", "adv_f1", "adv_rec", "success_rate"]
    result_csv = OUT_DIR / "pgd_at_results.csv"
    with open(result_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=result_fields)
        w.writeheader()
        w.writerows(all_results)
    print(f"\nWrote {result_csv} ({len(all_results)} rows)")

    # ── Summary + report ──────────────────────────────────────────────────────
    _write_report(all_results, args, OUT_DIR)
    _plot_results(all_results, args.epsilons, plot_dir)

    print(f"\nAll outputs -> {OUT_DIR.resolve()}")
    return 0


def _write_report(all_results, args, out_dir):
    lines = [
        "# PGD AT + PTQ Evaluation Report",
        "",
        f"- Model: `global_model.h5`",
        f"- AT: eps={args.at_epsilon}, steps={args.pgd_steps}, epochs={args.at_epochs}",
        f"- Eval epsilons: {args.epsilons}  |  Eval samples: {args.n_eval}",
        "",
        "## Design",
        "- **Phase A (pre_AT_h5)**: h5 evaluated directly, PGD from clean h5",
        "- **Phase B (post_AT_h5)**: h5 evaluated directly after PGD AT",
        "- **Phase C (float_tflite)**: float32 TFLite exported pre-AT, attacked via transfer from AT'd h5",
        "- **Phase D (ptq_tflite)**: PTQ TFLite exported pre-AT, attacked via transfer from AT'd h5",
        "",
        "**A→B**: Does AT improve h5 robustness?",
        "**A→C**: Does float32 TFLite conversion hurt robustness under transfer attack?",
        "**A→D**: Does PTQ further degrade robustness vs float TFLite?",
        "**B vs D**: AT'd h5 vs PTQ TFLite under same adversarial pressure.",
        "",
        "## Results by Phase",
        "",
        "| Phase | eps | Orig Acc | Adv Acc | Adv F1 | Adv Rec | Success |",
        "|-------|-----|----------|---------|--------|---------|---------|",
    ]
    for r in all_results:
        lines.append(
            f"| {r['phase']} | {r['epsilon']} "
            f"| {r['orig_acc']:.3f} | {r['adv_acc']:.3f} "
            f"| {r['adv_f1']:.3f} | {r['adv_rec']:.3f} | {r['success_rate']:+.3f} |"
        )

    # Summary: mean over epsilon sweep per phase
    lines += ["", "## Summary (mean over epsilon sweep)", "",
              "| Phase | Mean Adv Acc | Mean Adv F1 | Mean Adv Rec | Min Adv Acc |",
              "|-------|-------------|------------|-------------|------------|"]
    phases = list(dict.fromkeys(r["phase"] for r in all_results))
    for phase in phases:
        rows = [r for r in all_results if r["phase"] == phase]
        lines.append(
            f"| {phase} | {np.mean([r['adv_acc'] for r in rows]):.3f} "
            f"| {np.mean([r['adv_f1'] for r in rows]):.3f} "
            f"| {np.mean([r['adv_rec'] for r in rows]):.3f} "
            f"| {np.min([r['adv_acc'] for r in rows]):.3f} |"
        )

    p = out_dir / "pgd_at_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {p}")


PHASE_COLORS = {
    "pre_AT_h5":    "#4c72b0",
    "post_AT_h5":   "#e05c2a",
    "float_tflite": "#2ca02c",
    "ptq_tflite":   "#9467bd",
}
PHASE_LABELS = {
    "pre_AT_h5":    "h5 pre-AT",
    "post_AT_h5":   "h5 post-AT",
    "float_tflite": "float TFLite (transfer)",
    "ptq_tflite":   "PTQ TFLite (transfer)",
}


def _plot_results(all_results, epsilons, plot_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = list(PHASE_COLORS.keys())
    metrics = [("adv_acc", "Adv Accuracy"), ("adv_f1", "Adv F1"), ("adv_rec", "Adv Recall")]

    # Plot 1: epsilon curves — acc + F1 per phase
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for phase in phases:
            rows = sorted([r for r in all_results if r["phase"] == phase],
                          key=lambda r: r["epsilon"])
            if not rows:
                continue
            ax.plot([r["epsilon"] for r in rows],
                    [r[metric]   for r in rows],
                    marker="o", lw=2,
                    color=PHASE_COLORS[phase],
                    label=PHASE_LABELS[phase])
        ax.set_xlabel("Epsilon", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(epsilons)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("PGD AT + PTQ: Adversarial Robustness vs Epsilon",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    p = plot_dir / "epsilon_curves.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # Plot 2: bar chart at eps=0.1 for quick comparison
    eps_ref = 0.1
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    bar_w = 0.18
    for pi, phase in enumerate(phases):
        row = next((r for r in all_results
                    if r["phase"] == phase and abs(r["epsilon"] - eps_ref) < 1e-6), None)
        if row is None:
            continue
        vals = [row[m] for m, _ in metrics]
        offset = (pi - len(phases) / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      color=PHASE_COLORS[phase], alpha=0.85, label=PHASE_LABELS[phase])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + 0.01, f"{v:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in metrics], fontsize=10)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_title(f"Phase comparison at eps={eps_ref}  (global_model.h5 + TFLite exports)",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    p = plot_dir / "phase_comparison_bars.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")

    # Plot 3: AT gain (post - pre) and PTQ penalty (ptq - float) across epsilons
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (label, p1, p2, color) in zip(axes, [
        ("AT gain: post_AT - pre_AT h5", "pre_AT_h5", "post_AT_h5", "#e05c2a"),
        ("PTQ penalty: ptq_tflite - float_tflite", "float_tflite", "ptq_tflite", "#9467bd"),
    ]):
        deltas = []
        for eps in epsilons:
            r1 = next((r for r in all_results if r["phase"] == p1 and abs(r["epsilon"]-eps) < 1e-6), None)
            r2 = next((r for r in all_results if r["phase"] == p2 and abs(r["epsilon"]-eps) < 1e-6), None)
            deltas.append((r2["adv_acc"] - r1["adv_acc"]) if r1 and r2 else np.nan)

        bar_colors = ["#2ca02c" if d > 0 else "#d62728" for d in
                      [d if not np.isnan(d) else 0 for d in deltas]]
        bars = ax.bar(range(len(epsilons)), deltas, color=bar_colors, alpha=0.85)
        for bar, v in zip(bars, deltas):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + (0.002 if v >= 0 else -0.01),
                        f"{v:+.3f}", ha="center", fontsize=9)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(epsilons)))
        ax.set_xticklabels([str(e) for e in epsilons])
        ax.set_xlabel("Epsilon", fontsize=9)
        ax.set_ylabel("Delta Adv Acc", fontsize=9)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("AT gain and PTQ penalty by epsilon", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p = plot_dir / "at_gain_ptq_penalty.png"
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {p.name}")


if __name__ == "__main__":
    sys.exit(main())
