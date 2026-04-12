#!/usr/bin/env python3
"""
Run PGD evaluation on all models in sweep_results_3_4_2026 - sweep_results.csv.
Generates adversarial examples once (using global_model.h5 as surrogate),
then evaluates every TFLite model against them.

Outputs:
  data/processed/pgd_sweep/pgd_sweep_results.csv   — sweep CSV + pgd_adv_acc columns
  data/processed/pgd_sweep/pgd_sweep_summary.md    — top picks per PTQ group

Usage:
  python scripts/run_pgd_on_sweep.py --config config/federated_local_sky.yaml
"""
import argparse
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
from src.adversarial.fgsm_hook import (
    generate_pgd_attack,
    generate_adversarial_dataset_pgd,
)

CSV_PATH    = project_root / "sweep_results_3_4_2026 - sweep_results.csv"
TFLITE_DIR  = project_root / "models" / "tflite"
ATTACKER_H5 = project_root / "models" / "global_model.h5"
OUT_DIR     = project_root / "data" / "processed" / "pgd_sweep"


def _f(v, default=np.nan):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _norm_ptq(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def tag_to_tflite(tag: str) -> Path:
    """Convert CSV tag (double underscore) to tflite filename (single underscore)."""
    name = tag.replace("__", "_") + ".tflite"
    return TFLITE_DIR / name


def load_attacker(path: Path):
    """Load the surrogate Keras model for attack generation."""
    try:
        import tensorflow_model_optimization as tfmot
        with tfmot.quantization.keras.quantize_scope():
            m = tf.keras.models.load_model(str(path), compile=False)
    except Exception:
        m = tf.keras.models.load_model(str(path), compile=False)
    m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return m


def eval_tflite(tflite_path: Path, x_orig, x_adv, y_true, threshold=0.5):
    """Evaluate a TFLite model on original and adversarial examples."""
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]

    def predict(x):
        preds = []
        for i in range(len(x)):
            sample = x[i:i+1].astype(np.float32)
            interp.set_tensor(in_d["index"], sample)
            interp.invoke()
            out = interp.get_tensor(out_d["index"])
            preds.append(float(out.ravel()[-1]))
        return np.array(preds)

    p_orig = predict(x_orig)
    p_adv  = predict(x_adv)
    y = np.asarray(y_true).ravel().astype(int)

    orig_acc = float(((p_orig >= threshold).astype(int) == y).mean())
    adv_acc  = float(((p_adv  >= threshold).astype(int) == y).mean())
    success  = float(orig_acc - adv_acc)
    return {"orig_acc": orig_acc, "adv_acc": adv_acc, "success_rate": success}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/federated_local_sky.yaml")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--pgd-steps", type=int, default=10)
    parser.add_argument("--n-eval", type=int, default=5000,
                        help="Number of test samples to evaluate on")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Load data
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    print("Loading dataset ...")
    _, _, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)
    x_test = np.asarray(x_test, dtype=np.float32)[:args.n_eval]
    y_test = np.asarray(y_test, dtype=np.float32)[:args.n_eval]
    print(f"  Test samples: {len(x_test)}")

    # Load surrogate attacker
    print(f"\nLoading surrogate attacker: {ATTACKER_H5.name} ...")
    attacker = load_attacker(ATTACKER_H5)

    # Generate adversarial examples ONCE
    print(f"Generating PGD adversarial examples (eps={args.epsilon}, steps={args.pgd_steps}) ...")
    x_adv, _ = generate_adversarial_dataset_pgd(
        attacker, x_test, y_test,
        eps=args.epsilon, steps=args.pgd_steps, batch_size=128,
    )
    print(f"  Done. x_adv shape: {x_adv.shape}")

    # Load sweep CSV
    rows = []
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"\nEvaluating {len(rows)} models ...")

    results = []
    for i, row in enumerate(rows):
        tag = row["tag"].strip()
        tflite_path = tag_to_tflite(tag)
        ptq = _norm_ptq(row.get("ptq", ""))

        if not tflite_path.exists():
            print(f"  [{i+1:2d}/{len(rows)}] MISSING  {tflite_path.name}")
            results.append({**row, "pgd_orig_acc": "", "pgd_adv_acc": "",
                            "pgd_success_rate": "", "tflite_found": False})
            continue

        try:
            m = eval_tflite(tflite_path, x_test, x_adv, y_test, threshold=args.threshold)
            print(f"  [{i+1:2d}/{len(rows)}] {'PTQ' if ptq else 'noPTQ':5s}  "
                  f"orig={m['orig_acc']:.3f}  adv={m['adv_acc']:.3f}  "
                  f"success={m['success_rate']:+.3f}  {tag}")
            results.append({**row,
                            "pgd_orig_acc":    round(m["orig_acc"],    4),
                            "pgd_adv_acc":     round(m["adv_acc"],     4),
                            "pgd_success_rate":round(m["success_rate"],4),
                            "tflite_found":    True})
        except Exception as e:
            print(f"  [{i+1:2d}/{len(rows)}] ERROR  {tag}: {e}")
            results.append({**row, "pgd_orig_acc": "", "pgd_adv_acc": "",
                            "pgd_success_rate": "", "tflite_found": False})

    # Write enriched CSV
    fieldnames = list(rows[0].keys()) + ["pgd_orig_acc", "pgd_adv_acc",
                                          "pgd_success_rate", "tflite_found"]
    out_csv = OUT_DIR / "pgd_sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {out_csv}")

    # Analysis: best per PTQ group
    valid = [r for r in results if r.get("tflite_found") and r["pgd_adv_acc"] != ""]
    no_ptq = sorted([r for r in valid if not _norm_ptq(r["ptq"])],
                    key=lambda r: _f(r["pgd_adv_acc"]), reverse=True)
    yes_ptq = sorted([r for r in valid if _norm_ptq(r["ptq"])],
                     key=lambda r: _f(r["pgd_adv_acc"]), reverse=True)

    # Fixed 3 models
    fixed = [
        "yes_qat__distill_progressive__prune_5x10__ptq_yes",
        "no_qat__distill_progressive__prune_10x5__ptq_no",
        "no_qat__distill_progressive__prune_none__ptq_yes",
    ]
    fixed_rows = {r["tag"]: r for r in valid if r["tag"] in fixed}

    print("\n" + "="*70)
    print("TOP 5 - No PTQ (by PGD adv acc)")
    print("="*70)
    for r in no_ptq[:5]:
        print(f"  {r['tag']}")
        print(f"    F1={_f(r['final_f1']):.3f}  pgd_adv={_f(r['pgd_adv_acc']):.3f}  "
              f"size={_f(r.get('tflite_size_kb')):.1f}KB")

    print("\n" + "="*70)
    print("TOP 5 - Yes PTQ (by PGD adv acc)")
    print("="*70)
    for r in yes_ptq[:5]:
        print(f"  {r['tag']}")
        print(f"    F1={_f(r['final_f1']):.3f}  pgd_adv={_f(r['pgd_adv_acc']):.3f}  "
              f"size={_f(r.get('tflite_size_kb')):.1f}KB")

    # Pick best that aren't already in fixed 3
    best_no_ptq = next((r for r in no_ptq if r["tag"] not in fixed), no_ptq[0])
    best_yes_ptq = next((r for r in yes_ptq if r["tag"] not in fixed), yes_ptq[0])

    six = [
        ("Best no-PTQ (PGD)",  best_no_ptq),
        ("Best yes-PTQ (PGD)", best_yes_ptq),
        ("Fixed: extreme compression",    fixed_rows.get(fixed[0])),
        ("Fixed: moderate accuracy",      fixed_rows.get(fixed[1])),
        ("Fixed: compact+accurate",       fixed_rows.get(fixed[2])),
    ]
    # Deduplicate — if best_no_ptq or best_yes_ptq overlaps with a fixed model
    seen_tags = set()
    final_six = []
    for label, r in six:
        if r is None:
            continue
        if r["tag"] not in seen_tags:
            final_six.append((label, r))
            seen_tags.add(r["tag"])
    # Fill to 6 if dedup removed one
    for r in no_ptq + yes_ptq:
        if len(final_six) >= 6:
            break
        if r["tag"] not in seen_tags:
            final_six.append(("Additional pick", r))
            seen_tags.add(r["tag"])

    print("\n" + "="*70)
    print("FINAL 6 FOR FGSM AT + EVALUATION")
    print("=" * 70)
    for label, r in final_six:
        ptq_str = "PTQ" if _norm_ptq(r["ptq"]) else "noPTQ"
        print(f"  [{ptq_str}] {label}")
        print(f"    {r['tag']}")
        print(f"    F1={_f(r['final_f1']):.3f}  pgd_adv={_f(r['pgd_adv_acc']):.3f}  "
              f"size={_f(r.get('tflite_size_kb')):.1f}KB")

    # Write summary markdown
    md_lines = [
        "# PGD Sweep Results — Final 6 for FGSM",
        "",
        f"- Surrogate attacker: `{ATTACKER_H5.name}`",
        f"- Epsilon: {args.epsilon}  PGD steps: {args.pgd_steps}",
        f"- Eval samples: {args.n_eval}",
        "",
        "## Final 6 Models",
        "",
        "| Role | Tag | PTQ | F1 | PGD Adv Acc | Size KB |",
        "|------|-----|-----|----|-------------|---------|",
    ]
    for label, r in final_six:
        ptq_str = "Yes" if _norm_ptq(r["ptq"]) else "No"
        md_lines.append(
            f"| {label} | `{r['tag']}` | {ptq_str} "
            f"| {_f(r['final_f1']):.3f} | {_f(r['pgd_adv_acc']):.3f} "
            f"| {_f(r.get('tflite_size_kb')):.1f} |"
        )
    md_lines += [
        "",
        "## Top 5 No-PTQ by PGD Adv Acc",
        "",
        "| Tag | F1 | PGD Adv Acc | Size KB |",
        "|-----|----|-------------|---------|",
    ]
    for r in no_ptq[:5]:
        md_lines.append(
            f"| `{r['tag']}` | {_f(r['final_f1']):.3f} "
            f"| {_f(r['pgd_adv_acc']):.3f} | {_f(r.get('tflite_size_kb')):.1f} |"
        )
    md_lines += [
        "",
        "## Top 5 Yes-PTQ by PGD Adv Acc",
        "",
        "| Tag | F1 | PGD Adv Acc | Size KB |",
        "|-----|----|-------------|---------|",
    ]
    for r in yes_ptq[:5]:
        md_lines.append(
            f"| `{r['tag']}` | {_f(r['final_f1']):.3f} "
            f"| {_f(r['pgd_adv_acc']):.3f} | {_f(r.get('tflite_size_kb')):.1f} |"
        )

    md_path = OUT_DIR / "pgd_sweep_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nWrote {md_path}")

    # Save final 6 list for FGSM script
    six_path = OUT_DIR / "fgsm_final_six.json"
    with open(six_path, "w", encoding="utf-8") as f:
        json.dump([{"role": label, "tag": r["tag"],
                    "tflite": str(tag_to_tflite(r["tag"])),
                    "ptq": _norm_ptq(r["ptq"]),
                    "final_f1": _f(r["final_f1"]),
                    "pgd_adv_acc": _f(r["pgd_adv_acc"])}
                   for label, r in final_six], f, indent=2)
    print(f"Wrote {six_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
