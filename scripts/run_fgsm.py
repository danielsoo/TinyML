#!/usr/bin/env python3
"""
Run FGSM attack and write fgsm_report.md + fgsm_results.json to a run directory.
Used by run.py (Step 3b) so one run produces analysis/, eval/, fgsm/, models/.

Usage:
  python scripts/run_fgsm.py --model models/global_model.h5 --config config/federated.yaml --output-dir data/processed/runs/v18/2026-02-10_00-16-53/fgsm
  python scripts/run_fgsm.py --model path/to/model.h5 --fgsm-config config/fgsm.yaml --output-dir ./fgsm_out
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml
import tensorflow as tf
from src.data.loader import load_dataset
from src.adversarial.fgsm_hook import (
    generate_fgsm_attack,
    evaluate_attack_success,
    tune_epsilon,
    generate_adversarial_dataset,
)


def load_model_with_qat(model_path: str, use_qat: bool) -> tf.keras.Model:
    if use_qat:
        try:
            import tensorflow_model_optimization as tfmot
            with tfmot.quantization.keras.quantize_scope():
                model = tf.keras.models.load_model(model_path, compile=False)
        except Exception:
            model = tf.keras.models.load_model(model_path, compile=False)
    else:
        model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Run FGSM and write report to run dir")
    parser.add_argument("--model", required=True, help="Path to .h5 model")
    parser.add_argument("--config", default="config/federated.yaml", help="Federated/config for data")
    parser.add_argument("--fgsm-config", default="config/fgsm.yaml", help="FGSM params (epsilon, threshold)")
    parser.add_argument("--output-dir", required=True, help="Write fgsm_report.md and fgsm_results.json here")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fgsm_config, encoding="utf-8") as f:
        fgsm_cfg = yaml.safe_load(f)
    with open(args.config, encoding="utf-8") as f:
        fed_cfg = yaml.safe_load(f)

    data_cfg = fgsm_cfg.get("data") or fed_cfg.get("data", {})
    attack_cfg = fgsm_cfg.get("attack", {})
    eval_cfg = fgsm_cfg.get("eval", {})
    model_cfg = fgsm_cfg.get("model", {})

    dataset_name = data_cfg.get("name", "cicids2017")
    data_path = data_cfg.get("path", "data/raw/CIC-IDS2017")
    max_samples = data_cfg.get("max_samples", 2000000)
    threshold = attack_cfg.get("prediction_threshold", 0.3)
    epsilon_values = attack_cfg.get("epsilon_values", [0.01, 0.05, 0.1, 0.15, 0.2])
    epsilon_default = attack_cfg.get("epsilon_default", 0.1)
    tune_target = attack_cfg.get("tune_target_success_rate", 0.5)
    test_subset = eval_cfg.get("test_subset_size", 5000)
    tune_subset = eval_cfg.get("tune_subset_size", 5000)
    adv_subset = eval_cfg.get("adv_subset_size", 20000)
    batch_size = eval_cfg.get("batch_size", 64)

    use_qat = fed_cfg.get("federated", {}).get("use_qat", False)
    model = load_model_with_qat(str(model_path), use_qat)

    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")
    _, _, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    if hasattr(x_test, "values"):
        x_test = x_test.values
    if hasattr(y_test, "values"):
        y_test = y_test.values
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    x_test = x_test[:max(min(max_samples, len(x_test)), adv_subset + 1000)]
    y_test = y_test[:len(x_test)]

    def eval_success(x_orig, x_adv, y, thresh=threshold):
        return evaluate_attack_success(model, x_orig, x_adv, y, threshold=thresh)

    # Epsilon sweep
    sweep = []
    n_sw = min(test_subset, len(x_test))
    x_sw, y_sw = x_test[:n_sw], y_test[:n_sw]
    for eps in epsilon_values:
        x_adv, _ = generate_fgsm_attack(model, x_sw, y_sw, eps=eps)
        m = eval_success(x_sw, x_adv, y_sw)
        sweep.append({
            "epsilon": eps,
            "original_accuracy": m["original_accuracy"],
            "adversarial_accuracy": m["adversarial_accuracy"],
            "attack_success_rate": m["attack_success_rate"],
            "avg_perturbation": m["avg_perturbation"] if np.isfinite(m["avg_perturbation"]) else None,
        })

    # Tuning
    n_tune = min(tune_subset, len(x_test))
    tune_res = tune_epsilon(model, x_test[:n_tune], y_test[:n_tune], epsilon_range=epsilon_values, target_success_rate=tune_target)
    best_eps = tune_res["best_epsilon"]

    # Full adversarial eval
    n_adv = min(adv_subset, len(x_test))
    x_adv_full, y_adv_full = generate_adversarial_dataset(
        model, x_test[:n_adv], y_test[:n_adv], eps=epsilon_default, batch_size=batch_size
    )
    final = eval_success(x_test[:n_adv], x_adv_full, y_adv_full)

    generated = datetime.now().isoformat()
    results = {
        "model_path": str(model_path),
        "dataset": dataset_name,
        "data_path": data_path,
        "max_samples": max_samples,
        "threshold": threshold,
        "epsilon_sweep": sweep,
        "tuning": {"best_epsilon": best_eps, "target_success_rate": tune_target},
        "final": {
            "epsilon_used": epsilon_default,
            "original_accuracy": final["original_accuracy"],
            "adversarial_accuracy": final["adversarial_accuracy"],
            "attack_success_rate": final["attack_success_rate"],
            "attack_success_count": final["attack_success_count"],
            "total_samples": final["total_samples"],
            "avg_perturbation": final["avg_perturbation"] if np.isfinite(final["avg_perturbation"]) else None,
            "max_perturbation": final["max_perturbation"] if np.isfinite(final["max_perturbation"]) else None,
        },
        "generated": generated,
    }

    with open(out_dir / "fgsm_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    lines = [
        "# FGSM Attack Report",
        "",
        f"- **Model:** `{model_path}`",
        f"- **Dataset:** {dataset_name}",
        f"- **Data path:** {data_path}",
        f"- **Max samples:** {max_samples}",
        f"- **Prediction threshold:** {threshold}",
        f"- **Generated:** {generated}",
        "",
        "## Epsilon sweep",
        "",
        "| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |",
        "|---------|--------------|---------|--------------|-------------|",
    ]
    for r in sweep:
        avg_p = r["avg_perturbation"]
        avg_s = f"{avg_p:.6f}" if avg_p is not None else "nan"
        lines.append(f"| {r['epsilon']} | {r['original_accuracy']:.4f} | {r['adversarial_accuracy']:.4f} | {r['attack_success_rate']:.4f} | {avg_s} |")
    lines.extend([
        "",
        "## Epsilon tuning",
        "",
        f"- **Best epsilon:** {best_eps:.4f}",
        f"- **Target success rate:** {tune_target:.2f}",
        "",
        "## Final evaluation (adversarial dataset)",
        "",
        f"- **Epsilon used:** {epsilon_default}",
        f"- **Original Accuracy:** {final['original_accuracy']:.4f} ({final['original_accuracy']*100:.2f}%)",
        f"- **Adversarial Accuracy:** {final['adversarial_accuracy']:.4f} ({final['adversarial_accuracy']*100:.2f}%)",
        f"- **Attack Success Rate:** {final['attack_success_rate']:.4f} ({final['attack_success_rate']*100:.2f}%)",
        f"- **Attack Success Count:** {final['attack_success_count']}/{final['total_samples']}",
    ])
    ap = final.get("avg_perturbation")
    mp = final.get("max_perturbation")
    lines.append(f"- **Average Perturbation:** {ap:.6f}" if ap is not None and np.isfinite(ap) else "- **Average Perturbation:** nan")
    lines.append(f"- **Max Perturbation:** {mp:.6f}" if mp is not None and np.isfinite(mp) else "- **Max Perturbation:** nan")
    (out_dir / "fgsm_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'fgsm_report.md'} and fgsm_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
