#!/usr/bin/env python3
"""
Run FGSM attack and write fgsm_report.md + fgsm_results.json to a run directory.
Compares all models (Keras + TFLite) on the same adversarial examples (generated with the first Keras model).

Usage:
  python scripts/run_fgsm.py --model models/global_model.h5 --config config/federated.yaml --output-dir ./fgsm_out
  python scripts/run_fgsm.py --models models/global_model.h5 models/tflite/saved_model_original.tflite ... --config ... --output-dir ./fgsm_out
  (If --models not given, uses config: global_model.h5 + evaluation.ratio_sweep_models for comparison.)
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


def _predict_proba_keras(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    out = model.predict(x, verbose=0)
    if out.ndim > 1:
        out = out[:, -1] if out.shape[-1] > 1 else out.ravel()
    return np.asarray(out, dtype=np.float64)


def _predict_proba_tflite(interpreter, input_details, output_details, x: np.ndarray) -> np.ndarray:
    in_dtype = input_details[0]["dtype"]
    out_dtype = output_details[0]["dtype"]
    out_list = []
    for i in range(len(x)):
        batch = x[i : i + 1]
        if in_dtype == np.int8:
            s = input_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
            z = input_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
            batch = (batch / s + z).astype(np.int8)
        else:
            batch = batch.astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], batch)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        if out_dtype == np.int8:
            s = output_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
            z = output_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
            out = s * (out.astype(np.float32) - z)
        out_list.append(out)
    y_prob = np.concatenate(out_list, axis=0)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, -1] if y_prob.shape[-1] > 1 else y_prob.ravel()
    return np.asarray(y_prob, dtype=np.float64)


def evaluate_attack_success_with_predictor(pred_fn, x_original, x_adversarial, y_true, threshold=0.5):
    """Same metrics as evaluate_attack_success but with a generic pred_fn(x) -> proba (n,)."""
    pred_orig = pred_fn(x_original)
    pred_adv = pred_fn(x_adversarial)
    pred_orig = np.asarray(pred_orig).ravel()
    pred_adv = np.asarray(pred_adv).ravel()
    y_flat = np.asarray(y_true).ravel().astype(int)
    pred_orig_binary = (pred_orig >= threshold).astype(int)
    pred_adv_binary = (pred_adv >= threshold).astype(int)
    total = len(y_flat)
    orig_correct = (pred_orig_binary == y_flat).sum()
    adv_correct = (pred_adv_binary == y_flat).sum()
    attack_success = orig_correct - adv_correct
    perturbation = np.abs(np.asarray(x_adversarial, dtype=np.float64) - np.asarray(x_original, dtype=np.float64))
    perturbation = np.nan_to_num(perturbation, nan=0.0, posinf=0.0, neginf=0.0)
    avg_p = float(np.mean(perturbation)) if perturbation.size > 0 else 0.0
    max_p = float(np.max(perturbation)) if perturbation.size > 0 else 0.0
    return {
        "original_accuracy": float(orig_correct / total) if total else 0.0,
        "adversarial_accuracy": float(adv_correct / total) if total else 0.0,
        "attack_success_rate": float(attack_success / total) if total else 0.0,
        "attack_success_count": int(attack_success),
        "total_samples": int(total),
        "avg_perturbation": avg_p,
        "max_perturbation": max_p,
    }


def resolve_model_list(args, fed_cfg, out_dir_base: Path):
    """Resolve list of model paths: --models, or --model single, or from config (global_model.h5 + ratio_sweep_models)."""
    if getattr(args, "models", None) and len(args.models) > 0:
        return [Path(p) for p in args.models]
    if getattr(args, "model", None) and args.model:
        return [Path(args.model)]
    # From config: Keras + ratio_sweep_models (only existing)
    eval_cfg = fed_cfg.get("evaluation", {})
    ratio_models = eval_cfg.get("ratio_sweep_models") or []
    if isinstance(ratio_models, str):
        ratio_models = [ratio_models]
    # Prefer run_dir/models/ if we're writing to run_dir/fgsm (out_dir_base might be .../runs/v19/datetime/fgsm)
    run_dir = out_dir_base.parent if out_dir_base.name == "fgsm" else None
    candidates = []
    for h5_name in ["global_model.h5", "models/global_model.h5"]:
        for base in ([run_dir] if run_dir else []) + [None]:
            p = (Path(base) / h5_name) if base else Path(h5_name)
            if p and p.exists():
                candidates.append(p.resolve())
                break
        if candidates:
            break
    if not candidates:
        candidates = [Path("models/global_model.h5")]
    for m in ratio_models:
        m = m.strip() if isinstance(m, str) else str(m)
        for base in ([run_dir] if run_dir else []) + [None]:
            p = (Path(base) / m) if base else Path(m)
            if p.exists():
                candidates.append(p.resolve())
                break
    return list(dict.fromkeys(candidates))  # unique, order preserved


def get_display_name(model_path: Path) -> str:
    """Short display name for report (e.g. global_model.h5 -> Keras (global), saved_model_original -> Original)."""
    stem = model_path.stem
    if stem == "global_model":
        return "Keras (global_model.h5)"
    if stem == "saved_model_original":
        return "Original (TFLite)"
    if stem == "saved_model_qat_pruned_float32":
        return "QAT+Prune only"
    if stem == "saved_model_qat_ptq":
        return "QAT+PTQ"
    if stem == "saved_model_no_qat_ptq":
        return "noQAT+PTQ"
    if stem == "saved_model_traditional_qat":
        return "Traditional+QAT (no QAT in FL, QAT fine-tune)"
    if stem == "saved_model_pruned_qat":
        return "Compressed (QAT)"
    if stem == "saved_model_pruned_quantized":
        return "Compressed (PTQ)"
    return stem


def main():
    parser = argparse.ArgumentParser(description="Run FGSM and write report (single or multi-model comparison)")
    parser.add_argument("--model", default="", help="Single model path (.h5) for backward compat")
    parser.add_argument("--models", nargs="*", default=None, help="Multiple model paths (.h5 and .tflite) for comparison")
    parser.add_argument("--config", default="config/federated.yaml", help="Federated/config for data")
    parser.add_argument("--fgsm-config", default="config/fgsm.yaml", help="FGSM params (epsilon, threshold)")
    parser.add_argument("--output-dir", required=True, help="Write fgsm_report.md and fgsm_results.json here")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fgsm_config, encoding="utf-8") as f:
        fgsm_cfg = yaml.safe_load(f)
    with open(args.config, encoding="utf-8") as f:
        fed_cfg = yaml.safe_load(f)

    data_cfg = fgsm_cfg.get("data") or fed_cfg.get("data", {})
    attack_cfg = fgsm_cfg.get("attack", {})
    eval_cfg = fgsm_cfg.get("eval", {})
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

    model_paths = resolve_model_list(args, fed_cfg, out_dir)
    # Only existing paths
    model_paths = [p for p in model_paths if p.exists()]
    if not model_paths:
        print("No model paths found.", file=sys.stderr)
        return 1

    attack_model_path = None
    for p in model_paths:
        if str(p).endswith(".h5"):
            attack_model_path = p
            break
    if not attack_model_path:
        print("No .h5 model found for FGSM attack generation. Need at least one Keras model.", file=sys.stderr)
        return 1

    use_qat = fed_cfg.get("federated", {}).get("use_qat", False)
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
    x_test = x_test[: max(min(max_samples, len(x_test)), adv_subset + 1000)]
    y_test = y_test[: len(x_test)]

    # 1) Generate adversarial examples using the Keras (attack) model
    attack_model = load_model_with_qat(str(attack_model_path), use_qat)
    n_sw = min(test_subset, len(x_test))
    x_sw, y_sw = x_test[:n_sw], y_test[:n_sw]
    n_adv = min(adv_subset, len(x_test))
    x_adv_full, y_adv_full = generate_adversarial_dataset(
        attack_model, x_test[:n_adv], y_test[:n_adv], eps=epsilon_default, batch_size=batch_size
    )
    x_orig_eval = x_test[:n_adv]
    y_eval = y_adv_full  # same labels

    perturbation = np.abs(np.asarray(x_adv_full, dtype=np.float64) - np.asarray(x_orig_eval, dtype=np.float64))
    perturbation = np.nan_to_num(perturbation, nan=0.0, posinf=0.0, neginf=0.0)
    avg_perturb = float(np.mean(perturbation)) if perturbation.size > 0 else 0.0
    max_perturb = float(np.max(perturbation)) if perturbation.size > 0 else 0.0

    # 2) Epsilon sweep and tuning (attack model only) for report
    sweep = []
    for eps in epsilon_values:
        x_adv_sw, _ = generate_fgsm_attack(attack_model, x_sw, y_sw, eps=eps)
        m = evaluate_attack_success(attack_model, x_sw, x_adv_sw, y_sw, threshold=threshold)
        sweep.append({
            "epsilon": eps,
            "original_accuracy": m["original_accuracy"],
            "adversarial_accuracy": m["adversarial_accuracy"],
            "attack_success_rate": m["attack_success_rate"],
            "avg_perturbation": m["avg_perturbation"] if np.isfinite(m["avg_perturbation"]) else None,
        })
    n_tune = min(tune_subset, len(x_test))
    tune_res = tune_epsilon(
        attack_model, x_test[:n_tune], y_test[:n_tune],
        epsilon_range=epsilon_values, target_success_rate=tune_target,
    )
    best_eps = tune_res["best_epsilon"]

    # 3) Evaluate each model on (x_orig_eval, x_adv_full, y_eval)
    comparison = []
    for model_path in model_paths:
        path_str = str(model_path)
        display_name = get_display_name(model_path)
        if path_str.endswith(".h5"):
            model = load_model_with_qat(path_str, use_qat)
            pred_fn = lambda x, m=model: _predict_proba_keras(m, x)
        else:
            interp = tf.lite.Interpreter(model_path=path_str, experimental_preserve_all_tensors=True)
            interp.allocate_tensors()
            in_d = interp.get_input_details()
            out_d = interp.get_output_details()
            pred_fn = lambda x, i=interp, id_=in_d, od_=out_d: _predict_proba_tflite(i, id_, od_, x)
        metrics = evaluate_attack_success_with_predictor(
            pred_fn, x_orig_eval, x_adv_full, y_eval, threshold=threshold
        )
        comparison.append({
            "model_path": path_str,
            "display_name": display_name,
            "original_accuracy": metrics["original_accuracy"],
            "adversarial_accuracy": metrics["adversarial_accuracy"],
            "attack_success_rate": metrics["attack_success_rate"],
            "attack_success_count": metrics["attack_success_count"],
            "total_samples": metrics["total_samples"],
            "avg_perturbation": avg_perturb,
            "max_perturbation": max_perturb,
        })

    generated = datetime.now().isoformat()
    results = {
        "dataset": dataset_name,
        "data_path": data_path,
        "max_samples": max_samples,
        "threshold": threshold,
        "epsilon_used": epsilon_default,
        "attack_model": str(attack_model_path),
        "epsilon_sweep": sweep,
        "tuning": {"best_epsilon": best_eps, "target_success_rate": tune_target},
        "comparison": comparison,
        "generated": generated,
    }

    with open(out_dir / "fgsm_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Markdown: comparison table first (like other reports), then epsilon sweep / final for attack model
    lines = [
        "# FGSM Attack Report",
        "",
        f"- **Dataset:** {dataset_name}",
        f"- **Data path:** {data_path}",
        f"- **Max samples:** {max_samples}",
        f"- **Prediction threshold:** {threshold}",
        f"- **Adversarial examples generated with:** `{attack_model_path}`",
        f"- **Epsilon used:** {epsilon_default}",
        f"- **Generated:** {generated}",
        "",
        "## Model comparison (same adversarial examples)",
        "",
        "| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |",
        "|-------|--------------|---------|--------------|-------------|-------------|",
    ]
    for c in comparison:
        lines.append(
            f"| {c['display_name']} | {c['original_accuracy']:.4f} | {c['adversarial_accuracy']:.4f} | "
            f"{c['attack_success_rate']:.4f} | {c['avg_perturbation']:.6f} | {c['max_perturbation']:.6f} |"
        )
    lines.extend([
        "",
        "## Epsilon sweep (attack model)",
        "",
        "| Epsilon | Original Acc | Adv Acc | Success Rate | Avg Perturb |",
        "|---------|--------------|---------|--------------|-------------|",
    ])
    for r in sweep:
        avg_p = r["avg_perturbation"]
        avg_s = f"{avg_p:.6f}" if avg_p is not None else "nan"
        lines.append(
            f"| {r['epsilon']} | {r['original_accuracy']:.4f} | {r['adversarial_accuracy']:.4f} | "
            f"{r['attack_success_rate']:.4f} | {avg_s} |"
        )
    lines.extend([
        "",
        "## Epsilon tuning",
        "",
        f"- **Best epsilon:** {best_eps:.4f}",
        f"- **Target success rate:** {tune_target:.2f}",
        "",
    ])
    (out_dir / "fgsm_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'fgsm_report.md'} and fgsm_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
