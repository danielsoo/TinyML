#!/usr/bin/env python3
"""
Run PGD (or FGSM) attack and write pgd_report.md + pgd_results.json to a run directory.
Compares all models (Keras + TFLite) on the same adversarial examples (generated with the first Keras model).
Attack type (PGD vs FGSM) follows config adversarial_training.attack.

Usage:
  python scripts/run_pgd.py --model models/global_model.h5 --config config/federated.yaml --output-dir ./pgd_out
  python scripts/run_pgd.py --models models/global_model.h5 models/tflite/saved_model_original.tflite ... --config ... --output-dir ./pgd_out
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
    generate_pgd_attack,
    evaluate_attack_success,
    tune_epsilon,
    generate_adversarial_dataset,
    generate_adversarial_dataset_pgd,
)


def load_model_with_qat(model_path: str, use_qat: bool) -> tf.keras.Model:
    # Try 1: tfmot quantize_scope (handles QAT-saved .h5 files)
    try:
        import tensorflow_model_optimization as tfmot
        with tfmot.quantization.keras.quantize_scope():
            model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    except ImportError:
        pass
    except Exception:
        pass

    # Try 2: plain load (works if model has no QAT layers)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model
    except Exception:
        pass

    # Try 3: model has QAT layers but tfmot unavailable — load weights only into a
    # fresh equivalent architecture extracted from the .h5 file's weight arrays.
    import h5py, numpy as np
    print(f"  ℹ️  QAT model detected but tfmot unavailable. "
          f"Rebuilding plain model from weights only: {model_path}")
    with h5py.File(model_path, "r") as f:
        # Collect Dense layer weight arrays in order
        layer_weights = []
        for key in sorted(f["model_weights"].keys()):
            grp = f["model_weights"][key]
            sublayers = [k for k in grp.keys() if len(grp[k].keys()) > 0]
            for sub in sorted(sublayers):
                ws = [grp[sub][w][()] for w in sorted(grp[sub].keys())]
                if ws:
                    layer_weights.append(ws)

    # Infer architecture from weight shapes: each Dense = (in, out) kernel + (out,) bias
    dense_specs = []
    i = 0
    while i < len(layer_weights):
        ws = layer_weights[i]
        if len(ws) >= 2 and ws[0].ndim == 2:
            in_dim, out_dim = ws[0].shape
            dense_specs.append((out_dim, ws))
            i += 1
        else:
            i += 1

    if not dense_specs:
        raise RuntimeError(f"Could not extract weights from QAT model: {model_path}")

    layers_list = []
    for idx, (units, _) in enumerate(dense_specs):
        if idx == 0:
            layers_list.append(tf.keras.layers.Dense(
                units, activation="relu" if idx < len(dense_specs) - 1 else "sigmoid",
                input_shape=(dense_specs[0][1][0].shape[0],)))
        elif idx < len(dense_specs) - 1:
            layers_list.append(tf.keras.layers.Dense(units, activation="relu"))
        else:
            layers_list.append(tf.keras.layers.Dense(units, activation="sigmoid"))

    model = tf.keras.Sequential(layers_list)
    for layer, (_, ws) in zip([l for l in model.layers if "dense" in l.name.lower()], dense_specs):
        layer.set_weights(ws[:2])  # kernel + bias only

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def _predict_proba_keras(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    out = model.predict(x, verbose=0)
    if out.ndim > 1:
        out = out[:, -1] if out.shape[-1] > 1 else out.ravel()
    return np.asarray(out, dtype=np.float64)


def _tflite_input_feature_dim(input_details) -> int | None:
    """Return expected feature dimension from TFLite input (last dim, or None if dynamic)."""
    sh = input_details[0].get("shape")
    if not sh or len(sh) < 2:
        return None
    # shape is typically [batch, features] or [1, features]
    feat = sh[-1]
    return int(feat) if feat > 0 else None


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
    """Resolve list of model paths: --models-file, --models, --model single, or from config."""
    models_file = getattr(args, "models_file", None)
    if models_file and Path(models_file).exists():
        paths = []
        with open(models_file, encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p and not p.startswith("#"):
                    paths.append(Path(p))
        if paths:
            return paths
    if getattr(args, "models", None) and len(args.models) > 0:
        return [Path(p) for p in args.models]
    if getattr(args, "model", None) and args.model:
        return [Path(args.model)]
    # From config: Keras + ratio_sweep_models (only existing)
    eval_cfg = fed_cfg.get("evaluation", {})
    ratio_models = eval_cfg.get("ratio_sweep_models") or []
    if isinstance(ratio_models, str):
        ratio_models = [ratio_models]
    # Prefer run_dir/models/ if we're writing to run_dir/pgd (out_dir_base might be .../runs/v19/datetime/pgd)
    run_dir = out_dir_base.parent if out_dir_base.name == "pgd" else None
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
    """Short display name for report (match compression_analysis / PGD)."""
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
    parser = argparse.ArgumentParser(description="Run PGD (or FGSM) and write report (single or multi-model comparison)")
    parser.add_argument("--model", default="", help="Single model path (.h5) for backward compat")
    parser.add_argument("--models", nargs="*", default=None, help="Multiple model paths (.h5 and .tflite) for comparison")
    parser.add_argument("--models-file", default=None, help="Path to file with one model path per line (first .h5 = attack model); for 48 sweep PGD")
    parser.add_argument("--config", default="config/federated.yaml", help="Federated/config for data + adversarial_training.attack")
    parser.add_argument("--fgsm-config", default="config/fgsm.yaml", help="Attack params (epsilon, threshold); also used for PGD epsilon range")
    parser.add_argument("--output-dir", required=True, help="Write pgd_report.md and pgd_results.json here")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fgsm_config, encoding="utf-8") as f:
        fgsm_cfg = yaml.safe_load(f)
    with open(args.config, encoding="utf-8") as f:
        fed_cfg = yaml.safe_load(f)
    at_cfg = fed_cfg.get("adversarial_training", {})
    attack_type = (at_cfg.get("attack") or "fgsm").strip().lower()
    pgd_steps = int(at_cfg.get("pgd_steps", 10))
    pgd_alpha = at_cfg.get("pgd_alpha")
    at_epsilon = at_cfg.get("epsilon")

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
        print("No .h5 model found for PGD attack generation. Need at least one Keras model.", file=sys.stderr)
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

    # 1) Generate adversarial examples using the Keras (attack) model (PGD or FGSM per config)
    attack_model = load_model_with_qat(str(attack_model_path), use_qat)
    n_sw = min(test_subset, len(x_test))
    x_sw, y_sw = x_test[:n_sw], y_test[:n_sw]
    n_adv = min(adv_subset, len(x_test))
    eps_adv = at_epsilon if at_epsilon is not None else epsilon_default
    if attack_type == "pgd":
        x_adv_full, y_adv_full = generate_adversarial_dataset_pgd(
            attack_model, x_test[:n_adv], y_test[:n_adv],
            eps=eps_adv, steps=pgd_steps, alpha=pgd_alpha, batch_size=batch_size
        )
    else:
        x_adv_full, y_adv_full = generate_adversarial_dataset(
            attack_model, x_test[:n_adv], y_test[:n_adv], eps=eps_adv, batch_size=batch_size
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
        if attack_type == "pgd":
            x_adv_sw, _ = generate_pgd_attack(
                attack_model, x_sw, y_sw, eps=eps, steps=pgd_steps, alpha=pgd_alpha
            )
        else:
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
    data_feature_dim = x_orig_eval.shape[1] if x_orig_eval.ndim >= 2 else x_orig_eval.shape[0]
    comparison = []
    for model_path in model_paths:
        path_str = str(model_path)
        display_name = get_display_name(model_path)
        if path_str.endswith(".h5"):
            model = load_model_with_qat(path_str, use_qat)
            pred_fn = lambda x, m=model: _predict_proba_keras(m, x)
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
        else:
            interp = tf.lite.Interpreter(model_path=path_str, experimental_preserve_all_tensors=True)
            interp.allocate_tensors()
            in_d = interp.get_input_details()
            out_d = interp.get_output_details()
            model_feat = _tflite_input_feature_dim(in_d)
            if model_feat is not None and model_feat != data_feature_dim:
                print(
                    f"  [PGD] Skip {model_path.name}: input dimension mismatch "
                    f"(model expects {model_feat}, data has {data_feature_dim})",
                    file=sys.stderr,
                )
                comparison.append({
                    "model_path": path_str,
                    "display_name": display_name + " [dim_mismatch]",
                    "original_accuracy": None,
                    "adversarial_accuracy": None,
                    "attack_success_rate": None,
                    "attack_success_count": None,
                    "total_samples": len(y_eval),
                    "avg_perturbation": avg_perturb,
                    "max_perturbation": max_perturb,
                })
                continue
            pred_fn = lambda x, i=interp, id_=in_d, od_=out_d: _predict_proba_tflite(i, id_, od_, x)
            try:
                metrics = evaluate_attack_success_with_predictor(
                    pred_fn, x_orig_eval, x_adv_full, y_eval, threshold=threshold
                )
            except (ValueError, RuntimeError) as e:
                if "shape" in str(e).lower() or "dimension" in str(e).lower() or "incompatible" in str(e).lower():
                    print(
                        f"  [PGD] Skip {model_path.name}: input/output error — {e}",
                        file=sys.stderr,
                    )
                    comparison.append({
                        "model_path": path_str,
                        "display_name": display_name + " [eval_error]",
                        "original_accuracy": None,
                        "adversarial_accuracy": None,
                        "attack_success_rate": None,
                        "attack_success_count": None,
                        "total_samples": len(y_eval),
                        "avg_perturbation": avg_perturb,
                        "max_perturbation": max_perturb,
                    })
                    continue
                raise
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

    with open(out_dir / "pgd_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Experiment config from run dir (if this report is under runs/<version>/<run_id>/pgd)
    run_config_path = out_dir.parent / "run_config.yaml"
    run_cfg = {}
    if run_config_path.exists():
        try:
            with open(run_config_path, encoding="utf-8") as f:
                run_cfg = yaml.safe_load(f) or {}
        except Exception:
            pass
    eval_cfg = run_cfg.get("evaluation", {})
    at_cfg = run_cfg.get("adversarial_training", {})
    _pgd_top_n = eval_cfg.get("pgd_top_n", eval_cfg.get("fgsm_top_n", "-"))
    _pgd_metric = eval_cfg.get("pgd_metric", eval_cfg.get("fgsm_metric", "-"))

    # Markdown: comparison table first (like other reports), then epsilon sweep / final for attack model
    lines = [
        "# PGD Attack Report",
        "",
        f"- **Dataset:** {dataset_name}",
        f"- **Data path:** {data_path}",
        f"- **Max samples:** {max_samples}",
        f"- **Prediction threshold:** {threshold}",
        f"- **Adversarial examples generated with:** `{attack_model_path}`",
        f"- **Attack type:** {attack_type}",
        f"- **Epsilon used:** {epsilon_default}",
        f"- **Generated:** {generated}",
        "",
        "## 실험 설정 (이 실험에 사용된 요소)",
        "",
        "| 항목 | 값 |",
        "|------|-----|",
        f"| **PGD top-N** | {_pgd_top_n} |",
        f"| **PGD metric** | {_pgd_metric} |",
        f"| **AT enabled** | {at_cfg.get('enabled', '-')} |",
        f"| **AT attack** | {at_cfg.get('attack', '-')} |",
        f"| **AT epsilon** | {at_cfg.get('epsilon', '-')} |",
        f"| **평가 모델 수** | {len(model_paths)} |",
        "",
    ]
    lines.append("**평가한 모델:**")
    for p in model_paths:
        lines.append(f"- `{p}`")
    lines.extend([
        "",
        "전체 실험 설정: 동일 run 디렉터리의 `run_config.yaml` 및 `experiment_record.md` 참조.",
        "",
        "## Model comparison (same adversarial examples)",
        "",
        "| Model | Original Acc | Adv Acc | Success Rate | Avg Perturb | Max Perturb |",
        "|-------|--------------|---------|--------------|-------------|-------------|",
    ])
    for c in comparison:
        oa = c["original_accuracy"]
        aa = c["adversarial_accuracy"]
        sr = c["attack_success_rate"]
        oa_s = f"{oa:.4f}" if oa is not None else "-"
        aa_s = f"{aa:.4f}" if aa is not None else "-"
        sr_s = f"{sr:.4f}" if sr is not None else "-"
        lines.append(
            f"| {c['display_name']} | {oa_s} | {aa_s} | {sr_s} | "
            f"{c['avg_perturbation']:.6f} | {c['max_perturbation']:.6f} |"
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
    (out_dir / "pgd_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_dir / 'pgd_report.md'} and pgd_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
