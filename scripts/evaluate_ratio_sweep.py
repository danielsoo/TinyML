#!/usr/bin/env python3
"""
Test set ratio sweep: evaluate model(s) from 100% normal (100:0) to 100% attack (0:100)
in 10% steps. Reports Accuracy, Precision, Recall, F1, Normal Recall, Normal Precision per ratio.
With --models: runs sweep for all models and writes one comparison report (Model x Ratio).

Usage:
  python scripts/evaluate_ratio_sweep.py --config config/federated_local.yaml --model src/models/global_model.h5
  python scripts/evaluate_ratio_sweep.py --models m1.h5 m2.tflite ... --config ... --report ratio_sweep_report.md
"""
import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml


def get_display_name(model_path: Path) -> str:
    """Short display name for report (match compression_analysis / FGSM)."""
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
    if stem == "saved_model_pruned_qat":
        return "Compressed (QAT)"
    if stem == "saved_model_pruned_quantized":
        return "Compressed (PTQ)"
    return stem


def load_data_only(config_path: str):
    """Load config and dataset only; return (x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, dataset_name)."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "cicids2017")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")
    dataset_kwargs["binary"] = data_cfg.get("binary", True)
    from src.data.loader import load_dataset
    print("Loading dataset...")
    _, _, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    idx_normal = np.where(y_test == 0)[0]
    idx_attack = np.where(y_test == 1)[0]
    N_normal, N_attack = len(idx_normal), len(idx_attack)
    print(f"  Test: {len(y_test):,} (Normal={N_normal:,}, Attack={N_attack:,})")
    return x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, dataset_name, cfg


def load_model(model_path: Path, cfg: dict):
    """Load one model; return (model_tuple, display_name). model_tuple is (kind, ...) for _predict_proba."""
    import tensorflow as tf
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"Loading model: {model_path}")
    if str(model_path).endswith(".tflite"):
        interpreter = tf.lite.Interpreter(
            model_path=str(model_path),
            experimental_preserve_all_tensors=True,
        )
        interpreter.allocate_tensors()
        model = ("tflite", interpreter, interpreter.get_input_details(), interpreter.get_output_details())
    else:
        use_qat = cfg.get("federated", {}).get("use_qat", False)
        if use_qat:
            try:
                import tensorflow_model_optimization as tfmot
                with tfmot.quantization.keras.quantize_scope():
                    keras_model = tf.keras.models.load_model(model_path, compile=False)
            except Exception:
                keras_model = tf.keras.models.load_model(model_path, compile=False)
        else:
            try:
                keras_model = tf.keras.models.load_model(model_path, compile=False)
            except Exception as e:
                if "Unknown" in str(e) or "custom" in str(e).lower():
                    from src.models.nets import _focal_loss
                    keras_model = tf.keras.models.load_model(
                        model_path, compile=False,
                        custom_objects={"_focal_loss": _focal_loss, "loss_fn": _focal_loss()},
                    )
                else:
                    raise
        keras_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        model = ("keras", keras_model, None, None)
    return model, get_display_name(model_path)


def load_config_and_data(args):
    """Load config, dataset and single model; return (x_test, y_test, model, dataset_name, ...)."""
    x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, dataset_name, cfg = load_data_only(args.config)
    model, _ = load_model(Path(args.model), cfg)
    return x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, model, dataset_name


def subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, normal_pct, rng):
    """Return eval indices with normal_pct% normal, (100-normal_pct)% attack."""
    if normal_pct >= 100:
        size = min(len(idx_normal), 100000)
        return rng.choice(idx_normal, size=size, replace=False)
    if normal_pct <= 0:
        size = min(len(idx_attack), 100000)
        return rng.choice(idx_attack, size=size, replace=False)
    # normal_pct : (100-normal_pct) = n_normal : n_attack
    n_normal = min(N_normal, int(N_attack * normal_pct / (100 - normal_pct)))
    n_attack = min(N_attack, int(n_normal * (100 - normal_pct) / normal_pct))
    n_normal = min(N_normal, int(n_attack * normal_pct / (100 - normal_pct)))
    keep_n = rng.choice(idx_normal, size=n_normal, replace=False)
    keep_a = rng.choice(idx_attack, size=n_attack, replace=False)
    out = np.concatenate([keep_n, keep_a])
    rng.shuffle(out)
    return out


def _predict_proba(model, x_eval):
    """Get prediction probabilities from Keras or TFLite model."""
    kind = model[0]
    if kind == "keras":
        y_prob = model[1].predict(x_eval, verbose=0)
    else:
        interp, in_d, out_d = model[1], model[2], model[3]
        in_dtype = in_d[0]["dtype"]
        out_dtype = out_d[0]["dtype"]
        out_list = []
        for i in range(len(x_eval)):
            batch = x_eval[i : i + 1]
            if in_dtype == np.int8:
                s = in_d[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
                z = in_d[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
                batch = (batch / s + z).astype(np.int8)
            else:
                batch = batch.astype(np.float32)
            interp.set_tensor(in_d[0]["index"], batch)
            interp.invoke()
            out = interp.get_tensor(out_d[0]["index"])
            if out_dtype == np.int8:
                s = out_d[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
                z = out_d[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
                out = s * (out.astype(np.float32) - z)
            out_list.append(out)
        y_prob = np.concatenate(out_list, axis=0)
    if y_prob.ndim > 1:
        y_prob = y_prob[:, -1] if y_prob.shape[-1] > 1 else y_prob[:, 0]
    return np.asarray(y_prob, dtype=np.float64)


def evaluate_subset(model, x_test, y_test, eval_idx, threshold=0.5, verbose=0):
    """Run model on subset; return metrics dict. threshold: prob >= threshold → Attack (1)."""
    x_eval = x_test[eval_idx]
    y_eval = y_test[eval_idx]
    y_prob = _predict_proba(model, x_eval)
    y_pred = (y_prob >= threshold).astype(np.int32)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    n_classes = len(np.unique(y_eval))
    acc = accuracy_score(y_eval, y_pred)
    prec = precision_score(y_eval, y_pred, average="binary" if n_classes <= 2 else "weighted", zero_division=0)
    rec = recall_score(y_eval, y_pred, average="binary" if n_classes <= 2 else "weighted", zero_division=0)
    f1 = f1_score(y_eval, y_pred, average="binary" if n_classes <= 2 else "weighted", zero_division=0)
    out = {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}
    if n_classes <= 2 and np.any(y_eval == 0) and np.any(y_eval == 1):
        prec_per = precision_score(y_eval, y_pred, average=None, zero_division=0)
        rec_per = recall_score(y_eval, y_pred, average=None, zero_division=0)
        out["normal_recall"] = float(rec_per[0])
        out["normal_precision"] = float(prec_per[0])
    else:
        # only one class in eval set
        if np.all(y_eval == 0):
            out["normal_recall"] = float(np.mean(y_pred == 0))  # of actual normal, % predicted normal
            out["normal_precision"] = 1.0 if np.sum(y_pred == 0) > 0 else 0.0  # of predicted normal, % actual normal
        else:
            out["normal_recall"] = 0.0  # no normal samples
            out["normal_precision"] = 0.0  # no predicted normal, or undefined
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate model(s) across test set ratios (normal%:attack%)")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="", help="Single model path (used if --models not given)")
    parser.add_argument("--models", nargs="*", default=None, help="Multiple model paths for one comparison report")
    parser.add_argument("--ratios", type=str, default="100,90,80,70,60,50,40,30,20,10,0",
                        help="Comma-separated normal%% values (default: 100,90,...,0)")
    parser.add_argument("--out", type=str, default="", help="Optional CSV output path")
    parser.add_argument("--report", type=str, default="", help="Optional Markdown report path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    args = parser.parse_args()

    ratios = [int(x.strip()) for x in args.ratios.split(",")]
    rng = np.random.default_rng(args.seed)

    with open(args.config, encoding="utf-8") as f:
        threshold = float(yaml.safe_load(f).get("evaluation", {}).get("prediction_threshold", 0.5))

    multi = getattr(args, "models", None) and len(args.models) > 0
    if multi:
        # Multi-model: one report comparing all models (Model x Ratio)
        model_paths = [Path(p) for p in args.models if Path(p).exists()]
        if not model_paths:
            print("No model paths found.", file=sys.stderr)
            return 1
        x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, dataset_name, cfg = load_data_only(args.config)
        # Precompute eval indices per ratio (same for all models)
        eval_indices = {}
        for normal_pct in ratios:
            eval_indices[normal_pct] = subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, normal_pct, rng)
        all_results = []
        for model_path in model_paths:
            model, display_name = load_model(model_path, cfg)
            rows = []
            for normal_pct in ratios:
                eval_idx = eval_indices[normal_pct]
                n_n = int(np.sum(y_test[eval_idx] == 0))
                n_a = int(np.sum(y_test[eval_idx] == 1))
                metrics = evaluate_subset(model, x_test, y_test, eval_idx, threshold=threshold, verbose=0)
                row = {
                    "normal_pct": normal_pct,
                    "attack_pct": 100 - normal_pct,
                    "n_normal": n_n,
                    "n_attack": n_a,
                    "n_total": len(eval_idx),
                    **metrics,
                }
                rows.append(row)
            all_results.append((display_name, str(model_path), rows))
            print(f"  Done: {display_name}")
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _write_multi_model_report(all_results, report_path, args.config, ratios)
            print(f"Saved report: {report_path}")
    else:
        # Single model (original behavior)
        model_path = args.model or "src/models/global_model.h5"
        args.model = model_path
        x_test, y_test, idx_normal, idx_attack, N_normal, N_attack, model, _ = load_config_and_data(args)
        print(f"\nSweep: normal% = {ratios} (threshold={threshold})")
        print("=" * 100)
        rows = []
        for normal_pct in ratios:
            eval_idx = subsample_by_ratio(idx_normal, idx_attack, N_normal, N_attack, normal_pct, rng)
            n_n = int(np.sum(y_test[eval_idx] == 0))
            n_a = int(np.sum(y_test[eval_idx] == 1))
            metrics = evaluate_subset(model, x_test, y_test, eval_idx, threshold=threshold, verbose=0)
            row = {
                "normal_pct": normal_pct,
                "attack_pct": 100 - normal_pct,
                "n_normal": n_n,
                "n_attack": n_a,
                "n_total": len(eval_idx),
                **metrics,
            }
            rows.append(row)
            print(f"  Normal {normal_pct:3d}% : Attack {100-normal_pct:3d}%  |  n={len(eval_idx):,} (N={n_n:,}, A={n_a:,})  |  "
                  f"Acc={metrics['accuracy']:.4f}  F1={metrics['f1_score']:.4f}  |  "
                  f"NormRec={metrics.get('normal_recall', 0):.4f}  NormPrec={metrics.get('normal_precision', 0):.4f}")
        print("=" * 100)
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"Saved CSV: {out_path}")
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            _write_markdown_report(rows, report_path, args.model, args.config)
            print(f"Saved report: {report_path}")
    print("Done.")


def _write_multi_model_report(all_results, output_path: Path, config_path: str, ratios: list):
    """Write one report comparing all models (Model x Ratio) for paper / comparison."""
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})
    br = data_cfg.get("balance_ratio")
    br_desc = {1.0: "50:50", 4.0: "normal:attack 8:2", 9.0: "9:1", 19.0: "19:1"}.get(br) if br is not None else None
    br_str = f"{br} ({br_desc})" if br_desc else (str(br) if br is not None else "-")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Ratio Sweep Report (All Models)\n\n")
        f.write("| Item | Value |\n|------|-------|\n")
        f.write(f"| **Models** | {len(all_results)} models (same as compression_analysis) |\n")
        f.write(f"| **Config** | `{config_path}` |\n")
        f.write(f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")
        f.write("## Run / Training Configuration\n\n")
        f.write("| Item | Value |\n|------|-------|\n")
        f.write(f"| **Data** | {data_cfg.get('name', '-')} |\n")
        f.write(f"| **Max samples** | {data_cfg.get('max_samples', '-')} |\n")
        f.write(f"| **Balance ratio** | {br_str} |\n")
        f.write(f"| **Num clients** | {data_cfg.get('num_clients', '-')} |\n")
        f.write(f"| **Model** | {model_cfg.get('name', '-')} |\n")
        f.write(f"| **FL rounds** | {fed_cfg.get('num_rounds', '-')} |\n")
        f.write(f"| **Local epochs** | {fed_cfg.get('local_epochs', '-')} |\n")
        f.write(f"| **Batch size** | {fed_cfg.get('batch_size', '-')} |\n")
        f.write(f"| **Learning rate** | {fed_cfg.get('learning_rate', '-')} |\n")
        f.write(f"| **Use QAT** | {fed_cfg.get('use_qat', '-')} |\n\n")
        f.write("## Summary\n\n")
        f.write(f"Total models: {len(all_results)}, Total ratios: {len(ratios)}\n\n")

        # Comparison (Accuracy): Model | 100 | 90 | 80 | ... | 0
        f.write("## Comparison — Accuracy (Model × Normal%)\n\n")
        header = "| Model | " + " | ".join(str(p) for p in ratios) + " |\n"
        f.write(header)
        f.write("|" + "-------|" * (len(ratios) + 1) + "\n")
        for display_name, _, rows in all_results:
            by_pct = {r["normal_pct"]: r["accuracy"] for r in rows}
            cells = " | ".join(f"{by_pct.get(p, 0):.4f}" for p in ratios)
            f.write(f"| {display_name} | {cells} |\n")
        f.write("\n")

        # Comparison (F1-Score)
        f.write("## Comparison — F1-Score (Model × Normal%)\n\n")
        f.write(header)
        f.write("|" + "-------|" * (len(ratios) + 1) + "\n")
        for display_name, _, rows in all_results:
            by_pct = {r["normal_pct"]: r["f1_score"] for r in rows}
            cells = " | ".join(f"{by_pct.get(p, 0):.4f}" for p in ratios)
            f.write(f"| {display_name} | {cells} |\n")
        f.write("\n")

        # Comparison (Normal Recall)
        f.write("## Comparison — Normal Recall (Model × Normal%)\n\n")
        f.write(header)
        f.write("|" + "-------|" * (len(ratios) + 1) + "\n")
        for display_name, _, rows in all_results:
            by_pct = {r["normal_pct"]: r.get("normal_recall", 0) for r in rows}
            cells = " | ".join(f"{by_pct.get(p, 0):.4f}" for p in ratios)
            f.write(f"| {display_name} | {cells} |\n")
        f.write("\n")

        # Detailed per model (compact table each)
        f.write("## Detailed (per model)\n\n")
        for display_name, model_path, rows in all_results:
            f.write(f"### {display_name}\n\n")
            f.write("| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |\n")
            f.write("|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|\n")
            for r in rows:
                nr = r.get("normal_recall", 0)
                np_ = r.get("normal_precision", 0)
                f.write(f"| {r['normal_pct']} | {r['attack_pct']} | {r['n_total']:,} | "
                        f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | "
                        f"{nr:.4f} | {np_:.4f} |\n")
            f.write("\n")


def _write_markdown_report(rows, output_path: Path, model_path: str, config_path: str):
    """Write Markdown report in compression_analysis.md style (Detailed Metrics per ratio)."""
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Ratio Sweep Report\n\n")
        f.write(f"| Item | Value |\n")
        f.write(f"|------|-------|\n")
        f.write(f"| **Model** | `{model_path}` |\n")
        f.write(f"| **Config** | `{config_path}` |\n")
        f.write(f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |\n\n")
        f.write("## Run / Training Configuration\n\n")
        f.write("| Item | Value |\n|------|-------|\n")
        f.write(f"| **Data** | {data_cfg.get('name', '-')} |\n")
        f.write(f"| **Max samples** | {data_cfg.get('max_samples', '-')} |\n")
        br = data_cfg.get("balance_ratio")
        br_desc = {1.0: "50:50", 4.0: "normal:attack 8:2", 9.0: "9:1", 19.0: "19:1"}.get(br) if br is not None else None
        br_str = f"{br} ({br_desc})" if br_desc else (str(br) if br is not None else "-")
        f.write(f"| **Balance ratio** | {br_str} |\n")
        f.write(f"| **Num clients** | {data_cfg.get('num_clients', '-')} |\n")
        f.write(f"| **Model** | {model_cfg.get('name', '-')} |\n")
        f.write(f"| **FL rounds** | {fed_cfg.get('num_rounds', '-')} |\n")
        f.write(f"| **Local epochs** | {fed_cfg.get('local_epochs', '-')} |\n")
        f.write(f"| **Batch size** | {fed_cfg.get('batch_size', '-')} |\n")
        f.write(f"| **Learning rate** | {fed_cfg.get('learning_rate', '-')} |\n")
        f.write(f"| **Use QAT** | {fed_cfg.get('use_qat', '-')} |\n")
        f.write("\n")
        f.write("## Summary\n\n")
        f.write(f"Total ratios evaluated: {len(rows)}\n\n")
        f.write("## Comparison Table\n\n")
        f.write("| Normal% | Attack% | n_total | n_Normal | n_Attack | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |\n")
        f.write("|---------|----------|---------|----------|----------|----------|------------|--------|----------|---------------|-------------------|\n")
        for r in rows:
            nr = r.get("normal_recall", 0)
            np_ = r.get("normal_precision", 0)
            f.write(f"| {r['normal_pct']} | {r['attack_pct']} | {r['n_total']:,} | {r['n_normal']:,} | {r['n_attack']:,} | "
                    f"{r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1_score']:.4f} | "
                    f"{nr:.4f} | {np_:.4f} |\n")
        f.write("\n## Detailed Metrics\n\n")
        for r in rows:
            title = f"Normal {r['normal_pct']}% : Attack {r['attack_pct']}%"
            f.write(f"### {title}\n\n")
            f.write(f"- **Test samples**: {r['n_total']:,} (Normal={r['n_normal']:,}, Attack={r['n_attack']:,})\n")
            f.write(f"- **Accuracy**: {r['accuracy']:.4f}\n")
            f.write(f"- **Precision**: {r['precision']:.4f}\n")
            f.write(f"- **Recall**: {r['recall']:.4f}\n")
            f.write(f"- **F1-Score**: {r['f1_score']:.4f}\n")
            nr = r.get("normal_recall", 0)
            np_ = r.get("normal_precision", 0)
            f.write(f"- **Normal Recall** (of actual normal, predicted normal): {nr:.4f}\n")
            f.write(f"- **Normal Precision** (of predicted normal, actual normal): {np_:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
