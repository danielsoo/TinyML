#!/usr/bin/env python3
"""
Full compression grid sweep: fl_qat × distillation × pruning × ptq → 48 combinations.
Produces a CSV with columns: tag, fl_qat, distillation, pruning, ptq, fl_acc, fl_f1, ...,
final_acc, final_f1, tflite_size_kb, final_size_kb.
Optionally adds columns for AT, PGD, ratio_sweep (when those are run).

Usage:
  # 1) Train FL models first (run.py or train twice for qat + no_qat), then run distillation_first
  python run.py --config config/federated_local_sky.yaml  # or --skip-compression --skip-analysis
  # 2) Run grid sweep (uses existing models)
  python scripts/sweep_compression_grid.py --config config/federated_local_sky.yaml --output data/processed/sweep_compression_grid.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import yaml
from tensorflow import keras

from src.data.loader import load_dataset
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import export_tflite


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict, max_samples: int = None):
    data_cfg = cfg.get("data", {})
    name = data_cfg.get("name", "cicids2017")
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    if max_samples is not None:
        kwargs["max_samples"] = max_samples
    x_train, y_train, x_test, y_test = load_dataset(name, **kwargs)
    if hasattr(x_test, "values"):
        x_test = x_test.values
    if hasattr(y_test, "values"):
        y_test = y_test.values
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    return x_train, y_train, x_test, y_test


def eval_keras_binary(model: keras.Model, x: np.ndarray, y: np.ndarray, threshold: float = 0.5):
    """Returns acc, f1, prec, rec for binary classification."""
    pred = model.predict(x, verbose=0)
    if pred.ndim > 1:
        pred = pred[:, -1].ravel()
    pred_bin = (pred >= threshold).astype(int)
    y_flat = np.asarray(y).ravel().astype(int)
    tp = ((pred_bin == 1) & (y_flat == 1)).sum()
    fp = ((pred_bin == 1) & (y_flat == 0)).sum()
    fn = ((pred_bin == 0) & (y_flat == 1)).sum()
    tn = ((pred_bin == 0) & (y_flat == 0)).sum()
    acc = (tp + tn) / max(1, len(y_flat))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return float(acc), float(f1), float(prec), float(rec)


def eval_tflite_binary(tflite_path: str, x: np.ndarray, y: np.ndarray, threshold: float = 0.5):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    preds = []
    for i in range(len(x)):
        interp.set_tensor(in_d["index"], x[i : i + 1].astype(np.float32))
        interp.invoke()
        out = interp.get_tensor(out_d["index"])
        if out.ndim > 1:
            out = out[:, -1].ravel()
        preds.append(float(out[0]))
    pred = np.array(preds)
    pred_bin = (pred >= threshold).astype(int)
    y_flat = np.asarray(y).ravel().astype(int)
    tp = ((pred_bin == 1) & (y_flat == 1)).sum()
    fp = ((pred_bin == 1) & (y_flat == 0)).sum()
    fn = ((pred_bin == 0) & (y_flat == 1)).sum()
    tn = ((pred_bin == 0) & (y_flat == 0)).sum()
    acc = (tp + tn) / max(1, len(y_flat))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return float(acc), float(f1), float(prec), float(rec)


def get_model_size_kb(model: keras.Model) -> float:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name
    try:
        model.save(path)
        return Path(path).stat().st_size / 1024
    finally:
        Path(path).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compression grid sweep: fl_qat × distillation × pruning × ptq")
    parser.add_argument("--config", default="config/federated_local_sky.yaml")
    parser.add_argument("--output", default="data/processed/sweep_compression_grid.csv")
    parser.add_argument("--run-dir", default=None, help="If set, write CSV and pgd_model_list.txt here for 48 PGD")
    parser.add_argument("--max-test", type=int, default=5000, help="Max test samples for eval")
    parser.add_argument("--quick", action="store_true", help="Run only 4 combinations (no_qat/none/none/no, yes_qat/none/10x5/no, ...)")
    parser.add_argument("--resume-csv", default=None, help="Path to a partial CSV from a previous run; skip tags already present")
    args = parser.parse_args()

    # Load already-completed rows from a previous partial run (resume support)
    resume_rows = []
    done_tags = set()
    if args.resume_csv:
        resume_path = Path(args.resume_csv)
        if resume_path.exists():
            with open(resume_path, encoding="utf-8", newline="") as _rf:
                for r in csv.DictReader(_rf):
                    resume_rows.append(r)
                    done_tags.add(r.get("tag", ""))
            print(f"  [Resume] Loaded {len(resume_rows)} completed rows from {resume_path}")
            print(f"  [Resume] Skipping tags: {sorted(done_tags)}\n")

    cfg = load_config(args.config)
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.5))
    x_train, y_train, x_test, y_test = load_data(cfg, max_samples=args.max_test)
    x_train_sub = x_train[: min(10000, len(x_train))]
    if hasattr(x_train_sub, "values"):
        x_train_sub = x_train_sub.values
    y_train_sub = y_train[: len(x_train_sub)]
    if hasattr(y_train_sub, "values"):
        y_train_sub = y_train_sub.values
    x_train_sub = np.asarray(x_train_sub, dtype=np.float32)
    y_train_sub = np.asarray(y_train_sub, dtype=np.float32)

    # Paths
    qat_path = Path("models/global_model.h5")
    no_qat_path = Path("models/global_model_traditional.h5")
    distilled_dir = Path("models/distilled")
    out_dir = Path("models/tflite")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not qat_path.exists() or not no_qat_path.exists():
        print("Requires models/global_model.h5 and models/global_model_traditional.h5. Run train first.")
        return 1
    # Ensure 4 distilled models exist
    need = ["no_qat_direct.h5", "no_qat_progressive.h5", "qat_direct.h5", "qat_progressive.h5"]
    if not all((distilled_dir / n).exists() for n in need):
        print("Run: python scripts/run_distillation_first.py --config", args.config, "--skip-compress")
        print("  to create models/distilled/*.h5, then run this script again.")
        return 1

    def load_keras(path: str, use_qat_scope: bool = False):
        p = Path(path)
        if not p.exists():
            return None
        if use_qat_scope:
            try:
                import tensorflow_model_optimization as tfmot
                with tfmot.quantization.keras.quantize_scope():
                    m = keras.models.load_model(p, compile=False)
            except Exception:
                m = keras.models.load_model(p, compile=False)
        else:
            m = keras.models.load_model(p, compile=False)
        last = m.layers[-1]
        nclass = getattr(last, "units", 2)
        loss = "binary_crossentropy" if nclass <= 2 else "sparse_categorical_crossentropy"
        m.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        return m

    # Build 6 base models: (no_qat, none/direct/progressive), (yes_qat, none/direct/progressive)
    bases = {}
    bases[("no_qat", "none")] = load_keras(str(no_qat_path), use_qat_scope=False)
    bases[("no_qat", "direct")] = load_keras(str(distilled_dir / "no_qat_direct.h5"), use_qat_scope=False)
    bases[("no_qat", "progressive")] = load_keras(str(distilled_dir / "no_qat_progressive.h5"), use_qat_scope=False)
    bases[("yes_qat", "none")] = load_keras(str(qat_path), use_qat_scope=cfg.get("federated", {}).get("use_qat", False))
    bases[("yes_qat", "direct")] = load_keras(str(distilled_dir / "qat_direct.h5"), use_qat_scope=False)
    bases[("yes_qat", "progressive")] = load_keras(str(distilled_dir / "qat_progressive.h5"), use_qat_scope=False)

    fl_acc, fl_f1, fl_prec, fl_rec = {}, {}, {}, {}
    fl_size = {}
    for (fq, dist), m in bases.items():
        if m is None:
            continue
        if dist == "none":
            a, f1, p, r = eval_keras_binary(m, x_test, y_test, threshold)
            fl_acc[fq] = a
            fl_f1[fq] = f1
            fl_prec[fq] = p
            fl_rec[fq] = r
            fl_size[fq] = get_model_size_kb(m)

    # Grid: fl_qat, distillation, pruning, ptq
    pruning_opts = [
        ("prune_none", None),
        ("prune_10x5", 0.10),
        ("prune_10x2", 0.02),
        ("prune_5x10", 0.05),
    ]
    rows = []
    combos = []
    if args.quick:
        combos = [
            (False, "none", "prune_none", False),
            (True, "none", "prune_10x5", False),
            (False, "progressive", "prune_10x5", True),
            (True, "direct", "prune_5x10", True),
        ]
    else:
        for fl_qat in [False, True]:
            for dist in ["none", "direct", "progressive"]:
                for prune_name, ratio in pruning_opts:
                    for ptq in [False, True]:
                        combos.append((fl_qat, dist, prune_name, ptq))

    at_cfg = cfg.get("adversarial_training", {})
    at_enabled = at_cfg.get("enabled", False)
    at_attack = at_cfg.get("attack", "pgd")

    MIN_TFLITE_BYTES = 10_000   # anything smaller is a corrupt/empty export

    for fl_qat, dist, prune_name, ptq in combos:
        fq = "yes_qat" if fl_qat else "no_qat"
        tag = f"{fq}__distill_{dist}__{prune_name}__ptq_{'yes' if ptq else 'no'}"
        if tag in done_tags:
            print(f"  [Resume] Skipping {tag} (already done in CSV)")
            continue
        # Skip if a valid TFLite already exists on disk (resume from interrupted run)
        tflite_name_check = tag.replace("__", "_").replace(".", "_") + ".tflite"
        tflite_path_check = out_dir / tflite_name_check
        if tflite_path_check.exists() and tflite_path_check.stat().st_size >= MIN_TFLITE_BYTES:
            print(f"  [Skip] {tag} — TFLite exists ({tflite_path_check.stat().st_size // 1024}KB)")
            continue
        base = bases.get((fq, dist))
        if base is None:
            rows.append({"tag": tag, "tflite_path": "", "fl_qat": fl_qat, "distillation": dist, "pruning": prune_name, "ptq": ptq})
            continue

        # Prune
        _, ratio = next(((n, r) for n, r in pruning_opts if n == prune_name), (None, None))
        if ratio is not None:
            model_copy = keras.models.clone_model(base)
            model_copy.set_weights(base.get_weights())
            model_copy.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            pruned = apply_structured_pruning(model_copy, pruning_ratio=ratio, skip_last_layer=True, verbose=False)
            pruned.fit(x_train_sub, y_train_sub, epochs=2, batch_size=128, validation_split=0.1, verbose=0)
        else:
            pruned = base

        prune_acc, prune_f1, _, _ = eval_keras_binary(pruned, x_test, y_test, threshold)

        # Export and final eval
        tflite_name = tag.replace("__", "_").replace(".", "_") + ".tflite"
        tflite_path = out_dir / tflite_name
        size_bytes = export_tflite(
            pruned,
            str(tflite_path),
            quantize=ptq,
            representative_data=x_train_sub if ptq else None,
        )
        tflite_size_kb = size_bytes / 1024
        final_acc, final_f1, final_prec, final_rec = eval_tflite_binary(str(tflite_path), x_test, y_test, threshold)

        dist_acc = dist_f1 = ""
        if dist != "none":
            da, df, _, _ = eval_keras_binary(base, x_test, y_test, threshold)
            dist_acc = da
            dist_f1 = df

        tflite_path_str = str(tflite_path.resolve() if tflite_path.is_absolute() else tflite_path)
        row = {
            "tag": tag,
            "tflite_path": tflite_path_str,
            "fl_qat": fl_qat,
            "distillation": dist,
            "pruning": prune_name,
            "ptq": ptq,
            "fl_acc": fl_acc.get(fq, ""),
            "fl_f1": fl_f1.get(fq, ""),
            "fl_prec": fl_prec.get(fq, ""),
            "fl_rec": fl_rec.get(fq, ""),
            "fl_size_kb": fl_size.get(fq, ""),
            "dist_acc": dist_acc,
            "dist_f1": dist_f1,
            "prune_acc": prune_acc,
            "prune_f1": prune_f1,
            "final_acc": final_acc,
            "final_f1": final_f1,
            "final_prec": final_prec,
            "final_rec": final_rec,
            "tflite_size_kb": round(tflite_size_kb, 4),
            "final_size_kb": round(tflite_size_kb, 4),
            "at_enabled": at_enabled,
            "at_attack": at_attack,
            "pgd_adv_acc": "",
            "pgd_success_rate": "",
            "ratio_sweep_f1_50_50": "",
        }
        rows.append(row)
        print(f"  {tag} -> final_f1={final_f1:.4f} tflite_kb={tflite_size_kb:.2f}")

    # Merge resumed rows back in (prepend so order is preserved)
    all_rows = resume_rows + rows

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag", "tflite_path", "fl_qat", "distillation", "pruning", "ptq",
        "fl_acc", "fl_f1", "fl_prec", "fl_rec", "fl_size_kb",
        "dist_acc", "dist_f1",
        "prune_acc", "prune_f1",
        "final_acc", "final_f1", "final_prec", "final_rec",
        "tflite_size_kb", "final_size_kb",
        "at_enabled", "at_attack",
        "pgd_adv_acc", "pgd_success_rate", "ratio_sweep_f1_50_50",
    ]
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        out_path = run_dir / "sweep_compression_grid.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out_path} ({len(all_rows)} rows, {len(rows)} new + {len(resume_rows)} resumed)")

    # Write pgd_model_list.txt for run_pgd.py --models-file (attack model first, then 48 TFLite)
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir)
        list_path = run_dir / "pgd_model_list.txt"
        keras_path = Path("models/global_model.h5").resolve()
        with open(list_path, "w", encoding="utf-8") as f:
            f.write(str(keras_path) + "\n")
            for r in all_rows:
                p = r.get("tflite_path", "").strip()
                if p and Path(p).exists():
                    f.write(p + "\n")
        print(f"Wrote {list_path} (1 Keras + {sum(1 for r in all_rows if r.get('tflite_path'))} TFLite paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
