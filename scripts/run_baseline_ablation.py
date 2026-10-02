#!/usr/bin/env python3
"""
Reviewer B baseline ablation: (a) centralized, (b) FL, (c) FL+compression PTQ,
(d) FL+compression QAT, plus optional failed_config narrative row.

Outputs CSV + Markdown under data/processed/ablation/<timestamp>/.

Usage:
  python scripts/run_baseline_ablation.py
  python scripts/run_baseline_ablation.py --quick --skip-train
  python scripts/run_baseline_ablation.py --with-pgd --with-failed
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import (
    apply_quick_mode,
    load_keras_for_eval,
    load_yaml,
    make_output_dir,
    rows_to_csv,
    rows_to_markdown,
    run_cmd,
    save_json,
    save_yaml,
)
from src.data.loader import load_dataset
from src.evaluation.metrics import evaluate_keras_model, evaluate_tflite_model


def _dataset_kwargs(cfg: dict) -> dict:
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    return kwargs


def _load_test_data(cfg: dict):
    data_cfg = cfg.get("data", {})
    name = data_cfg.get("name", "cicids2017")
    _, _, x_test, y_test = load_dataset(name, **_dataset_kwargs(cfg))
    return x_test, y_test


def _evaluate_keras_path(
    model_path: Path,
    cfg: dict,
    row_id: str,
    label: str,
) -> Dict[str, Any]:
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
    x_test, y_test = _load_test_data(cfg)
    model = load_keras_for_eval(model_path)
    metrics = evaluate_keras_model(model, x_test, y_test, threshold=threshold)
    size_kb = model_path.stat().st_size / 1024.0
    result = {
        "row_id": row_id,
        "label": label,
        "model_path": str(model_path),
        "format": "keras",
        "size_kb": round(size_kb, 2),
        "latency_ms": "",
        **{k: metrics[k] for k in [
            "accuracy", "precision", "recall", "attack_recall", "f1",
            "tp", "tn", "fp", "fn", "missed_attacks", "false_alarms",
            "false_alarm_rate", "threshold",
        ]},
    }
    return result


def _evaluate_tflite_path(
    tflite_path: Path,
    cfg: dict,
    row_id: str,
    label: str,
) -> Dict[str, Any]:
    if not tflite_path.exists():
        return {
            "row_id": row_id,
            "label": label,
            "model_path": str(tflite_path),
            "format": "tflite",
            "error": "missing",
        }
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
    x_test, y_test = _load_test_data(cfg)
    metrics = evaluate_tflite_model(
        tflite_path, x_test, y_test, threshold=threshold, latency_runs=50
    )
    return {
        "row_id": row_id,
        "label": label,
        "model_path": str(tflite_path),
        "format": "tflite",
        "size_kb": round(metrics.get("size_kb", 0), 2),
        "latency_ms": round(metrics.get("latency_ms", 0), 4),
        **{k: metrics[k] for k in [
            "accuracy", "precision", "recall", "attack_recall", "f1",
            "tp", "tn", "fp", "fn", "missed_attacks", "false_alarms",
            "false_alarm_rate", "threshold",
        ]},
    }


def _run_compression(
    cfg_path: Path,
    model_path: Path,
    out_tflite_dir: Path,
    traditional_copy: Path,
) -> None:
    out_tflite_dir.mkdir(parents=True, exist_ok=True)
    project_models = ROOT / "models" / "tflite"
    project_models.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(cfg_path)
    comp = cfg.setdefault("compression", {})
    comp["traditional_model_path"] = str(traditional_copy)
    save_yaml(cfg_path, cfg)
    shutil.copy2(model_path, traditional_copy)

    cmd = [
        sys.executable,
        str(ROOT / "compression.py"),
        "--use-trained",
        "--config",
        str(cfg_path),
        "--model-path",
        str(model_path),
    ]
    if not run_cmd(cmd, f"Compression for {model_path.name}"):
        raise RuntimeError(f"Compression failed for {model_path}")

    for src in project_models.glob("*.tflite"):
        shutil.copy2(src, out_tflite_dir / src.name)


def _run_pgd_on_models(
    cfg_path: Path,
    models: List[Path],
    pgd_out: Path,
    attack: str = "pgd",
) -> None:
    pgd_out.mkdir(parents=True, exist_ok=True)
    model_args = []
    for m in models:
        if m.exists():
            model_args.extend(["--models", str(m)])
    if not model_args:
        return
    # PGD needs a Keras model to generate adversarial examples
    keras_models = [m for m in models if str(m).endswith(".h5") and m.exists()]
    if keras_models:
        model_args = ["--models", str(keras_models[0])]
        for m in models:
            if m.suffix == ".tflite" and m.exists():
                model_args.extend(["--models", str(m)])
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pgd.py"),
        *model_args,
        "--config",
        str(cfg_path),
        "--output-dir",
        str(pgd_out),
        "--attack",
        attack,
    ]
    run_cmd(cmd, f"Adversarial eval ({attack.upper()})")


def main():
    parser = argparse.ArgumentParser(description="Reviewer B baseline ablation")
    parser.add_argument("--base-config", default="config/ablation/base.yaml")
    parser.add_argument(
        "--config-dir",
        default="config/ablation",
        help="Directory holding centralized.yaml, fl_baseline.yaml, failed_config.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--quick", action="store_true", help="Fewer FL rounds for smoke test")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--with-failed", action="store_true", help="Include failed_config row")
    parser.add_argument("--with-pgd", action="store_true", help="Run PGD+FGSM on compressed models")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else make_output_dir("ablation")
    models_dir = out_dir / "models"
    tflite_dir = out_dir / "tflite"
    configs_dir = out_dir / "configs"
    models_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = ROOT / args.base_config
    base_cfg = apply_quick_mode(load_yaml(base_cfg_path), args.quick)

    rows: List[Dict[str, Any]] = []

    # --- (a) Centralized ---
    cfg_a = configs_dir / "a_centralized.yaml"
    shutil.copy2(ROOT / args.config_dir / "centralized.yaml", cfg_a)
    save_yaml(cfg_a, apply_quick_mode(load_yaml(cfg_a), args.quick))
    model_a = models_dir / "a_centralized.h5"
    if not args.skip_train:
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_centralized.py"),
                "--config",
                str(cfg_a),
                "--save-model",
                str(model_a),
            ],
            "(a) Centralized training",
        )
    if model_a.exists():
        rows.append(_evaluate_keras_path(model_a, load_yaml(cfg_a), "a", "Centralized+cosine+focal"))

    # --- (b) FL ---
    cfg_b = configs_dir / "b_fl.yaml"
    shutil.copy2(ROOT / args.config_dir / "fl_baseline.yaml", cfg_b)
    save_yaml(cfg_b, apply_quick_mode(load_yaml(cfg_b), args.quick))
    model_b = models_dir / "b_fl.h5"
    if not args.skip_train:
        run_cmd(
            [
                sys.executable,
                "-m",
                "src.federated.client",
                "--config",
                str(cfg_b),
                "--save-model",
                str(model_b),
            ],
            "(b) FL training",
        )
    if model_b.exists():
        rows.append(_evaluate_keras_path(model_b, load_yaml(cfg_b), "b", "FL+cosine+focal"))

    # --- (c)(d) Compression ---
    tflite_c = tflite_dir / "saved_model_no_qat_ptq.tflite"
    tflite_d = tflite_dir / "saved_model_traditional_qat.tflite"
    tflite_ptq_main = tflite_dir / "saved_model_qat_ptq.tflite"
    if model_b.exists() and not args.skip_compression:
        trad_copy = models_dir / "b_fl_traditional.h5"
        _run_compression(cfg_b, model_b, tflite_dir, trad_copy)
        ptq_path = tflite_c if tflite_c.exists() else tflite_ptq_main
        rows.append(_evaluate_tflite_path(ptq_path, load_yaml(cfg_b), "c", "FL+compression+PTQ"))
        # QAT-trained FL models (paper_v12) emit pruned_qat instead of traditional_qat
        qat_path = tflite_d if tflite_d.exists() else tflite_dir / "saved_model_pruned_qat.tflite"
        rows.append(_evaluate_tflite_path(qat_path, load_yaml(cfg_b), "d", "FL+compression+QAT"))
    elif args.skip_compression:
        for path, rid, lbl in [
            (tflite_c, "c", "FL+compression+PTQ"),
            (tflite_d, "d", "FL+compression+QAT"),
            (tflite_ptq_main, "c_alt", "FL+compression+PTQ (main)"),
            (tflite_dir / "saved_model_pruned_qat.tflite", "d_alt", "FL+compression+QAT (QAT-trained)"),
        ]:
            if path.exists():
                rows.append(_evaluate_tflite_path(path, load_yaml(cfg_b), rid, lbl))

    # --- failed narrative ---
    if args.with_failed:
        cfg_f = configs_dir / "failed.yaml"
        shutil.copy2(ROOT / args.config_dir / "failed_config.yaml", cfg_f)
        save_yaml(cfg_f, apply_quick_mode(load_yaml(cfg_f), args.quick))
        model_f = models_dir / "failed_fl.h5"
        if not args.skip_train:
            run_cmd(
                [
                    sys.executable,
                    "-m",
                    "src.federated.client",
                    "--config",
                    str(cfg_f),
                    "--save-model",
                    str(model_f),
                ],
                "Failed config FL (fixed LR)",
            )
        if model_f.exists():
            rows.append(
                _evaluate_keras_path(
                    model_f, load_yaml(cfg_f), "failed", "FL fixed LR (narrative)"
                )
            )

    rows_to_csv(rows, out_dir / "baseline_ablation.csv")
    rows_to_markdown(rows, out_dir / "baseline_ablation.md", "Baseline Ablation (Reviewer B)")
    save_json({"rows": rows, "output_dir": str(out_dir)}, out_dir / "baseline_ablation.json")
    print(f"\n✅ Baseline ablation saved to {out_dir}")

    if args.with_pgd:
        pgd_dir = out_dir / "pgd"
        tflite_models = [
            tflite_dir / "saved_model_no_qat_ptq.tflite",
            tflite_dir / "saved_model_traditional_qat.tflite",
            tflite_dir / "saved_model_qat_ptq.tflite",
            tflite_dir / "saved_model_pruned_qat.tflite",
        ]
        _run_pgd_on_models(cfg_b, [model_b] + tflite_models, pgd_dir / "pgd", attack="pgd")
        _run_pgd_on_models(cfg_b, [model_b] + tflite_models, pgd_dir / "fgsm", attack="fgsm")
        print(f"   PGD/FGSM results: {pgd_dir}")


if __name__ == "__main__":
    main()
