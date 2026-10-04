#!/usr/bin/env python3
"""
Compression-pipeline combinations: PTQ, QAT, structured / unstructured pruning, distillation
and their order, all FL-faithful (only one client's local data after federated training).

Every pipeline starts from the float near-IID federated model and is repeated over several
random fine-tuning draws of client `ft_client`, because a single draw can mislead (paper 5.9).

Pipelines (CLIP = calibration inputs clipped to +-clip SD, see ptq_calibration_check.py):
  fp32                 federated model, no compression (reference)
  q_ptq                INT8 PTQ
  q_ptq_clip           INT8 PTQ, clipped calibration
  q_qat                QAT fine-tune (no pruning)
  s_ptq                structured prune 50% -> FT -> PTQ
  s_ptq_clip           structured prune 50% -> FT -> PTQ (clipped)
  s_qat                structured prune 50% -> FT -> QAT FT        (deployed recipe)
  s_qat_clipft         structured prune 50% -> FT -> QAT FT on clipped inputs
  s_kd_qat             structured prune 50% -> FT with teacher soft targets -> QAT FT (soft targets)
  s_kd_ptq_clip        structured prune 50% -> FT with teacher soft targets -> PTQ (clipped)
  o_qat_s_qat          QAT FT -> strip -> structured prune 50% -> FT -> QAT FT (quantize first)
  u{p}_ptq_clip        magnitude (unstructured) prune p -> PTQ (clipped)
  u{p}_pqat            magnitude prune p -> sparsity-preserving QAT (PQAT)
  u80_qat              magnitude prune 80% -> standard QAT (does it keep the sparsity?)
  su_pqat              structured 50% -> FT -> magnitude 50% -> PQAT (both pruning types)

TFLite Micro stores weights densely, so unstructured sparsity does not shrink flash use; the
gzip size column shows what it would save with a compressed weight format.

Spec (YAML):
  ft_samples: 10000
  draws: 3
  clip: 5.0
  sparsities: [0.5, 0.8]
  models:
    - {name: ton_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>, ft_client: 0}

Usage:
  python scripts/compression_combos.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import gzip
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_model_optimization as tfmot

from compression import strip_qat_layers
from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import (
    PRUNE_FT_EPOCHS,
    PRUNE_RATIO,
    QAT_FT_EPOCHS,
    _clone,
    _fit,
    _load_fl_model,
    _qat,
)
from scripts.ptq_calibration_check import convert_int8
from scripts.quant_distill_ablation import convert, kd_targets
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_qat, export_tflite_qat

METRICS = ["accuracy", "precision", "attack_recall", "f1", "false_alarm_rate", "fn", "fp"]
DESCRIPTIONS = {
    "fp32": "no compression (reference)",
    "q_ptq": "INT8 PTQ",
    "q_ptq_clip": "INT8 PTQ, clipped calibration",
    "q_qat": "QAT fine-tune, no pruning",
    "s_ptq": "structured 50% -> FT -> PTQ",
    "s_ptq_clip": "structured 50% -> FT -> PTQ (clipped)",
    "s_qat": "structured 50% -> FT -> QAT (deployed)",
    "s_qat_clipft": "structured 50% -> FT -> QAT on clipped inputs",
    "s_kd_qat": "structured 50% -> KD fine-tune -> QAT",
    "s_kd_ptq_clip": "structured 50% -> KD fine-tune -> PTQ (clipped)",
    "o_qat_s_qat": "QAT first -> structured 50% -> FT -> QAT",
    "u80_qat": "magnitude 80% -> standard QAT",
    "su_pqat": "structured 50% -> FT -> magnitude 50% -> PQAT",
}


def _steps(n: int, epochs: int, batch: int = 128) -> int:
    return int(np.ceil(0.9 * n / batch)) * epochs  # _fit holds out 10% for validation


def magnitude_prune(model: keras.Model, x, y, sparsity: float, epochs: int = PRUNE_FT_EPOCHS) -> keras.Model:
    """Gradual magnitude pruning of the hidden Dense layers during fine-tuning (output layer kept)."""
    total = _steps(len(y), epochs)
    sched = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=float(sparsity), begin_step=0,
        end_step=max(1, int(total * 2 / 3)), frequency=max(1, total // 30))
    work = _clone(model)
    last = work.layers[-1].name

    def wrap(layer):
        if isinstance(layer, keras.layers.Dense) and layer.name != last:
            return tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule=sched)
        return layer

    pm = keras.models.clone_model(work, clone_function=wrap)
    pm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    pm.fit(x, y, epochs=epochs, batch_size=128, validation_split=0.1, verbose=0,
           callbacks=[tfmot.sparsity.keras.UpdatePruningStep()])
    out = tfmot.sparsity.keras.strip_pruning(pm)
    out.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return out


def pqat(model: keras.Model, x, y) -> keras.Model:
    """Sparsity-preserving QAT (tfmot collaborative optimization)."""
    annotated = tfmot.quantization.keras.quantize_annotate_model(model)
    q = tfmot.quantization.keras.quantize_apply(
        annotated, tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme())
    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    _fit(q, x, y, QAT_FT_EPOCHS)
    return q


def tflite_weight_sparsity(path: Path) -> float:
    """Fraction of exact zeros in the 2-D weight tensors of a TFLite model (output layer excluded)."""
    try:
        interp = tf.lite.Interpreter(model_path=str(path))
        interp.allocate_tensors()
        zeros = total = 0
        for d in interp.get_tensor_details():
            shape = d["shape"]
            if len(shape) != 2 or min(shape) <= 1 or d["dtype"] not in (np.int8, np.float32):
                continue
            try:
                w = interp.get_tensor(d["index"])
            except ValueError:
                continue  # activation tensor
            if d["dtype"] == np.int8:
                zp = d["quantization_parameters"]["zero_points"]
                w = w.astype(np.int32) - (int(zp[0]) if len(zp) else 0)
            zeros += int(np.sum(w == 0))
            total += w.size
        return round(zeros / total, 4) if total else float("nan")
    except Exception:
        return float("nan")


def run_model(entry: dict, spec: dict, out_dir: Path) -> List[Dict[str, Any]]:
    name = entry["name"]
    cfg = load_yaml(ROOT / entry["config"])
    data_cfg = dict(cfg.get("data", {}))
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_train, y_train, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)
    parts = partition_data(
        x_train, y_train, int(data_cfg.get("num_clients", 4)),
        strategy=data_cfg.get("partition_strategy", "label_balanced"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
    )
    threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
    base = _load_fl_model(str(ROOT / entry["model"]))
    k = int(entry.get("ft_client", 0))
    cx, cy = parts[k]["x"], parts[k]["y"].astype(np.float32)
    n_ft = int(spec.get("ft_samples", 10000))
    clip = float(spec.get("clip", 5.0))
    sparsities = [float(s) for s in spec.get("sparsities", [0.5, 0.8])]
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def record(pipeline: str, draw: int, build: Callable[[Path], None]):
        p = tdir / f"{pipeline}_d{draw}.tflite"
        row: Dict[str, Any] = {"model": name, "pipeline": pipeline, "draw": draw,
                               "description": DESCRIPTIONS.get(pipeline, pipeline)}
        try:
            build(p)
            m = evaluate_tflite_model(p, x_test, y_test, threshold=threshold, latency_runs=0)
            data = p.read_bytes()
            row.update({"size_kb": round(len(data) / 1024, 2),
                        "gzip_kb": round(len(gzip.compress(data, 9)) / 1024, 2),
                        "weight_sparsity": tflite_weight_sparsity(p)})
            row.update({key: m[key] for key in METRICS})
            print(f"  [{name} d{draw}] {pipeline:16s} {row['size_kb']:7.1f} KB (gz {row['gzip_kb']:6.1f}) "
                  f"sparsity={row['weight_sparsity']} f1={row['f1']:.4f} far={row['false_alarm_rate']:.4f}")
            if draw > 0 or pipeline.startswith("u") or pipeline.startswith("su"):
                p.unlink()  # keep draw-0 models of the main pipelines only
        except Exception as err:
            traceback.print_exc()
            row["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            print(f"  ⚠️ [{name} d{draw}] {pipeline} failed: {row['error']}")
        rows.append(row)
        rows_to_csv(rows, out_dir / f"{name}_partial.csv")

    record("fp32", 0, lambda p: convert(base, "fp32", cx[:500], p))
    for draw in range(int(spec.get("draws", 3))):
        tf.keras.utils.set_random_seed(100 + draw)
        idx = np.random.default_rng(100 + draw).permutation(len(cy))[:n_ft]
        xf, yf = cx[idx], cy[idx]
        xf_clip = np.clip(xf, -clip, clip)
        soft = kd_targets(base, xf, yf, 2.0, 0.5)

        # ---- quantization only ----
        record("q_ptq", draw, lambda p: convert_int8(base, xf[:500], "builtins_int8", p))
        record("q_ptq_clip", draw, lambda p: convert_int8(base, xf_clip[:500], "builtins_int8", p))
        record("q_qat", draw, lambda p: export_tflite_qat(_qat(_clone(base), xf, yf), str(p)))

        # ---- structured pruning + quantization ----
        pr = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
        _fit(pr, xf, yf, PRUNE_FT_EPOCHS)
        record("s_ptq", draw, lambda p: convert_int8(pr, xf[:500], "builtins_int8", p))
        record("s_ptq_clip", draw, lambda p: convert_int8(pr, xf_clip[:500], "builtins_int8", p))
        record("s_qat", draw, lambda p: export_tflite_qat(_qat(pr, xf, yf), str(p)))
        record("s_qat_clipft", draw, lambda p: export_tflite_qat(_qat(pr, xf_clip, yf), str(p)))

        # ---- structured pruning + distillation fine-tuning ----
        pk = apply_structured_pruning(_clone(base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True, verbose=False)
        _fit(pk, xf, soft, PRUNE_FT_EPOCHS)
        record("s_kd_qat", draw, lambda p: export_tflite_qat(_qat(pk, xf, soft), str(p)))
        record("s_kd_ptq_clip", draw, lambda p: convert_int8(pk, xf_clip[:500], "builtins_int8", p))

        # ---- order: quantization-aware fine-tuning before pruning ----
        def quantize_first(p):
            fl = strip_qat_layers(_qat(_clone(base), xf, yf))
            fl.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            po = apply_structured_pruning(_strip_bn_dropout_for_qat(fl), pruning_ratio=PRUNE_RATIO,
                                          skip_last_layer=True, verbose=False)
            _fit(po, xf, yf, PRUNE_FT_EPOCHS)
            export_tflite_qat(_qat(po, xf, yf), str(p))
        record("o_qat_s_qat", draw, quantize_first)

        # ---- unstructured (magnitude) pruning ----
        for sp in sparsities:
            tag = f"u{int(round(sp * 100))}"
            DESCRIPTIONS[f"{tag}_ptq_clip"] = f"magnitude {sp:.0%} -> PTQ (clipped)"
            DESCRIPTIONS[f"{tag}_pqat"] = f"magnitude {sp:.0%} -> sparsity-preserving QAT"
            try:
                um = magnitude_prune(base, xf, yf, sp)
            except Exception as err:
                traceback.print_exc()
                rows.append({"model": name, "pipeline": tag, "draw": draw, "error": str(err)[:200]})
                continue
            record(f"{tag}_ptq_clip", draw, lambda p, um=um: convert_int8(um, xf_clip[:500], "builtins_int8", p))
            record(f"{tag}_pqat", draw, lambda p, um=um: export_tflite_qat(pqat(um, xf, yf), str(p)))
            if abs(sp - 0.8) < 1e-9:
                record("u80_qat", draw, lambda p, um=um: export_tflite_qat(_qat(um, xf, yf), str(p)))

        # ---- both pruning types ----
        record("su_pqat", draw, lambda p: export_tflite_qat(pqat(magnitude_prune(pr, xf, yf, 0.5), xf, yf), str(p)))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    keys = []
    for r in rows:
        if (r["model"], r["pipeline"]) not in keys:
            keys.append((r["model"], r["pipeline"]))
    for model, pipe in keys:
        ok = [r for r in rows if r["model"] == model and r["pipeline"] == pipe and "f1" in r]
        bad = [r for r in rows if r["model"] == model and r["pipeline"] == pipe and "error" in r]
        row: Dict[str, Any] = {"model": model, "pipeline": pipe, "description": DESCRIPTIONS.get(pipe, pipe),
                               "draws_ok": len(ok), "draws_failed": len(bad)}
        if ok:
            f1 = [r["f1"] * 100 for r in ok]
            far = [r["false_alarm_rate"] * 100 for r in ok]
            rec = [r["attack_recall"] * 100 for r in ok]
            row.update({
                "size_kb": ok[0]["size_kb"], "gzip_kb": round(statistics.mean(r["gzip_kb"] for r in ok), 1),
                "weight_sparsity": round(statistics.mean(r["weight_sparsity"] for r in ok), 3),
                "f1_mean": round(statistics.mean(f1), 2), "f1_min": round(min(f1), 2), "f1_max": round(max(f1), 2),
                "recall_mean": round(statistics.mean(rec), 2), "far_mean": round(statistics.mean(far), 2),
            })
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compression pipeline combinations")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for entry in spec["models"]:
        rows.extend(run_model(entry, spec, out_dir))
        rows_to_csv(rows, out_dir / "compression_combos.csv")
        summary = summarize(rows)
        rows_to_csv(summary, out_dir / "compression_combos_summary.csv")
    rows_to_markdown(summarize(rows), out_dir / "compression_combos.md",
                     "Compression pipeline combinations (mean / min / max over fine-tuning draws)")
    save_json({"rows": rows, "summary": summarize(rows)}, out_dir / "compression_combos.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
