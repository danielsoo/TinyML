#!/usr/bin/env python3
"""
Recall-priority compression: make the 65 KB INT8 model miss as few attacks as the float
federated model does, using only the fine-tuning client's own data.

Two levers, combined:
  1. Fine-tuning targets that keep the teacher's recall: distillation from the federated model
     (alpha = weight of the hard label; alpha 0 = pure teacher) and attack-weighted loss.
  2. A decision threshold chosen on the client's own held-out data (disjoint from the
     fine-tuning draw) to reach a target attack recall, instead of the fixed 0.3.

Every variant: structured prune 50% -> fine-tune 3 ep -> QAT fine-tune 2 ep -> INT8 (65 KB),
repeated over fine-tuning draws. Test metrics are reported at fixed thresholds and at the
thresholds selected on the client validation set. The test split is never used for selection.

Spec (YAML):
  ft_samples: 10000
  val_samples: 20000
  draws: 3
  clip: 5.0
  recall_targets: [0.999, 0.9995]
  fixed_thresholds: [0.05, 0.1, 0.2, 0.3, 0.5]
  variants:            # name -> {kd_alpha: null|float, kd_temperature: float, attack_weight: float,
                       #          clip: bool, ft_samples: int, prune_ratio: float, ft_epochs: int,
                       #          qat_epochs: int, federated: bool (fine-tune on all clients with FedAvg)}
    hard: {}
    kd05: {kd_alpha: 0.5}
  models:
    - {name: cic_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>, ft_client: 0}

Usage:
  python scripts/recall_priority.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import collections
import json
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_model_optimization as tfmot

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import PRUNE_FT_EPOCHS, PRUNE_RATIO, QAT_FT_EPOCHS, _clone, _load_fl_model
from scripts.quant_distill_ablation import convert, kd_targets
from src.data.loader import load_dataset, partition_data
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_qat, export_tflite_qat


def _fit_w(model, x, y, w, epochs: int) -> None:
    # Same settings as compression_ablation._fit, plus optional per-sample weights
    model.fit(x, y, sample_weight=w, epochs=epochs, batch_size=128, validation_split=0.1, verbose=0)


def _fedavg_round(global_model, clients, epochs: int, qat: bool = False) -> None:
    """One FedAvg round: every client fine-tunes a copy of global_model; weights are averaged by size."""
    states, sizes = [], []
    for x, t, w in clients:
        if qat:
            with tfmot.quantization.keras.quantize_scope():
                m = keras.models.clone_model(global_model)
        else:
            m = keras.models.clone_model(global_model)
        m.set_weights(global_model.get_weights())
        m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        _fit_w(m, x, t, w, epochs)
        states.append(m.get_weights())
        sizes.append(len(t))
    total = float(sum(sizes))
    global_model.set_weights([sum(st[i] * (n / total) for st, n in zip(states, sizes))
                              for i in range(len(states[0]))])


def tflite_probs(path: Path, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    interp = tf.lite.Interpreter(model_path=str(path))
    inp = interp.get_input_details()[0]["index"]
    out = interp.get_output_details()[0]["index"]
    probs = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].astype(np.float32)
        interp.resize_tensor_input(inp, list(xb.shape))
        interp.allocate_tensors()
        interp.set_tensor(inp, xb)
        interp.invoke()
        probs.append(np.asarray(interp.get_tensor(out)).reshape(-1))
    return np.concatenate(probs)


def metrics_at(p: np.ndarray, y: np.ndarray, t: float) -> Dict[str, Any]:
    pred = p >= t
    pos = y == 1
    tp = int(np.sum(pred & pos)); fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos)); tn = int(np.sum(~pred & ~pos))
    rec = tp / max(tp + fn, 1); far = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"threshold": round(float(t), 6), "attack_recall": rec, "false_alarm_rate": far,
            "precision": prec, "f1": f1, "accuracy": (tp + tn) / len(y), "fn": fn, "fp": fp}


def threshold_for_recall(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """Largest threshold whose recall on (p, y) is >= target."""
    att = np.sort(p[y == 1])
    k = int(np.floor((1.0 - target) * len(att)))  # attacks we may miss
    return float(att[k]) if len(att) else 0.5


def run_model(entry: dict, spec: dict, out_dir: Path) -> List[Dict[str, Any]]:
    name = entry["name"]
    cfg = load_yaml(ROOT / entry["config"])
    data_cfg = dict(cfg.get("data", {}))
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    dname = data_cfg.get("name", "cicids2017")
    attack_test = None
    if "cic" in dname.lower() or "ton" in dname.lower():
        x_train, y_train, x_test, y_test, _, attack_test = load_dataset(dname, return_attack_labels=True, **kwargs)
        attack_test = np.asarray(attack_test).astype(str)
    else:
        x_train, y_train, x_test, y_test = load_dataset(dname, **kwargs)
    parts = partition_data(
        x_train, y_train, int(data_cfg.get("num_clients", 4)),
        strategy=data_cfg.get("partition_strategy", "label_balanced"),
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
    )
    base = _load_fl_model(str(ROOT / entry["model"]))
    k = int(entry.get("ft_client", 0))
    cx, cy = parts[k]["x"], parts[k]["y"].astype(np.float32)
    n_val = int(spec.get("val_samples", 20000))
    perm = np.random.default_rng(7).permutation(len(cy))
    xv, yv = cx[perm[:n_val]], cy[perm[:n_val]]          # client-local validation set
    pool_x, pool_y = cx[perm[n_val:]], cy[perm[n_val:]]  # fine-tuning draws come from the rest
    clip = float(spec.get("clip", 5.0))
    targets = [float(t) for t in spec.get("recall_targets", [0.999, 0.9995])]
    fixed = [float(t) for t in spec.get("fixed_thresholds", [0.05, 0.1, 0.2, 0.3, 0.5])]
    tdir = out_dir / name
    tdir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    def missed_by_type(pt: np.ndarray, t: float) -> str:
        if attack_test is None:
            return ""
        miss = (y_test == 1) & (pt < t)
        return json.dumps(dict(collections.Counter(attack_test[miss]).most_common()))

    def evaluate(variant: str, draw: int, path: Path, size_kb: float):
        pt, pv = tflite_probs(path, x_test), tflite_probs(path, xv)
        for t in fixed:
            r = {"model": name, "variant": variant, "draw": draw, "size_kb": size_kb,
                 "selection": f"fixed {t}", **metrics_at(pt, y_test, t)}
            if abs(t - 0.3) < 1e-9:
                r["missed_by_type"] = missed_by_type(pt, t)
            rows.append(r)
        for tr in targets:
            t = threshold_for_recall(pv, yv, tr)
            r = {"model": name, "variant": variant, "draw": draw, "size_kb": size_kb,
                 "selection": f"val recall>={tr}", **metrics_at(pt, y_test, t)}
            r["val_far"] = metrics_at(pv, yv, t)["false_alarm_rate"]
            r["missed_by_type"] = missed_by_type(pt, t)
            rows.append(r)
        sel = [r for r in rows if r["variant"] == variant and r["draw"] == draw]
        print(f"  [{name} d{draw}] {variant:12s} " + " | ".join(
            f"{r['selection']}: t={r['threshold']:.4f} fn={r['fn']} far={r['false_alarm_rate']:.4f}"
            for r in sel if r["selection"] in ("fixed 0.3", f"val recall>={targets[0]}")))

    p = tdir / "fp32.tflite"
    convert(base, "fp32", xv[:500], p)
    evaluate("fp32_federated", 0, p, round(p.stat().st_size / 1024, 2))

    variants = spec.get("variants", {"hard": {}})
    for draw in range(int(spec.get("draws", 3))):
        for vname, v in variants.items():
            v = v or {}
            try:
                tf.keras.utils.set_random_seed(100 + draw)
                n_ft = int(v.get("ft_samples", spec.get("ft_samples", 10000)))
                idx = np.random.default_rng(100 + draw).permutation(len(pool_y))[:n_ft]
                xf, yf = pool_x[idx], pool_y[idx]
                if v.get("clip"):
                    xf = np.clip(xf, -clip, clip)
                if v.get("kd_alpha") is not None:
                    tgt = kd_targets(base, xf, yf, float(v.get("kd_temperature", 2.0)), float(v["kd_alpha"]))
                else:
                    tgt = yf
                w = np.where(yf == 1, float(v.get("attack_weight", 1.0)), 1.0).astype(np.float32)
                ratio = float(v.get("prune_ratio", PRUNE_RATIO))
                ft_ep = int(v.get("ft_epochs", PRUNE_FT_EPOCHS))
                qat_ep = int(v.get("qat_epochs", QAT_FT_EPOCHS))
                pr = _clone(base)
                if ratio > 0:
                    pr = apply_structured_pruning(pr, pruning_ratio=ratio, skip_last_layer=True, verbose=False)
                    pr.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                if v.get("federated"):
                    # every client fine-tunes on its own data; FedAvg after each epoch (client 0 = xf)
                    clients = [(xf, tgt, w)]
                    for cid in range(len(parts)):
                        if cid == k:
                            continue
                        jdx = np.random.default_rng(1000 * (cid + 1) + draw).permutation(len(parts[cid]["y"]))[:n_ft]
                        xc, yc = parts[cid]["x"][jdx], parts[cid]["y"][jdx].astype(np.float32)
                        if v.get("clip"):
                            xc = np.clip(xc, -clip, clip)
                        tc = (kd_targets(base, xc, yc, float(v.get("kd_temperature", 2.0)), float(v["kd_alpha"]))
                              if v.get("kd_alpha") is not None else yc)
                        clients.append((xc, tc, np.where(yc == 1, float(v.get("attack_weight", 1.0)), 1.0).astype(np.float32)))
                    for _ in range(ft_ep):
                        _fedavg_round(pr, clients, 1)
                    q = tfmot.quantization.keras.quantize_model(_strip_bn_dropout_for_qat(pr))
                    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    for _ in range(qat_ep):
                        _fedavg_round(q, clients, 1, qat=True)
                else:
                    _fit_w(pr, xf, tgt, w, ft_ep)
                    q = tfmot.quantization.keras.quantize_model(_strip_bn_dropout_for_qat(pr))
                    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                    _fit_w(q, xf, tgt, w, qat_ep)
                p = tdir / f"{vname}_d{draw}.tflite"
                export_tflite_qat(q, str(p))
                evaluate(vname, draw, p, round(p.stat().st_size / 1024, 2))
                if draw > 0:
                    p.unlink()
            except Exception as err:
                traceback.print_exc()
                rows.append({"model": name, "variant": vname, "draw": draw, "error": str(err)[:200]})
            rows_to_csv(rows, out_dir / f"{name}_partial.csv")
    return rows


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out, keys = [], []
    for r in rows:
        if "selection" in r and (r["model"], r["variant"], r["selection"]) not in keys:
            keys.append((r["model"], r["variant"], r["selection"]))
    for m, v, s in keys:
        rs = [r for r in rows if r.get("model") == m and r.get("variant") == v and r.get("selection") == s]
        out.append({"model": m, "variant": v, "selection": s, "draws": len(rs), "size_kb": rs[0]["size_kb"],
                    "threshold_mean": round(statistics.mean(r["threshold"] for r in rs), 4),
                    "fn_mean": round(statistics.mean(r["fn"] for r in rs), 1),
                    "fn_max": max(r["fn"] for r in rs),
                    "recall_mean": round(100 * statistics.mean(r["attack_recall"] for r in rs), 3),
                    "far_mean": round(100 * statistics.mean(r["false_alarm_rate"] for r in rs), 2),
                    "far_max": round(100 * max(r["false_alarm_rate"] for r in rs), 2),
                    "f1_mean": round(100 * statistics.mean(r["f1"] for r in rs), 2)})
    return out


def main():
    parser = argparse.ArgumentParser(description="Recall-priority compression and threshold selection")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for entry in spec["models"]:
        rows.extend(run_model(entry, spec, out_dir))
        rows_to_csv(rows, out_dir / "recall_priority.csv")
    summary = summarize(rows)
    rows_to_csv(summary, out_dir / "recall_priority_summary.csv")
    rows_to_markdown(summary, out_dir / "recall_priority.md",
                     "Recall-priority compression (means over fine-tuning draws; thresholds from client validation data)")
    save_json({"rows": rows, "summary": summary}, out_dir / "recall_priority.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
