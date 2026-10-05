#!/usr/bin/env python3
"""
Federated-training sweep for TON_IoT with selection on the validation split only.

Each variant overrides the base federated config, is trained with data.eval_split = "val"
(10% of the training split held out, test split never loaded) and is scored on that
validation set. The selection criterion is recall-priority: the false-alarm rate at the
threshold that reaches `selection_recall` attack recall on validation (lower is better),
with validation F1 at the fixed threshold as tie-break. The selected variant is then
retrained on the full training split and evaluated once on the test split, together with
the reference model (the federated model used so far), including missed attacks per type.

Spec (YAML):
  base_config: <b_fl.yaml>
  reference_model: <b_fl.h5>          # current federated model, evaluated on test for comparison
  selection_recall: 0.999
  variants:                           # name -> nested overrides of the base config
    base: {}
    alpha05: {federated: {focal_loss_alpha: 0.5}}
  retrain_best: true

Usage:
  python scripts/ton_sweep.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import collections
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from scripts.ablation_utils import load_keras_for_eval, load_yaml, rows_to_csv, rows_to_markdown, save_json
from src.data.loader import load_dataset


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_split(cfg: dict, eval_split: str):
    data_cfg = dict(cfg["data"])
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients", "eval_split"}}
    if "path" in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    return load_dataset(data_cfg["name"], eval_split=eval_split, return_attack_labels=True, **kwargs)


def scores(model, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict(x, batch_size=4096, verbose=0)).reshape(-1)


def metrics(p: np.ndarray, y: np.ndarray, t: float) -> Dict[str, Any]:
    pred, pos = p >= t, y == 1
    tp, fn = int(np.sum(pred & pos)), int(np.sum(~pred & pos))
    fp, tn = int(np.sum(pred & ~pos)), int(np.sum(~pred & ~pos))
    rec, far = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    prec = tp / max(tp + fp, 1)
    return {"threshold": round(float(t), 6), "recall": rec, "far": far, "precision": prec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-12), "fn": fn, "fp": fp}


def threshold_for_recall(p: np.ndarray, y: np.ndarray, target: float) -> float:
    att = np.sort(p[y == 1])
    return float(att[int(np.floor((1.0 - target) * len(att)))])


def summarize(name: str, split: str, p: np.ndarray, y: np.ndarray, types: np.ndarray,
              target: float, fixed: float) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    at_fixed = metrics(p, y, fixed)
    t = threshold_for_recall(p, y, target)
    at_target = metrics(p, y, t)
    miss = (y == 1) & (p < fixed)
    return {"variant": name, "split": split, "n": int(len(y)), "attacks": int(np.sum(y)),
            "roc_auc": round(float(roc_auc_score(y, p)), 6), "pr_auc": round(float(average_precision_score(y, p)), 6),
            "f1_fixed": round(100 * at_fixed["f1"], 3), "recall_fixed": round(100 * at_fixed["recall"], 3),
            "far_fixed": round(100 * at_fixed["far"], 3), "fn_fixed": at_fixed["fn"],
            "threshold_target": round(t, 6), "recall_target": round(100 * at_target["recall"], 3),
            "far_at_target": round(100 * at_target["far"], 3), "fn_target": at_target["fn"],
            "missed_by_type_fixed": json.dumps(dict(collections.Counter(np.asarray(types)[miss].astype(str)).most_common()))}


def train(cfg: dict, name: str, out_dir: Path) -> Path:
    cfg_path = out_dir / "configs" / f"{name}.yaml"
    model_path = out_dir / "models" / f"{name}.h5"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log = out_dir / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.run([sys.executable, "-m", "src.federated.client", "--config", str(cfg_path),
                             "--save-model", str(model_path)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"  trained {name} in {(time.time() - t0) / 60:.1f} min (rc={rc})")
    if rc != 0 or not model_path.exists():
        raise RuntimeError(f"training {name} failed (rc={rc}); see {log}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="TON_IoT federated training sweep (val selection)")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = load_yaml(ROOT / spec["base_config"])
    target = float(spec.get("selection_recall", 0.999))
    fixed = float(base.get("evaluation", {}).get("prediction_threshold", 0.3))
    rows: List[Dict[str, Any]] = []

    for name, over in spec["variants"].items():
        cfg = deep_merge(base, over)
        cfg["data"]["eval_split"] = "val"
        cfg.setdefault("compression", {})
        try:
            mp = train(cfg, name, out_dir)
            _, _, xv, yv, _, tv = load_split(cfg, "val")
            row = summarize(name, "val", scores(load_keras_for_eval(mp), xv), yv, tv, target, fixed)
            row["overrides"] = json.dumps(over)
        except Exception as err:
            row = {"variant": name, "split": "val", "error": str(err)[:300]}
        rows.append(row)
        print(f"[val] {row}")
        rows_to_csv(rows, out_dir / "ton_sweep.csv")

    ok = [r for r in rows if r.get("split") == "val" and "far_at_target" in r]
    if ok and spec.get("retrain_best", True):
        best = min(ok, key=lambda r: (r["far_at_target"], -r["f1_fixed"]))
        print(f"selected on validation: {best['variant']} (FAR {best['far_at_target']}% at recall >= {target})")
        cfg = deep_merge(base, spec["variants"][best["variant"]])
        cfg["data"]["eval_split"] = "test"
        try:
            mp = train(cfg, f"best_{best['variant']}_test", out_dir)
            _, _, xt, yt, _, tt = load_split(cfg, "test")
            r = summarize(f"best_{best['variant']}", "test", scores(load_keras_for_eval(mp), xt), yt, tt, target, fixed)
            r["model_path"] = str(mp.relative_to(ROOT)) if mp.is_relative_to(ROOT) else str(mp)
            r["config_path"] = str((out_dir / "configs" / f"best_{best['variant']}_test.yaml"))
            rows.append(r)
            print(f"[test] {r}")
        except Exception as err:
            rows.append({"variant": f"best_{best['variant']}", "split": "test", "error": str(err)[:300]})
    if spec.get("reference_model"):
        _, _, xt, yt, _, tt = load_split(base, "test")
        r = summarize("reference", "test", scores(load_keras_for_eval(ROOT / spec["reference_model"]), xt),
                      yt, tt, target, fixed)
        rows.append(r)
        print(f"[test] {r}")

    rows_to_csv(rows, out_dir / "ton_sweep.csv")
    rows_to_markdown(rows, out_dir / "ton_sweep.md", "TON_IoT federated training sweep (selection on validation)")
    save_json(rows, out_dir / "ton_sweep.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
