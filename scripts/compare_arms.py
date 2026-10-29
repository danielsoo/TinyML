#!/usr/bin/env python3
"""
F / P / Q / R / S comparison under validation-chosen false-alarm budgets (Dr. Saha's guide E1, E2).

For every teacher seed and memory budget:
  F   federated teacher (float), plus F8 = teacher with strong PTQ (over budget; reference)
  P   structured prune -> federated fine-tune on client data -> strong calibrated PTQ
  Q   the same pruned + fine-tuned model -> QAT on client data -> INT8
  R   prune -> fine-tune distilled from F on clipped inputs -> QAT on clipped inputs -> INT8
  S   compact model trained federatively from scratch (train_fl step) -> strong calibrated PTQ
  SQ  S -> QAT on client data -> INT8
  QC / SQC  as Q / SQ, but QAT on inputs clipped to +-clip SD (the learned input range is then bounded)
"Strong calibrated PTQ" picks, per model, the calibration (TFLite min/max per channel, with or
without +-c SD input clipping, or per-tensor ranges from the 99.9 / 99.99th percentile, MSE or KL
estimators) with the best mean validation attack recall at the FAR budgets.

Thresholds: for each model and stage, the smallest threshold whose false-alarm rate on the pooled
client validation slices is <= the budget. They are frozen and applied once to the test split.
Stages: teacher, compact FP32 (after fine-tuning), INT8. E2 counts, per attack family, the test
attacks lost / recovered between consecutive stages (each at its own frozen threshold).

The split is rebuilt from dataset_config with the v13 pipeline and checked against the manifest
hashes saved with the checkpoints, so every arm sees exactly the training-time partitions.

Spec (YAML):
  dataset_config: config/v13/ton_fl.yaml
  train_dir: data/processed/revision/2026-10-28_y2_train_ton/train_fl
  seeds: [1, 2, 3]                 # teacher F_s<seed>, scratch S80_s<seed> / S128_s<seed>
  budgets: {80: {prune_ratio: 0.5, scratch: S80}, 128: {prune_ratio: 0.3, scratch: S128}}
  far_budgets: [0.01, 0.05]
  arms: [F8, P, Q, R, S, SQ]
  fine_tune: {mode: federated, samples_per_client: 10000, epochs: 3, qat_epochs: 2,
              kd_alpha: 0.5, kd_temperature: 2.0, clip: 5.0, client: 0}
  calib: {n: 2000, client: 0, estimators: [tfl_max, tfl_clip5, pct999, pct9999, mse, kl, clip5_pct9999]}

Usage:
  python scripts/compare_arms.py --spec <spec.yaml> --output-dir <dir>
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
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from sklearn.metrics import average_precision_score, roc_auc_score
from tensorflow import keras

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import _clone, _load_fl_model
from scripts.quant_distill_ablation import convert, kd_targets
from scripts.quant_lit_methods import FreezeRanges, estimate_ranges, fq_export, set_ranges
from scripts.recall_priority import _fedavg_round, _fit_w, tflite_probs
from src.data.fed_pipeline import build_federated
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_qat, export_tflite_qat

BENIGN = {"benign", "normal"}


# --------------------------------------------------------------------------- thresholds / metrics
def threshold_at_far(pv: np.ndarray, yv: np.ndarray, far: float) -> float:
    """Smallest threshold t with mean(pv[benign] >= t) <= far (ties in INT8 outputs respected)."""
    ben = np.sort(pv[yv == 0].astype(np.float32))
    k = int(np.floor(far * len(ben)))  # false alarms allowed
    if k >= len(ben):
        return 0.0
    # next float32 above the (k+1)-th largest benign score. Kept in float32: with NumPy 1.x a
    # float64 nextafter is cast back to float32 in `p >= t`, which re-admits the whole tie group
    # (a large one for discretized INT8 outputs).
    return float(np.nextafter(ben[len(ben) - k - 1], np.float32(np.inf), dtype=np.float32))


def metrics(p: np.ndarray, y: np.ndarray, t: float) -> Dict[str, Any]:
    pred, pos = p >= t, y == 1
    tp = int(np.sum(pred & pos)); fn = int(np.sum(~pred & pos))
    fp = int(np.sum(pred & ~pos)); tn = int(np.sum(~pred & ~pos))
    rec = tp / max(tp + fn, 1); far = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    return {"recall": rec, "far": far, "precision": prec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "fn": fn, "fp": fp}


def family_recall(p: np.ndarray, y: np.ndarray, fam: np.ndarray, t: float) -> Dict[str, str]:
    out = {}
    for f in sorted(set(fam[y == 1])):
        m = (fam == f) & (y == 1)
        out[f] = f"{int(np.sum(p[m] >= t))}/{int(m.sum())}"
    return out


# --------------------------------------------------------------------------- data
def load_data(spec: dict):
    cfg = load_yaml(ROOT / spec["dataset_config"])
    d = dict(cfg["data"])
    kw = {k: v for k, v in d.items() if k not in {"name", "path", "data_path", "num_clients", "pipeline"}}
    fd = build_federated(d["name"], d.get("path") or d.get("data_path"), num_clients=int(d.get("num_clients", 4)), **kw)
    return fd


def check_manifest(fd, man_path: Path) -> None:
    m = json.loads(man_path.read_text())
    ok = m["test"]["hash"] == fd.manifest["test"]["hash"] and all(
        a["train"]["hash"] == b["train"]["hash"] and a["val"]["hash"] == b["val"]["hash"]
        for a, b in zip(m["clients"], fd.manifest["clients"]))
    if not ok:
        raise RuntimeError(f"split differs from the one used to train {man_path.name}")


# --------------------------------------------------------------------------- compression arms
class Ctx:
    def __init__(self, fd, spec: dict, out: Path):
        self.fd, self.spec, self.out = fd, spec, out
        self.ft = {"mode": "federated", "samples_per_client": 10000, "epochs": 3, "qat_epochs": 2,
                   "kd_alpha": 0.5, "kd_temperature": 2.0, "clip": 5.0, "client": 0, **spec.get("fine_tune", {})}
        cal = {"n": 2000, "client": 0, "estimators": ["tfl_max", "tfl_clip5", "pct999", "pct9999", "mse", "kl",
                                                      "clip5_pct9999"], **spec.get("calib", {})}
        self.cal = cal
        val = fd.pooled_val()
        self.xv, self.yv = val.x, val.y.astype(np.float32)
        self.xt, self.yt, self.fam_t = fd.test.x, fd.test.y.astype(np.float32), np.asarray(fd.test.attack).astype(str)
        cx = fd.clients[int(cal["client"])].train.x
        self.xcal = cx[np.random.default_rng(11).permutation(len(cx))[:int(cal["n"])]]
        self.fars = [float(b) for b in spec.get("far_budgets", [0.01, 0.05])]

    def client_sets(self, seed: int, clip: Optional[float] = None, teacher=None):
        """Fine-tuning sets [(x, target, weight)] (one per participating client)."""
        ids = range(len(self.fd.clients)) if self.ft["mode"] == "federated" else [int(self.ft["client"])]
        sets = []
        for cid in ids:
            tr = self.fd.clients[cid].train
            idx = np.random.default_rng(1000 * (cid + 1) + seed).permutation(len(tr.y))[:int(self.ft["samples_per_client"])]
            x, y = tr.x[idx], tr.y[idx].astype(np.float32)
            if clip:
                x = np.clip(x, -clip, clip)
            t = kd_targets(teacher, x, y, float(self.ft["kd_temperature"]), float(self.ft["kd_alpha"])) if teacher is not None else y
            sets.append((x, t, np.ones(len(y), np.float32)))
        return sets

    def train(self, model, sets, epochs: int, qat: bool = False, callbacks=None) -> None:
        if len(sets) == 1:
            _fit_w(model, *sets[0], epochs, callbacks)
        else:
            for _ in range(epochs):
                _fedavg_round(model, sets, 1, qat=qat, callbacks=callbacks)

    def qat(self, model, sets) -> keras.Model:
        q = tfmot.quantization.keras.quantize_model(_strip_bn_dropout_for_qat(_clone(model)))
        q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        self.train(q, sets, int(self.ft["qat_epochs"]), qat=True)
        return q

    def strong_ptq(self, model, path: Path) -> Tuple[Path, str, Dict[str, float]]:
        """Build every calibration candidate, keep the one with the best validation recall."""
        scores = {}
        best = None
        for est in self.cal["estimators"]:
            p = path.with_name(f"{path.stem}__{est}.tflite")
            try:
                if est.startswith("tfl_"):
                    rep = self.xcal if est == "tfl_max" else np.clip(self.xcal, -float(est[8:]), float(est[8:]))
                    convert(model, "int8", rep, p)
                else:
                    fq_export(model, estimate_ranges(model, self.xcal, est), p)
                pv = tflite_probs(p, self.xv)
                s = statistics.mean(metrics(pv, self.yv, threshold_at_far(pv, self.yv, b))["recall"] for b in self.fars)
                scores[est] = round(s, 6)
                if best is None or s > best[1]:
                    best = (est, s)
            except Exception as err:  # an estimator can fail on degenerate ranges
                scores[est] = f"error: {str(err)[:80]}"
        for est in self.cal["estimators"]:
            p = path.with_name(f"{path.stem}__{est}.tflite")
            if best and est == best[0]:
                p.replace(path)
            elif p.exists():
                p.unlink()
        return path, best[0], scores


def keras_probs(model, x):
    return model.predict(x, batch_size=4096, verbose=0).ravel()


# --------------------------------------------------------------------------- evaluation
def run_seed(ctx: Ctx, seed: int, rows: list, trans: list, sel: dict) -> None:
    spec, out, fd = ctx.spec, ctx.out, ctx.fd
    tdir = Path(spec["train_dir"]).resolve() if Path(spec["train_dir"]).is_absolute() else ROOT / spec["train_dir"]
    sdir = out / f"s{seed}"
    sdir.mkdir(parents=True, exist_ok=True)
    check_manifest(fd, tdir / "models" / f"F_s{seed}_manifest.json")
    teacher = _load_fl_model(str(tdir / "models" / f"F_s{seed}.h5"))
    clip = float(ctx.ft["clip"])
    arms = spec.get("arms", ["F8", "P", "Q", "R", "S", "SQ"])
    scores: Dict[str, np.ndarray] = {}
    val_scores: Dict[str, np.ndarray] = {}  # saved too, so thresholds can be re-derived offline

    def record(arm: str, budget: str, stage: str, pv: np.ndarray, pt: np.ndarray, size_kb: Optional[float]):
        key = f"{arm}|{budget}|{stage}"
        scores[key] = pt.astype(np.float32)
        val_scores[key] = pv.astype(np.float32)  # saved as float16 (enough for INT8 outputs, 1/256 steps)
        thr = {}
        try:
            auc, ap = roc_auc_score(ctx.yt, pt), average_precision_score(ctx.yt, pt)
        except ValueError:
            auc = ap = float("nan")
        for b in ctx.fars:
            t = threshold_at_far(pv, ctx.yv, b)
            thr[b] = t
            vm, tm = metrics(pv, ctx.yv, t), metrics(pt, ctx.yt, t)
            assert vm["far"] <= b + 1e-12, f"{arm} {stage}: validation FAR {vm['far']:.4f} above budget {b}"
            rows.append({"seed": seed, "arm": arm, "budget_kb": budget, "stage": stage, "far_budget": b,
                         "size_kb": size_kb, "threshold": t, "val_far": vm["far"], "val_recall": vm["recall"],
                         **{f"test_{k}": v for k, v in tm.items()}, "test_roc_auc": auc, "test_pr_auc": ap,
                         "family_recall": json.dumps(family_recall(pt, ctx.yt, ctx.fam_t, t))})
        print(f"  s{seed} {arm:3s} {budget:>4s} {stage:5s} size={size_kb} " + " | ".join(
            f"FAR{b:g}: rec={r['test_recall']:.4f} far={r['test_far']:.4f}"
            for b in ctx.fars for r in rows[-len(ctx.fars):] if r["far_budget"] == b), flush=True)
        return thr

    def transitions(arm: str, budget: str, chain: List[Tuple[str, np.ndarray, Dict[float, float]]]):
        att = ctx.yt == 1
        for b in ctx.fars:
            for (s0, p0, t0), (s1, p1, t1) in zip(chain[:-1], chain[1:]):
                d0, d1 = p0 >= t0[b], p1 >= t1[b]
                lost, rec = att & d0 & ~d1, att & ~d0 & d1
                ben = ~att
                trans.append({"seed": seed, "arm": arm, "budget_kb": budget, "far_budget": b, "from": s0, "to": s1,
                              "attacks_lost": int(lost.sum()), "attacks_recovered": int(rec.sum()),
                              "false_alarms_added": int((ben & ~d0 & d1).sum()),
                              "false_alarms_removed": int((ben & d0 & ~d1).sum()),
                              "lost_by_family": json.dumps(dict(collections.Counter(ctx.fam_t[lost]).most_common())),
                              "recovered_by_family": json.dumps(dict(collections.Counter(ctx.fam_t[rec]).most_common()))})

    # teacher
    pf = tdir / "models" / f"F_s{seed}.h5"
    f_v, f_t = keras_probs(teacher, ctx.xv), keras_probs(teacher, ctx.xt)
    f_thr = record("F", "-", "fp32", f_v, f_t, None)
    teacher_link = ("teacher", f_t, f_thr)
    if "F8" in arms:
        p, est, sc = ctx.strong_ptq(teacher, sdir / "F8.tflite")
        sel[f"s{seed}|F8"] = {"chosen": est, "val_scores": sc}
        thr8 = record("F8", "-", "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
        transitions("F8", "-", [teacher_link, ("int8", scores["F8|-|int8"], thr8)])

    for budget, bcfg in spec.get("budgets", {}).items():
        budget = str(budget)
        ratio = float(bcfg["prune_ratio"])
        try:
            tf.keras.utils.set_random_seed(100 * seed + 1)
            if {"P", "Q", "QC"} & set(arms):
                pr = apply_structured_pruning(_clone(teacher), pruning_ratio=ratio, skip_last_layer=True, verbose=False)
                pr.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                sets = ctx.client_sets(seed)
                ctx.train(pr, sets, int(ctx.ft["epochs"]))
                pr.save(sdir / f"PQ_{budget}.h5")  # compact FP32 model, reused by range_mechanism.py (E3/E4)
                c_thr = record("PQ", budget, "fp32", keras_probs(pr, ctx.xv), keras_probs(pr, ctx.xt), None)
                c_link = ("compact_fp32", scores[f"PQ|{budget}|fp32"], c_thr)
                if "P" in arms:
                    p, est, sc = ctx.strong_ptq(pr, sdir / f"P_{budget}.tflite")
                    sel[f"s{seed}|P|{budget}"] = {"chosen": est, "val_scores": sc}
                    t8 = record("P", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("P", budget, [teacher_link, c_link, ("int8", scores[f"P|{budget}|int8"], t8)])
                if "Q" in arms:
                    q = ctx.qat(pr, sets)
                    qf = record("Q", budget, "qat", keras_probs(q, ctx.xv), keras_probs(q, ctx.xt), None)
                    p = sdir / f"Q_{budget}.tflite"
                    export_tflite_qat(q, str(p))
                    t8 = record("Q", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("Q", budget, [teacher_link, c_link, ("qat_float", scores[f"Q|{budget}|qat"], qf),
                                              ("int8", scores[f"Q|{budget}|int8"], t8)])
                if "QC" in arms:
                    # QAT on inputs clipped to +-clip SD (hard labels): isolates the learned input range
                    tf.keras.utils.set_random_seed(100 * seed + 4)
                    q = ctx.qat(pr, ctx.client_sets(seed, clip=clip))
                    p = sdir / f"QC_{budget}.tflite"
                    export_tflite_qat(q, str(p))
                    t8 = record("QC", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("QC", budget, [teacher_link, c_link, ("int8", scores[f"QC|{budget}|int8"], t8)])
            if "R" in arms:
                tf.keras.utils.set_random_seed(100 * seed + 2)
                pr = apply_structured_pruning(_clone(teacher), pruning_ratio=ratio, skip_last_layer=True, verbose=False)
                pr.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                sets = ctx.client_sets(seed, clip=clip, teacher=teacher)
                ctx.train(pr, sets, int(ctx.ft["epochs"]))
                xv_c, xt_c = np.clip(ctx.xv, -clip, clip), np.clip(ctx.xt, -clip, clip)
                pr.save(sdir / f"R_{budget}.h5")
                c_thr = record("R", budget, "fp32", keras_probs(pr, xv_c), keras_probs(pr, xt_c), None)
                q = ctx.qat(pr, sets)
                p = sdir / f"R_{budget}.tflite"
                export_tflite_qat(q, str(p))
                t8 = record("R", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                transitions("R", budget, [teacher_link, ("compact_fp32", scores[f"R|{budget}|fp32"], c_thr),
                                          ("int8", scores[f"R|{budget}|int8"], t8)])
            if {"S", "SQ", "SQC"} & set(arms) and bcfg.get("scratch"):
                sname = f"{bcfg['scratch']}_s{seed}"
                check_manifest(fd, tdir / "models" / f"{sname}_manifest.json")
                sm = _load_fl_model(str(tdir / "models" / f"{sname}.h5"))
                s_thr = record("S", budget, "fp32", keras_probs(sm, ctx.xv), keras_probs(sm, ctx.xt), None)
                s_link = ("compact_fp32", scores[f"S|{budget}|fp32"], s_thr)
                if "S" in arms:
                    p, est, sc = ctx.strong_ptq(sm, sdir / f"S_{budget}.tflite")
                    sel[f"s{seed}|S|{budget}"] = {"chosen": est, "val_scores": sc}
                    t8 = record("S", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("S", budget, [teacher_link, s_link, ("int8", scores[f"S|{budget}|int8"], t8)])
                if "SQ" in arms:
                    tf.keras.utils.set_random_seed(100 * seed + 3)
                    q = ctx.qat(sm, ctx.client_sets(seed))
                    p = sdir / f"SQ_{budget}.tflite"
                    export_tflite_qat(q, str(p))
                    t8 = record("SQ", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("SQ", budget, [teacher_link, s_link, ("int8", scores[f"SQ|{budget}|int8"], t8)])
                if "SQC" in arms:
                    tf.keras.utils.set_random_seed(100 * seed + 5)
                    q = ctx.qat(sm, ctx.client_sets(seed, clip=clip))
                    p = sdir / f"SQC_{budget}.tflite"
                    export_tflite_qat(q, str(p))
                    t8 = record("SQC", budget, "int8", tflite_probs(p, ctx.xv), tflite_probs(p, ctx.xt), round(p.stat().st_size / 1024, 2))
                    transitions("SQC", budget, [teacher_link, s_link, ("int8", scores[f"SQC|{budget}|int8"], t8)])
        except Exception as err:
            traceback.print_exc()
            rows.append({"seed": seed, "budget_kb": budget, "error": str(err)[:200]})
    np.savez_compressed(sdir / "test_scores.npz", **{k.replace("|", "__"): v.astype(np.float16) for k, v in scores.items()})
    np.savez_compressed(sdir / "val_scores.npz", **{k.replace("|", "__"): v.astype(np.float16) for k, v in val_scores.items()})


def summarize(rows: list) -> list:
    groups = collections.defaultdict(list)
    for r in rows:
        if "error" not in r:
            groups[(r["arm"], r["budget_kb"], r["stage"], r["far_budget"])].append(r)
    out = []
    for (arm, budget, stage, b), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][1]), kv[0][0], kv[0][2], kv[0][3])):
        def ms(k, scale=100.0, nd=2):
            v = [float(r[k]) * scale for r in rs]
            return f"{statistics.mean(v):.{nd}f} ± {statistics.stdev(v):.{nd}f}" if len(v) > 1 else f"{v[0]:.{nd}f}"
        out.append({"arm": arm, "budget_kb": budget, "stage": stage, "far_budget": b, "seeds": len(rs),
                    "size_kb": rs[0]["size_kb"], "test_recall_%": ms("test_recall"), "test_far_%": ms("test_far"),
                    "val_far_%": ms("val_far"), "test_f1_%": ms("test_f1"), "test_fn": ms("test_fn", 1.0, 0),
                    "test_pr_auc_%": ms("test_pr_auc")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    spec = load_yaml(args.spec)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fd = load_data(spec)
    ctx = Ctx(fd, spec, out)
    rows, trans, sel = [], [], {}
    for seed in spec.get("seeds", [1, 2, 3]):
        print(f"=== teacher seed {seed}", flush=True)
        try:
            run_seed(ctx, int(seed), rows, trans, sel)
        except Exception as err:
            traceback.print_exc()
            rows.append({"seed": seed, "error": str(err)[:200]})
        rows_to_csv(rows, out / "compare_arms_rows.csv")
        rows_to_csv(trans, out / "compare_arms_transitions.csv")
        save_json(sel, out / "ptq_selection.json")
    summ = summarize(rows)
    rows_to_csv(summ, out / "compare_arms_summary.csv")
    rows_to_markdown(summ, out / "compare_arms.md",
                     "F/P/Q/R/S under validation-chosen FAR budgets (mean ± SD over teacher seeds; test applied once)")
    errors = [r for r in rows if "error" in r]
    if errors:
        print(f"❌ {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
