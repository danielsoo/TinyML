#!/usr/bin/env python3
"""
Checks requested after the v13.1 re-run (Dr. Saha, 2026-11-05), computed from existing checkpoints,
predictions and thresholds; nothing is retrained and every threshold stays frozen.

  provenance  For the random v13.1 split of each dataset and the host-grouped TON_IoT split (E6):
              how the new test / client-train / client-validation records relate to the old
              pipeline's split (old test, old training pool, records the old pipeline dropped),
              plus the manifest hashes of the rebuilt split.
  exports     Strong PTQ restricted to per-tensor weight scales (deployable on the TFLM 1.0.0 runtime)
              for every strong-PTQ arm (F8, P, S) and seed: test recall and FAR at thresholds chosen on
              validation (1% / 5%), size, SHA-256 and the selected calibration, next to the Table 2
              export it replaces. The files flashed in E7 are hashed and evaluated as well.
  paired      Record-level paired comparisons between arms on the same test records, from the saved
              test scores and the frozen thresholds of compare_arms_rows.csv: per run, attacks
              detected by one arm and missed by the other (and the same for false alarms), with an
              exact two-sided McNemar p-value. Recomputed recall / FAR are checked against the rows.

Spec (YAML): see config/jobs/<job>/saha_checks.yaml.
Usage: python scripts/saha_checks.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.ablation_utils import load_yaml
from src.data import fed_pipeline as fp


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(cfg_path: str):
    cfg = load_yaml(ROOT / cfg_path)
    d = dict(cfg["data"])
    kw = {k: v for k, v in d.items() if k not in {"name", "path", "data_path", "num_clients", "pipeline"}}
    return d, fp.build_federated(d["name"], d["path"], num_clients=int(d.get("num_clients", 4)), **kw)


# --------------------------------------------------------------------------- provenance
def old_split(name: str, raw) -> tuple[set, set]:
    """Old pipeline (src/data/loader.py): record ids kept by its dedup, and its test ids."""
    if name.lower().startswith("cic"):
        df = pd.DataFrame(raw.X)
        df["_a"] = raw.attack
    else:
        df = pd.DataFrame(raw.X[:, :raw.info["n_base_features"]])
        df["_y"] = raw.y
    keep = ~df.duplicated().values
    return set(raw.ids[keep].astype(str)), set(fp.old_test_ids(name, raw))


def provenance(entry: dict) -> dict:
    d, fd = build(entry["dataset_config"])
    raw = fp.load_raw(d["name"], d["path"], text_features=d.get("text_features", True))
    old_kept, old_test = old_split(d["name"], raw)
    old_train = old_kept - old_test

    def rel(ids):
        ids = set(np.asarray(ids).astype(str))
        return {"records": len(ids), "in_old_test": len(ids & old_test), "in_old_training_pool": len(ids & old_train),
                "dropped_by_old_dedup": len(ids - old_kept)}

    out = {"config": entry["dataset_config"], "manifest_test_hash": fd.manifest["test"]["hash"],
           "manifest_client_hashes": [(c["train"]["hash"], c["val"]["hash"]) for c in fd.manifest["clients"]],
           "old_pipeline": {"records_after_old_dedup": len(old_kept), "old_test": len(old_test)},
           "new_test": rel(fd.test.ids),
           "new_client_train": rel(np.concatenate([c.train.ids for c in fd.clients])),
           "new_client_val": rel(np.concatenate([c.val.ids for c in fd.clients])),
           "group_split": fd.manifest.get("group_split")}
    return out


# --------------------------------------------------------------------------- exports
def evaluate_tflite(path: Path, xv, yv, xt, yt, fars) -> dict:
    from scripts.compare_arms import metrics, threshold_at_far
    from scripts.recall_priority import tflite_probs
    pv, pt = tflite_probs(path, xv), tflite_probs(path, xt)
    res = {"file": str(Path(path).relative_to(ROOT)) if str(path).startswith(str(ROOT)) else str(path),
           "bytes": Path(path).stat().st_size, "sha256": sha256(path)}
    for b in fars:
        t = threshold_at_far(pv, yv, b)
        m = metrics(pt, yt, t)
        res[str(b)] = {"threshold": t, "val_far": metrics(pv, yv, t)["far"], "test_recall": m["recall"],
                       "test_far": m["far"], "test_fn": m["fn"], "test_fp": m["fp"]}
    return res


def exports(entry: dict, out: Path, fars) -> dict:
    """Strong PTQ restricted to per-tensor weight scales (the deployable form on the TFLM 1.0.0 runtime),
    selected exactly like compare_arms.strong_ptq (best mean validation recall at the FAR budgets, same
    2,000-record client-0 calibration draw), next to the Table 2 export it replaces."""
    import statistics
    from scripts.compare_arms import metrics, threshold_at_far
    from scripts.compression_ablation import _load_fl_model
    from scripts.quant_lit_methods import convert_int8_per_tensor, estimate_ranges, fq_export
    from scripts.recall_priority import tflite_probs
    d, fd = build(entry["dataset_config"])
    val = fd.pooled_val()
    xv, yv, xt, yt = val.x, val.y.astype(np.float32), fd.test.x, fd.test.y.astype(np.float32)
    xc = fd.clients[0].train.x
    xcal = xc[np.random.default_rng(11).permutation(len(xc))[:2000]]   # compare_arms calibration draw
    cands = ["tfl_max_pt", "tfl_clip5_pt", "pct999", "pct9999", "mse", "kl", "clip5_pct9999"]
    edir = out / "exports" / entry["name"]
    edir.mkdir(parents=True, exist_ok=True)
    res = {}
    for seed in entry.get("seeds", [1, 2, 3]):
        for arm, keras_tpl, pc_tpl in entry["models"]:
            k = f"{arm}_s{seed}"
            model = _load_fl_model(str(ROOT / keras_tpl.format(seed=seed)))
            scores, best = {}, None
            for est in cands:
                p = edir / f"{k}__{est}.tflite"
                try:
                    if est.startswith("tfl_"):
                        rep = xcal if est == "tfl_max_pt" else np.clip(xcal, -5.0, 5.0)
                        convert_int8_per_tensor(model, rep, p)
                    else:
                        fq_export(model, estimate_ranges(model, xcal, est), p)
                    pv = tflite_probs(p, xv)
                    sc = statistics.mean(metrics(pv, yv, threshold_at_far(pv, yv, b))["recall"] for b in fars)
                    scores[est] = round(sc, 6)
                    if best is None or sc > best[1]:
                        best = (est, sc)
                except Exception as err:  # noqa: BLE001
                    scores[est] = f"error: {str(err)[:80]}"
            for est in cands:
                p = edir / f"{k}__{est}.tflite"
                if best and est == best[0]:
                    p.replace(edir / f"{k}_per_tensor.tflite")
                elif p.exists():
                    p.unlink()
            res[k] = {"table2_export": evaluate_tflite(ROOT / pc_tpl.format(seed=seed), xv, yv, xt, yt, fars),
                      "per_tensor_strong_ptq": evaluate_tflite(edir / f"{k}_per_tensor.tflite", xv, yv, xt, yt, fars),
                      "per_tensor_selected": best[0] if best else None, "per_tensor_val_scores": scores}
            print(f"  {k}: Table-2 export recall@1% {res[k]['table2_export']['0.01']['test_recall']:.4f} "
                  f"per-tensor ({best[0] if best else '-'}) {res[k]['per_tensor_strong_ptq']['0.01']['test_recall']:.4f}",
                  flush=True)
    for name, path in entry.get("bundle_files", {}).items():
        res[f"bundle:{name}"] = evaluate_tflite(ROOT / path, xv, yv, xt, yt, fars)
    return res


# --------------------------------------------------------------------------- paired
def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(entry: dict) -> dict:
    d, fd = build(entry["dataset_config"])
    yt = fd.test.y.astype(int)
    att, ben = yt == 1, yt == 0
    rows = {}
    for f in entry["rows_files"]:
        for r in csv.DictReader(open(ROOT / f)):
            if r.get("error"):
                continue
            rows[(int(r["seed"]), r["arm"], r["budget_kb"], r["stage"], r["far_budget"])] = (r, f)
    res = {"checks": [], "pairs": []}
    for seed in entry.get("seeds", [1, 2, 3]):
        dec = {}
        for arm, budget, stage in entry["arms"]:
            for b in entry.get("far_budgets", ["0.01", "0.05"]):
                key = (seed, arm, budget, stage, b)
                if key not in rows:
                    continue
                r, f = rows[key]
                npz = ROOT / Path(f).parent / f"s{seed}" / "test_scores.npz"
                s = np.load(npz)[f"{arm}__{budget}__{stage}"].astype(np.float32)
                t = float(r["threshold"])
                dd = s >= np.float32(t)
                dec[(arm, b)] = dd
                rec, far = float(dd[att].mean()), float(dd[ben].mean())
                res["checks"].append({"seed": seed, "arm": arm, "far_budget": b, "recall_recomputed": rec,
                                      "recall_rows": float(r["test_recall"]), "far_recomputed": far,
                                      "far_rows": float(r["test_far"]),
                                      "attack_decisions_differing_from_rows": int(round(abs(rec - float(r["test_recall"])) * att.sum()))})
        for a, bb in entry["pairs"]:
            for b in entry.get("far_budgets", ["0.01", "0.05"]):
                if (a, b) not in dec or (bb, b) not in dec:
                    continue
                A, B = dec[(a, b)], dec[(bb, b)]
                a_only, b_only = int((att & A & ~B).sum()), int((att & ~A & B).sum())
                fa_a, fa_b = int((ben & A & ~B).sum()), int((ben & ~A & B).sum())
                res["pairs"].append({"seed": seed, "far_budget": b, "pair": f"{a} vs {bb}",
                                     "attacks_only_A_detects": a_only, "attacks_only_B_detects": b_only,
                                     "recall_diff_pts": 100 * (a_only - b_only) / att.sum(),
                                     "mcnemar_p_attacks": mcnemar_p(a_only, b_only),
                                     "false_alarms_only_A": fa_a, "false_alarms_only_B": fa_b,
                                     "far_diff_pts": 100 * (fa_a - fa_b) / ben.sum(),
                                     "mcnemar_p_benign": mcnemar_p(fa_a, fa_b)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    a = ap.parse_args()
    spec = load_yaml(a.spec)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fars = [float(b) for b in spec.get("far_budgets", [0.01, 0.05])]
    report = {}
    for name, entry in spec.get("provenance", {}).items():
        print(f"=== provenance {name}", flush=True)
        report.setdefault("provenance", {})[name] = provenance(entry)
        (out / "saha_checks.json").write_text(json.dumps(report, indent=2, default=str))
    for name, entry in spec.get("exports", {}).items():
        print(f"=== exports {name}", flush=True)
        report.setdefault("exports", {})[name] = exports({**entry, "name": name}, out, fars)
        (out / "saha_checks.json").write_text(json.dumps(report, indent=2, default=str))
    for name, entry in spec.get("paired", {}).items():
        print(f"=== paired {name}", flush=True)
        report.setdefault("paired", {})[name] = paired(entry)
        (out / "saha_checks.json").write_text(json.dumps(report, indent=2, default=str))
    try:
        import tensorflow as tf
        report["environment"] = {"tensorflow": tf.__version__, "numpy": np.__version__, "python": sys.version.split()[0]}
    except Exception as err:  # noqa: BLE001
        report["environment"] = {"error": repr(err)}
    (out / "saha_checks.json").write_text(json.dumps(report, indent=2, default=str))
    lines = ["# Checks for Dr. Saha's completion list", "", "See saha_checks.json for every number.", ""]
    for name, p in report.get("provenance", {}).items():
        lines.append(f"- provenance {name}: new test {p['new_test']}; client train {p['new_client_train']}; "
                     f"client val {p['new_client_val']}")
    for name, e in report.get("exports", {}).items():
        for k, v in e.items():
            if "per_tensor_strong_ptq" in v:
                a, b = v["table2_export"]["0.01"], v["per_tensor_strong_ptq"]["0.01"]
                lines.append(f"- {name} {k}: Table-2 export recall/FAR@1% {a['test_recall']:.4f}/{a['test_far']:.4f}; "
                             f"per-tensor ({v['per_tensor_selected']}) {b['test_recall']:.4f}/{b['test_far']:.4f} "
                             f"({v['per_tensor_strong_ptq']['bytes']} B)")
    (out / "saha_checks.md").write_text("\n".join(lines) + "\n")
    print("✅ saved", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
