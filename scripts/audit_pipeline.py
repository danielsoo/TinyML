#!/usr/bin/env python3
"""
Data / implementation audit (docs/AUDIT.md, Dr. Saha's experimental guide section 2).

For each dataset it records, as JSON + Markdown:
  1. raw-data facts: rows, inf / NaN replacements, negative values per column, constant columns;
  2. duplicates: exact duplicate feature vectors and vectors that occur with both labels;
  3. old pipeline (src/data/loader.py + partition_data, as used for every result so far):
     a. SMOTE validity: share of synthetic rows whose value in a discrete column (label-encoded
        category or integer count with <= 64 distinct real values) is not a value that occurs
        in real data;
     b. client ownership: synthetic rows per client (SMOTE ran on the pooled training set,
        before the client partition);
     c. threshold-validation leakage: the recall-priority validation slice (client 0, 20,000 /
        8,000 rows) is part of client 0's federated training data; share of synthetic rows in it;
     d. old test-split record ids (re-derived), for the overlap check below;
  4. extreme records: the records with the largest |z| in the v13 training pool, with raw value,
     feature, attack family and source id, to separate genuine extremes from data errors;
  5. v13 pipeline manifest (fresh split seed) and its overlap with the old test split.

Spec (YAML):
  datasets:
    - {name: cicids2017, data_path: data/raw/CIC-IDS2017, val_samples: 20000}
    - {name: ton_iot, data_path: data/raw/TON_IoT, val_samples: 8000, text_features: true}
  v13: {split_seed: 2027, partition_seed: 42, val_frac: 0.1, balance_ratio: 4.0}

Usage:
  python scripts/audit_pipeline.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.ablation_utils import load_yaml
from src.data import fed_pipeline as fp
from src.data.loader import load_dataset, partition_data


def raw_facts(raw: fp.RawData) -> Dict[str, Any]:
    X = raw.X
    neg = {raw.feature_names[j]: int((X[:, j] < 0).sum()) for j in range(X.shape[1]) if (X[:, j] < 0).any()}
    const = [raw.feature_names[j] for j in range(X.shape[1]) if np.all(X[:, j] == X[0, j])]
    return {"records": int(len(raw.y)), "attack": int(raw.y.sum()), "benign": int((raw.y == 0).sum()),
            "attack_families": pd.Series(raw.attack).value_counts().to_dict(),
            "negative_values_per_column": neg, "constant_columns": const,
            "categorical_label_encoded": raw.categorical, **raw.info}


def old_split_ids(name: str, raw_all: fp.RawData, text_features: bool) -> np.ndarray:
    """Re-derive the record ids of the old (src/data/loader.py) test split."""
    if name.startswith("cic"):
        df = pd.DataFrame(raw_all.X)
        df["_y"] = raw_all.attack
        df["_a"] = raw_all.attack
        keep = ~df.duplicated().values
        ids, y = raw_all.ids[keep], (raw_all.attack[keep] != "BENIGN").astype(int)
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(ids))
        ids, y = ids[perm], y[perm]
    else:
        nb = raw_all.info["n_base_features"]
        df = pd.DataFrame(raw_all.X[:, :nb])
        df["_y"] = raw_all.y
        keep = ~df.duplicated().values
        ids, y = raw_all.ids[keep], raw_all.y[keep]
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(ids))
        ids, y = ids[perm], y[perm]
    _, test_ids = train_test_split(ids, test_size=0.2, stratify=y, random_state=42)
    return np.asarray(test_ids)


def old_pipeline_checks(name: str, data_path: str, val_samples: int, text_features: bool) -> Dict[str, Any]:
    kw = {"data_path": data_path, "binary": True, "use_smote": True, "balance_ratio": 4.0,
          "return_attack_labels": True}
    if name.startswith("ton"):
        kw["text_features"] = text_features
    x_tr, y_tr, _, _, a_tr, _ = load_dataset(name, **kw)
    synth = np.asarray(a_tr).astype(str) == "SMOTE"
    real = ~synth
    out: Dict[str, Any] = {"train_rows": int(len(y_tr)), "synthetic_rows": int(synth.sum()),
                           "synthetic_share": round(float(synth.mean()), 4)}
    invalid = {}
    for j in range(x_tr.shape[1]):
        vals = np.unique(np.round(x_tr[real, j], 5))
        if len(vals) > 64:
            continue
        sv = np.round(x_tr[synth, j], 5)
        bad = ~np.isin(sv, vals)
        if bad.any():
            invalid[str(j)] = round(float(bad.mean()), 4)
    out["smote_invalid_discrete_share_per_column"] = invalid
    out["smote_rows_with_any_invalid_discrete_value"] = None
    if invalid:
        cols = [int(j) for j in invalid]
        any_bad = np.zeros(int(synth.sum()), bool)
        for j in cols:
            vals = np.unique(np.round(x_tr[real, j], 5))
            any_bad |= ~np.isin(np.round(x_tr[synth, j], 5), vals)
        out["smote_rows_with_any_invalid_discrete_value"] = round(float(any_bad.mean()), 4)
    # client ownership: partition indices, synthetic rows per client
    parts = partition_data(np.arange(len(y_tr))[:, None], y_tr, 4, strategy="label_balanced")
    per_client = []
    for k, p in enumerate(parts):
        idx = p["x"].ravel().astype(int)
        per_client.append({"rows": int(len(idx)), "synthetic": int(synth[idx].sum())})
    out["clients"] = per_client
    # recall-priority validation slice (scripts/recall_priority.py: client 0, perm seed 7)
    idx0 = parts[0]["x"].ravel().astype(int)
    perm = np.random.default_rng(7).permutation(len(idx0))
    val = idx0[perm[:val_samples]]
    out["threshold_validation"] = {
        "rows": int(len(val)), "part_of_client0_federated_training": True,
        "synthetic_rows": int(synth[val].sum()),
        "synthetic_share": round(float(synth[val].mean()), 4),
        "synthetic_share_of_attack_rows": round(float((synth[val] & (y_tr[val] == 1)).sum() / max(1, (y_tr[val] == 1).sum())), 4),
        "synthetic_share_of_benign_rows": round(float((synth[val] & (y_tr[val] == 0)).sum() / max(1, (y_tr[val] == 0).sum())), 4)}
    return out


def extreme_records(fd: fp.FederatedData, top: int = 30) -> Dict[str, Any]:
    xs = np.concatenate([c.train.x for c in fd.clients])
    ids = np.concatenate([c.train.ids for c in fd.clients])
    att = np.concatenate([c.train.attack for c in fd.clients])
    absz = np.abs(xs)
    rowmax = absz.max(axis=1)
    order = np.argsort(-rowmax)[:top]
    recs = []
    for i in order:
        j = int(absz[i].argmax())
        recs.append({"id": str(ids[i]), "attack": str(att[i]), "feature": fd.feature_names[j],
                     "z": round(float(xs[i, j]), 1),
                     "raw_value": float(xs[i, j] * fd.scaler_scale[j] + fd.scaler_mean[j])})
    q = {f"share_rows_max_abs_z_gt_{t}": round(float((rowmax > t).mean()), 6) for t in (5, 10, 100, 1000)}
    return {"top_records": recs, **q, "max_abs_z": round(float(rowmax.max()), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    v13 = spec.get("v13", {})
    report: Dict[str, Any] = {}
    for d in spec["datasets"]:
        name, path = d["name"], d["data_path"]
        tf_ = bool(d.get("text_features", True))
        print(f"=== {name}", flush=True)
        r: Dict[str, Any] = {}
        raw_all = fp.load_raw(name, path, text_features=tf_)
        r["raw"] = raw_facts(raw_all)
        old_test = old_split_ids(name, raw_all, tf_)
        r["old_pipeline"] = old_pipeline_checks(name, path, int(d.get("val_samples", 20000)), tf_)
        fd = fp.build_federated(name, path, text_features=tf_, **v13)
        fd.save_manifest(out_dir / f"{name}_v13_manifest")
        r["v13_manifest"] = fd.manifest
        new_test = fd.test.ids.astype(str)
        ov = np.isin(new_test, old_test.astype(str))
        r["split_overlap"] = {"new_test": int(len(new_test)), "old_test": int(len(old_test)),
                              "new_test_records_that_were_in_old_test": int(ov.sum()),
                              "share": round(float(ov.mean()), 4)}
        r["extremes"] = extreme_records(fd)
        if name.startswith("ton") and "ts" in raw_all.meta:
            ts = pd.to_numeric(raw_all.meta["ts"], errors="coerce")
            r["metadata"] = {"ts_min": float(ts.min()), "ts_max": float(ts.max()),
                             "ts_missing": int(ts.isna().sum()),
                             "distinct_src_ip": int(raw_all.meta["src_ip"].nunique()),
                             "distinct_dst_ip": int(raw_all.meta["dst_ip"].nunique()),
                             "top_src_ip_share": round(float(raw_all.meta["src_ip"].value_counts(normalize=True).iloc[0]), 4)}
            # attack families over time (deciles)
            dec = pd.qcut(ts.rank(method="first"), 10, labels=False)
            r["metadata"]["attack_share_by_time_decile"] = (
                pd.Series(raw_all.y).groupby(dec.values).mean().round(3).tolist())
            r["metadata"]["families_by_time_decile"] = {
                int(k): sorted(set(v)) for k, v in pd.Series(raw_all.attack).groupby(dec.values)}
        report[name] = r
        (out_dir / "audit.json").write_text(json.dumps(report, indent=2, default=str))
    # short markdown
    lines = ["# Pipeline audit", ""]
    for name, r in report.items():
        op, dd = r["old_pipeline"], r["v13_manifest"]["dedup"]
        lines += [f"## {name}",
                  f"- raw records {r['raw']['records']:,}; duplicates removed {dd['duplicate_records_removed']:,}; "
                  f"conflicting-label vectors {dd['conflicting_label_vectors']:,} ({dd['conflicting_label_records']:,} records)",
                  f"- old pipeline: synthetic (SMOTE) share of training {op['synthetic_share']}; rows with an invalid "
                  f"discrete value {op['smote_rows_with_any_invalid_discrete_value']}; synthetic rows per client "
                  f"{[c['synthetic'] for c in op['clients']]}",
                  f"- old threshold-validation slice: {op['threshold_validation']}",
                  f"- negative values: {r['raw']['negative_values_per_column']}",
                  f"- new test overlap with old test: {r['split_overlap']}",
                  f"- max |z| {r['extremes']['max_abs_z']}; top extreme: {r['extremes']['top_records'][:5]}", ""]
    (out_dir / "audit.md").write_text("\n".join(lines))
    print("✅ audit saved", out_dir)


if __name__ == "__main__":
    main()
