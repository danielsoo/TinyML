#!/usr/bin/env python3
"""
Number of flows per attack type in the train and test splits (CIC-IDS2017), so that per-type
missed attacks can be reported as "missed of total".

Spec (YAML):
  entries:
    - {name: cic, config: <b_fl.yaml>}

Usage:
  python scripts/attack_type_counts.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from src.data.loader import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Attack-type counts per split")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in spec["entries"]:
        cfg = load_yaml(ROOT / entry["config"])
        data_cfg = dict(cfg["data"])
        kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
        if "path" in kwargs:
            kwargs["data_path"] = kwargs.pop("path")
        _, _, _, y_test, attack_train, attack_test = load_dataset(
            data_cfg["name"], return_attack_labels=True, **kwargs)
        tr = collections.Counter(np.asarray(attack_train).astype(str))
        te = collections.Counter(np.asarray(attack_test).astype(str))
        for t in sorted(set(tr) | set(te), key=lambda k: -te.get(k, 0)):
            rows.append({"entry": entry["name"], "type": t, "train": int(tr.get(t, 0)), "test": int(te.get(t, 0))})
        print(f"[{entry['name']}] test attacks={int(np.sum(y_test))}; " +
              ", ".join(f"{t}={n}" for t, n in te.most_common()))
    rows_to_csv(rows, out_dir / "attack_type_counts.csv")
    rows_to_markdown(rows, out_dir / "attack_type_counts.md", "Flows per attack type (train after balancing/SMOTE, test)")
    save_json(rows, out_dir / "attack_type_counts.json")


if __name__ == "__main__":
    main()
