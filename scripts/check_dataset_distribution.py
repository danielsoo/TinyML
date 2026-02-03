#!/usr/bin/env python3
"""
Check class distribution (BENIGN vs ATTACK) in CIC-IDS2017 without sampling.
Usage: python scripts/check_dataset_distribution.py [--path /path/to/data]
"""
import argparse
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Check CIC-IDS2017 class distribution")
    parser.add_argument(
        "--path",
        type=str,
        default="/scratch/yqp5187/Bot-IoT",
        help="Directory containing *.pcap_ISCX.csv files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (uses data.path if set)",
    )
    args = parser.parse_args()

    data_path = Path(args.path)
    if args.config and Path(args.config).exists():
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        p = cfg.get("data", {}).get("path")
        if p:
            data_path = Path(p)
            print(f"Using path from config: {data_path}")

    if not data_path.exists():
        print(f"Path not found: {data_path}")
        sys.exit(1)

    csv_files = list(data_path.glob("*.pcap_ISCX.csv"))
    if not csv_files:
        print(f"No *.pcap_ISCX.csv found in {data_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("CIC-IDS2017 Full Dataset Class Distribution")
    print(f"{'='*60}")
    print(f"Path: {data_path}")
    print(f"Files: {len(csv_files)}")
    print(f"{'='*60}\n")

    all_labels = []
    per_file = []

    for f in sorted(csv_files):
        print(f"Loading {f.name}...", end=" ", flush=True)
        df = pd.read_csv(f)
        label_col = df.columns[-1]
        y = df[label_col].values
        ben = int(np.sum(y == "BENIGN"))
        att = int(np.sum(y != "BENIGN"))
        total = len(y)
        per_file.append((f.name, ben, att, total))
        all_labels.extend(y)
        print(f"BENIGN={ben:,} ATTACK={att:,} total={total:,}")

    y_all = np.array(all_labels)
    ben_total = int(np.sum(y_all == "BENIGN"))
    att_total = int(np.sum(y_all != "BENIGN"))
    total_all = len(y_all)

    print(f"\n{'='*60}")
    print("TOTAL (full dataset)")
    print(f"{'='*60}")
    print(f"BENIGN (normal):  {ben_total:>12,}  ({100*ben_total/total_all:.2f}%)")
    print(f"ATTACK:           {att_total:>12,}  ({100*att_total/total_all:.2f}%)")
    print(f"Total:            {total_all:>12,}")
    print(f"Ratio (BENIGN:ATTACK): {ben_total/total_all:.2f} : {att_total/total_all:.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
