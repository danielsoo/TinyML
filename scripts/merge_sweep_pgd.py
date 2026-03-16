#!/usr/bin/env python3
"""
Merge PGD results into sweep CSV.
Reads sweep_compression_grid.csv and pgd_results.json (from run_dir/pgd/),
joins by model path (tflite_path vs comparison[].model_path), and writes
sweep_compression_grid_with_pgd.csv with pgd_adv_acc and pgd_success_rate filled.

Usage:
  python scripts/merge_sweep_pgd.py --sweep-csv data/processed/runs/v1_PGD/sweep_2026-03-14/sweep_compression_grid.csv --pgd-json data/processed/runs/v1_PGD/sweep_2026-03-14/pgd/pgd_results.json --output data/processed/runs/v1_PGD/sweep_2026-03-14/sweep_compression_grid_with_pgd.csv
  # Or with run-dir (sweep CSV and pgd/ under it):
  python scripts/merge_sweep_pgd.py --run-dir data/processed/runs/v1_PGD/sweep_2026-03-14
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge PGD results into sweep CSV")
    parser.add_argument("--sweep-csv", default=None, help="Path to sweep_compression_grid.csv")
    parser.add_argument("--pgd-json", default=None, help="Path to pgd_results.json")
    parser.add_argument("--output", default=None, help="Output CSV path (default: run_dir/sweep_compression_grid_with_pgd.csv)")
    parser.add_argument("--run-dir", default=None, help="If set, sweep-csv=run_dir/sweep_compression_grid.csv, pgd-json=run_dir/pgd/pgd_results.json, output=run_dir/sweep_compression_grid_with_pgd.csv")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        sweep_csv = run_dir / "sweep_compression_grid.csv"
        pgd_json = run_dir / "pgd" / "pgd_results.json"
        output = run_dir / "sweep_compression_grid_with_pgd.csv"
    else:
        if not args.sweep_csv or not args.pgd_json:
            print("Either --run-dir or both --sweep-csv and --pgd-json are required.", file=sys.stderr)
            return 1
        sweep_csv = Path(args.sweep_csv)
        pgd_json = Path(args.pgd_json)
        output = Path(args.output) if args.output else sweep_csv.parent / "sweep_compression_grid_with_pgd.csv"

    if not sweep_csv.exists():
        print(f"Sweep CSV not found: {sweep_csv}", file=sys.stderr)
        return 1
    if not pgd_json.exists():
        print(f"PGD JSON not found: {pgd_json}", file=sys.stderr)
        return 1

    with open(pgd_json, encoding="utf-8") as f:
        pgd_data = json.load(f)
    comparison = pgd_data.get("comparison", [])
    # path -> {adversarial_accuracy, attack_success_rate}
    pgd_by_path = {}
    for c in comparison:
        p = c.get("model_path", "")
        if not p:
            continue
        norm = str(Path(p).resolve())
        pgd_by_path[norm] = {
            "adversarial_accuracy": c.get("adversarial_accuracy"),
            "attack_success_rate": c.get("attack_success_rate"),
        }
        # also store with original path for matching
        pgd_by_path[p] = pgd_by_path[norm]

    rows = []
    with open(sweep_csv, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        for row in r:
            tflite_path = (row.get("tflite_path") or "").strip()
            if tflite_path:
                norm = str(Path(tflite_path).resolve())
                info = pgd_by_path.get(norm) or pgd_by_path.get(tflite_path)
                if info:
                    row["pgd_adv_acc"] = info.get("adversarial_accuracy", "")
                    row["pgd_success_rate"] = info.get("attack_success_rate", "")
            rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {output} ({len(rows)} rows with PGD columns merged)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
