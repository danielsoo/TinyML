#!/usr/bin/env python3
"""
Generate RUN_LEVEL_DIFFERENCES.md - detailed comparison of runs within each version.

Each version can have multiple runs (different timestamps). This script scans
analysis folders and produces per-run metrics plus inferred code/config changes.

Usage:
  python scripts/generate_run_level_changelog.py
  python scripts/generate_run_level_changelog.py --analysis-dir TinyML-results/processed/analysis
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List


def collect_runs(root: Path) -> List[Dict]:
    """Collect all runs from analysis directory."""
    runs = []
    for ver_dir in sorted(root.iterdir()):
        if not ver_dir.is_dir() or ver_dir.name.startswith("."):
            continue
        for run_dir in sorted(ver_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            json_path = run_dir / "compression_analysis.json"
            if not json_path.exists():
                continue
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            res = data.get("results", [])
            if not res:
                continue
            orig = res[0]
            comp = next((r for r in res if "compress" in r["stage"].lower()), res[-1] if len(res) > 1 else orig)
            runs.append({
                "version": ver_dir.name,
                "run_id": run_dir.name,
                "data_version": data.get("data_version", ""),
                "generated_at": data.get("generated_at", ""),
                "orig": orig,
                "comp": comp,
            })
    return sorted(runs, key=lambda x: (x["version"], x["run_id"]))


def infer_run_notes(runs: List[Dict], version: str) -> Dict[str, str]:
    """Infer what changed between runs based on metrics."""
    vruns = [r for r in runs if r["version"] == version]
    notes = {}
    for r in vruns:
        o, c = r["orig"], r["comp"]
        o_p, o_r = o.get("precision", 0), o.get("recall", 0)
        c_p, c_r = c.get("precision", 0), c.get("recall", 0)
        params = o.get("parameter_count", 0)
        comp_size = c.get("file_size_mb", 0)

        parts = []
        if o_p == 0 and o_r == 0 and o.get("accuracy", 0) > 0.5:
            parts.append("Orig TFLite NaN/collapse")
        if c_p == 0 and c_r == 0 and c.get("accuracy", 0) > 0.5:
            parts.append("Comp Full INT8 collapse")
        elif o_p > 0 and o_r > 0 and (c_p == 0 or c_r == 0):
            parts.append("BN strip O, PTQ(Full INT8) X")
        if o_p > 0 and o_r > 0 and c_p > 0 and c_r > 0:
            parts.append("DRQ applied, both OK")
        if params < 100000:
            parts.append("small 38feat")
        elif params > 200000:
            parts.append("large 78feat")
        if comp_size < 0.025:
            parts.append("Comp~21KB")
        elif comp_size > 0.065 and comp_size < 0.076:
            parts.append("Comp~73KB(Full INT8)")
        elif comp_size > 0.065:
            parts.append("Comp~68KB(DRQ)")

        notes[r["run_id"]] = "; ".join(parts) if parts else "-"
    return notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", default="data/processed/analysis")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.analysis_dir)
    if not root.exists():
        print(f"Not found: {root}")
        return 1

    runs = collect_runs(root)
    if not runs:
        print("No runs found")
        return 1

    out_path = Path(args.output) if args.output else root / "RUN_LEVEL_DIFFERENCES.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    versions = sorted(set(r["version"] for r in runs))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Run-Level Differences (per-run by timestamp)\n\n")
        f.write("Detailed comparison of multiple runs (timestamps) within each version. For reference in papers/reports.\n\n")

        for ver in versions:
            vruns = [r for r in runs if r["version"] == ver]
            if len(vruns) <= 1:
                continue
            f.write(f"## {ver} (runs: {len(vruns)})\n\n")
            notes = infer_run_notes(runs, ver)
            f.write("| Run | Data | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig (MB) | Comp (MB) | Params | Estimated Changes |\n")
            f.write("|-----|------|----------|----------|----------|----------|-----------|-----------|--------|----------------|\n")
            for r in vruns:
                o, c = r["orig"], r["comp"]
                o_pr = f"{o.get('precision',0):.2f}/{o.get('recall',0):.2f}"
                c_pr = f"{c.get('precision',0):.2f}/{c.get('recall',0):.2f}"
                n = notes.get(r["run_id"], "")
                f.write(
                    f"| {r['run_id']} | {r['data_version'][:25]} | {o.get('accuracy',0):.4f} | {o_pr} | "
                    f"{c.get('accuracy',0):.4f} | {c_pr} | {o.get('file_size_mb',0):.3f} | "
                    f"{c.get('file_size_mb',0):.3f} | {o.get('parameter_count',0):,} | {n} |\n"
                )
            f.write("\n")

        f.write("---\n\n### Notes on Estimated Changes\n\n")
        f.write("- **Original TFLite NaN/prediction collapse**: BN strip not applied; TFLite conversion yields NaN or single-class predictions\n")
        f.write("- **Compressed Full INT8 collapse**: Full INT8 (int8 in/out) yields P/R/F1=0\n")
        f.write("- **BN strip applied but PTQ not applied**: Original OK, Compressed still Full INT8\n")
        f.write("- **DRQ applied**: Dynamic Range Quantization (int8 weights, float32 I/O), both OK\n")
        f.write("- **Small/large model**: 38 feat vs 78 feat, Bot-IoT vs CIC-IDS2017\n")

    print(f"✅ Wrote {out_path}")
    return 0


if __name__ == "__main__":
    exit(main())
