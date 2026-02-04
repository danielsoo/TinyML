#!/usr/bin/env python3
"""
Migrate existing data/processed/analysis/<version>/<run_id>/ into the new layout:
  data/processed/runs/<version>/<run_id>/analysis/
  data/processed/runs/<version>/<run_id>/eval/ratio_sweep_report.md (if present)

Also copies .last_run_id and VERSIONS.md to data/processed/runs/.

Usage:
  python scripts/migrate_analysis_to_runs.py [--dry-run] [--source DIR]
  --source defaults to data/processed/analysis (relative to project root).
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

# Project root = parent of scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def is_run_id_like(name: str) -> bool:
    """Heuristic: run_id is usually datetime like 2026-02-02_23-28-45 or similar."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name))


def migrate_one_run(
    source_dir: Path,
    runs_root: Path,
    version: str,
    run_id: str,
    dry_run: bool,
) -> bool:
    """Move/copy one analysis run to runs/<version>/<run_id>/analysis/ and eval/ if needed."""
    analysis_dst = runs_root / version / run_id / "analysis"
    eval_dst = runs_root / version / run_id / "eval"

    # Analysis files (compression + viz); exclude ratio_sweep_report.md (goes to eval)
    analysis_files = [
        "compression_analysis.csv",
        "compression_analysis.json",
        "compression_analysis.md",
        "compression_metrics.png",
        "compression_ratio.png",
        "size_vs_accuracy.png",
    ]
    # Include any "compression_analysis 2.*" etc.
    extra = [f.name for f in source_dir.iterdir() if f.is_file() and f.name not in analysis_files and f.name != "ratio_sweep_report.md" and not f.name.startswith(".")]
    all_analysis_names = set(analysis_files) | set(extra)

    if dry_run:
        if analysis_dst.exists() and any(analysis_dst.iterdir()):
            print(f"  [skip] {version}/{run_id}/analysis (already exists)")
        else:
            print(f"  [would create] {analysis_dst} and copy {len(all_analysis_names)} file(s)")
        if (source_dir / "ratio_sweep_report.md").exists():
            print(f"  [would create] {eval_dst}/ratio_sweep_report.md")
        return True

    analysis_dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        if f.name == "ratio_sweep_report.md":
            eval_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, eval_dst / "ratio_sweep_report.md")
            copied += 1
            continue
        if f.name.startswith("."):
            continue
        dst_file = analysis_dst / f.name
        if dst_file.exists():
            continue  # don't overwrite
        shutil.copy2(f, dst_file)
        copied += 1
    if copied:
        print(f"  {version}/{run_id}: {copied} file(s) -> runs/{version}/{run_id}/")
    return True


def main():
    ap = argparse.ArgumentParser(description="Migrate analysis/ to runs/<version>/<run_id>/analysis and eval/")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    ap.add_argument("--source", type=str, default=None, help="Source analysis dir (default: data/processed/analysis)")
    args = ap.parse_args()

    source_root = Path(args.source) if args.source else (PROJECT_ROOT / "data" / "processed" / "analysis")
    if not source_root.is_absolute():
        source_root = PROJECT_ROOT / source_root
    runs_root = PROJECT_ROOT / "data" / "processed" / "runs"
    source_root = source_root.resolve()
    runs_root = runs_root.resolve()

    if not source_root.exists():
        print(f"Source not found: {source_root}", file=sys.stderr)
        return 1

    print(f"Source: {source_root}")
    print(f"Target: {runs_root}")
    if args.dry_run:
        print("(dry run)\n")

    migrated = 0
    # Walk version dirs
    for ver_dir in sorted(source_root.iterdir()):
        if not ver_dir.is_dir() or ver_dir.name.startswith("."):
            continue
        version = ver_dir.name
        # Check for run_id subdirs (e.g. v11/2026-02-02_23-28-45)
        subdirs = [d for d in ver_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if subdirs:
            for run_dir in sorted(subdirs):
                run_id = run_dir.name
                migrate_one_run(run_dir, runs_root, version, run_id, args.dry_run)
                migrated += 1
        else:
            # Files directly under version (e.g. v1/, v2_20260130_145009/)
            if any(ver_dir.iterdir()):
                migrate_one_run(ver_dir, runs_root, version, "run", args.dry_run)
                migrated += 1

    # .last_run_id
    last_run_file = source_root / ".last_run_id"
    if last_run_file.exists():
        dst_last = runs_root / ".last_run_id"
        if args.dry_run:
            print(f"  [would copy] .last_run_id -> {dst_last}")
        else:
            runs_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(last_run_file, dst_last)
            print(f"  .last_run_id -> {dst_last}")

    # VERSIONS.md
    versions_md = source_root / "VERSIONS.md"
    if versions_md.exists():
        dst_ver = runs_root / "VERSIONS.md"
        if args.dry_run:
            print(f"  [would copy] VERSIONS.md -> {dst_ver}")
        else:
            runs_root.mkdir(parents=True, exist_ok=True)
            if dst_ver.exists():
                # Append new lines from source that are not in dst (simple merge)
                src_text = versions_md.read_text(encoding="utf-8")
                dst_text = dst_ver.read_text(encoding="utf-8")
                for line in src_text.splitlines():
                    if line.strip() and line not in dst_text:
                        dst_ver.write_text(dst_text.rstrip() + "\n" + line + "\n", encoding="utf-8")
                        dst_text = dst_ver.read_text(encoding="utf-8")
            else:
                shutil.copy2(versions_md, dst_ver)
            print(f"  VERSIONS.md -> {dst_ver}")

    print(f"\nDone. Migrated {migrated} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
