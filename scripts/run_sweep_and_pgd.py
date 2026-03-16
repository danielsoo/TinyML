#!/usr/bin/env python3
"""
Run 48 compression grid sweep, then PGD on all 48 models, then merge results.
Creates a run directory with sweep_compression_grid.csv, pgd/, and sweep_compression_grid_with_pgd.csv.

Prerequisites:
  - models/global_model.h5, models/global_model_traditional.h5
  - models/distilled/*.h5 (from run_distillation_first.py)

Usage:
  python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml
  python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml --version v1_PGD --quick
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from scripts.experiment_record import write_experiment_record


def run_cmd(cmd: list, desc: str) -> bool:
    print(f"\n{'='*60}\n  {desc}\n{'='*60}\n")
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n  {desc} done.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Failed: {e}\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="48 sweep + PGD on all + merge")
    parser.add_argument("--config", default="config/federated_local_sky.yaml")
    parser.add_argument("--version", default="sweep_pgd", help="Version prefix for run dir, e.g. v1_PGD")
    parser.add_argument("--quick", action="store_true", help="Run only 4 combinations (sweep --quick)")
    parser.add_argument("--skip-sweep", action="store_true", help="Use existing sweep CSV in run-dir (run PGD + merge only)")
    parser.add_argument("--run-dir", default=None, help="Use this run dir (default: data/processed/runs/<version>/<timestamp>)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = Path("data/processed/runs") / args.version
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir.resolve()}")

    # Step 0: Save run config and experiment record (so pgd_report etc. can refer to them)
    with open(args.config, encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    run_config_path = run_dir / "run_config.yaml"
    with open(run_config_path, "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"   📋 Saved run config: {run_config_path}\n")
    write_experiment_record(run_cfg, run_dir)

    if not args.skip_sweep:
        sweep_cmd = [
            sys.executable,
            str(project_root / "scripts" / "sweep_compression_grid.py"),
            "--config", args.config,
            "--run-dir", str(run_dir),
        ]
        if args.quick:
            sweep_cmd.append("--quick")
        if not run_cmd(sweep_cmd, "Step 1: 48 (or 4) compression grid sweep"):
            return 1

    list_file = run_dir / "pgd_model_list.txt"
    if not list_file.exists():
        print(f"  pgd_model_list.txt not found. Run sweep without --skip-sweep first.", file=sys.stderr)
        return 1

    pgd_out = run_dir / "pgd"
    pgd_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pgd.py"),
        "--models-file", str(list_file),
        "--config", args.config,
        "--output-dir", str(pgd_out),
    ]
    if not run_cmd(pgd_cmd, "Step 2: PGD on all models"):
        return 1

    merge_cmd = [
        sys.executable,
        str(project_root / "scripts" / "merge_sweep_pgd.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(merge_cmd, "Step 3: Merge PGD into sweep CSV"):
        return 1

    # Step 4: Sweep summary (sweep_summary.md)
    summary_cmd = [
        sys.executable,
        str(project_root / "scripts" / "write_sweep_summary.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(summary_cmd, "Step 4: Write sweep summary"):
        pass  # non-fatal

    # Step 5: Heatmaps (sweep_heatmap_*.png)
    heatmap_cmd = [
        sys.executable,
        str(project_root / "scripts" / "plot_sweep_heatmaps.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(heatmap_cmd, "Step 5: Plot sweep heatmaps"):
        pass  # non-fatal

    out_csv = run_dir / "sweep_compression_grid_with_pgd.csv"
    print(f"\n  Result: {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
