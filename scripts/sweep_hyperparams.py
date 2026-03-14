#!/usr/bin/env python3
"""
Grid sweep over training hyperparameters (num_rounds, batch_size, local_epochs,
learning_rate, focal_loss_alpha). Runs the full pipeline per combination and
collects QAT+PTQ metrics into one report under a single sweep run dir.

Usage:
  python scripts/sweep_hyperparams.py --config config/federated.yaml
  python scripts/sweep_hyperparams.py --config config/federated.yaml --quick
  python scripts/sweep_hyperparams.py --config config/federated.yaml --skip-viz --skip-pgd
"""

import argparse
import csv
import itertools
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


# Default sweep values (plan: meaningful ranges)
DEFAULT_NUM_ROUNDS = [60, 80, 100]
DEFAULT_BATCH_SIZE = [64, 128, 256]
DEFAULT_LOCAL_EPOCHS = [1, 2]
DEFAULT_LEARNING_RATE = [0.0003, 0.0005, 0.001]
DEFAULT_FOCAL_LOSS_ALPHA = [0.5, 0.65, 0.8]

# Quick sweep: 2 values each -> 32 runs
QUICK_NUM_ROUNDS = [60, 100]
QUICK_BATCH_SIZE = [64, 256]
QUICK_LOCAL_EPOCHS = [1, 2]
QUICK_LEARNING_RATE = [0.0003, 0.001]
QUICK_FOCAL_LOSS_ALPHA = [0.5, 0.8]


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def run_pipeline(
    config_path: str,
    extra_args: list,
    sweep_run_index: int | None = None,
    sweep_run_total: int | None = None,
) -> bool:
    cmd = [sys.executable, "run.py", "--config", config_path] + extra_args
    env = os.environ.copy()
    if sweep_run_index is not None and sweep_run_total is not None:
        env["SWEEP_RUN_INDEX"] = str(sweep_run_index)
        env["SWEEP_RUN_TOTAL"] = str(sweep_run_total)
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True, env=env)
        return True
    except subprocess.CalledProcessError:
        return False


def get_last_run_dir(runs_base: Path) -> Path | None:
    last_run_file = runs_base / ".last_run_id"
    if not last_run_file.exists():
        return None
    rel = last_run_file.read_text(encoding="utf-8").strip()
    return runs_base / rel


def parse_summary_qat_ptq(md_path: Path) -> dict | None:
    """Parse compression_analysis.md Summary table; return QAT+PTQ row as dict or None."""
    if not md_path.exists():
        return None
    text = md_path.read_text(encoding="utf-8")
    in_summary = False
    headers = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| Stage |"):
            in_summary = True
            headers = [c.strip() for c in line.split("|")[1:-1]]
            continue
        if in_summary and line.startswith("|-------"):
            continue
        if in_summary and line.startswith("| QAT+PTQ |"):
            parts = [c.strip() for c in line.split("|")[1:-1]]
            if len(parts) >= len(headers):
                return dict(zip(headers, parts))
            break
        if in_summary and line.startswith("|") and "QAT+PTQ" not in line and "Stage" not in line:
            if "Summary" in line or line.startswith("##"):
                break
    return None


def parse_summary_best_f1(md_path: Path) -> dict | None:
    """Parse Summary table and return the row with highest F1-Score (excluding Original)."""
    if not md_path.exists():
        return None
    text = md_path.read_text(encoding="utf-8")
    in_summary = False
    headers = []
    best_row = None
    best_f1 = -1.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("| Stage |"):
            in_summary = True
            headers = [c.strip() for c in line.split("|")[1:-1]]
            continue
        if in_summary and line.startswith("|-------"):
            continue
        if in_summary and line.startswith("|") and "Stage" not in line and "---" not in line:
            parts = [c.strip() for c in line.split("|")[1:-1]]
            if len(parts) >= len(headers):
                row = dict(zip(headers, parts))
                stage = row.get("Stage", "")
                if stage == "Original":
                    continue
                try:
                    f1 = float(row.get("F1-Score", "0").replace(",", "."))
                except ValueError:
                    continue
                if f1 > best_f1:
                    best_f1 = f1
                    best_row = row
        if in_summary and line.startswith("##"):
            break
    return best_row


def lr_to_slug(lr: float) -> str:
    if lr >= 0.001:
        return "1e-3"
    if lr >= 0.0005:
        return "5e-4"
    if lr >= 0.0003:
        return "3e-4"
    return str(lr).replace(".", "_").replace("-", "m")


def combo_to_slug(combo: dict) -> str:
    r = combo["num_rounds"]
    b = combo["batch_size"]
    e = combo["local_epochs"]
    lr = combo["learning_rate"]
    a = round(combo["focal_loss_alpha"], 2)
    return f"r{r}_b{b}_e{e}_lr{lr_to_slug(lr)}_a{a}"


def main():
    parser = argparse.ArgumentParser(
        description="Grid sweep over training hyperparameters and compare QAT+PTQ metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default="config/federated.yaml", help="Base config YAML")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 2 values per param (32 runs) instead of full grid (162 runs)",
    )
    parser.add_argument("--skip-viz", action="store_true", help="Pass --skip-viz to run.py each time")
    parser.add_argument("--skip-pgd", action="store_true", help="Pass --skip-pgd to run.py each time")
    parser.add_argument("--skip-ratio-sweep", action="store_true", help="Pass --skip-ratio-sweep to run.py each time")
    parser.add_argument(
        "--output-report",
        type=str,
        default="",
        help="Report path (default: <sweep_run_dir>/hyperparam_sweep_report.md)",
    )
    parser.add_argument(
        "--best-by",
        type=str,
        choices=("f1", "accuracy", "normal_recall"),
        default="f1",
        help="Metric to pick best combination",
    )
    args = parser.parse_args()

    base_path = Path(args.config)
    if not base_path.exists():
        print(f"Config not found: {base_path}")
        return 1

    cfg = load_config(args.config)
    if "federated" not in cfg:
        print("Config has no 'federated' section.")
        return 1

    if args.quick:
        num_rounds_list = QUICK_NUM_ROUNDS
        batch_size_list = QUICK_BATCH_SIZE
        local_epochs_list = QUICK_LOCAL_EPOCHS
        learning_rate_list = QUICK_LEARNING_RATE
        focal_loss_alpha_list = QUICK_FOCAL_LOSS_ALPHA
    else:
        num_rounds_list = DEFAULT_NUM_ROUNDS
        batch_size_list = DEFAULT_BATCH_SIZE
        local_epochs_list = DEFAULT_LOCAL_EPOCHS
        learning_rate_list = DEFAULT_LEARNING_RATE
        focal_loss_alpha_list = DEFAULT_FOCAL_LOSS_ALPHA

    combinations = list(
        itertools.product(
            num_rounds_list,
            batch_size_list,
            local_epochs_list,
            learning_rate_list,
            focal_loss_alpha_list,
        )
    )
    combos = [
        {
            "num_rounds": r,
            "batch_size": b,
            "local_epochs": e,
            "learning_rate": lr,
            "focal_loss_alpha": a,
        }
        for (r, b, e, lr, a) in combinations
    ]

    extra = []
    if args.skip_viz:
        extra.append("--skip-viz")
    if args.skip_pgd:
        extra.append("--skip-pgd")
    if args.skip_ratio_sweep:
        extra.append("--skip-ratio-sweep")

    runs_base = Path("data/processed/runs")
    version = cfg.get("version", "run")
    sweep_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_hyperparam_sweep"
    sweep_base = runs_base / version / sweep_run_id
    sweep_base.mkdir(parents=True, exist_ok=True)
    total = len(combos)
    print(f"\nSweep run dir: {sweep_base}")
    print(f"  총 {total} runs (Run 1/{total} ~ Run {total}/{total})")
    print(f"  Each run: run_<slug>/ (analysis/, pgd/, eval/, models/)\n")

    sweep_config_dir = Path("config/hyperparam_sweep")
    sweep_config_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, combo in enumerate(combos):
        slug = combo_to_slug(combo)
        run_num = i + 1
        print(f"\n{'='*80}")
        print(f"  Run {run_num}/{total} (총 {total}개 중 {run_num}번째): {slug}")
        print(f"{'='*80}\n")

        cfg_copy = yaml.safe_load(yaml.dump(cfg))
        cfg_copy["federated"]["num_rounds"] = combo["num_rounds"]
        cfg_copy["federated"]["batch_size"] = combo["batch_size"]
        cfg_copy["federated"]["local_epochs"] = combo["local_epochs"]
        cfg_copy["federated"]["learning_rate"] = combo["learning_rate"]
        cfg_copy["federated"]["focal_loss_alpha"] = combo["focal_loss_alpha"]

        temp_config = sweep_config_dir / f"federated_{slug}.yaml"
        save_config(cfg_copy, temp_config)

        ok = run_pipeline(
            str(temp_config), extra,
            sweep_run_index=run_num,
            sweep_run_total=total,
        )
        status = "OK" if ok else "FAILED"
        print(f"\n  → Run {run_num}/{total} 완료 ({status})\n")
        last_run = get_last_run_dir(runs_base) if ok else None

        run_dir = sweep_base / f"run_{slug}"
        if last_run and last_run.exists() and last_run != run_dir:
            if run_dir.exists():
                shutil.rmtree(run_dir)
            shutil.move(str(last_run), str(run_dir))

        row = None
        if run_dir.exists():
            analysis_md = run_dir / "analysis" / "compression_analysis.md"
            row = parse_summary_qat_ptq(analysis_md)
            if row is None:
                row = parse_summary_best_f1(analysis_md)

        if row:
            try:
                acc = float(row.get("Accuracy", "0").replace(",", "."))
                f1 = float(row.get("F1-Score", "0").replace(",", "."))
                nr = float(row.get("Normal Recall", "0").replace(",", "."))
            except ValueError:
                acc = f1 = nr = 0.0
            results.append({
                **combo,
                "run_dir": str(run_dir),
                "slug": slug,
                "stage": row.get("Stage", "?"),
                "accuracy": acc,
                "f1": f1,
                "normal_recall": nr,
            })
        else:
            results.append({
                **combo,
                "run_dir": str(run_dir) if run_dir.exists() else "failed",
                "slug": slug,
                "stage": "-",
                "accuracy": 0.0,
                "f1": 0.0,
                "normal_recall": 0.0,
            })

    key = "f1" if args.best_by == "f1" else ("accuracy" if args.best_by == "accuracy" else "normal_recall")
    best_idx = max(range(len(results)), key=lambda i: results[i][key])
    best = results[best_idx]

    out_path = Path(args.output_report) if args.output_report else sweep_base / "hyperparam_sweep_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Hyperparameter Sweep Report",
        "",
        f"Config: `{args.config}`  \nBest by: **{args.best_by}**  \nBest combo: "
        f"num_rounds={best['num_rounds']}, batch_size={best['batch_size']}, local_epochs={best['local_epochs']}, "
        f"learning_rate={best['learning_rate']}, focal_loss_alpha={best['focal_loss_alpha']}",
        "",
        "이 스윕 run 아래 `run_<slug>/` 에 조합별 보고서(analysis/, pgd/, eval/, models/)가 있습니다.",
        "",
        "| num_rounds | batch_size | local_epochs | learning_rate | focal_loss_alpha | Accuracy | F1-Score | Normal Recall | Best? |",
        "|-------------|------------|--------------|---------------|------------------|----------|----------|---------------|-------|",
    ]
    for r in results:
        mark = "Y" if r["slug"] == best["slug"] else ""
        lines.append(
            f"| {r['num_rounds']} | {r['batch_size']} | {r['local_epochs']} | {r['learning_rate']} | "
            f"{r['focal_loss_alpha']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['normal_recall']:.4f} | {mark} |"
        )
    lines.extend([
        "",
        "## Recommendation",
        f"- **Best (by {args.best_by}):** run_{best['slug']}",
        f"- num_rounds: {best['num_rounds']}, batch_size: {best['batch_size']}, local_epochs: {best['local_epochs']}, "
        f"learning_rate: {best['learning_rate']}, focal_loss_alpha: {best['focal_loss_alpha']}",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")

    csv_path = sweep_base / "runs_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "slug", "num_rounds", "batch_size", "local_epochs", "learning_rate", "focal_loss_alpha",
                "accuracy", "f1", "normal_recall", "stage",
            ],
        )
        w.writeheader()
        w.writerows(
            {
                "slug": r["slug"],
                "num_rounds": r["num_rounds"],
                "batch_size": r["batch_size"],
                "local_epochs": r["local_epochs"],
                "learning_rate": r["learning_rate"],
                "focal_loss_alpha": r["focal_loss_alpha"],
                "accuracy": r["accuracy"],
                "f1": r["f1"],
                "normal_recall": r["normal_recall"],
                "stage": r["stage"],
            }
            for r in results
        )

    print(f"\nReport written: {out_path}")
    print(f"Manifest: {csv_path}")
    print(f"Best (by {args.best_by}): run_{best['slug']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
