#!/usr/bin/env python3
"""
Sweep focal_loss_alpha to find a good value without changing config by hand.

Runs the full pipeline (train → compression → analysis) for each alpha,
then collects QAT+PTQ metrics and writes a comparison report.

Usage:
  python scripts/sweep_focal_alpha.py --config config/federated.yaml --alphas 0.4 0.5 0.6 0.65 0.7
  python scripts/sweep_focal_alpha.py --config config/federated.yaml --alphas-min 0.05 --alphas-max 0.95 --alphas-step 0.05
  python scripts/sweep_focal_alpha.py --config config/federated.yaml --alphas-min 0.4 --alphas-max 0.7 --alphas-num 5
  python scripts/sweep_focal_alpha.py --config config/federated.yaml --alphas 0.5 0.6 0.65 --skip-viz --skip-fgsm
"""

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def run_pipeline(config_path: str, extra_args: list) -> bool:
    cmd = [sys.executable, "run.py", "--config", config_path] + extra_args
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
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
    # Find Summary table: | Stage | Size | ... | then rows like | QAT+PTQ | 0.0676 | ...
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
            # End of table or other row
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


def main():
    parser = argparse.ArgumentParser(
        description="Sweep focal_loss_alpha and compare QAT+PTQ metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=str, default="config/federated.yaml", help="Base config YAML")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="List of focal_loss_alpha values (e.g. 0.4 0.5 0.6 0.65 0.7)",
    )
    parser.add_argument("--alphas-min", type=float, default=0.4, help="Min alpha (with --alphas-num or --alphas-step)")
    parser.add_argument("--alphas-max", type=float, default=0.7, help="Max alpha (with --alphas-num or --alphas-step)")
    parser.add_argument("--alphas-num", type=int, default=5, help="Number of alphas in [min,max] (linear, used if --alphas-step not set)")
    parser.add_argument("--alphas-step", type=float, default=None, help="Step between alphas (e.g. 0.05 → 0.05, 0.10, ..., 0.95)")
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Pass --skip-viz to run.py each time",
    )
    parser.add_argument(
        "--skip-fgsm",
        action="store_true",
        help="Pass --skip-fgsm to run.py each time",
    )
    parser.add_argument(
        "--skip-ratio-sweep",
        action="store_true",
        help="Pass --skip-ratio-sweep to run.py each time",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="",
        help="Report path (default: <sweep_run_dir>/alpha_sweep_report.md)",
    )
    parser.add_argument(
        "--best-by",
        type=str,
        choices=("f1", "accuracy", "normal_recall"),
        default="f1",
        help="Which metric to use for 'best' alpha",
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

    if args.alphas is not None:
        alphas = sorted(set(args.alphas))
    elif args.alphas_step is not None:
        step = args.alphas_step
        alphas = []
        a = args.alphas_min
        while a <= args.alphas_max + 1e-9:
            alphas.append(round(a, 3))
            a += step
    else:
        n = max(1, args.alphas_num - 1)
        alphas = [args.alphas_min + (args.alphas_max - args.alphas_min) * i / n for i in range(args.alphas_num)]
        alphas = [round(a, 3) for a in alphas]

    extra = []
    if args.skip_viz:
        extra.append("--skip-viz")
    if args.skip_fgsm:
        extra.append("--skip-fgsm")
    if args.skip_ratio_sweep:
        extra.append("--skip-ratio-sweep")

    runs_base = Path("data/processed/runs")
    version = cfg.get("version", "run")
    sweep_run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_alpha_sweep"
    sweep_base = runs_base / version / sweep_run_id
    sweep_base.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Sweep run dir: {sweep_base}\n   (각 alpha별 보고서는 여기 아래 alpha_<값>/ 에 정리됩니다)\n")

    sweep_config_dir = Path("config/alpha_sweep")
    sweep_config_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, alpha in enumerate(alphas):
        print(f"\n{'='*80}")
        print(f"  Alpha sweep {i+1}/{len(alphas)}: focal_loss_alpha = {alpha}")
        print(f"{'='*80}\n")

        cfg_copy = yaml.safe_load(yaml.dump(cfg))
        cfg_copy["federated"]["focal_loss_alpha"] = alpha
        temp_config = sweep_config_dir / f"federated_alpha_{alpha}.yaml"
        save_config(cfg_copy, temp_config)

        ok = run_pipeline(str(temp_config), extra)
        last_run = get_last_run_dir(runs_base) if ok else None

        # 이번 alpha 결과를 sweep_run/alpha_<값>/ 로 이동 → 알파별로 한 곳에서 보고서 확인
        alpha_folder = f"alpha_{alpha}"
        run_dir = sweep_base / alpha_folder
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
                "alpha": alpha,
                "run_dir": str(run_dir),
                "stage": row.get("Stage", "?"),
                "accuracy": acc,
                "f1": f1,
                "normal_recall": nr,
            })
        else:
            results.append({
                "alpha": alpha,
                "run_dir": str(run_dir) if run_dir.exists() else "failed",
                "stage": "-",
                "accuracy": 0.0,
                "f1": 0.0,
                "normal_recall": 0.0,
            })

    # Best by chosen metric
    key = "f1" if args.best_by == "f1" else ("accuracy" if args.best_by == "accuracy" else "normal_recall")
    best_idx = max(range(len(results)), key=lambda i: results[i][key])
    best_alpha = results[best_idx]["alpha"]

    # Write report (기본: 이번 스윕 run 디렉터리 안에 저장 → 연월일_시분초_alpha_sweep 안에서 한 번에 확인)
    out_path = Path(args.output_report) if args.output_report else sweep_base / "alpha_sweep_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Focal Loss Alpha Sweep Report",
        "",
        f"Config: `{args.config}`  \nAlphas: `{alphas}`  \nBest by: **{args.best_by}** → **{best_alpha}**",
        "",
        "이 스윕 run 아래에 **alpha별 폴더** (`alpha_0.05/`, `alpha_0.10/`, …)가 있으며, 각 폴더에 해당 alpha에 대한 `analysis/`, `fgsm/`, `eval/`, `models/` 등 보고서·산출물이 있습니다.",
        "",
        "| Alpha | 폴더 (이 run 내) | Stage | Accuracy | F1-Score | Normal Recall | Best? |",
        "|-------|------------------|-------|----------|----------|---------------|-------|",
    ]
    for r in results:
        mark = "✅" if r["alpha"] == best_alpha else ""
        run_short = Path(r["run_dir"]).name if r["run_dir"] else r["run_dir"]
        lines.append(
            f"| {r['alpha']} | {run_short} | {r['stage']} | {r['accuracy']:.4f} | {r['f1']:.4f} | {r['normal_recall']:.4f} | {mark} |"
        )
    lines.extend([
        "",
        "## Recommendation",
        f"- **Best alpha (by {args.best_by}):** **{best_alpha}**",
        f"- Set in config: `federated.focal_loss_alpha: {best_alpha}`",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Report written: {out_path}")
    print(f"   Best alpha (by {args.best_by}): {best_alpha}")
    print(f"   Alpha별 보고서: {sweep_base}/alpha_<값>/ (analysis/, fgsm/, eval/, models/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
