#!/usr/bin/env python3
"""
Backfill "Run / Training Configuration" into existing run reports and save run_config.yaml.

- Scans data/processed/runs/<version>/<run_id>/
- If run_config.yaml missing: infer config from version and save it
- If compression_analysis.md or eval/ratio_sweep_report.md exist and lack the section: insert it

Usage:
  python scripts/backfill_run_config_to_reports.py
  python scripts/backfill_run_config_to_reports.py --dry-run
"""
from pathlib import Path
import argparse
import re
import sys

# Optional: use PyYAML if available; else minimal parser (no extra deps)
try:
    import yaml
    def _load_yaml(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    def _save_yaml(path, cfg):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
except ImportError:
    def _load_yaml(path):
        return _parse_yaml_minimal(Path(path).read_text(encoding="utf-8"))
    def _save_yaml(path, cfg):
        Path(path).write_text(_dump_yaml_minimal(cfg), encoding="utf-8")

def _parse_yaml_minimal(text: str) -> dict:
    """Minimal YAML-like parser: data/, model/, federated/ sections only."""
    data, model, fed = {}, {}, {}
    section = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if line and not line[0].isspace() and ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip().rstrip(" #").strip().strip("'\"")
            if k == "data":
                section = data
            elif k == "model":
                section = model
            elif k == "federated":
                section = fed
            elif section is not None and k and not k.startswith("#"):
                if v.lower() == "true": v = True
                elif v.lower() == "false": v = False
                elif v.isdigit(): v = int(v)
                elif v and re.match(r"^-?\d+\.\d+$", v): v = float(v)
                section[k] = v
        elif section is not None and line.startswith(" ") and ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip().rstrip(" #").strip().strip("'\"")
            if k and not k.startswith("#"):
                if v.lower() == "true": v = True
                elif v.lower() == "false": v = False
                elif v.isdigit(): v = int(v)
                elif v and re.match(r"^-?\d+\.\d+$", v): v = float(v)
                section[k] = v
    return {"data": data, "model": model, "federated": fed}

def _dump_yaml_minimal(cfg: dict) -> str:
    """Write minimal YAML (enough for run_config.yaml)."""
    lines = []
    for top in ("data", "model", "federated"):
        if top not in cfg or not isinstance(cfg[top], dict):
            continue
        lines.append(f"{top}:")
        for k, v in cfg[top].items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines) + "\n"

# Project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RUNS_BASE = PROJECT_ROOT / "data" / "processed" / "runs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Version -> config file name (without path). Used when run_config.yaml is missing.
VERSION_TO_CONFIG = {
    "v11": "federated_scratch.yaml",
    "v8": "federated_local.yaml",
    "v8_centralized": "federated_local.yaml",
    "v9": "federated_local.yaml",
    "v7": "federated_local.yaml",
    "v6": "federated_local.yaml",
    "v5": "federated_local.yaml",
    "v4": "federated_local.yaml",
    "v3": "federated_local.yaml",
    "v1": "federated_local.yaml",
    "v2_20260130_145009": "federated_local.yaml",
}
DEFAULT_CONFIG = "federated_local.yaml"


def get_config_for_run(run_dir: Path) -> tuple[dict, Path | None]:
    """Load config for a run: from run_config.yaml if present, else infer from version and load yaml."""
    version = run_dir.parent.name
    run_config_path = run_dir / "run_config.yaml"
    if run_config_path.exists():
        return _load_yaml(run_config_path), run_config_path
    config_name = VERSION_TO_CONFIG.get(version) or DEFAULT_CONFIG
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        return {}, None
    return _load_yaml(config_path), run_config_path


def config_section_markdown(cfg: dict, for_compression_report: bool = True) -> str:
    """Generate the 'Run / Training Configuration' markdown block (same format as analyze_compression)."""
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})
    lines = [
        "## Run / Training Configuration\n\n",
        "| Item | Value |\n|------|-------|\n",
        f"| **Data** | {data_cfg.get('name', '-')} |\n",
        f"| **Data path** | {data_cfg.get('path', '-')} |\n",
        f"| **Max samples** | {data_cfg.get('max_samples', '-')} |\n",
    ]
    br = data_cfg.get("balance_ratio")
    br_desc = {1.0: "50:50", 4.0: "정상:공격 8:2", 9.0: "9:1", 19.0: "19:1"}.get(br) if br is not None else None
    br_str = f"{br} ({br_desc})" if br_desc else (str(br) if br is not None else "-")
    lines.append(f"| **Balance ratio** (정상:공격) | {br_str} |\n")
    lines.extend([
        f"| **Num clients** | {data_cfg.get('num_clients', '-')} |\n",
        f"| **Binary** | {data_cfg.get('binary', '-')} |\n",
        f"| **Use SMOTE** | {data_cfg.get('use_smote', '-')} |\n",
        f"| **Model** | {model_cfg.get('name', '-')} |\n",
        f"| **FL rounds** | {fed_cfg.get('num_rounds', '-')} |\n",
        f"| **Local epochs** | {fed_cfg.get('local_epochs', '-')} |\n",
        f"| **Batch size** | {fed_cfg.get('batch_size', '-')} |\n",
        f"| **Learning rate** | {fed_cfg.get('learning_rate', '-')} |\n",
        f"| **Fraction fit** | {fed_cfg.get('fraction_fit', '-')} |\n",
        f"| **Fraction evaluate** | {fed_cfg.get('fraction_evaluate', '-')} |\n",
        f"| **Use class weights** | {fed_cfg.get('use_class_weights', '-')} |\n",
        f"| **Use focal loss** | {fed_cfg.get('use_focal_loss', '-')} |\n",
        f"| **Focal loss alpha** | {fed_cfg.get('focal_loss_alpha', '-')} |\n",
        f"| **Use QAT** | {fed_cfg.get('use_qat', '-')} |\n",
        f"| **Server momentum** | {fed_cfg.get('server_momentum', '-')} |\n",
        f"| **Server learning rate** | {fed_cfg.get('server_learning_rate', '-')} |\n",
        f"| **Min fit clients** | {fed_cfg.get('min_fit_clients', '-')} |\n",
        f"| **Min evaluate clients** | {fed_cfg.get('min_evaluate_clients', '-')} |\n",
        "\n",
    ])
    return "".join(lines)


def config_section_markdown_short(cfg: dict) -> str:
    """Shorter block for ratio_sweep_report (same as evaluate_ratio_sweep)."""
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})
    br = data_cfg.get("balance_ratio")
    br_desc = {1.0: "50:50", 4.0: "정상:공격 8:2", 9.0: "9:1", 19.0: "19:1"}.get(br) if br is not None else None
    br_str = f"{br} ({br_desc})" if br_desc else (str(br) if br is not None else "-")
    return "".join([
        "## Run / Training Configuration\n\n",
        "| Item | Value |\n|------|-------|\n",
        f"| **Data** | {data_cfg.get('name', '-')} |\n",
        f"| **Max samples** | {data_cfg.get('max_samples', '-')} |\n",
        f"| **Balance ratio** | {br_str} |\n",
        f"| **Num clients** | {data_cfg.get('num_clients', '-')} |\n",
        f"| **Model** | {model_cfg.get('name', '-')} |\n",
        f"| **FL rounds** | {fed_cfg.get('num_rounds', '-')} |\n",
        f"| **Local epochs** | {fed_cfg.get('local_epochs', '-')} |\n",
        f"| **Batch size** | {fed_cfg.get('batch_size', '-')} |\n",
        f"| **Learning rate** | {fed_cfg.get('learning_rate', '-')} |\n",
        f"| **Use QAT** | {fed_cfg.get('use_qat', '-')} |\n",
        "\n",
    ])


def insert_before_summary(content: str, section: str, marker: str = "\n\n## Summary\n\n") -> str:
    """Insert section before the first occurrence of marker. If section already in content, return as-is."""
    if "## Run / Training Configuration" in content:
        return content
    if marker not in content:
        return content
    idx = content.index(marker)
    return content[:idx] + section + marker + content[idx + len(marker):]


def main():
    ap = argparse.ArgumentParser(description="Backfill Run/Training Configuration into existing reports")
    ap.add_argument("--dry-run", action="store_true", help="Only print what would be done")
    args = ap.parse_args()
    dry = args.dry_run

    if not RUNS_BASE.exists():
        print(f"Run base not found: {RUNS_BASE}")
        sys.exit(1)

    # Collect run dirs: data/processed/runs/<version>/<run_id>/ that have at least one report
    run_dirs = []
    for version_dir in RUNS_BASE.iterdir():
        if not version_dir.is_dir() or version_dir.name.startswith("."):
            continue
        for run_id_dir in version_dir.iterdir():
            if not run_id_dir.is_dir():
                continue
            # Standard: run_id/analysis/compression_analysis.md or run_id/eval/ratio_sweep_report.md
            # Legacy: run_id/compression_analysis.md (e.g. v2_.../analysis/compression_analysis.md)
            if (
                (run_id_dir / "analysis" / "compression_analysis.md").exists()
                or (run_id_dir / "eval" / "ratio_sweep_report.md").exists()
                or (run_id_dir / "compression_analysis.md").exists()
            ):
                run_dirs.append(run_id_dir)

    run_dirs.sort(key=lambda p: (p.parent.name, p.name))
    print(f"Found {len(run_dirs)} run dirs under {RUNS_BASE}\n")

    updated_compression = 0
    updated_ratio_sweep = 0
    saved_config = 0

    for run_dir in run_dirs:
        rel = run_dir.relative_to(PROJECT_ROOT)
        cfg, run_config_path = get_config_for_run(run_dir)
        if not cfg:
            print(f"  ⏭️  {rel} — no config (skip)")
            continue

        # Save run_config.yaml if missing
        if run_config_path and not (run_dir / "run_config.yaml").exists():
            if not dry:
                _save_yaml(run_dir / "run_config.yaml", cfg)
            print(f"  📋 {rel} — saved run_config.yaml")
            saved_config += 1

        section_full = config_section_markdown(cfg, for_compression_report=True)
        section_short = config_section_markdown_short(cfg)

        # compression_analysis.md: insert before "## Summary" (standard: run_dir/analysis/, legacy: run_dir/)
        for comp_path in [run_dir / "analysis" / "compression_analysis.md", run_dir / "compression_analysis.md"]:
            if comp_path.exists():
                text = comp_path.read_text(encoding="utf-8")
                if "## Run / Training Configuration" not in text:
                    new_text = insert_before_summary(text, section_full, "\n\n## Summary\n\n")
                    if new_text != text:
                        if not dry:
                            comp_path.write_text(new_text, encoding="utf-8")
                        print(f"  📊 {rel}/{comp_path.relative_to(run_dir)} — added Run/Training Configuration")
                        updated_compression += 1

        # eval/ratio_sweep_report.md: insert before "## Summary"
        ratio_path = run_dir / "eval" / "ratio_sweep_report.md"
        if ratio_path.exists():
            text = ratio_path.read_text(encoding="utf-8")
            if "## Run / Training Configuration" not in text:
                new_text = insert_before_summary(text, section_short, "\n\n## Summary\n\n")
                if new_text != text:
                    if not dry:
                        ratio_path.write_text(new_text, encoding="utf-8")
                    print(f"  📊 {rel}/eval/ratio_sweep_report.md — added Run/Training Configuration")
                    updated_ratio_sweep += 1

    print()
    print(f"Done. run_config.yaml saved: {saved_config}, compression_analysis.md updated: {updated_compression}, ratio_sweep_report.md updated: {updated_ratio_sweep}")
    if dry:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
