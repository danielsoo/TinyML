#!/usr/bin/env python3
"""
Generate VERSIONS.md - a comprehensive comparison of all experiment runs.

Scans analysis folders, extracts config + metrics, and produces a single
markdown table for paper/report use.

Usage:
  python scripts/generate_version_summary.py
  python scripts/generate_version_summary.py --analysis-dir data/processed/analysis
  python scripts/generate_version_summary.py --output docs/VERSIONS.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    yaml = None


def load_config(analysis_dir: Path) -> Optional[Dict]:
    """Load config from config_snapshot.yaml or compression_analysis.json."""
    config_path = analysis_dir / "config_snapshot.yaml"
    if config_path.exists() and yaml:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    json_path = analysis_dir / "compression_analysis.json"
    if json_path.exists():
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
            if data.get("config"):
                return data["config"]
            # Fallback: build minimal config from data_version (e.g. cicids2017_max1500k_bal1.0)
            dv = data.get("data_version", "")
            if dv:
                return {"_data_version": dv, "_inferred": True}
    return None


def load_metrics(analysis_dir: Path) -> Optional[Dict]:
    """Load metrics from compression_analysis.json."""
    json_path = analysis_dir / "compression_analysis.json"
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def config_to_summary(cfg: Optional[Dict]) -> str:
    """Condense config to a short summary string."""
    if not cfg:
        return "-"
    # Inferred from data_version only
    if cfg.get("_inferred") and cfg.get("_data_version"):
        return cfg["_data_version"]
    parts = []
    data = cfg.get("data", {})
    fed = cfg.get("federated", {})

    parts.append(data.get("name", "?"))
    if data.get("max_samples"):
        parts.append(f"max{data['max_samples']//1000}k")
    if data.get("balance_ratio") is not None:
        parts.append(f"bal{data['balance_ratio']}")
    if data.get("use_smote"):
        parts.append("smote")
    parts.append(f"r{fed.get('num_rounds', '?')}ep{fed.get('local_epochs', '?')}")
    if fed.get("use_focal_loss"):
        parts.append(f"fl{fed.get('focal_loss_alpha', '')}")
    if fed.get("server_momentum"):
        parts.append("FedAvgM")
    return " | ".join(str(p) for p in parts)


def load_version_changes(changelog_path: Optional[Path] = None) -> Dict:
    """Load detailed changelog from config/version_changes.yaml."""
    if changelog_path is None:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        changelog_path = project_root / "config" / "version_changes.yaml"
    changelog_path = Path(changelog_path)
    if not changelog_path.exists():
        return {}
    if yaml is None and changelog_path.suffix in (".yaml", ".yml"):
        return {}
    try:
        if changelog_path.suffix in (".yaml", ".yml") and yaml:
            with open(changelog_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def changes_to_bullets(changes: Dict) -> str:
    """Format version changes as markdown bullets."""
    if not changes:
        return ""
    lines = []
    for category, items in changes.items():
        if isinstance(items, list):
            for item in items:
                lines.append(f"  - **{category}**: {item}")
        elif isinstance(items, str):
            lines.append(f"  - **{category}**: {items}")
    return "\n".join(lines) if lines else ""


def main():
    parser = argparse.ArgumentParser(description="Generate VERSIONS.md from analysis runs")
    parser.add_argument(
        "--analysis-dir",
        type=str,
        default="data/processed/analysis",
        help="Root analysis directory (e.g. data/processed/analysis or TinyML-results/processed/analysis)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for VERSIONS.md (default: <analysis-dir>/VERSIONS.md)",
    )
    parser.add_argument(
        "--changelog",
        type=str,
        default=None,
        help="Path to version_changes.yaml or VERSION_CHANGELOG.md (default: config/version_changes.yaml)",
    )
    args = parser.parse_args()

    root = Path(args.analysis_dir)
    if not root.exists():
        print(f"Analysis dir not found: {root}")
        return 1

    # Collect all version/run folders
    runs: List[tuple] = []
    for ver_dir in sorted(root.iterdir()):
        if not ver_dir.is_dir() or ver_dir.name.startswith("."):
            continue
        for run_dir in sorted(ver_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            json_path = run_dir / "compression_analysis.json"
            if not json_path.exists():
                continue
            runs.append((ver_dir.name, run_dir.name, run_dir))

    if not runs:
        print(f"No analysis runs found under {root}")
        return 1

    # Sort by (version, run_id)
    runs.sort(key=lambda x: (x[0], x[1]))

    changelog_path = Path(args.changelog) if args.changelog else None
    version_changes = load_version_changes(changelog_path)

    # Build table rows
    rows = []
    for version, run_id, run_path in runs:
        metrics_data = load_metrics(run_path)
        config = load_config(run_path)
        if not metrics_data:
            continue

        res = metrics_data.get("results", [])
        if not res:
            continue

        best_acc = max(r["accuracy"] for r in res)
        best_f1 = max(r["f1_score"] for r in res)
        orig = res[0]
        comp = next((r for r in res if "compress" in r["stage"].lower()), res[-1] if len(res) > 1 else orig)

        precision_orig = orig.get("precision", 0)
        recall_orig = orig.get("recall", 0)
        precision_comp = comp.get("precision", 0)
        recall_comp = comp.get("recall", 0)

        ratio = orig["file_size_mb"] / comp["file_size_mb"] if comp["file_size_mb"] > 0 else 0
        config_summary = config_to_summary(config)

        rows.append({
            "version": version,
            "run_id": run_id,
            "config": config_summary,
            "best_acc": best_acc,
            "best_f1": best_f1,
            "orig_mb": orig["file_size_mb"],
            "comp_mb": comp["file_size_mb"],
            "ratio": ratio,
            "p_orig": precision_orig,
            "r_orig": recall_orig,
            "p_comp": precision_comp,
            "r_comp": recall_comp,
        })

    # Write VERSIONS.md
    out_path = Path(args.output) if args.output else root / "VERSIONS.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Versions Summary\n\n")
        f.write("All runs with configuration and compression metrics for paper/report.\n\n")
        f.write("## Full Comparison Table\n\n")
        f.write("| Version | Run | Config Summary | Best Acc | Best F1 | Orig P/R | Comp P/R | Orig (MB) | Comp (MB) | Ratio |\n")
        f.write("|---------|-----|----------------|----------|---------|----------|----------|-----------|-----------|-------|\n")

        for r in rows:
            orig_pr = f"{r['p_orig']:.2f}/{r['r_orig']:.2f}"
            comp_pr = f"{r['p_comp']:.2f}/{r['r_comp']:.2f}"
            cfg = (r['config'][:50] + "…") if len(r['config']) > 50 else r['config']
            f.write(
                f"| {r['version']} | {r['run_id']} | {cfg} | "
                f"{r['best_acc']:.4f} | {r['best_f1']:.4f} | {orig_pr} | {comp_pr} | "
                f"{r['orig_mb']:.3f} | {r['comp_mb']:.3f} | {r['ratio']:.1f}x |\n"
            )

        f.write("\n## Configuration Key\n\n")
        f.write("- **dataset**: cicids2017, bot_iot, etc.\n")
        f.write("- **maxNk**: max_samples (e.g. max1500k = 1.5M)\n")
        f.write("- **balR**: balance_ratio (e.g. bal1.0 = 50:50)\n")
        f.write("- **smote**: use_smote=true\n")
        f.write("- **rN**: num_rounds\n")
        f.write("- **epN**: local_epochs\n")
        f.write("- **flα**: focal_loss + alpha\n")
        f.write("- **FedAvgM**: server momentum\n")
        f.write("\n*P/R = Precision / Recall (Original | Compressed)*\n")

        # Detailed version changelog (from config/version_changes.yaml)
        if version_changes:
            f.write("\n---\n\n")
            f.write("## Version Changelog (Detailed)\n\n")
            f.write("What changed in each version (training, compression, quantization).\n\n")
            seen_versions = set()
            for r in rows:
                ver = r["version"]
                if ver in seen_versions:
                    continue
                seen_versions.add(ver)
                changes = version_changes.get(ver, {})
                if not changes:
                    f.write(f"### {ver}\n\n- *(No changelog entry)*\n\n")
                    continue
                f.write(f"### {ver}\n\n")
                for category, items in changes.items():
                    if isinstance(items, list):
                        f.write(f"- **{category}**\n")
                        for item in items:
                            f.write(f"  - {item}\n")
                        f.write("\n")
                    elif isinstance(items, str):
                        f.write(f"- **{category}**: {items}\n\n")
                f.write("\n")
        else:
            # Fallback: include standalone changelog if available
            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent
            md_changelog = project_root / "docs" / "VERSION_CHANGELOG.md"
            if md_changelog.exists():
                f.write("\n---\n\n")
                f.write(md_changelog.read_text(encoding="utf-8"))

    print(f"✅ Wrote {out_path} ({len(rows)} runs)")
    return 0


if __name__ == "__main__":
    exit(main())
