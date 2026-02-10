#!/usr/bin/env python3
"""
Run threshold tuning for multiple normal:attack ratios and append results to ratio_sweep_report.md.
Used by run.py after evaluate_ratio_sweep.py.

Usage:
  python scripts/tune_threshold_all_ratios.py --config config/federated.yaml --model models/tflite/saved_model_original.tflite --append-to data/processed/runs/v18/.../eval/ratio_sweep_report.md
"""
import argparse
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _stage_label_from_path(model_path: str) -> str:
    """Return a short display label for the model path (e.g. for report section headers)."""
    p = str(model_path).lower()
    if "saved_model_qat_pruned_float32" in p or "qat_pruned_float32" in p:
        return "QAT+Prune only"
    if "saved_model_qat_ptq" in p:
        return "QAT+PTQ"
    if "saved_model_no_qat_ptq" in p:
        return "noQAT+PTQ"
    if "saved_model_original" in p:
        return "Original"
    if "saved_model_pruned_quantized" in p:
        return "Compressed (PTQ)"
    return Path(model_path).stem


def main():
    parser = argparse.ArgumentParser(description="Tune threshold for multiple ratios and append to report")
    parser.add_argument("--config", default="config/federated_local.yaml", help="Config YAML")
    parser.add_argument("--model", default="models/tflite/saved_model_original.tflite", help="Model path")
    parser.add_argument("--append-to", type=str, required=True, help="Path to ratio_sweep_report.md to append to")
    parser.add_argument("--ratios", type=str, default="90,80,70,60,50", help="Comma-separated normal%% ratios")
    parser.add_argument("--metric", type=str, default="f1", choices=["f1", "attack_recall", "normal_recall", "balanced"])
    args = parser.parse_args()

    report_path = Path(args.append_to)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    ratios = [int(x.strip()) for x in args.ratios.split(",")]
    stage_label = _stage_label_from_path(args.model)

    with open(report_path, "a", encoding="utf-8") as f:
        f.write(f"\n## Threshold Tuning ({stage_label})\n\n")
        f.write(f"Model: `{args.model}`\n\n")

    for ratio in ratios:
        cmd = [
            sys.executable,
            str(project_root / "scripts" / "tune_threshold.py"),
            "--config", args.config,
            "--model", args.model,
            "--ratio", str(ratio),
            "--metric", args.metric,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        if result.returncode != 0:
            print(f"Warning: tune_threshold --ratio {ratio} failed: {result.stderr}", file=sys.stderr)
            continue
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"### Normal {ratio}% : Attack {100-ratio}%\n\n")
            f.write("```\n")
            f.write(result.stdout)
            f.write("```\n\n")

    print(f"Appended threshold tuning to {report_path}")


if __name__ == "__main__":
    main()
