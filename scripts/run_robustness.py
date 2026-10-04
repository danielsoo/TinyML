#!/usr/bin/env python3
"""
FGSM + PGD robustness for a set of models per dataset (wrapper around run_pgd.py).

Adversarial examples are generated on the first .h5 model of each entry (white-box on the
float FL model) and transferred to every other model (Keras and TFLite), as in run_pgd.py.

Spec (YAML):
  attack_config: config/attack/default.yaml
  entries:
    - name: cic
      config: config/paper_v12_float/fl_baseline.yaml
      models: [path/to/b_fl.h5, path/to/a_centralized.h5, path/to/deploy.tflite, ...]

Usage:
  python scripts/run_robustness.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, run_cmd


def main():
    parser = argparse.ArgumentParser(description="FGSM/PGD robustness over model sets")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    spec = load_yaml(args.spec)
    out_dir = Path(args.output_dir)
    attack_cfg = spec.get("attack_config", "config/attack/default.yaml")
    rows, failed = [], []
    for entry in spec["entries"]:
        missing = [m for m in entry["models"] if not (ROOT / m).exists()]
        if missing:
            print(f"⚠️ {entry['name']}: missing models {missing}")
        for attack in spec.get("attacks", ["fgsm", "pgd"]):
            run_dir = out_dir / entry["name"] / attack
            ok = run_cmd(
                [sys.executable, str(ROOT / "scripts" / "run_pgd.py"),
                 "--models", *entry["models"],
                 "--config", entry["config"],
                 "--fgsm-config", attack_cfg,
                 "--attack", attack,
                 "--output-dir", str(run_dir)],
                f"{attack.upper()} on {entry['name']}",
            )
            if not ok:
                failed.append(f"{entry['name']}/{attack}")
                continue
            res = json.loads((run_dir / "pgd_results.json").read_text(encoding="utf-8"))
            for r in res.get("comparison", res.get("models", [])):
                rows.append({"dataset": entry["name"], "attack": attack, **{
                    k: r.get(k) for k in ["display_name", "original_accuracy",
                                          "adversarial_accuracy", "attack_success_rate"]}})
            rows_to_csv(rows, out_dir / "robustness.csv")
    rows_to_markdown(rows, out_dir / "robustness.md", "FGSM / PGD robustness (transfer from float FL model)")
    if failed:
        print(f"❌ failed: {failed}")
        return 1
    print(f"✅ Robustness results in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
