#!/usr/bin/env python3
"""
Train a list of federated models (v13 pipeline) and keep every checkpoint and manifest.

Spec (YAML):
  base_config: config/v13/ton_fl.yaml
  runs:
    - {name: F_s1, seed: 1}
    - {name: S80_s1, seed: 1, overrides: {model: {hidden_units: [256, 128, 64]}}}

Each run writes <out>/configs/<name>.yaml, <out>/models/<name>.h5, <name>_manifest.{json,npz}
and <out>/logs/<name>.log. Finished runs (model file present) are skipped, so the step resumes.
The test split is not touched here; runs.json records training time and the manifest hashes.

Usage:
  python scripts/train_fl_runs.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import load_yaml


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    spec = load_yaml(args.spec)
    out = Path(args.output_dir)
    for sub in ("configs", "models", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    base = load_yaml(ROOT / spec["base_config"])
    summary_path = out / "runs.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    failed = 0
    for run in spec["runs"]:
        name = run["name"]
        model_path = out / "models" / f"{name}.h5"
        if model_path.exists():
            print(f"⏭  {name} exists")
            continue
        cfg = deep_merge(base, run.get("overrides", {}))
        cfg.setdefault("federated", {})["seed"] = int(run["seed"])
        cfg_path = out / "configs" / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        t0 = time.time()
        print(f"▶ {name} (seed {run['seed']})", flush=True)
        with open(out / "logs" / f"{name}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-m", "src.federated.client", "--config", str(cfg_path),
                                 "--save-model", str(model_path)], cwd=ROOT, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        minutes = round((time.time() - t0) / 60, 1)
        ok = rc == 0 and model_path.exists()
        man = out / "models" / f"{name}_manifest.json"
        entry = {"seed": run["seed"], "minutes": minutes, "ok": ok, "config": str(cfg_path.relative_to(out))}
        if man.exists():
            m = json.loads(man.read_text())
            entry["test_hash"] = m["test"]["hash"]
            entry["client_hashes"] = [(c["train"]["hash"], c["val"]["hash"]) for c in m["clients"]]
        summary[name] = entry
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"{'✅' if ok else '❌'} {name} in {minutes} min (rc={rc})", flush=True)
        failed += 0 if ok else 1
    lines = ["# Federated training runs (v13)", "", "| run | seed | minutes | ok |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['seed']} | {v['minutes']} | {v['ok']} |" for k, v in summary.items()]
    (out / "train_fl.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
