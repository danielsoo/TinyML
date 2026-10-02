"""Shared helpers for LCTES review ablation scripts."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_keras_for_eval(path: str | Path):
    """Load a saved .h5 for evaluation; QAT checkpoints need tf_keras + quantize_scope."""
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(str(path), compile=False)
    except ValueError as err:
        if "Quantize" not in str(err):
            raise
        import tensorflow_model_optimization as tfmot
        import tf_keras

        with tfmot.quantization.keras.quantize_scope():
            model = tf_keras.models.load_model(str(path), compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def save_yaml(path: str | Path, cfg: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)


def make_output_dir(prefix: str = "ablation") -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ROOT / "data" / "processed" / prefix / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def run_cmd(cmd: List[str], desc: str, cwd: Optional[Path] = None) -> bool:
    print(f"\n{'=' * 72}\n  {desc}\n  $ {' '.join(cmd)}\n{'=' * 72}\n")
    result = subprocess.run(cmd, cwd=str(cwd or ROOT), check=False)
    if result.returncode != 0:
        print(f"❌ Failed: {desc} (exit {result.returncode})")
        return False
    print(f"✅ Done: {desc}")
    return True


def apply_quick_mode(cfg: dict, quick: bool) -> dict:
    if not quick:
        return cfg
    cfg = deepcopy(cfg)
    fed = cfg.setdefault("federated", {})
    fed["num_rounds"] = min(int(fed.get("num_rounds", 80)), 5)
    fed["local_epochs"] = min(int(fed.get("local_epochs", 2)), 1)
    return cfg


def write_config_variant(base_path: Path, overrides: dict, out_path: Path) -> Path:
    cfg = load_yaml(base_path)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = {**cfg[key], **val}
        else:
            cfg[key] = val
    save_yaml(out_path, cfg)
    return out_path


def _union_keys(rows: List[Dict[str, Any]]) -> List[str]:
    # Rows can differ (e.g. a "missing_model" row has no metric columns)
    return list(dict.fromkeys(k for row in rows for k in row))


def rows_to_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _union_keys(rows)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_markdown(rows: List[Dict[str, Any]], path: Path, title: str) -> None:
    if not rows:
        return
    cols = _union_keys(rows)
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
