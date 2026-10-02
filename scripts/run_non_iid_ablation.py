#!/usr/bin/env python3
"""
Non-IID ablation: FL baseline × partition strategies × client counts.

Usage:
  python scripts/run_non_iid_ablation.py
  python scripts/run_non_iid_ablation.py --quick --strategies label_balanced,dirichlet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ablation_utils import (
    apply_quick_mode,
    load_keras_for_eval,
    load_yaml,
    make_output_dir,
    rows_to_csv,
    rows_to_markdown,
    run_cmd,
    save_json,
    save_yaml,
)
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_keras_model


def _client_distribution(
    cfg: dict, strategy: str, num_clients: int
) -> List[Dict[str, float]]:
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    kwargs["return_attack_labels"] = strategy in {"attack_class", "attack_shift"}

    name = data_cfg.get("name", "cicids2017")
    if kwargs.get("return_attack_labels"):
        x_train, y_train, _, _, attack_train, _ = load_dataset(name, **kwargs)
    else:
        x_train, y_train, _, _ = load_dataset(name, **kwargs)
        attack_train = None

    parts = partition_data(
        x_train,
        y_train,
        num_clients,
        strategy=strategy,
        dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
        attack_labels=attack_train,
    )
    stats = []
    for cid, part in enumerate(parts):
        y = part["y"]
        n = max(len(y), 1)
        stats.append(
            {
                "client": cid,
                "total": len(y),
                "attack_pct": round(100.0 * float((y == 1).sum()) / n, 2),
                "normal_pct": round(100.0 * float((y == 0).sum()) / n, 2),
            }
        )
    return stats


def main():
    parser = argparse.ArgumentParser(description="Non-IID FL ablation")
    parser.add_argument("--base-config", default="config/ablation/fl_baseline.yaml")
    parser.add_argument("--client-counts", default="4,8,16")
    parser.add_argument(
        "--strategies",
        default="label_balanced,dirichlet,attack_class",
        help="Comma-separated partition strategies",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else make_output_dir("ablation/non_iid")
    configs_dir = out_dir / "configs"
    models_dir = out_dir / "models"
    configs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    client_counts = [int(x) for x in args.client_counts.split(",") if x.strip()]
    rows: List[Dict[str, Any]] = []

    for strategy in strategies:
        for n_clients in client_counts:
            cfg_path = configs_dir / f"fl_{strategy}_c{n_clients}.yaml"
            cfg = apply_quick_mode(load_yaml(ROOT / args.base_config), args.quick)
            cfg.setdefault("data", {})["partition_strategy"] = strategy
            cfg["data"]["num_clients"] = n_clients
            cfg["federated"]["min_fit_clients"] = n_clients
            cfg["federated"]["min_evaluate_clients"] = n_clients
            cfg["federated"]["min_available_clients"] = n_clients
            save_yaml(cfg_path, cfg)

            dist = _client_distribution(cfg, strategy, n_clients)
            model_path = models_dir / f"fl_{strategy}_c{n_clients}.h5"

            if not args.skip_train:
                ok = run_cmd(
                    [
                        sys.executable,
                        "-m",
                        "src.federated.client",
                        "--config",
                        str(cfg_path),
                        "--save-model",
                        str(model_path),
                    ],
                    f"FL train {strategy} clients={n_clients}",
                )
                if not ok:
                    continue

            if not model_path.exists():
                rows.append(
                    {
                        "strategy": strategy,
                        "num_clients": n_clients,
                        "status": "missing_model",
                    }
                )
                continue

            threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
            kwargs = {k: v for k, v in cfg["data"].items() if k not in {"name", "num_clients"}}
            if "path" in kwargs:
                kwargs["data_path"] = kwargs.pop("path")
            _, _, x_test, y_test = load_dataset(cfg["data"]["name"], **kwargs)
            model = load_keras_for_eval(model_path)
            metrics = evaluate_keras_model(model, x_test, y_test, threshold=threshold)

            row = {
                "strategy": strategy,
                "num_clients": n_clients,
                "client_distribution": str(dist),
                **{k: metrics[k] for k in [
                    "accuracy", "precision", "recall", "attack_recall", "f1",
                    "fn", "fp", "missed_attacks", "false_alarm_rate",
                ]},
            }
            rows.append(row)

    rows_to_csv(rows, out_dir / "non_iid_ablation.csv")
    rows_to_markdown(rows, out_dir / "non_iid_ablation.md", "Non-IID FL Ablation")
    save_json(rows, out_dir / "non_iid_ablation.json")
    print(f"✅ Non-IID ablation saved to {out_dir}")


if __name__ == "__main__":
    main()
