#!/usr/bin/env python3
"""
Single client fit test - isolate Simulation crash cause.
On server: python scripts/test_client_fit.py --config config/federated_scratch.yaml
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# TMPDIR etc (avoid server home quota)
if "TMPDIR" not in os.environ and os.path.exists("/scratch"):
    user = os.environ.get("USER", "")
    if user:
        scratch_tmp = f"/scratch/{user}/tmp"
        if os.path.exists(scratch_tmp):
            os.environ.setdefault("TMPDIR", scratch_tmp)
            os.environ.setdefault("TEMP", scratch_tmp)
            os.environ.setdefault("TMP", scratch_tmp)

import argparse
import yaml
import numpy as np
from src.data.loader import load_dataset, partition_non_iid
from src.federated.client import _build_model, KerasClient, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/federated_scratch.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})

    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    x_train, y_train, x_test, y_test = load_dataset(data_cfg.get("name", "cicids2017"), **dataset_kwargs)

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    if num_classes == 1:
        num_classes = 2
    input_shape = (x_train.shape[1],) if x_train.ndim == 2 else x_train.shape[1:]
    train_parts = partition_non_iid(x_train, y_train, 1)
    test_parts = partition_non_iid(x_test, y_test, 1)

    model = _build_model(
        model_cfg.get("name", "mlp"),
        input_shape,
        num_classes,
        float(fed_cfg.get("learning_rate", 0.001)),
        fed_cfg.get("use_focal_loss", False),
    )
    client = KerasClient(
        model,
        train_parts[0]["x"],
        train_parts[0]["y"],
        test_parts[0]["x"],
        test_parts[0]["y"],
        cid=0,
        num_classes=num_classes,
        use_class_weights=fed_cfg.get("use_class_weights", False),
    )

    print("Running single client fit (1 epoch)...")
    config = {"local_epochs": 1, "batch_size": 128, "use_callbacks": False}
    try:
        weights, num_examples, _ = client.fit(client.get_parameters(config), config)
        print(f"OK: fit completed, {num_examples} samples, {len(weights)} weight arrays")
    except Exception as e:
        import traceback
        print("=" * 60)
        print("CLIENT FIT FAILED - Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
