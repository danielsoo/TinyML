"""
test_train.py — Federated Learning Training Only

Runs FL training on CICIDS2017 dataset, displays results, and stores:
  1. Trained global model (.h5)
  2. Training metrics (JSON) for use by test_distillation.py

Usage:
    python test_train.py
    python test_train.py --config config/federated_cicids.yaml
    python test_train.py --output-dir models/cicids/
    python test_train.py --qat  # Enable Quantization-Aware Training
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import yaml
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Project root (one level up from test/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import load_dataset
from src.federated.client import load_config, simulate_clients, _build_model, _apply_qat, _extract_base_weights_from_qat


def run_fedavg(state: dict, fed_cfg: dict, verbose: bool = True, use_qat: bool = False):
    """
    FedAvg simulation with per-round metrics tracking.

    Args:
        state: Client simulation state
        fed_cfg: Federated learning configuration
        verbose: Print progress
        use_qat: Enable Quantization-Aware Training (clients train with fake quantization)

    Returns:
        global_model: Trained Keras model
        history: Dict with per-round metrics
    """
    num_clients = state["num_clients"]
    num_rounds = int(fed_cfg.get("num_rounds", 100))
    batch_size = int(fed_cfg.get("batch_size", 32))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
    )
    global_weights = global_model.get_weights()

    history = {
        "rounds": [],
        "accuracy": [],
        "loss": [],
    }

    qat_status = " [QAT]" if use_qat else ""
    if verbose:
        print(f"\n  FedAvg: {num_clients} clients, {num_rounds} rounds{qat_status}")
        print(f"  Model: {state['model_name']}, Params: {global_model.count_params():,}")
        print(f"  Local epochs: {local_epochs}, Batch size: {batch_size}\n")

    for round_num in range(1, num_rounds + 1):
        if verbose:
            print(f"{'─'*60}")
            print(f"  Round {round_num}/{num_rounds}")
            print(f"{'─'*60}")

        client_weights = []
        client_sizes = []

        for cid in range(num_clients):
            client_model = _build_model(
                state["model_name"],
                state["input_shape"],
                state["num_classes"],
            )
            client_model.set_weights(global_weights)

            # Apply QAT if enabled (wrap with fake quantization)
            if use_qat:
                client_model = _apply_qat(client_model)

            x_tr = state["train_parts"][cid]["x"]
            y_tr = state["train_parts"][cid]["y"]

            client_model.fit(
                x_tr, y_tr,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
            )

            # Extract weights to send back to server
            if use_qat:
                # Build fresh base model to extract weights into
                base_model = _build_model(
                    state["model_name"],
                    state["input_shape"],
                    state["num_classes"],
                )
                base_model = _extract_base_weights_from_qat(client_model, base_model)
                client_weights.append(base_model.get_weights())
            else:
                client_weights.append(client_model.get_weights())

            client_sizes.append(len(x_tr))

        # FedAvg: weighted average
        total_samples = sum(client_sizes)
        avg_weights = []
        for layer_idx in range(len(global_weights)):
            layer_avg = np.zeros_like(global_weights[layer_idx])
            for cid in range(num_clients):
                weight = client_sizes[cid] / total_samples
                layer_avg += weight * client_weights[cid][layer_idx]
            avg_weights.append(layer_avg)

        global_weights = avg_weights
        global_model.set_weights(global_weights)

        # Evaluate on each client's test data
        all_metrics = []
        for cid in range(num_clients):
            x_te = state["test_parts"][cid]["x"]
            y_te = state["test_parts"][cid]["y"]
            loss, acc = global_model.evaluate(x_te, y_te, verbose=0)
            all_metrics.append({"accuracy": acc, "loss": loss})

        mean_acc = float(np.mean([m["accuracy"] for m in all_metrics]))
        mean_loss = float(np.mean([m["loss"] for m in all_metrics]))

        history["rounds"].append(round_num)
        history["accuracy"].append(mean_acc)
        history["loss"].append(mean_loss)

        if verbose:
            print(f"  Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%), Loss: {mean_loss:.4f}")

    return global_model, history


def evaluate_final(model, state: dict):
    """Final evaluation with detailed metrics."""
    num_clients = state["num_clients"]

    all_x_test = np.concatenate([state["test_parts"][c]["x"] for c in range(num_clients)])
    all_y_test = np.concatenate([state["test_parts"][c]["y"] for c in range(num_clients)])

    final_loss, final_acc = model.evaluate(all_x_test, all_y_test, verbose=0)
    y_prob = model.predict(all_x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1) if y_prob.shape[1] > 1 else (y_prob.ravel() >= 0.5).astype(int)
    y_true = all_y_test.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        "accuracy": float(final_acc),
        "loss": float(final_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total_samples": len(all_y_test),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="FL Training (CICIDS2017)")
    parser.add_argument('--config', type=str, default='config/federated_cicids.yaml')
    parser.add_argument('--output-dir', type=str, default='models/cicids/')
    parser.add_argument('--model-name', type=str, default=None,
                        help='Override model name (mlp, lightweight, bottleneck, tiny)')
    parser.add_argument('--qat', action='store_true',
                        help='Enable Quantization-Aware Training (QAT) during FL')
    args = parser.parse_args()

    qat_label = " [QAT]" if args.qat else ""
    print("\n" + "=" * 70)
    print(f"TEST_TRAIN: Federated Learning Training{qat_label}")
    print("=" * 70)

    # Load config
    config = load_config(args.config)

    if args.model_name:
        config["model"]["name"] = args.model_name

    model_name = config.get("model", {}).get("name", "mlp")
    print(f"\n  Config: {args.config}")
    print(f"  Model: {model_name}")
    print(f"  Output: {args.output_dir}")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate clients (loads data + partitions)
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING + PARTITIONING")
    print("=" * 70)

    state = simulate_clients(config)

    print(f"\n  Clients: {state['num_clients']}")
    print(f"  Input shape: {state['input_shape']}")
    print(f"  Classes: {state['num_classes']}")
    print(f"  Model: {state['model_name']}")

    # Compute per-client data sizes
    for cid in range(state["num_clients"]):
        train_size = len(state["train_parts"][cid]["x"])
        test_size = len(state["test_parts"][cid]["x"])
        print(f"    Client {cid}: train={train_size:,}, test={test_size:,}")

    # Run FL training
    print("\n" + "=" * 70)
    print("STEP 2: FEDERATED LEARNING (FedAvg)")
    print("=" * 70)

    fed_cfg = config.get("federated", {})
    global_model, history = run_fedavg(state, fed_cfg, verbose=True, use_qat=args.qat)

    # Final evaluation
    print(f"\n{'─'*60}")
    print(f"  Final Evaluation")
    print(f"{'─'*60}")

    final_metrics = evaluate_final(global_model, state)

    print(f"  Accuracy:  {final_metrics['accuracy']:.4f} ({final_metrics['accuracy']*100:.2f}%)")
    print(f"  Loss:      {final_metrics['loss']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1_score']:.4f}")
    print(f"  TP={final_metrics['tp']}, TN={final_metrics['tn']}, FP={final_metrics['fp']}, FN={final_metrics['fn']}")

    # Save model
    model_path = str(output_dir / "global_model.h5")
    global_model.save(model_path)
    print(f"\n  Model saved: {model_path}")

    # Save results as JSON (for test_distillation.py to use)
    results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": args.config,
        "model_name": model_name,
        "model_path": model_path,
        "model_params": int(global_model.count_params()),
        "input_shape": list(state["input_shape"]),
        "num_classes": state["num_classes"],
        "num_clients": state["num_clients"],
        "qat_enabled": args.qat,
        "federated": {
            "num_rounds": int(fed_cfg.get("num_rounds", 100)),
            "local_epochs": int(fed_cfg.get("local_epochs", 1)),
            "batch_size": int(fed_cfg.get("batch_size", 32)),
        },
        "history": history,
        "final_metrics": final_metrics,
    }

    results_path = str(output_dir / "train_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path}")

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Model: {model_name} ({global_model.count_params():,} params)")
    print(f"  Final accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Final F1-score: {final_metrics['f1_score']:.4f}")
    print(f"\n  Files saved:")
    print(f"    {model_path}")
    print(f"    {results_path}")
    print(f"\n  Next step: python test_distillation.py --train-results {results_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
