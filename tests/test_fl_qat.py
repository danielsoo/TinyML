"""
Test FL + QAT (Quantization-Aware Training) without ray dependency.
Manually orchestrates FL training loop with QAT-enabled clients.
"""
import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from src.federated.client import load_config, KerasClient, _build_model
from src.data.loader import load_dataset, partition_non_iid
from src.modelcompression.quantization import (
    calculate_quantization_params,
    quantize_array,
    dequantize_array,
)


def quantize_weights(weights):
    """Quantize float32 weights to int8."""
    quantized = []
    quant_params = []
    for w in weights:
        if w.dtype in [np.float32, np.float64]:
            params = calculate_quantization_params(w, symmetric=True)
            q_w = quantize_array(w, params)
            quantized.append(q_w)
            quant_params.append(params)
        else:
            quantized.append(w)
            quant_params.append(None)
    return quantized, quant_params


def dequantize_weights(weights, quant_params=None):
    """Dequantize int8 weights back to float32."""
    dequantized = []
    for i, w in enumerate(weights):
        if w.dtype == np.int8:
            # Calculate params from the data if not provided
            params = quant_params[i] if quant_params and quant_params[i] else \
                     calculate_quantization_params(w.astype(np.float32), symmetric=True)
            dequantized.append(dequantize_array(w, params))
        else:
            dequantized.append(w)
    return dequantized


def fedavg_aggregate(client_weights_list, client_samples):
    """FedAvg: weighted average of client model weights."""
    total_samples = sum(client_samples)
    aggregated = []

    num_layers = len(client_weights_list[0])
    for layer_idx in range(num_layers):
        layer_weights = np.zeros_like(client_weights_list[0][layer_idx])
        for client_idx, weights in enumerate(client_weights_list):
            weight_factor = client_samples[client_idx] / total_samples
            layer_weights += weights[layer_idx] * weight_factor
        aggregated.append(layer_weights)

    return aggregated


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute precision, recall, F1 for binary classification."""
    y_true = y_true.astype(int).flatten()
    y_pred = y_pred.astype(int).flatten()

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def get_model_info(model):
    """Get model size and parameter count."""
    total_params = model.count_params()

    # Estimate model size (4 bytes per float32 param)
    model_size_bytes = 0
    for w in model.get_weights():
        model_size_bytes += w.nbytes

    return {
        'total_params': total_params,
        'model_size_kb': model_size_bytes / 1024,
        'model_size_mb': model_size_bytes / (1024 * 1024),
    }


def run_fl_qat_test(config_path: str = "config/federated_local_sky.yaml"):
    """Run FL with QAT without using ray/simulation."""
    print("\n" + "="*70)
    print("  FL + QAT Test (Manual Orchestration - No Ray)")
    print("="*70 + "\n")

    # Load config
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})

    # Extract settings
    num_clients = int(data_cfg.get("num_clients", 4))
    num_rounds = int(fed_cfg.get("num_rounds", 10))
    local_epochs = int(fed_cfg.get("local_epochs", 2))
    batch_size = int(fed_cfg.get("batch_size", 128))
    learning_rate = float(fed_cfg.get("learning_rate", 0.001))
    use_qat = fed_cfg.get("use_qat", True)
    use_class_weights = fed_cfg.get("use_class_weights", False)
    use_focal_loss = fed_cfg.get("use_focal_loss", False)
    focal_loss_alpha = float(fed_cfg.get("focal_loss_alpha", 0.75))
    model_name = model_cfg.get("name", "mlp")

    print(f"Configuration:")
    print(f"  - Clients: {num_clients}")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Local epochs: {local_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - QAT enabled: {use_qat}")
    print(f"  - Class weights: {use_class_weights}")
    print(f"  - Focal loss: {use_focal_loss}")
    print()

    # Load dataset
    dataset_name = data_cfg.get("name", "cicids2017")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    print(f"Loading dataset: {dataset_name}...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Determine num_classes
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

    input_shape = (x_train.shape[1],) if x_train.ndim == 2 else x_train.shape[1:]

    print(f"  - Train samples: {len(x_train):,}")
    print(f"  - Test samples: {len(x_test):,}")
    print(f"  - Features: {input_shape}")
    print(f"  - Classes: {num_classes}")
    print()

    # Partition data across clients (non-IID)
    train_parts = partition_non_iid(x_train, y_train, num_clients)
    test_parts = partition_non_iid(x_test, y_test, num_clients)

    print("Client data distribution:")
    for cid in range(num_clients):
        y_client = train_parts[cid]["y"]
        unique, counts = np.unique(y_client, return_counts=True)
        total = len(y_client)
        dist_str = ", ".join([f"Class {l}: {c} ({100*c/total:.1f}%)" for l, c in zip(unique, counts)])
        print(f"  Client {cid}: {total:,} samples - {dist_str}")
    print()

    # Build global model (server)
    print("Building global model...")
    global_model = _build_model(
        model_name, input_shape, num_classes, learning_rate,
        use_focal_loss=use_focal_loss, focal_loss_alpha=focal_loss_alpha
    )

    model_info = get_model_info(global_model)
    print(f"  - Parameters: {model_info['total_params']:,}")
    print(f"  - Model size: {model_info['model_size_kb']:.2f} KB")
    print()

    # Create clients with QAT
    print(f"Creating {num_clients} clients (QAT={use_qat})...")
    clients = []
    for cid in range(num_clients):
        # Build fresh model for each client
        client_model = _build_model(
            model_name, input_shape, num_classes, learning_rate,
            use_focal_loss=use_focal_loss, focal_loss_alpha=focal_loss_alpha
        )

        client = KerasClient(
            model=client_model,
            x_train=train_parts[cid]["x"],
            y_train=train_parts[cid]["y"],
            x_test=test_parts[cid]["x"],
            y_test=test_parts[cid]["y"],
            cid=cid,
            num_classes=num_classes,
            use_class_weights=use_class_weights,
            use_qat=use_qat,
            learning_rate=learning_rate,
        )
        clients.append(client)
    print()

    # Training history
    history = {
        'rounds': [],
        'accuracy': [],
        'loss': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'client_metrics': [],
    }

    # Get initial global weights
    global_weights = global_model.get_weights()

    # Quantization params cache for server->client communication
    server_quant_params = None

    # FL Training Loop
    print("="*70)
    print("  Starting Federated Learning")
    if use_qat:
        print("  [QAT Flow: Client -> quantize -> Server -> dequantize -> aggregate -> quantize -> Client]")
    print("="*70)

    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num}/{num_rounds} ---")

        client_weights_list = []
        client_samples = []
        round_client_metrics = []

        # Server: Quantize global weights before sending to clients (if QAT)
        if use_qat:
            weights_to_send, server_quant_params = quantize_weights(global_weights)
            if round_num == 1:
                orig_size = sum(w.nbytes for w in global_weights)
                quant_size = sum(w.nbytes for w in weights_to_send)
                print(f"  [Server->Client] Quantized: {orig_size/1024:.2f}KB -> {quant_size/1024:.2f}KB ({orig_size/quant_size:.1f}x)")
        else:
            weights_to_send = global_weights

        # Client training
        for cid, client in enumerate(clients):
            # Client: Dequantize weights received from server (if QAT)
            if use_qat:
                client_weights = dequantize_weights(weights_to_send, server_quant_params)
            else:
                client_weights = weights_to_send

            # Local training (client.fit handles internal dequantize/quantize)
            config = {"local_epochs": local_epochs, "batch_size": batch_size}
            new_weights, num_samples, _ = client.fit(client_weights, config)

            # Client: weights returned are already quantized if QAT (handled in client.fit)
            # Server: Dequantize received weights for aggregation
            if use_qat and new_weights[0].dtype == np.int8:
                new_weights = dequantize_weights(new_weights)

            client_weights_list.append(new_weights)
            client_samples.append(num_samples)

            # Evaluate client
            loss, num_eval, metrics = client.evaluate(new_weights, {})
            round_client_metrics.append(metrics)

            print(f"  Client {cid}: samples={num_samples:,}, "
                  f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")

        # Server: Aggregate in float32 (FedAvg)
        global_weights = fedavg_aggregate(client_weights_list, client_samples)

        # Evaluate global model on aggregated test set
        global_model.set_weights(global_weights)

        # Combine all test data for global evaluation
        x_test_all = np.concatenate([test_parts[i]["x"] for i in range(num_clients)])
        y_test_all = np.concatenate([test_parts[i]["y"] for i in range(num_clients)])

        global_loss, global_acc = global_model.evaluate(x_test_all, y_test_all, verbose=0)
        y_pred_prob = global_model.predict(x_test_all, verbose=0)
        y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

        metrics = compute_metrics(y_test_all, y_pred)

        # Record history
        history['rounds'].append(round_num)
        history['accuracy'].append(global_acc)
        history['loss'].append(global_loss)
        history['precision'].append(metrics['precision'])
        history['recall'].append(metrics['recall'])
        history['f1_score'].append(metrics['f1_score'])
        history['client_metrics'].append(round_client_metrics)

        print(f"\n  [Global] Loss: {global_loss:.4f}, Acc: {global_acc:.4f} ({global_acc*100:.2f}%)")
        print(f"           Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")

    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)

    # Final metrics
    final_acc = history['accuracy'][-1]
    final_f1 = history['f1_score'][-1]
    print(f"\nFinal Results:")
    print(f"  - Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  - F1 Score: {final_f1:.4f} ({final_f1*100:.2f}%)")
    print(f"  - Model Parameters: {model_info['total_params']:,}")
    print(f"  - Model Size: {model_info['model_size_kb']:.2f} KB")

    # Save results
    results_dir = Path("tests/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save history JSON
    results = {
        'config': {
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'local_epochs': local_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_qat': use_qat,
            'use_class_weights': use_class_weights,
            'use_focal_loss': use_focal_loss,
        },
        'model_info': model_info,
        'history': {
            'rounds': history['rounds'],
            'accuracy': [float(x) for x in history['accuracy']],
            'loss': [float(x) for x in history['loss']],
            'precision': [float(x) for x in history['precision']],
            'recall': [float(x) for x in history['recall']],
            'f1_score': [float(x) for x in history['f1_score']],
        },
        'final_metrics': {
            'accuracy': float(final_acc),
            'f1_score': float(final_f1),
        }
    }

    json_path = results_dir / "fl_qat_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Generate plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Accuracy plot
    axes[0, 0].plot(history['rounds'], history['accuracy'], 'b-o', label='Accuracy')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Global Model Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss plot
    axes[0, 1].plot(history['rounds'], history['loss'], 'r-o', label='Loss')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Global Model Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # F1 Score plot
    axes[1, 0].plot(history['rounds'], history['f1_score'], 'g-o', label='F1 Score')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Global Model F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Precision/Recall plot
    axes[1, 1].plot(history['rounds'], history['precision'], 'm-o', label='Precision')
    axes[1, 1].plot(history['rounds'], history['recall'], 'c-s', label='Recall')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Precision & Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.suptitle(f'FL + QAT Training Results\n({num_clients} clients, {num_rounds} rounds, QAT={use_qat})')
    plt.tight_layout()

    plot_path = results_dir / "fl_qat_training_plot.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.close()

    # Save global model
    model_path = results_dir / "fl_qat_global_model.h5"
    global_model.save(model_path)
    print(f"Model saved to: {model_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FL + QAT Test")
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated_local_sky.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    results = run_fl_qat_test(args.config)
