from pathlib import Path
from typing import Tuple
import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*protobuf.*')

import numpy as np
import yaml

from src.data.loader import load_dataset, partition_non_iid
from src.models import nets

# Suppress TensorFlow logging after import
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    pass


# ------------------------------------------------
# Configuration Loading
# ------------------------------------------------

def load_config(config_path: str = None):
    """Load configuration file. Defaults to environment variable or auto-detected path.
    
    If config_path is None, automatically detects environment (Colab vs local)
    and selects appropriate config file.
    """
    if config_path is None:
        # Try environment variable first
        config_path = os.getenv("FEDERATED_CONFIG", None)
        
        # If not set, auto-detect environment
        if config_path is None:
            try:
                from src.utils.env_utils import get_default_config_path
                config_path = get_default_config_path()
            except ImportError:
                # Fallback to local config if env_utils not available
                config_path = "config/federated_local.yaml"
    
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path.resolve()}")
    
    with cfg_path.open() as f:
        return yaml.safe_load(f)

# Default config load (can be reloaded in main)
CFG = None




def _build_model(model_name: str, input_shape: Tuple[int, ...], num_classes: int):
    """Build model based on model name. Supports both standard (nets.py) and lightweight architectures."""
    name = (model_name or "mlp").lower()

    # Check if it's a lightweight architecture
    if name in ["lightweight", "bottleneck", "bottleneck_mlp", "tiny", "tiny_mlp"]:
        from src.models.lightweight import get_lightweight_model
        return get_lightweight_model(name, input_shape, num_classes)

    # Standard architectures (nets.py)
    if hasattr(nets, "get_model"):
        return nets.get_model(model_name, input_shape, num_classes)

    if name in ["cnn", "small_cnn"] and hasattr(nets, "make_small_cnn"):
        return nets.make_small_cnn(input_shape, num_classes)
    if hasattr(nets, "make_mlp"):
        return nets.make_mlp(input_shape, num_classes)

    raise AttributeError(
        f"Model '{model_name}' not found. Options: mlp, cnn, lightweight, bottleneck, tiny"
    )


# ------------------------------------------------
# Simulation Preparation
# ------------------------------------------------

def simulate_clients(config: dict = None):
    """Prepare client simulation."""
    if config is None:
        config = CFG
    data_cfg = config.get("data", {})
    fed_cfg = config.get("federated", {})
    model_cfg = config.get("model", {})

    dataset_name = data_cfg.get("name", "bot_iot")

    # Pass all config except name and num_clients to data loader
    dataset_kwargs = {
        k: v
        for k, v in data_cfg.items()
        if k not in {"name", "num_clients"}
    }

    num_clients = int(data_cfg.get("num_clients", 1))

    try:
        x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    except FileNotFoundError as err:
        data_path = dataset_kwargs.get("data_path") or dataset_kwargs.get("path")
        hint = f" (checked path: {data_path})" if data_path else ""
        raise FileNotFoundError(
            f"Failed to load dataset '{dataset_name}'{hint}. "
            f"If using Colab, please check if CSV files exist in the specified path."
        ) from err

    # Calculate num_classes
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)

    # Defense code for binary classification (0/1) when only one class exists
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

    # Normalize input shape (2D for MLP)
    if x_train.ndim == 2:
        input_shape = (x_train.shape[1],)
    else:
        input_shape = x_train.shape[1:]

    # Partition data
    train_parts = partition_non_iid(x_train, y_train, num_clients)
    test_parts = partition_non_iid(x_test, y_test, num_clients)

    model_name = model_cfg.get("name", "mlp")

    state = {
        "num_clients": num_clients,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "model_name": model_name,
        "train_parts": train_parts,
        "test_parts": test_parts,
    }

    return state


# ------------------------------------------------
# Execution
# ------------------------------------------------

def _fedavg_simulation(state: dict, fed_cfg: dict):
    """
    Manual FedAvg simulation loop (no Ray dependency).

    Implements the same logic as fl.simulation.start_simulation but runs
    entirely in-process, compatible with any Python version.
    """
    num_clients = state["num_clients"]
    num_rounds = int(fed_cfg.get("num_rounds", 3))
    batch_size = int(fed_cfg.get("batch_size", 32))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    # Build initial global model
    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
    )
    global_weights = global_model.get_weights()

    print(f"\n  FedAvg Simulation: {num_clients} clients, {num_rounds} rounds")
    print(f"  Model: {state['model_name']}, Params: {global_model.count_params():,}\n")

    for round_num in range(1, num_rounds + 1):
        print(f"{'─'*60}")
        print(f"  Round {round_num}/{num_rounds}")
        print(f"{'─'*60}")

        # Each client trains locally
        client_weights = []
        client_sizes = []

        for cid in range(num_clients):
            # Build fresh model and set global weights
            client_model = _build_model(
                state["model_name"],
                state["input_shape"],
                state["num_classes"],
            )
            client_model.set_weights(global_weights)

            # Get client data
            x_tr = state["train_parts"][cid]["x"]
            y_tr = state["train_parts"][cid]["y"]

            # Local training
            client_model.fit(
                x_tr, y_tr,
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
            )

            client_weights.append(client_model.get_weights())
            client_sizes.append(len(x_tr))

        # FedAvg: weighted average of client weights
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

        mean_acc = np.mean([m["accuracy"] for m in all_metrics])
        mean_loss = np.mean([m["loss"] for m in all_metrics])
        print(f"  Accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%), Loss: {mean_loss:.4f}")

    # Final evaluation with detailed metrics
    print(f"\n{'─'*60}")
    print(f"  Final Evaluation")
    print(f"{'─'*60}")

    # Combine all test data for final eval
    all_x_test = np.concatenate([state["test_parts"][c]["x"] for c in range(num_clients)])
    all_y_test = np.concatenate([state["test_parts"][c]["y"] for c in range(num_clients)])

    final_loss, final_acc = global_model.evaluate(all_x_test, all_y_test, verbose=0)
    y_prob = global_model.predict(all_x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1) if y_prob.shape[1] > 1 else (y_prob.ravel() >= 0.5).astype(int)
    y_true = all_y_test.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"  Accuracy:  {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  Loss:      {final_loss:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return global_model


def main(save_path: str = "src/models/global_model.h5", config_path: str = None):
    """Main execution function."""
    global CFG
    CFG = load_config(config_path)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    state = simulate_clients(CFG)
    fed_cfg = CFG.get("federated", {})

    # Use manual FedAvg simulation (no Ray dependency)
    global_model = _fedavg_simulation(state, fed_cfg)

    # Save model
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    global_model.save(save_path)
    print(f"\n✅ Saved global model to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-model",
        type=str,
        default="src/models/global_model.h5",
        help="Path to save the global model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: FEDERATED_CONFIG env var or config/federated_local.yaml)",
    )
    args = parser.parse_args()
    try:
        main(save_path=args.save_model, config_path=args.config)
    except Exception as e:
        import traceback
        print(f"\n❌ Training failed with error: {type(e).__name__}: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

