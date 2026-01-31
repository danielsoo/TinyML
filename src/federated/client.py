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
# QAT (Quantization-Aware Training) Utilities
# ------------------------------------------------

def _apply_qat(model):
    """
    Apply quantization-aware training to a model.

    Wraps the model with fake quantization nodes that simulate INT8
    quantization during training. The model still uses float32 weights
    but learns to be robust to quantization noise.

    Returns:
        QAT-wrapped model (still float32, but quantization-aware)
    """
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError(
            "tensorflow-model-optimization is required for QAT. "
            "Install with: pip install tensorflow-model-optimization"
        )

    # tfmot requires tf.keras models (not keras 3.x)
    # Clone the model using tf.keras to ensure compatibility
    import tensorflow as tf

    try:
        # Try direct quantization first (works with tf.keras Sequential)
        qat_model = tfmot.quantization.keras.quantize_model(model)
    except ValueError:
        # Model might be keras 3.x - rebuild as tf.keras Sequential
        # Get config and weights from original model
        config = model.get_config()
        weights = model.get_weights()

        # Rebuild using tf.keras
        from tf_keras import Sequential, layers as tf_layers, Input

        new_model = Sequential()
        for layer_config in config['layers']:
            layer_class = layer_config['class_name']
            layer_cfg = layer_config['config']

            if layer_class == 'InputLayer':
                new_model.add(Input(shape=layer_cfg['batch_shape'][1:]))
            elif layer_class == 'Dense':
                new_model.add(tf_layers.Dense(
                    units=layer_cfg['units'],
                    activation=layer_cfg['activation'],
                    name=layer_cfg['name']
                ))

        new_model.set_weights(weights)
        new_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        qat_model = tfmot.quantization.keras.quantize_model(new_model)

    # Recompile with same settings
    qat_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return qat_model


def _strip_qat(qat_model):
    """
    Strip QAT wrappers from model, keeping the learned weights.

    After QAT training, this removes the fake quantization nodes
    and returns a regular Keras model with float32 weights that
    have been trained to be quantization-friendly.

    Returns:
        Regular Keras model (float32 weights, no QAT wrappers)
    """
    try:
        import tensorflow_model_optimization as tfmot
    except ImportError:
        raise ImportError("tensorflow-model-optimization is required")

    return tfmot.quantization.keras.quantize_model(qat_model)


def _extract_base_weights_from_qat(qat_model, base_model):
    """
    Extract the core weights from a QAT model and apply to base model.

    QAT models have extra quantization-related variables. This function
    extracts only the trainable kernel/bias weights and applies them
    to a fresh base (non-QAT) model for aggregation.

    Args:
        qat_model: QAT-wrapped model after training
        base_model: Fresh base model (same architecture, no QAT)

    Returns:
        base_model with weights copied from qat_model
    """
    # Get weight names from base model to know what to extract
    base_weight_names = [w.name for w in base_model.weights]

    # Build mapping from QAT weights to base weights
    qat_weights = qat_model.get_weights()
    base_weights = base_model.get_weights()

    # QAT adds extra weights (quantize_layer, etc.)
    # We need to match kernel/bias weights by position in layer order

    # Simpler approach: match by counting Dense/Conv layers
    new_weights = []
    qat_idx = 0

    for base_weight in base_weights:
        # Find corresponding weight in QAT model by shape
        found = False
        for i in range(qat_idx, len(qat_weights)):
            if qat_weights[i].shape == base_weight.shape:
                new_weights.append(qat_weights[i])
                qat_idx = i + 1
                found = True
                break

        if not found:
            # Keep original weight if no match (shouldn't happen)
            new_weights.append(base_weight)

    base_model.set_weights(new_weights)
    return base_model


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

def _fedavg_simulation(state: dict, fed_cfg: dict, use_qat: bool = False):
    """
    Manual FedAvg simulation loop (no Ray dependency).

    Implements the same logic as fl.simulation.start_simulation but runs
    entirely in-process, compatible with any Python version.

    Args:
        state: Client simulation state from simulate_clients()
        fed_cfg: Federated learning configuration
        use_qat: If True, clients train with quantization-aware training (QAT)

    QAT Flow (when use_qat=True):
        1. Server holds float32 global weights
        2. Client receives weights → applies QAT → trains with fake quantization
        3. Client extracts float32 weights (trained to be quantization-robust)
        4. Server aggregates float32 weights via FedAvg
        5. Final model can be converted to INT8 with minimal accuracy loss
    """
    num_clients = state["num_clients"]
    num_rounds = int(fed_cfg.get("num_rounds", 3))
    batch_size = int(fed_cfg.get("batch_size", 32))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    # Build initial global model (float32, no QAT)
    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
    )
    global_weights = global_model.get_weights()

    qat_status = " [QAT enabled]" if use_qat else ""
    print(f"\n  FedAvg Simulation: {num_clients} clients, {num_rounds} rounds{qat_status}")
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

            # Apply QAT if enabled (wrap with fake quantization)
            if use_qat:
                client_model = _apply_qat(client_model)

            # Get client data
            x_tr = state["train_parts"][cid]["x"]
            y_tr = state["train_parts"][cid]["y"]

            # Local training (with QAT fake-quantization if enabled)
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

        # FedAvg: weighted average of client weights (float32)
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


def main(save_path: str = "src/models/global_model.h5", config_path: str = None, use_qat: bool = False):
    """Main execution function."""
    global CFG
    CFG = load_config(config_path)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    state = simulate_clients(CFG)
    fed_cfg = CFG.get("federated", {})

    # Use manual FedAvg simulation (no Ray dependency)
    global_model = _fedavg_simulation(state, fed_cfg, use_qat=use_qat)

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
    parser.add_argument(
        "--qat",
        action="store_true",
        help="Enable Quantization-Aware Training (QAT) during FL",
    )
    args = parser.parse_args()
    try:
        main(save_path=args.save_model, config_path=args.config, use_qat=args.qat)
    except Exception as e:
        import traceback
        print(f"\n❌ Training failed with error: {type(e).__name__}: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
