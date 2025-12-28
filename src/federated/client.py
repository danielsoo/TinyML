from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*protobuf.*')

import flwr as fl
import numpy as np
import yaml
from flwr.common import parameters_to_ndarrays

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
    """Load configuration file. Defaults to environment variable or default path."""
    if config_path is None:
        # Check environment variable for config file path
        import os
        config_path = os.getenv("FEDERATED_CONFIG", "config/federated_local.yaml")
    
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path.resolve()}")
    
    with cfg_path.open() as f:
        return yaml.safe_load(f)

# Default config load (can be reloaded in main)
CFG = None


class SaveModelStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy that stores latest parameters for saving."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters: Optional[fl.common.Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


def _build_model(model_name: str, input_shape: Tuple[int, ...], num_classes: int):
    """Build model based on model name. Falls back to default functions if get_model is not available in nets module."""
    if hasattr(nets, "get_model"):
        return nets.get_model(model_name, input_shape, num_classes)

    name = (model_name or "mlp").lower()
    if name in ["cnn", "small_cnn"] and hasattr(nets, "make_small_cnn"):
        return nets.make_small_cnn(input_shape, num_classes)
    if hasattr(nets, "make_mlp"):
        return nets.make_mlp(input_shape, num_classes)

    raise AttributeError(
        "Model creation function not found. Please check if 'get_model' or 'make_mlp' is defined."
    )


# ------------------------------------------------
# Flower NumPyClient
# ------------------------------------------------

class KerasClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, cid: int, num_classes: int):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cid = cid
        self.num_classes = num_classes

    def get_parameters(self, config: Dict[str, Any]):
        return self.model.get_weights()

    def fit(self, parameters, config: Dict[str, Any]):
        self.model.set_weights(parameters)

        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))

        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )

        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config: Dict[str, Any]):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        y_true = self.y_test.astype(int)
        y_prob = self.model.predict(self.x_test, verbose=0)

        if y_prob.ndim == 2 and y_prob.shape[1] == 1:
            y_prob = y_prob.ravel()

        if self.num_classes <= 2:
            if y_prob.ndim > 1:
                y_prob = y_prob[:, 0]
            y_pred = (y_prob >= 0.5).astype(int)
        else:
            if y_prob.ndim == 1:
                raise ValueError("Expected 2D probability array for multi-class output.")
            y_pred = np.argmax(y_prob, axis=1)

        total = len(y_true)
        attack_actual = int(np.sum(y_true == 1))
        normal_actual = int(np.sum(y_true == 0))
        attack_predicted = int(np.sum(y_pred == 1))
        normal_predicted = int(np.sum(y_pred == 0))

        true_positives = int(np.sum((y_true == 1) & (y_pred == 1)))
        true_negatives = int(np.sum((y_true == 0) & (y_pred == 0)))
        false_positives = int(np.sum((y_true == 0) & (y_pred == 1)))
        false_negatives = int(np.sum((y_true == 1) & (y_pred == 0)))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "accuracy": float(acc),
            "loss": float(loss),
            "total_samples": total,
            "actual_attack": attack_actual,
            "actual_normal": normal_actual,
            "predicted_attack": attack_predicted,
            "predicted_normal": normal_predicted,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
        }

        return float(loss), len(self.x_test), metrics


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

    # State to use in client_fn
    state = {
        "num_clients": num_clients,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "model_name": model_name,
        "train_parts": train_parts,
        "test_parts": test_parts,
    }

    def client_fn(context: fl.common.Context) -> fl.client.Client:
        # Safely map cid from Flower/Ray to index
        raw_cid = None

        # VCE / New API
        if hasattr(context, "node_config") and isinstance(context.node_config, dict):
            raw_cid = context.node_config.get("cid", None)

        # Legacy
        if raw_cid is None and hasattr(context, "cid"):
            raw_cid = context.cid

        try:
            cid_int = int(raw_cid)
        except (TypeError, ValueError):
            cid_int = hash(str(raw_cid))

        idx = cid_int % state["num_clients"]

        part_tr = state["train_parts"][idx]
        part_te = state["test_parts"][idx]

        model = _build_model(
            state["model_name"],
            state["input_shape"],
            state["num_classes"],
        )

        client = KerasClient(
            model,
            part_tr["x"],
            part_tr["y"],
            part_te["x"],
            part_te["y"],
            cid=idx,
            num_classes=state["num_classes"],
        )

        return client.to_client()

    return state, client_fn


# ------------------------------------------------
# Execution
# ------------------------------------------------

def main(save_path: str = "src/models/global_model.h5", config_path: str = None):
    """Main execution function."""
    global CFG
    # Load configuration file
    CFG = load_config(config_path)
    
    # Ensure model directory exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    (state, client_fn) = simulate_clients(CFG)
    fed_cfg = CFG.get("federated", {})

    num_rounds = int(fed_cfg.get("num_rounds", 3))
    batch_size = int(fed_cfg.get("batch_size", 32))
    local_epochs = int(fed_cfg.get("local_epochs", 1))

    def evaluate_metrics_aggregation_fn(results):
        if not results:
            return {}

        metrics_list = [m[1] for m in results if m[1] is not None]
        if not metrics_list:
            return {}

        mean_accuracy = float(np.mean([m.get("accuracy", 0.0) for m in metrics_list]))
        mean_loss = float(np.mean([m.get("loss", 0.0) for m in metrics_list]))

        aggregated = {"accuracy": mean_accuracy, "loss": mean_loss}

        first = metrics_list[0]
        for key in [
            "total_samples",
            "actual_attack",
            "actual_normal",
            "predicted_attack",
            "predicted_normal",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1_score",
        ]:
            if key in first:
                aggregated[key] = first[key]

        print("\n📊 Evaluation Summary")
        print("=" * 60)
        print(f"Accuracy: {aggregated['accuracy']:.4f} ({aggregated['accuracy']*100:.2f}%)")
        print(f"Loss: {aggregated['loss']:.4f}\n")

        if "actual_attack" in aggregated and "actual_normal" in aggregated and "total_samples" in aggregated:
            print("📈 Ground Truth:")
            print(f"  - Attack samples: {aggregated['actual_attack']}")
            print(f"  - Normal samples: {aggregated['actual_normal']}")
            print(f"  - Total samples: {aggregated['total_samples']}\n")

        if "predicted_attack" in aggregated and "predicted_normal" in aggregated:
            print("🔮 Predictions:")
            print(f"  - Predicted Attack: {aggregated['predicted_attack']}")
            print(f"  - Predicted Normal: {aggregated['predicted_normal']}\n")

        if {"true_positives", "true_negatives", "false_positives", "false_negatives"} <= aggregated.keys():
            print("✅ Confusion Matrix:")
            print(f"  - True Positives (TP): {aggregated['true_positives']}")
            print(f"  - True Negatives (TN): {aggregated['true_negatives']}")
            print(f"  - False Positives (FP): {aggregated['false_positives']}")
            print(f"  - False Negatives (FN): {aggregated['false_negatives']}\n")

        metric_present = all(k in aggregated for k in ["precision", "recall", "f1_score"])
        if metric_present:
            precision_pct = aggregated["precision"] * 100.0
            recall_pct = aggregated["recall"] * 100.0
            f1_pct = aggregated["f1_score"] * 100.0
            print("📏 Metrics:")
            print(f"  - Precision: {aggregated['precision']:.4f} ({precision_pct:.2f}%)")
            print(f"  - Recall: {aggregated['recall']:.4f} ({recall_pct:.2f}%)")
            print(f"  - F1-Score: {aggregated['f1_score']:.4f} ({f1_pct:.2f}%)")

        print("=" * 60 + "\n")

        return aggregated

    strategy = SaveModelStrategy(
        fraction_fit=fed_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fed_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fed_cfg.get("min_fit_clients", state["num_clients"]),
        min_evaluate_clients=fed_cfg.get("min_evaluate_clients", state["num_clients"]),
        min_available_clients=fed_cfg.get("min_available_clients", state["num_clients"]),
        on_fit_config_fn=lambda rnd: {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
        },
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=state["num_clients"],
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Save global model
    # Use weights from last round
    # Assume start_simulation has filled strategy.parameters with final values
    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
    )
    if getattr(strategy, "latest_parameters", None) is not None:
        weights = parameters_to_ndarrays(strategy.latest_parameters)
        global_model.set_weights(weights)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        global_model.save(save_path)
        print(f"✅ Saved global model to {save_path}")
    else:
        print("⚠️ Could not find saveable parameters in Federated strategy.")


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
    main(save_path=args.save_model, config_path=args.config)

