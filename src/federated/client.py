from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import csv
import json
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Reduce Ray backend log noise (e.g. metrics exporter, large object dumps) when using FL simulation
os.environ.setdefault('RAY_BACKEND_LOG_LEVEL', 'warning')
warnings.filterwarnings('ignore', category=UserWarning)

# Avoid home disk quota: redirect temp to /scratch
if 'TMPDIR' not in os.environ and os.path.exists('/scratch'):
    user = os.environ.get('USER', '')
    if user:
        scratch_tmp = f'/scratch/{user}/tmp'
        if not os.path.exists(scratch_tmp):
            try:
                os.makedirs(scratch_tmp, exist_ok=True)
            except OSError:
                pass
        if os.path.exists(scratch_tmp):
            os.environ['TMPDIR'] = scratch_tmp
            os.environ['TEMP'] = scratch_tmp
            os.environ['TMP'] = scratch_tmp
warnings.filterwarnings('ignore', message='.*protobuf.*')

import flwr as fl
import numpy as np
import yaml
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.data.loader import load_dataset, partition_non_iid
from src.models import nets
from src.modelcompression.quantization import (
    QuantizationParams,
    calculate_quantization_params,
    quantize_array,
    dequantize_array,
)

# Suppress TensorFlow logging after import
try:
    import tensorflow as tf
    import tensorflow_model_optimization as tfmot
    tf.get_logger().setLevel('ERROR')
    _KERAS_CALLBACKS = tf.keras.callbacks
except ImportError:
    try:
        from keras import callbacks as _KERAS_CALLBACKS
    except ImportError:
        _KERAS_CALLBACKS = None


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
    
    with cfg_path.open(encoding='utf-8') as f:
        return yaml.safe_load(f)

# Default config load (can be reloaded in main)
CFG = None


def _get_strategy_base():
    """Use FedAvgM (momentum) when available to mitigate client drift."""
    if hasattr(fl.server.strategy, "FedAvgM"):
        return fl.server.strategy.FedAvgM
    return fl.server.strategy.FedAvg


class SaveModelStrategy(_get_strategy_base()):
    """FedAvg/FedAvgM strategy that stores latest parameters for saving.
    
    FedAvgM: server-side momentum mitigates client drift and improves convergence stability.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters: Optional[fl.common.Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate fit results (sample-weighted, with momentum if FedAvgM)."""
        if not results:
            return None, {}
        
        total_examples = sum([num_examples for _, fit_res in results for num_examples in [fit_res.num_examples]])
        if server_round % 5 == 0:
            print(f"\n[Round {server_round}] Client contributions:")
            for client_proxy, fit_res in results:
                weight = fit_res.num_examples / total_examples
                print(f"  Client samples: {fit_res.num_examples:,} (weight: {weight:.4f})")
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics


def _build_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False,
    focal_loss_alpha: float = 0.75,
):
    """Build model based on model name."""
    if hasattr(nets, "get_model"):
        return nets.get_model(
            model_name, input_shape, num_classes, learning_rate,
            use_focal_loss=use_focal_loss, focal_loss_alpha=focal_loss_alpha,
        )

    name = (model_name or "mlp").lower()
    if name in ["cnn", "small_cnn"] and hasattr(nets, "make_small_cnn"):
        return nets.make_small_cnn(input_shape, num_classes, learning_rate)
    if hasattr(nets, "make_mlp"):
        return nets.make_mlp(
            input_shape, num_classes, learning_rate,
            use_focal_loss=use_focal_loss, focal_loss_alpha=focal_loss_alpha,
        )

    raise AttributeError(
        "Model creation function not found. Please check if 'get_model' or 'make_mlp' is defined."
    )


# ------------------------------------------------
# Flower NumPyClient
# ------------------------------------------------

class KerasClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        x_train,
        y_train,
        x_test,
        y_test,
        cid: int,
        num_classes: int,
        use_class_weights: bool = False,
        use_qat: bool = False,
        learning_rate: float = 0.001,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.cid = cid
        self.num_classes = num_classes
        self.use_class_weights = use_class_weights
        self.use_qat = use_qat
        self.learning_rate = learning_rate

        # Apply QAT if enabled
        if use_qat:
            self.model = self._apply_qat(model)
            # Log only once (client 0) to avoid repeated messages when using Ray/simulation
            if cid == 0:
                print("[Client 0] QAT enabled - model quantization-aware")
        else:
            self.model = model

        # Compute class weights for imbalanced data (manual computation with smoothing)
        if self.use_class_weights:
            y_labels = np.argmax(y_train, axis=1) if len(y_train.shape) > 1 else y_train
            classes, counts = np.unique(y_labels, return_counts=True)
            
            # Smoothed class weights: use sqrt to mitigate extreme weights
            n_samples = len(y_labels)
            n_classes = len(classes)
            # Base weight calculation
            raw_weights = n_samples / (n_classes * counts)
            # Apply square root for smoothing
            smoothed_weights = np.sqrt(raw_weights)
            # Normalize
            class_weights_array = smoothed_weights / np.mean(smoothed_weights)
            
            self.class_weight = {int(cls): float(weight) for cls, weight in zip(classes, class_weights_array)}
            # Compact log to avoid huge (0,0.0),(1,0.0),... dumps when many classes or Ray wraps output
            if len(self.class_weight) <= 8:
                print(f"[Client {cid}] Class weights (smoothed): {self.class_weight}")
            else:
                w_vals = list(self.class_weight.values())
                print(f"[Client {cid}] Class weights: {len(self.class_weight)} classes, min={min(w_vals):.3f}, max={max(w_vals):.3f}")
        else:
            self.class_weight = None

        # Cache for quantization parameters (for dequantizing received weights)
        self.quant_params_cache: List[Optional[QuantizationParams]] = []

    def _quantize_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Quantize float32 weights to int8 before sending to server.
        Stores quantization params for potential later use.
        """
        if not self.use_qat:
            return weights

        quantized = []
        self.quant_params_cache = []

        for w in weights:
            if w.dtype in [np.float32, np.float64]:
                params = calculate_quantization_params(w, symmetric=True)
                q_w = quantize_array(w, params)
                quantized.append(q_w)
                self.quant_params_cache.append(params)
            else:
                # Already quantized or non-float type
                quantized.append(w)
                self.quant_params_cache.append(None)

        return quantized

    def _dequantize_weights(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        Dequantize int8 weights received from server back to float32.
        """
        if not self.use_qat:
            return weights

        dequantized = []

        for i, w in enumerate(weights):
            if w.dtype == np.int8:
                # Need to dequantize - calculate params from the data
                params = calculate_quantization_params(w.astype(np.float32), symmetric=True)
                dequantized.append(dequantize_array(w, params))
            else:
                # Already float32
                dequantized.append(w)

        return dequantized

    def _apply_qat(self, model):
        """
        For QAT in federated learning, we keep the original model architecture
        to ensure weight compatibility between clients and server.

        The quantization happens only during communication:
        - Weights are quantized (int8) before sending to server
        - Weights are dequantized (float32) when received from server

        This approach ensures:
        1. Model architecture stays the same (compatible with FedAvg)
        2. Communication is compressed (4x reduction)
        3. Training still happens in float32 for accuracy

        Note: For true QAT with tfmot.quantize_model(), the model structure changes
        which breaks weight compatibility. That should be done post-training instead.
        """
        # Keep the original model - QAT for FL is implemented via
        # quantize/dequantize in communication only
        print(f"[Client {self.cid}] QAT mode: quantized communication enabled")
        return model

    def get_parameters(self, config: Dict[str, Any]):
        """Get model weights, quantized if QAT is enabled."""
        weights = self.model.get_weights()
        if self.use_qat:
            return self._quantize_weights(weights)
        return weights

    def fit(self, parameters, config: Dict[str, Any]):
        # Dequantize weights received from server if QAT is enabled
        if self.use_qat:
            parameters = self._dequantize_weights(parameters)
        self.model.set_weights(parameters)

        # Learning rate decay by round: lr = base_lr * (lr_decay ** (round-1))
        lr_base = config.get("learning_rate")
        lr_decay = float(config.get("lr_decay", 1.0))
        server_round = int(config.get("server_round", 1))
        if lr_base is not None and getattr(self.model.optimizer, "learning_rate", None) is not None:
            if lr_decay < 1.0:
                lr = float(lr_base) * (lr_decay ** (server_round - 1))
                lr = max(lr, 1e-6)
            else:
                lr = float(lr_base)
            self.model.optimizer.learning_rate.assign(lr)

        epochs = int(config.get("local_epochs", 1))
        batch_size = int(config.get("batch_size", 32))

        # Prepare fit arguments
        fit_kwargs = {
            'x': self.x_train,
            'y': self.y_train,
            'epochs': epochs,
            'batch_size': batch_size,
            'verbose': 0,
        }
        use_callbacks = config.get("use_callbacks", False) and _KERAS_CALLBACKS is not None
        if use_callbacks:
            fit_kwargs['validation_data'] = (self.x_test, self.y_test)
            fit_kwargs['callbacks'] = [
                _KERAS_CALLBACKS.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=0
                ),
                _KERAS_CALLBACKS.EarlyStopping(
                    monitor='val_loss', patience=4, restore_best_weights=True, verbose=0
                ),
            ]
        if self.class_weight is not None:
            fit_kwargs['class_weight'] = self.class_weight

        self.model.fit(**fit_kwargs)

        # Quantize weights before sending to server if QAT is enabled
        weights = self.model.get_weights()
        if self.use_qat:
            weights = self._quantize_weights(weights)

        return weights, len(self.x_train), {}

    def evaluate(self, parameters, config: Dict[str, Any]):
        # Dequantize weights received from server if QAT is enabled
        if self.use_qat:
            parameters = self._dequantize_weights(parameters)
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

    # Print client data distribution for debugging class imbalance
    print("\n" + "="*60)
    print("CLIENT DATA DISTRIBUTION")
    print("="*60)
    for cid in range(num_clients):
        y_client = train_parts[cid]["y"]
        unique, counts = np.unique(y_client, return_counts=True)
        total = len(y_client)
        print(f"Client {cid}: Total={total:,} samples")
        for label, count in zip(unique, counts):
            pct = 100.0 * count / total
            print(f"  - Class {label}: {count:,} ({pct:.2f}%)")
    print("="*60 + "\n")

    model_name = model_cfg.get("name", "mlp")
    use_class_weights = fed_cfg.get("use_class_weights", False)
    use_focal_loss = fed_cfg.get("use_focal_loss", False)
    focal_loss_alpha = float(fed_cfg.get("focal_loss_alpha", 0.75))
    learning_rate = float(fed_cfg.get("learning_rate", 0.001))
    use_qat = fed_cfg.get("use_qat", False)

    if use_qat:
        print("[Config] QAT (Quantization-Aware Training) enabled for clients")

    # State to use in client_fn
    state = {
        "num_clients": num_clients,
        "input_shape": input_shape,
        "num_classes": num_classes,
        "model_name": model_name,
        "train_parts": train_parts,
        "test_parts": test_parts,
        "use_class_weights": use_class_weights,
        "use_focal_loss": use_focal_loss,
        "focal_loss_alpha": focal_loss_alpha,
        "learning_rate": learning_rate,
        "use_qat": use_qat,
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
            state["learning_rate"],
            state.get("use_focal_loss", False),
        )

        client = KerasClient(
            model,
            part_tr["x"],
            part_tr["y"],
            part_te["x"],
            part_te["y"],
            cid=idx,
            num_classes=state["num_classes"],
            use_class_weights=state["use_class_weights"],
            use_qat=state.get("use_qat", False),
            learning_rate=state["learning_rate"],
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

    # Per-round metrics storage (for reports and problem diagnosis)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "fl_evaluation_history.csv"
    json_path = output_dir / "fl_evaluation_history.json"
    round_counter: List[int] = [0]
    metrics_history: List[Dict[str, Any]] = []

    num_clients = state["num_clients"]
    # CSV header (including per-device accuracy columns)
    header = [
        "round", "accuracy", "accuracy_pct", "loss",
        "precision", "precision_pct", "recall", "recall_pct", "f1_score", "f1_pct",
    ]
    for i in range(num_clients):
        header.extend([f"client_{i}_accuracy", f"client_{i}_accuracy_pct"])
    header.extend([
        "total_samples", "actual_attack", "actual_normal",
        "predicted_attack", "predicted_normal",
        "true_positives", "true_negatives", "false_positives", "false_negatives",
    ])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

    def _get_metrics(res_item):
        r = res_item[1]
        if r is None:
            return None
        # Flower 1.x: EvaluateRes has .metrics attribute (Record/dict-like)
        if hasattr(r, "metrics") and r.metrics is not None:
            m = r.metrics
            if isinstance(m, dict):
                return m
            if hasattr(m, "items"):
                return dict(m)
            return {}
        # Legacy: tuple (loss, num_examples, metrics)
        if isinstance(r, (tuple, list)) and len(r) >= 3:
            return r[2] if isinstance(r[2], dict) else {}
        return None

    def evaluate_metrics_aggregation_fn(results):
        if not results:
            return {}

        metrics_list = [_get_metrics(x) for x in results if x[1] is not None]
        metrics_list = [m for m in metrics_list if m is not None]
        if not metrics_list:
            return {}

        round_counter[0] += 1
        current_round = round_counter[0]

        mean_accuracy = float(np.mean([m.get("accuracy", 0.0) for m in metrics_list]))
        mean_loss = float(np.mean([m.get("loss", 0.0) for m in metrics_list]))
        mean_precision = float(np.mean([m.get("precision", 0.0) for m in metrics_list]))
        mean_recall = float(np.mean([m.get("recall", 0.0) for m in metrics_list]))
        mean_f1 = float(np.mean([m.get("f1_score", 0.0) for m in metrics_list]))

        accuracy_pct = mean_accuracy * 100.0
        precision_pct = mean_precision * 100.0
        recall_pct = mean_recall * 100.0
        f1_pct = mean_f1 * 100.0

        # Per-device accuracy (client 0, 1, 2, 3 in order)
        client_accuracies = []
        client_accuracies_pct = []
        for i in range(num_clients):
            if i < len(metrics_list):
                acc = float(metrics_list[i].get("accuracy", 0.0))
                client_accuracies.append(acc)
                client_accuracies_pct.append(acc * 100.0)
            else:
                client_accuracies.append(0.0)
                client_accuracies_pct.append(0.0)

        aggregated = {
            "accuracy": mean_accuracy,
            "loss": mean_loss,
            "precision": mean_precision,
            "recall": mean_recall,
            "f1_score": mean_f1,
        }

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
        ]:
            if key in first:
                aggregated[key] = first[key]

        # Per-round record (including per-device metrics)
        row = {
            "round": current_round,
            "accuracy": mean_accuracy,
            "accuracy_pct": accuracy_pct,
            "loss": mean_loss,
            "precision": mean_precision,
            "precision_pct": precision_pct,
            "recall": mean_recall,
            "recall_pct": recall_pct,
            "f1_score": mean_f1,
            "f1_pct": f1_pct,
            "client_accuracies": client_accuracies,
            "client_accuracies_pct": client_accuracies_pct,
            "total_samples": aggregated.get("total_samples", ""),
            "actual_attack": aggregated.get("actual_attack", ""),
            "actual_normal": aggregated.get("actual_normal", ""),
            "predicted_attack": aggregated.get("predicted_attack", ""),
            "predicted_normal": aggregated.get("predicted_normal", ""),
            "true_positives": aggregated.get("true_positives", ""),
            "true_negatives": aggregated.get("true_negatives", ""),
            "false_positives": aggregated.get("false_positives", ""),
            "false_negatives": aggregated.get("false_negatives", ""),
        }
        metrics_history.append(row)

        csv_row = [
            row["round"], row["accuracy"], f"{row['accuracy_pct']:.2f}", row["loss"],
            row["precision"], f"{row['precision_pct']:.2f}",
            row["recall"], f"{row['recall_pct']:.2f}",
            row["f1_score"], f"{row['f1_pct']:.2f}",
        ]
        for i in range(num_clients):
            csv_row.append(row["client_accuracies"][i] if i < len(row["client_accuracies"]) else "")
            csv_row.append(f"{row['client_accuracies_pct'][i]:.2f}" if i < len(row["client_accuracies_pct"]) else "")
        csv_row.extend([
            row["total_samples"], row["actual_attack"], row["actual_normal"],
            row["predicted_attack"], row["predicted_normal"],
            row["true_positives"], row["true_negatives"], row["false_positives"], row["false_negatives"],
        ])
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(csv_row)

        # Console output: mean + per-device accuracy (decimal + percent)
        print("\n[Evaluation Summary]")
        print("=" * 60)
        print(f"Round: {current_round} / {num_rounds}")
        print(f"Accuracy (mean):  {aggregated['accuracy']:.4f}  ({accuracy_pct:.2f}%)")
        print(f"Loss:             {aggregated['loss']:.4f}")
        print("[Per-Device Accuracy]")
        for i in range(num_clients):
            if i < len(client_accuracies_pct):
                print(f"  - Device {i}:  {client_accuracies[i]:.4f}  ({client_accuracies_pct[i]:.2f}%)")
        print()

        if "actual_attack" in aggregated and "actual_normal" in aggregated and "total_samples" in aggregated:
            print("[Ground Truth]")
            print(f"  - Attack samples: {aggregated['actual_attack']}")
            print(f"  - Normal samples: {aggregated['actual_normal']}")
            print(f"  - Total samples:  {aggregated['total_samples']}\n")

        if "predicted_attack" in aggregated and "predicted_normal" in aggregated:
            print("[Predictions]")
            print(f"  - Predicted Attack:  {aggregated['predicted_attack']}")
            print(f"  - Predicted Normal:  {aggregated['predicted_normal']}\n")

        if {"true_positives", "true_negatives", "false_positives", "false_negatives"} <= aggregated.keys():
            print("[Confusion Matrix]")
            print(f"  - True Positives (TP):  {aggregated['true_positives']}")
            print(f"  - True Negatives (TN):  {aggregated['true_negatives']}")
            print(f"  - False Positives (FP): {aggregated['false_positives']}")
            print(f"  - False Negatives (FN): {aggregated['false_negatives']}\n")

        print("[Metrics]")
        print(f"  - Precision:  {aggregated['precision']:.4f}  ({precision_pct:.2f}%)")
        print(f"  - Recall:     {aggregated['recall']:.4f}  ({recall_pct:.2f}%)")
        print(f"  - F1-Score:   {aggregated['f1_score']:.4f}  ({f1_pct:.2f}%)")

        print("=" * 60 + "\n")

        return aggregated

    strategy_kw = dict(
        fraction_fit=fed_cfg.get("fraction_fit", 1.0),
        fraction_evaluate=fed_cfg.get("fraction_evaluate", 1.0),
        min_fit_clients=fed_cfg.get("min_fit_clients", state["num_clients"]),
        min_evaluate_clients=fed_cfg.get("min_evaluate_clients", state["num_clients"]),
        min_available_clients=fed_cfg.get("min_available_clients", state["num_clients"]),
        on_fit_config_fn=lambda rnd: {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "use_callbacks": fed_cfg.get("use_callbacks", False),
            "server_round": rnd,
            "learning_rate": state["learning_rate"],
            "lr_decay": fed_cfg.get("lr_decay", 1.0),
        },
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
    )
    if hasattr(fl.server.strategy, "FedAvgM"):
        strategy_kw["server_momentum"] = fed_cfg.get("server_momentum", 0.9)
        strategy_kw["server_learning_rate"] = fed_cfg.get("server_learning_rate", 1.0)
        init_model = _build_model(
            state["model_name"],
            state["input_shape"],
            state["num_classes"],
            state["learning_rate"],
            state.get("use_focal_loss", False),
            state.get("focal_loss_alpha", 0.75),
        )
        strategy_kw["initial_parameters"] = ndarrays_to_parameters(init_model.get_weights())
        print(f"[Strategy] FedAvgM (momentum={strategy_kw['server_momentum']})")
    strategy = SaveModelStrategy(**strategy_kw)

    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=state["num_clients"],
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception as ex:
        import traceback
        project_root = Path(__file__).resolve().parent.parent.parent
        err_log = project_root / "data" / "processed" / "simulation_crash.log"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        with open(err_log, "w", encoding="utf-8") as f:
            f.write("=== Simulation Crash ===\n\n")
            traceback.print_exc(file=f)
            if hasattr(ex, "__cause__") and ex.__cause__ is not None:
                f.write("\n--- Caused by ---\n\n")
                traceback.print_exception(type(ex.__cause__), ex.__cause__, getattr(ex.__cause__, "__traceback__", None), file=f)
                f.write(str(ex.__cause__) + "\n")
        print("\n" + "=" * 60)
        print("  Simulation Crash - Full traceback saved to:")
        print(f"  {err_log}")
        print("=" * 60)
        traceback.print_exc()
        if hasattr(ex, "__cause__") and ex.__cause__ is not None:
            print("\n  Caused by:", ex.__cause__)
            traceback.print_exception(type(ex.__cause__), ex.__cause__, getattr(ex.__cause__, "__traceback__", None))
        raise

    # Per-round metrics JSON (per-device included, for reports and graphs)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "num_clients": num_clients,
            "num_rounds": num_rounds,
            "rounds": metrics_history,
        }, f, indent=2, ensure_ascii=False)
    print(f"[Report] Round-by-round metrics saved:")
    print(f"  - CSV:  {csv_path.resolve()}  (round, accuracy, device 0~{num_clients-1} accuracy_pct, ...)")
    print(f"  - JSON: {json_path.resolve()}  (same + per-device for scripts / graphs)")
    print("  → Use these to find which round or which device accuracy drops.\n")

    # Generate report (Markdown) and graph
    import subprocess
    import sys
    project_root = Path(__file__).resolve().parent.parent.parent
    try:
        subprocess.run(
            [sys.executable, "scripts/generate_fl_report.py", "--input", str(json_path), "--output-dir", str(output_dir)],
            cwd=project_root,
            check=False,
            timeout=15,
        )
    except Exception as e:
        print(f"[Report] Markdown: run manually:  python scripts/generate_fl_report.py  ({e})")
    try:
        subprocess.run(
            [sys.executable, "scripts/visualize_fl_history.py", "--input", str(json_path), "--output-dir", str(output_dir)],
            cwd=project_root,
            check=False,
            timeout=30,
        )
    except Exception as e:
        print(f"[Report] Graph: run manually:  python scripts/visualize_fl_history.py  ({e})\n")

    # Save global model
    # Use weights from last round
    # Assume start_simulation has filled strategy.parameters with final values
    global_model = _build_model(
        state["model_name"],
        state["input_shape"],
        state["num_classes"],
        state["learning_rate"],
        state.get("use_focal_loss", False),
        state.get("focal_loss_alpha", 0.75),
    )
    if getattr(strategy, "latest_parameters", None) is not None:
        weights = parameters_to_ndarrays(strategy.latest_parameters)
        global_model.set_weights(weights)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        global_model.save(save_path)
        print(f"[SUCCESS] Saved global model to {save_path}")
    else:
        print("[WARNING] Could not find saveable parameters in Federated strategy.")
        raise RuntimeError("Failed to save model: No parameters available in strategy.")


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

