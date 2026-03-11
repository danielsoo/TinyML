"""
PGD-10 Adversarial Training (Per-Client, FedAvg Aggregation)

Loads a pre-trained FL model, partitions data identically to FL training,
performs PGD-10 adversarial training locally on each client with a configurable
clean/adversarial ratio, then aggregates weights via FedAvg and saves the
hardened model.

Usage:
    python -m src.adversarial.pgd_adversarial_training \
        --model models/global_model.h5 \
        --config config/federated_local_sly.yaml \
        --pgd-config config/fgsm.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# PGD Attack
# ---------------------------------------------------------------------------

def pgd_attack(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    steps: int = 10,
    loss_fn=None,
) -> np.ndarray:
    """
    Generate PGD adversarial examples (Projected Gradient Descent).

    Iterates FGSM steps with projection back into the epsilon-ball.
    Data may be StandardScaler'd (not bounded to [0,1]), so we only
    enforce the L-inf epsilon-ball around the original input.

    Args:
        model: Trained Keras model.
        x: Clean inputs (batch_size, features).
        y: True labels.
        epsilon: L-inf perturbation budget.
        alpha: Step size per iteration.
        steps: Number of PGD iterations (e.g. 10).
        loss_fn: Loss function (default: model's compiled loss).

    Returns:
        Adversarial examples with same shape as x.
    """
    if loss_fn is None:
        loss_fn = model.loss
    if isinstance(loss_fn, str):
        loss_fn = keras.losses.get(loss_fn)

    x_orig = tf.constant(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)

    # Random initialisation inside epsilon-ball
    x_adv = x_orig + tf.random.uniform(x_orig.shape, -epsilon, epsilon)

    for _ in range(steps):
        x_adv_var = tf.Variable(x_adv, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_adv_var)
            predictions = model(x_adv_var, training=False)
            # Align shapes for binary sigmoid output (batch,1) vs labels (batch,)
            y_for_loss = y_tensor
            if len(predictions.shape) == 2 and len(y_tensor.shape) == 1:
                y_for_loss = tf.reshape(y_tensor, [-1, 1])
            loss = loss_fn(y_for_loss, predictions)

        gradients = tape.gradient(loss, x_adv_var)
        grad_np = gradients.numpy()
        if not np.isfinite(grad_np).all():
            grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)

        # FGSM step
        x_adv = x_adv + alpha * tf.sign(tf.constant(grad_np, dtype=tf.float32))

        # Project back into epsilon-ball around original
        x_adv = tf.clip_by_value(x_adv, x_orig - epsilon, x_orig + epsilon)

    return x_adv.numpy()


# ---------------------------------------------------------------------------
# Class Weight Computation (mirrors client.py sqrt-smoothed inverse frequency)
# ---------------------------------------------------------------------------

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    Compute sqrt-smoothed inverse-frequency class weights.

    Same logic as KerasClient in src/federated/client.py.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    raw_weights = n_samples / (n_classes * counts)
    smoothed = np.sqrt(raw_weights)
    smoothed = smoothed / np.mean(smoothed)

    return {int(c): float(w) for c, w in zip(classes, smoothed)}


# ---------------------------------------------------------------------------
# Per-Client Adversarial Training
# ---------------------------------------------------------------------------

def adversarial_train_client(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    pgd_cfg: dict,
    client_id: int,
) -> Tuple[List[np.ndarray], float]:
    """
    Run PGD adversarial training for a single client.

    Each batch is split into clean and adversarial portions according to
    ``adv_ratio``. PGD examples are generated for the adversarial portion,
    then the combined batch is used for a single training step.

    Args:
        model: Keras model (weights already set to global model).
        x_train: Client's local training data.
        y_train: Client's local labels.
        pgd_cfg: PGD config dict (epsilon, alpha, steps, adv_ratio, etc.).
        client_id: Client identifier (for logging).

    Returns:
        Tuple of (trained weights, final batch loss).
    """
    epsilon = pgd_cfg.get("epsilon", 0.1)
    alpha = pgd_cfg.get("alpha", 0.01)
    steps = pgd_cfg.get("steps", 10)
    adv_ratio = pgd_cfg.get("adv_ratio", 0.5)
    epochs = pgd_cfg.get("epochs", 5)
    batch_size = pgd_cfg.get("batch_size", 128)

    class_weights = compute_class_weights(y_train)
    n_samples = len(x_train)
    loss_fn = model.loss

    last_loss = 0.0

    for epoch in range(epochs):
        # Shuffle each epoch
        perm = np.random.permutation(n_samples)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]

        n_batches = (n_samples + batch_size - 1) // batch_size
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_samples)
            x_batch = x_shuf[start:end]
            y_batch = y_shuf[start:end]

            # Split batch into clean / adversarial portions
            n_adv = max(1, int(len(x_batch) * adv_ratio))
            n_clean = len(x_batch) - n_adv

            x_clean = x_batch[:n_clean]
            y_clean = y_batch[:n_clean]
            x_to_attack = x_batch[n_clean:]
            y_to_attack = y_batch[n_clean:]

            # Generate PGD adversarial examples
            x_adv = pgd_attack(
                model, x_to_attack, y_to_attack,
                epsilon=epsilon, alpha=alpha, steps=steps,
                loss_fn=loss_fn,
            )

            # Combine clean + adversarial
            if n_clean > 0:
                x_combined = np.concatenate([x_clean, x_adv], axis=0)
                y_combined = np.concatenate([y_clean, y_to_attack], axis=0)
            else:
                x_combined = x_adv
                y_combined = y_to_attack

            batch_loss = model.train_on_batch(
                x_combined, y_combined, class_weight=class_weights,
            )
            # train_on_batch returns loss or [loss, metric, ...]
            if isinstance(batch_loss, (list, tuple)):
                batch_loss = batch_loss[0]
            epoch_loss += float(batch_loss)

        avg_loss = epoch_loss / max(n_batches, 1)
        last_loss = avg_loss
        print(
            f"  [Client {client_id}] Epoch {epoch + 1}/{epochs} "
            f"- loss: {avg_loss:.4f}"
        )

    return model.get_weights(), last_loss


# ---------------------------------------------------------------------------
# FedAvg Weight Aggregation
# ---------------------------------------------------------------------------

def aggregate_weights(
    client_weights: List[List[np.ndarray]],
    sample_counts: List[int],
) -> List[np.ndarray]:
    """Sample-weighted FedAvg aggregation."""
    total = sum(sample_counts)
    fractions = [n / total for n in sample_counts]

    averaged = []
    num_layers = len(client_weights[0])
    for i in range(num_layers):
        layer_avg = sum(
            w[i] * frac for w, frac in zip(client_weights, fractions)
        )
        averaged.append(layer_avg)

    return averaged


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model_for_at(
    model_path: str,
    fed_cfg: dict,
) -> keras.Model:
    """
    Load a saved .h5 model and recompile with the correct loss.

    Handles focal loss deserialization (custom object) by recompiling
    after loading with ``compile=False``.
    """
    from src.models.nets import _focal_loss

    model = keras.models.load_model(model_path, compile=False)

    use_focal = fed_cfg.get("use_focal_loss", False)
    focal_alpha = fed_cfg.get("focal_loss_alpha", 0.25)

    if use_focal:
        loss = _focal_loss(gamma=2.0, alpha=focal_alpha)
    else:
        loss = "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def run_pgd_adversarial_training(
    model_path: str,
    config_path: str,
    pgd_config_path: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    End-to-end PGD-10 adversarial training with per-client FedAvg.

    Args:
        model_path: Path to trained FL model (.h5).
        config_path: FL training config (data, federated settings).
        pgd_config_path: PGD training config (fgsm.yaml → pgd_training section).
        output_path: Where to save the hardened model.

    Returns:
        Summary dict with per-client stats and output paths.
    """
    import yaml
    from src.data.loader import load_dataset, partition_non_iid

    # --- Load configs ---
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(pgd_config_path, encoding="utf-8") as f:
        pgd_full_cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})
    pgd_cfg = pgd_full_cfg.get("pgd_training", {})

    if not pgd_cfg.get("enabled", True):
        print("PGD adversarial training is disabled in config.")
        return {}

    num_clients = int(data_cfg.get("num_clients", 4))

    if output_path is None:
        base = Path(model_path)
        output_path = str(base.parent / f"{base.stem}_pgd_at{base.suffix}")

    print("\n" + "=" * 70)
    print("  PGD-10 Adversarial Training (Per-Client)")
    print("=" * 70)
    print(f"  Model:       {model_path}")
    print(f"  Clients:     {num_clients}")
    print(f"  Epsilon:     {pgd_cfg.get('epsilon', 0.1)}")
    print(f"  Alpha:       {pgd_cfg.get('alpha', 0.01)}")
    print(f"  Steps:       {pgd_cfg.get('steps', 10)}")
    print(f"  Adv ratio:   {pgd_cfg.get('adv_ratio', 0.5)}")
    print(f"  Epochs:      {pgd_cfg.get('epochs', 5)}")
    print(f"  Batch size:  {pgd_cfg.get('batch_size', 128)}")
    print(f"  LR:          {pgd_cfg.get('learning_rate', 0.0001)}")
    print("=" * 70 + "\n")

    # --- Load data (same preprocessing as FL) ---
    print("Loading dataset...")
    dataset_name = data_cfg.get("name", "cicids2017")
    dataset_kwargs = {
        k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}
    }
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    print(f"  Train: {x_train.shape}, Test: {x_test.shape}\n")

    # --- Partition (same as FL, deterministic seed) ---
    print("Partitioning data across clients...")
    train_parts = partition_non_iid(x_train, y_train, num_clients)
    for cid in range(num_clients):
        yc = train_parts[cid]["y"]
        unique, counts = np.unique(yc, return_counts=True)
        dist = ", ".join(f"class {c}: {n}" for c, n in zip(unique, counts))
        print(f"  Client {cid}: {len(yc)} samples ({dist})")
    print()

    # --- Load model ---
    print("Loading model...")
    model = load_model_for_at(model_path, fed_cfg)

    # Override optimizer LR with PGD config
    pgd_lr = pgd_cfg.get("learning_rate", 0.0001)
    model.optimizer.learning_rate.assign(pgd_lr)

    global_weights = model.get_weights()
    print(f"  Parameters: {model.count_params():,}\n")

    # --- Per-client adversarial training ---
    client_weights_list = []
    sample_counts = []
    client_reports = []

    for cid in range(num_clients):
        x_c = train_parts[cid]["x"]
        y_c = train_parts[cid]["y"]

        print(f"--- Client {cid} ({len(y_c)} samples) ---")

        # Reset to global weights for each client
        model.set_weights(global_weights)
        model.optimizer.learning_rate.assign(pgd_lr)

        weights, final_loss = adversarial_train_client(
            model, x_c, y_c, pgd_cfg, cid,
        )

        client_weights_list.append(weights)
        sample_counts.append(len(y_c))

        unique, counts = np.unique(y_c, return_counts=True)
        client_reports.append({
            "client_id": cid,
            "num_samples": int(len(y_c)),
            "class_distribution": {str(c): int(n) for c, n in zip(unique, counts)},
            "final_loss": float(final_loss),
        })
        print()

    # --- Aggregate weights ---
    print("Aggregating client weights (FedAvg)...")
    avg_weights = aggregate_weights(client_weights_list, sample_counts)
    model.set_weights(avg_weights)

    # --- Evaluate on test set ---
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Aggregated model - Accuracy: {test_acc:.2%}, Loss: {test_loss:.4f}\n")

    # --- Save ---
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"  Saved hardened model: {output_path}")

    # --- Report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_input": model_path,
        "model_output": output_path,
        "pgd_config": {k: v for k, v in pgd_cfg.items()},
        "num_clients": num_clients,
        "clients": client_reports,
        "aggregation": "fedavg_sample_weighted",
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
    }

    report_path = str(Path(output_path).parent / "pgd_at_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved report: {report_path}")

    print("\n" + "=" * 70)
    print("  PGD-10 Adversarial Training Complete")
    print("=" * 70 + "\n")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGD-10 Adversarial Training (per-client, FedAvg aggregation)",
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to trained FL model (.h5)",
    )
    parser.add_argument(
        "--config", default="config/federated_local.yaml",
        help="FL config (data, federated settings)",
    )
    parser.add_argument(
        "--pgd-config", default="config/fgsm.yaml",
        help="PGD training config (reads pgd_training section)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output model path (default: <model>_pgd_at.h5)",
    )
    args = parser.parse_args()

    run_pgd_adversarial_training(
        model_path=args.model,
        config_path=args.config,
        pgd_config_path=args.pgd_config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
