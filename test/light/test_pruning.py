"""
test_pruning.py — Structured Pruning for Lightweight FL Model (No Distillation)

The lightweight path skips distillation because the FL model is already small.
This script loads the FL-trained lightweight model directly and prunes it.

Pipeline: FL(lightweight) → Prune → Fine-tune → Save for quantization

Shows metrics before/after fine-tuning. Saves pruned model for quantization.

Usage:
    python test/light/test_pruning.py
    python test/light/test_pruning.py --train-results models/cicids/train_results.json
    python test/light/test_pruning.py --pruning-ratio 0.3
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Project root (two levels up from test/light/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras

from src.data.loader import load_dataset
from src.compression.pruning import apply_structured_pruning, fine_tune_pruned_model


def load_train_results(results_path: str) -> dict:
    """Load training results from test_train.py."""
    with open(results_path, 'r') as f:
        return json.load(f)


def load_data_from_config(config_path: str):
    """Load dataset using config file."""
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("name", "cicids2017")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = max(2, len(unique_labels))

    if np.min(unique_labels) != 0 or np.max(unique_labels) != num_classes - 1:
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])

    return x_train, y_train, x_test, y_test, num_classes


def evaluate_model(model, x_test, y_test):
    """Evaluate model and return metrics dict."""
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1) if y_prob.shape[1] > 1 else (y_prob.ravel() >= 0.5).astype(int)
    y_true = y_test.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "params": int(model.count_params()),
    }


def print_metrics(metrics: dict, name: str):
    print(f"\n  {name}:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"    Loss:      {metrics['loss']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1_score']:.4f}")
    print(f"    Params:    {metrics['params']:,}")


def main():
    parser = argparse.ArgumentParser(description="Pruning — Lightweight FL Model (no distillation)")
    parser.add_argument('--train-results', type=str, default='models/cicids/train_results.json',
                        help='Path to train_results.json from test_train.py (lightweight FL model)')
    parser.add_argument('--config', type=str, default='config/federated_cicids.yaml')
    parser.add_argument('--output-dir', type=str, default='models/cicids/')
    parser.add_argument('--pruning-ratio', type=float, default=0.5,
                        help='Fraction of neurons to remove (0.0-1.0)')
    parser.add_argument('--fine-tune-epochs', type=int, default=5)
    parser.add_argument('--fine-tune-lr', type=float, default=0.0001)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TEST_PRUNING (Light): FL(lightweight) → Prune → Fine-tune")
    print("  No distillation — model is already small")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load train results ---
    train_results = load_train_results(args.train_results)
    model_path = train_results["model_path"]
    model_name = train_results["model_name"]
    num_classes = train_results["num_classes"]
    input_shape = tuple(train_results["input_shape"])

    print(f"\n  FL model: {model_path}")
    print(f"  Architecture: {model_name}")
    print(f"  Params: {train_results['model_params']:,}")
    print(f"  FL accuracy: {train_results['final_metrics']['accuracy']:.4f}")
    print(f"  Input shape: {input_shape}, Classes: {num_classes}")
    print(f"  Pruning ratio: {args.pruning_ratio}")

    # --- Load data ---
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    x_train, y_train, x_test, y_test, num_classes = load_data_from_config(args.config)
    print(f"  Train: {len(x_train):,}, Test: {len(x_test):,}, Features: {x_train.shape[1]}")

    val_split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:val_split], y_train[:val_split]
    x_val, y_val = x_train[val_split:], y_train[val_split:]

    # --- Load FL model directly (no distillation needed) ---
    print("\n" + "=" * 70)
    print("STEP 2: LOAD FL-TRAINED LIGHTWEIGHT MODEL")
    print("=" * 70)

    model = keras.models.load_model(model_path)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    pre_prune_metrics = evaluate_model(model, x_test, y_test)
    print_metrics(pre_prune_metrics, f"FL Model ({model_name}) BEFORE Pruning")

    # --- Pruning ---
    print("\n" + "=" * 70)
    print("STEP 3: STRUCTURED PRUNING")
    print("=" * 70)

    pruned_model = apply_structured_pruning(
        model=model, pruning_ratio=args.pruning_ratio,
        skip_last_layer=True, verbose=True
    )

    post_prune_metrics = evaluate_model(pruned_model, x_test, y_test)
    print_metrics(post_prune_metrics, f"Model AFTER Pruning (before fine-tune)")

    # --- Fine-tuning ---
    print("\n" + "=" * 70)
    print("STEP 4: FINE-TUNING")
    print("=" * 70)

    print(f"  Epochs: {args.fine_tune_epochs}")
    print(f"  Learning rate: {args.fine_tune_lr}")

    batch_size = train_results.get("federated", {}).get("batch_size", 128)

    pruned_model = fine_tune_pruned_model(
        pruned_model=pruned_model,
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        epochs=args.fine_tune_epochs,
        batch_size=batch_size,
        learning_rate=args.fine_tune_lr,
        verbose=True
    )

    post_finetune_metrics = evaluate_model(pruned_model, x_test, y_test)
    print_metrics(post_finetune_metrics, f"Model AFTER Fine-tuning")

    # --- Save pruned model ---
    pruned_path = str(output_dir / f"pruned_{model_name}.h5")
    pruned_model.save(pruned_path)
    print(f"\n  Pruned model saved: {pruned_path}")

    # --- Save results ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": f"pruning_lightweight_{model_name}",
        "model_name": model_name,
        "pruned_model_path": pruned_path,
        "config": {
            "pruning_ratio": args.pruning_ratio,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_lr": args.fine_tune_lr,
            "batch_size": batch_size,
        },
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "pre_prune_metrics": pre_prune_metrics,
        "post_prune_metrics": post_prune_metrics,
        "post_finetune_metrics": post_finetune_metrics,
    }

    results_path = str(output_dir / f"pruning_l_{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Comparison table
    print(f"\n  {'Stage':<35} {'Params':>10} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'─'*35} {'─'*10} {'─'*10} {'─'*10}")
    rows = [
        ("FL model (before pruning)", pre_prune_metrics),
        ("After pruning (no fine-tune)", post_prune_metrics),
        ("After fine-tuning", post_finetune_metrics),
    ]
    for name, m in rows:
        print(f"  {name:<35} {m['params']:>10,} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f}")

    print(f"\n  Pruned model: {pruned_path}")
    print(f"  Results: {results_path}")
    print(f"\n  Next step: use pruned model in quantization")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
