"""
test_distillation.py — Knowledge Distillation using nets.py (Standard MLP Student)

Uses the trained FL model from test_train.py as teacher,
creates a compressed student via create_student_model (nets.py-style architecture).

Distillation ONLY — saves student model for later pruning step.

Usage:
    python test/normal/test_distillation.py
    python test/normal/test_distillation.py --train-results models/cicids/train_results.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# Project root (two levels up from test/normal/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tensorflow import keras

from src.data.loader import load_dataset
from src.compression.distillation import train_with_distillation, create_student_model


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
    parser = argparse.ArgumentParser(description="Distillation Only — Standard MLP Student (nets.py)")
    parser.add_argument('--train-results', type=str, default='models/cicids/train_results.json')
    parser.add_argument('--config', type=str, default='config/federated_cicids.yaml')
    parser.add_argument('--output-dir', type=str, default='models/cicids/')
    parser.add_argument('--compression-ratio', type=float, default=0.5,
                        help='Student size relative to teacher (0.5 = half)')
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TEST_DISTILLATION: Knowledge Distillation (Standard MLP Student)")
    print("  Student: nets.py (create_student_model)")
    print("  Mode: Distillation ONLY (no pruning)")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load train results ---
    train_results = load_train_results(args.train_results)
    model_path = train_results["model_path"]
    num_classes = train_results["num_classes"]
    input_shape = tuple(train_results["input_shape"])

    print(f"\n  Teacher model: {model_path}")
    print(f"  Teacher params: {train_results['model_params']:,}")
    print(f"  Teacher accuracy: {train_results['final_metrics']['accuracy']:.4f}")
    print(f"  Input shape: {input_shape}, Classes: {num_classes}")

    # --- Load data ---
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    x_train, y_train, x_test, y_test, num_classes = load_data_from_config(args.config)
    print(f"  Train: {len(x_train):,}, Test: {len(x_test):,}, Features: {x_train.shape[1]}")

    # --- Load teacher ---
    print("\n" + "=" * 70)
    print("STEP 2: LOAD TEACHER MODEL")
    print("=" * 70)

    teacher_model = keras.models.load_model(model_path)
    teacher_metrics = evaluate_model(teacher_model, x_test, y_test)
    print_metrics(teacher_metrics, "Teacher (FL-trained MLP)")

    # --- Distillation ---
    print("\n" + "=" * 70)
    print("STEP 3: KNOWLEDGE DISTILLATION")
    print("=" * 70)

    student_model = create_student_model(
        teacher_model, compression_ratio=args.compression_ratio, num_classes=num_classes
    )

    print(f"\n  Student architecture (nets.py style):")
    print(f"    Compression ratio: {args.compression_ratio}")
    print(f"    Student params: {student_model.count_params():,}")
    print(f"    Teacher params: {teacher_model.count_params():,}")
    print(f"    Reduction: {teacher_model.count_params() / student_model.count_params():.1f}x")

    val_split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:val_split], y_train[:val_split]
    x_val, y_val = x_train[val_split:], y_train[val_split:]

    batch_size = train_results.get("federated", {}).get("batch_size", 128)

    student_model, distill_history = train_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Compile for standalone evaluation
    student_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    student_metrics = evaluate_model(student_model, x_test, y_test)
    print_metrics(student_metrics, "Student after Distillation")

    # --- Save student model ---
    student_path = str(output_dir / "student_standard.h5")
    student_model.save(student_path)
    print(f"\n  Student model saved: {student_path}")

    # --- Save results ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": "distillation_standard_mlp",
        "student_type": "nets.py (create_student_model)",
        "student_model_path": student_path,
        "config": {
            "compression_ratio": args.compression_ratio,
            "temperature": args.temperature,
            "alpha": args.alpha,
            "epochs": args.epochs,
            "batch_size": batch_size,
        },
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "teacher_metrics": teacher_metrics,
        "student_metrics": student_metrics,
    }

    results_path = str(output_dir / "distillation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Comparison table
    print(f"\n  {'Model':<30} {'Params':>10} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10}")
    for name, m in [("Teacher (FL MLP)", teacher_metrics), ("Student (distilled)", student_metrics)]:
        print(f"  {name:<30} {m['params']:>10,} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f}")

    print(f"\n  Student model: {student_path}")
    print(f"  Results: {results_path}")
    print(f"\n  Next step: use student model in pruning")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
