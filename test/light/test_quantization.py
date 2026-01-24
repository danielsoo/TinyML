"""
test_quantization.py — TFLite Quantization for Lightweight Path (No Distillation)

Loads the pruned lightweight model from test_pruning.py,
exports to TFLite with INT8 quantization.

Pipeline: Pruned model → TFLite (float) → TFLite (INT8 quantized)

Usage:
    python test/light/test_quantization.py
    python test/light/test_quantization.py --pruning-results models/cicids/pruning_l_lightweight_results.json
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
from src.tinyml.export_tflite import export_tflite


def load_pruning_results(results_path: str) -> dict:
    """Load pruning results JSON."""
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


def evaluate_tflite(tflite_path: str, x_test, y_test):
    """Evaluate a TFLite model and return metrics."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]['dtype']

    y_preds = []
    for i in range(len(x_test)):
        sample = x_test[i:i+1].astype(np.float32)

        # Handle quantized input
        if input_dtype == np.int8 or input_dtype == np.uint8:
            input_scale = input_details[0]['quantization_parameters']['scales'][0]
            input_zero_point = input_details[0]['quantization_parameters']['zero_points'][0]
            sample = (sample / input_scale + input_zero_point).astype(input_dtype)

        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        # Handle quantized output
        output_dtype = output_details[0]['dtype']
        if output_dtype == np.int8 or output_dtype == np.uint8:
            output_scale = output_details[0]['quantization_parameters']['scales'][0]
            output_zero_point = output_details[0]['quantization_parameters']['zero_points'][0]
            output = (output.astype(np.float32) - output_zero_point) * output_scale

        y_preds.append(output[0])

    y_prob = np.array(y_preds)
    y_pred = np.argmax(y_prob, axis=1) if y_prob.shape[1] > 1 else (y_prob.ravel() >= 0.5).astype(int)
    y_true = y_test.astype(int)

    accuracy = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def print_metrics(metrics: dict, name: str):
    print(f"\n  {name}:")
    print(f"    Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    if 'loss' in metrics:
        print(f"    Loss:      {metrics['loss']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1-Score:  {metrics['f1_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Quantization — Lightweight (no distillation)")
    parser.add_argument('--pruning-results', type=str, default=None,
                        help='Path to pruning results JSON (auto-detected from train_results)')
    parser.add_argument('--config', type=str, default='config/federated_cicids.yaml')
    parser.add_argument('--output-dir', type=str, default='models/cicids/')
    parser.add_argument('--eval-samples', type=int, default=2000,
                        help='Number of test samples for TFLite evaluation')
    args = parser.parse_args()

    # Auto-detect pruning results from train_results
    if args.pruning_results is None:
        # Try to read model_name from train_results
        train_results_path = Path(args.output_dir) / "train_results.json"
        if train_results_path.exists():
            with open(train_results_path) as f:
                tr = json.load(f)
            model_name = tr.get("model_name", "lightweight")
        else:
            model_name = "lightweight"
        args.pruning_results = f'models/cicids/pruning_l_{model_name}_results.json'

    print("\n" + "=" * 70)
    print("TEST_QUANTIZATION (Light): TFLite Export + INT8 Quantization")
    print("  Light path: FL(lightweight) → Prune → Quantize")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load pruning results ---
    pruning_results = load_pruning_results(args.pruning_results)
    pruned_path = pruning_results["pruned_model_path"]
    model_name = pruning_results.get("model_name", "lightweight")
    num_classes = pruning_results["num_classes"]
    input_shape = tuple(pruning_results["input_shape"])

    print(f"\n  Pruned model: {pruned_path}")
    print(f"  Architecture: {model_name}")
    print(f"  Input shape: {input_shape}, Classes: {num_classes}")
    print(f"  Post-finetune accuracy: {pruning_results['post_finetune_metrics']['accuracy']:.4f}")

    # --- Load data ---
    print("\n" + "=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    x_train, y_train, x_test, y_test, num_classes = load_data_from_config(args.config)
    print(f"  Train: {len(x_train):,}, Test: {len(x_test):,}, Features: {x_train.shape[1]}")

    # --- Load pruned model ---
    print("\n" + "=" * 70)
    print("STEP 2: LOAD PRUNED MODEL")
    print("=" * 70)

    model = keras.models.load_model(pruned_path)
    print(f"  Params: {model.count_params():,}")
    print(f"  Size (float32): {model.count_params() * 4 / 1024:.2f} KB")

    # --- Export TFLite (float32) ---
    print("\n" + "=" * 70)
    print("STEP 3: TFLITE EXPORT (float32)")
    print("=" * 70)

    float_path = str(output_dir / f"{model_name}_float.tflite")
    float_size_bytes = export_tflite(
        model=model,
        out_path=float_path,
        quantize=False
    )
    float_size_kb = float_size_bytes / 1024
    print(f"  Float TFLite: {float_path}")
    print(f"  Size: {float_size_kb:.2f} KB")

    # --- Export TFLite (INT8 quantized) ---
    print("\n" + "=" * 70)
    print("STEP 4: TFLITE EXPORT (INT8 quantized)")
    print("=" * 70)

    representative_data = x_train[:1000]
    quant_path = str(output_dir / f"{model_name}_int8.tflite")
    quant_size_bytes = export_tflite(
        model=model,
        out_path=quant_path,
        quantize=True,
        representative_data=representative_data
    )
    quant_size_kb = quant_size_bytes / 1024
    print(f"  Quantized TFLite: {quant_path}")
    print(f"  Size: {quant_size_kb:.2f} KB")
    print(f"  Compression: {float_size_kb / quant_size_kb:.1f}x vs float TFLite")

    # --- Evaluate TFLite models ---
    print("\n" + "=" * 70)
    print("STEP 5: TFLITE EVALUATION")
    print("=" * 70)

    eval_samples = min(args.eval_samples, len(x_test))
    x_eval = x_test[:eval_samples]
    y_eval = y_test[:eval_samples]
    print(f"  Evaluating on {eval_samples} samples...")

    float_metrics = evaluate_tflite(float_path, x_eval, y_eval)
    print_metrics(float_metrics, "TFLite (float32)")

    quant_metrics = evaluate_tflite(quant_path, x_eval, y_eval)
    print_metrics(quant_metrics, "TFLite (INT8)")

    # --- Save results ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "method": f"quantization_lightweight_{model_name}",
        "model_name": model_name,
        "pruned_model_path": pruned_path,
        "tflite_float": {
            "path": float_path,
            "size_kb": float_size_kb,
            "metrics": float_metrics,
        },
        "tflite_int8": {
            "path": quant_path,
            "size_kb": quant_size_kb,
            "metrics": quant_metrics,
        },
        "input_shape": list(input_shape),
        "num_classes": num_classes,
        "keras_params": int(model.count_params()),
    }

    results_path = str(output_dir / f"quantization_l_{model_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Final table
    print(f"\n  {'Format':<25} {'Size(KB)':>10} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10}")
    print(f"  {'Keras (float32)':<25} {model.count_params()*4/1024:>10.2f} {pruning_results['post_finetune_metrics']['accuracy']:>10.4f} {pruning_results['post_finetune_metrics']['f1_score']:>10.4f}")
    print(f"  {'TFLite (float32)':<25} {float_size_kb:>10.2f} {float_metrics['accuracy']:>10.4f} {float_metrics['f1_score']:>10.4f}")
    print(f"  {'TFLite (INT8)':<25} {quant_size_kb:>10.2f} {quant_metrics['accuracy']:>10.4f} {quant_metrics['f1_score']:>10.4f}")

    print(f"\n  Results: {results_path}")
    print(f"  Deploy: {quant_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
