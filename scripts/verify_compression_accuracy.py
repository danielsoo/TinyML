#!/usr/bin/env python3
"""
Verify why Original TFLite shows lower accuracy than Compressed.
Compare: Keras .h5 vs TFLite Original vs TFLite Compressed on same test set.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import yaml
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_dataset

def load_config():
    cfg_path = project_root / "config" / "federated.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def evaluate_keras(model_path: str, x_test, y_test, threshold: float = 0.5):
    model = tf.keras.models.load_model(model_path, compile=False)
    y_pred_proba = model.predict(x_test, verbose=0)
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 0]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)

def evaluate_tflite(model_path: str, x_test, y_test, threshold: float = 0.5):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]["dtype"]
    
    y_pred_list = []
    for i in range(len(x_test)):
        batch_x = x_test[i:i+1]
        if input_dtype == np.int8:
            s = input_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
            z = input_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
            batch_x = (batch_x / s + z).astype(np.int8)
        else:
            batch_x = batch_x.astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], batch_x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        if output_details[0]["dtype"] == np.int8:
            s = output_details[0].get("quantization_parameters", {}).get("scales", [1.0])[0]
            z = output_details[0].get("quantization_parameters", {}).get("zero_points", [0])[0]
            output = s * (output.astype(np.float32) - z)
        y_pred_list.append(output)
    
    y_pred_proba = np.concatenate(y_pred_list, axis=0)
    if y_pred_proba.ndim > 1:
        y_pred_proba = y_pred_proba[:, 0]
    y_pred = (y_pred_proba >= threshold).astype(int)
    return accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average="binary", pos_label=1, zero_division=0)

def main():
    cfg = load_config()
    eval_cfg = cfg.get("evaluation", {})
    threshold = float(eval_cfg.get("prediction_threshold", 0.5))
    data_cfg = cfg["data"]
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")
    
    print("Loading test data (cicids2017)...")
    _, _, x_test, y_test = load_dataset(data_cfg["name"], **dataset_kwargs)
    print(f"Test samples: {len(x_test)}")
    print(f"Prediction threshold: {threshold} (from config evaluation.prediction_threshold)\n")
    
    results = []
    
    # 1. Keras global_model.h5 (if exists)
    keras_path = project_root / "models" / "global_model.h5"
    if keras_path.exists():
        acc, f1 = evaluate_keras(str(keras_path), x_test, y_test, threshold)
        results.append(("Keras (global_model.h5)", acc, f1))
    else:
        keras_path = project_root / "src" / "models" / "global_model.h5"
        if keras_path.exists():
            acc, f1 = evaluate_keras(str(keras_path), x_test, y_test, threshold)
            results.append(("Keras (src/models/global_model.h5)", acc, f1))
    
    # 2. TFLite Original
    orig_path = project_root / "models" / "tflite" / "saved_model_original.tflite"
    if orig_path.exists():
        acc, f1 = evaluate_tflite(orig_path, x_test, y_test, threshold)
        results.append(("TFLite Original", acc, f1))
    
    # 3. TFLite Compressed
    comp_path = project_root / "models" / "tflite" / "saved_model_pruned_quantized.tflite"
    if comp_path.exists():
        acc, f1 = evaluate_tflite(comp_path, x_test, y_test, threshold)
        results.append(("TFLite Compressed (PTQ)", acc, f1))
    
    print("=" * 60)
    print("ACCURACY VERIFICATION (same test set)")
    print("=" * 60)
    for name, acc, f1 in results:
        print(f"{name:<40} Acc: {acc:.4f}  F1: {f1:.4f}")
    print("=" * 60)
    
    if len(results) >= 2:
        keras_acc = results[0][1] if "Keras" in results[0][0] else None
        orig_acc = next((r[1] for r in results if "Original" in r[0]), None)
        comp_acc = next((r[1] for r in results if "Compressed" in r[0]), None)
        if keras_acc is not None and orig_acc is not None:
            diff = orig_acc - keras_acc
            print(f"\n⚠️  TFLite Original vs Keras: {diff:+.4f} ({'TFLite worse' if diff < 0 else 'TFLite same/better'})")
        if orig_acc is not None and comp_acc is not None:
            print(f"⚠️  Compressed vs Original: {comp_acc - orig_acc:+.4f} (Compressed {'better' if comp_acc > orig_acc else 'worse'})")

if __name__ == "__main__":
    main()
