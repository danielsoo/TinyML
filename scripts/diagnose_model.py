#!/usr/bin/env python3
"""
Diagnose why the model might be collapsed (predicting all normal).
Checks: Keras model predictions, TFLite predictions, data balance per client.
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/federated_scratch.yaml")
    parser.add_argument("--model", default="models/global_model.h5")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DIAGNOSTIC: Model collapse check")
    print("=" * 60)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    from src.data.loader import load_dataset, partition_non_iid

    print("\n1. Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(
        data_cfg["name"], **dataset_kwargs
    )
    print(f"   Train: {len(y_train):,} (BENIGN={np.sum(y_train==0):,}, ATTACK={np.sum(y_train==1):,})")
    print(f"   Test:  {len(y_test):,} (BENIGN={np.sum(y_test==0):,}, ATTACK={np.sum(y_test==1):,})")

    print("\n2. Client partition balance (train data)...")
    num_clients = int(data_cfg.get("num_clients", 4))
    parts = partition_non_iid(x_train, y_train, num_clients)
    for cid in range(num_clients):
        y_c = parts[cid]["y"]
        n0, n1 = int(np.sum(y_c == 0)), int(np.sum(y_c == 1))
        pct0 = 100 * n0 / len(y_c)
        print(f"   Client {cid}: BENIGN={n0:,} ({pct0:.1f}%), ATTACK={n1:,} ({100-pct0:.1f}%)")

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n3. Model not found: {model_path}")
        print("   (Run training first)")
        return

    print(f"\n3. Loading Keras model: {model_path}")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    print("\n4. Keras model predictions on test set...")
    y_prob = model.predict(x_test[:5000], verbose=0)  # Subset for speed
    if y_prob.ndim > 1:
        y_prob = y_prob[:, 0]
    y_pred = (y_prob >= 0.5).astype(int)
    n_pred_0 = int(np.sum(y_pred == 0))
    n_pred_1 = int(np.sum(y_pred == 1))
    print(f"   Predicted BENIGN: {n_pred_0:,} ({100*n_pred_0/len(y_pred):.1f}%)")
    print(f"   Predicted ATTACK: {n_pred_1:,} ({100*n_pred_1/len(y_pred):.1f}%)")
    print(f"   Prob range: min={y_prob.min():.4f}, max={y_prob.max():.4f}, mean={y_prob.mean():.4f}")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_te = y_test[:5000]
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    print(f"   Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    tflite_path = Path("models/tflite/saved_model_original.tflite")
    if tflite_path.exists():
        print(f"\n5. TFLite (original) predictions...")
        # XNNPACK delegate can cause NaN; disable via experimental_preserve_all_tensors
        interp = tf.lite.Interpreter(
            model_path=str(tflite_path),
            experimental_preserve_all_tensors=True,
        )
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]
        print(f"   TFLite input shape: {inp['shape']}, dtype: {inp['dtype']}")
        print(f"   TFLite output shape: {out['shape']}, dtype: {out['dtype']}")
        print(f"   Data shape: {x_test.shape}")
        preds = []
        tflite_probs = []
        for i in range(min(5000, len(x_test))):
            interp.set_tensor(inp["index"], x_test[i:i+1].astype(np.float32))
            interp.invoke()
            p = interp.get_tensor(out["index"])[0, 0]
            if hasattr(p, "__iter__") and len(np.array(p).shape) > 0:
                p = float(np.array(p).flatten()[0])
            else:
                p = float(p)
            tflite_probs.append(p)
            preds.append(1 if p >= 0.5 else 0)
        preds = np.array(preds)
        tflite_probs = np.array(tflite_probs)
        n0_t, n1_t = int(np.sum(preds == 0)), int(np.sum(preds == 1))
        print(f"   Predicted BENIGN: {n0_t:,}, ATTACK: {n1_t:,}")
        print(f"   TFLite prob range: min={tflite_probs.min():.4f}, max={tflite_probs.max():.4f}, mean={tflite_probs.mean():.4f}")
        # Compare first 5 samples: Keras vs TFLite
        print(f"   Sample comparison (first 5): Keras vs TFLite")
        for i in range(min(5, len(y_prob))):
            print(f"      [{i}] Keras={y_prob[i]:.4f} TFLite={tflite_probs[i]:.4f}")
        acc_t = accuracy_score(y_test[:len(preds)], preds)
        prec_t = precision_score(y_test[:len(preds)], preds, zero_division=0)
        rec_t = recall_score(y_test[:len(preds)], preds, zero_division=0)
        print(f"   Accuracy={acc_t:.4f}, Precision={prec_t:.4f}, Recall={rec_t:.4f}")
    else:
        print(f"\n5. TFLite not found: {tflite_path}")

    print("\n" + "=" * 60)
    if n_pred_1 == 0:
        print("  RESULT: Model is COLLAPSED (predicts all BENIGN)")
        print("  → Check: training config, aggregation, or run centralized to compare")
    else:
        print("  RESULT: Model predicts both classes (not collapsed)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
