"""
Windows-compatible training script (no Ray dependency).
Trains a centralized model on Bot-IoT data.
"""
import yaml
from pathlib import Path
import numpy as np
from src.data.loader import load_dataset
from src.models.nets import get_model

def train():
    # Load config
    with open("config/federated.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fed_cfg = cfg.get("federated", {})

    # Load dataset
    dataset_name = data_cfg.get("name", "bot_iot")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    print("📂 Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Get data info
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

    if x_train.ndim == 2:
        input_shape = (x_train.shape[1],)
    else:
        input_shape = x_train.shape[1:]

    print(f"📊 Dataset loaded:")
    print(f"  - Training samples: {len(x_train)}")
    print(f"  - Test samples: {len(x_test)}")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Attack samples (train): {np.sum(y_train == 1)}")
    print(f"  - Normal samples (train): {np.sum(y_train == 0)}\n")

    # Build model
    model_name = model_cfg.get("name", "mlp")
    print(f"🏗️  Building {model_name.upper()} model...")
    model = get_model(model_name, input_shape, num_classes)

    # Training parameters
    epochs = fed_cfg.get("local_epochs", 2)
    batch_size = fed_cfg.get("batch_size", 128)

    print(f"🚀 Starting training...")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}\n")

    # Train model
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("📊 Final Evaluation")
    print("="*60)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred_prob = model.predict(x_test, verbose=0)

    # Binary classification predictions
    if num_classes <= 2:
        if y_pred_prob.ndim > 1:
            y_pred_prob = y_pred_prob.ravel()
        y_pred = (y_pred_prob >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    y_true = y_test.astype(int)

    # Calculate metrics
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Loss: {loss:.4f}\n")

    print("📈 Ground Truth:")
    print(f"  - Attack samples: {np.sum(y_true == 1)}")
    print(f"  - Normal samples: {np.sum(y_true == 0)}")
    print(f"  - Total samples: {len(y_true)}\n")

    print("🔮 Predictions:")
    print(f"  - Predicted Attack: {np.sum(y_pred == 1)}")
    print(f"  - Predicted Normal: {np.sum(y_pred == 0)}\n")

    print("✅ Confusion Matrix:")
    print(f"  - True Positives (TP): {tp}")
    print(f"  - True Negatives (TN): {tn}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}\n")

    print("📏 Metrics:")
    print(f"  - Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  - Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"  - F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print("="*60 + "\n")

    # Save model
    output_path = Path("models/global_model.h5")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_path)
    print(f"✅ Model saved to {output_path}")

    # Also save as TFLite
    try:
        import tensorflow as tf
        tflite_path = Path("models/global_model.tflite")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f"✅ TFLite model saved to {tflite_path}")
        print(f"   Model size: {len(tflite_model) / 1024:.2f} KB")
    except Exception as e:
        print(f"⚠️  Could not convert to TFLite: {e}")

if __name__ == "__main__":
    train()
