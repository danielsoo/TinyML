"""
Standalone script to apply structured pruning to a trained model.
"""
import yaml
import argparse
from pathlib import Path
import numpy as np
from tensorflow import keras

from src.data.loader import load_dataset
from src.modelcompression.pruning import (
    apply_structured_pruning,
    compare_models,
    fine_tune_pruned_model,
    get_model_size
)


def safe_evaluate(model, x, y, verbose=0):
    """Safely evaluate model and return (loss, accuracy)."""
    result = model.evaluate(x, y, verbose=verbose)
    if isinstance(result, (list, tuple)):
        return result[0], result[1]
    else:
        return result, 0.0


def main():
    parser = argparse.ArgumentParser(description="Apply structured pruning to a trained model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/global_model.h5",
        help="Path to trained model (.h5 file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/pruned_model.h5",
        help="Path to save pruned model"
    )
    parser.add_argument(
        "--pruning-ratio",
        type=float,
        default=0.5,
        help="Fraction of neurons/filters to remove (0.0 to 1.0)"
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=5,
        help="Number of epochs to fine-tune after pruning"
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Skip fine-tuning after pruning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated.yaml",
        help="Path to config file"
    )

    args = parser.parse_args()

    # Load config
    print(f"\n{'='*60}")
    print(f"🔧 Structured Pruning Pipeline")
    print(f"{'='*60}\n")

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    fed_cfg = cfg.get("federated", {})

    # Load dataset for fine-tuning
    dataset_name = data_cfg.get("name", "bot_iot")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    print("📂 Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    print(f"  - Training samples: {len(x_train)}")
    print(f"  - Test samples: {len(x_test)}\n")

    # Load original model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print(f"   Please train a model first using train_windows.py")
        return

    print(f"📦 Loading model from {model_path}...")
    original_model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully\n")

    # Evaluate original model
    print("📊 Evaluating original model...")
    orig_loss, orig_acc = safe_evaluate(original_model, x_test, y_test, verbose=0)
    print(f"  - Accuracy: {orig_acc:.4f} ({orig_acc*100:.2f}%)")
    print(f"  - Loss: {orig_loss:.4f}\n")

    # Apply structured pruning
    pruned_model = apply_structured_pruning(
        original_model,
        pruning_ratio=args.pruning_ratio,
        skip_last_layer=True,
        verbose=True
    )

    # Compare model sizes
    compare_models(original_model, pruned_model)

    # Evaluate pruned model before fine-tuning
    print("📊 Evaluating pruned model (before fine-tuning)...")
    pruned_loss, pruned_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
    print(f"  - Accuracy: {pruned_acc:.4f} ({pruned_acc*100:.2f}%)")
    print(f"  - Loss: {pruned_loss:.4f}")
    print(f"  - Accuracy drop: {(orig_acc - pruned_acc)*100:.2f}%\n")

    # Fine-tune if requested
    if not args.no_finetune:
        batch_size = fed_cfg.get("batch_size", 128)
        pruned_model = fine_tune_pruned_model(
            pruned_model,
            x_train, y_train,
            x_test, y_test,
            epochs=args.finetune_epochs,
            batch_size=batch_size,
            learning_rate=0.0001,
            verbose=True
        )

        # Evaluate after fine-tuning
        print("📊 Evaluating pruned model (after fine-tuning)...")
        final_loss, final_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
        print(f"  - Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
        print(f"  - Loss: {final_loss:.4f}")
        print(f"  - Accuracy recovery: {(final_acc - pruned_acc)*100:.2f}%")
        print(f"  - Final vs original: {(final_acc - orig_acc)*100:.2f}%\n")

    # Save pruned model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pruned_model.save(output_path)
    print(f"✅ Pruned model saved to {output_path}")

    # Also save as TFLite
    try:
        import tensorflow as tf
        tflite_path = output_path.with_suffix('.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f"✅ TFLite model saved to {tflite_path}")
        print(f"   Model size: {len(tflite_model) / 1024:.2f} KB")

        # Compare TFLite sizes
        if model_path.with_suffix('.tflite').exists():
            orig_tflite_size = model_path.with_suffix('.tflite').stat().st_size / 1024
            pruned_tflite_size = len(tflite_model) / 1024
            size_reduction = (1 - pruned_tflite_size / orig_tflite_size) * 100
            print(f"   Size reduction: {size_reduction:.1f}%")
    except Exception as e:
        print(f"⚠️  Could not convert to TFLite: {e}")

    print(f"\n{'='*60}")
    print(f"✅ Pruning Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
