"""
Complete TinyML Production Pipeline
FL/TL → Pruning → Fine-tuning → Quantization → TFLite Export

This is the main entry point for the complete model compression pipeline.
Use this for deploying models to microcontrollers.

Usage:
    python main_pipeline.py --dataset mnist --model cnn --pruning-ratio 0.5
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import time

from src.models.nets import get_model
from src.data.loader import load_dataset
from src.modelcompression.pruning import (
    apply_structured_pruning,
    fine_tune_pruned_model,
    compare_models,
    get_model_size
)
from src.modelcompression.quantization import (
    quantize_model,
    compare_model_sizes,
    evaluate_quantization_accuracy
)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def train_initial_model(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 10,
    batch_size: int = 128
) -> keras.Model:
    """
    Train initial model (simulates FL/TL result).

    In production, this would be your:
    - Federated Learning (FL) aggregated weights
    - Transfer Learning (TL) fine-tuned weights
    - Standard centralized training
    """
    print_section("STAGE 1: Training Initial Model (FL/TL)")

    print(f"Training for {epochs} epochs with batch_size={batch_size}")
    print("(In production, this would be your FL/TL trained model)\n")

    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        verbose=1
    )

    train_time = time.time() - start_time

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"\n✅ Training completed in {train_time:.2f} seconds")
    print(f"✅ Test Accuracy: {acc*100:.2f}%")
    print(f"✅ Test Loss: {loss:.4f}")

    params, size_kb = get_model_size(model)
    print(f"✅ Model Parameters: {params:,}")
    print(f"✅ Model Size: {size_kb:.2f} KB")

    return model


def apply_pruning_step(
    model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    pruning_ratio: float,
    finetune_epochs: int
) -> keras.Model:
    """Apply structured pruning and fine-tune."""
    print_section("STAGE 2: Applying Structured Pruning")

    pruned_model = apply_structured_pruning(
        model,
        pruning_ratio=pruning_ratio,
        skip_last_layer=True,
        verbose=True
    )

    compare_models(model, pruned_model)

    print("Evaluating pruned model (before fine-tuning)...")
    pruned_loss, pruned_acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Pruned Model Accuracy: {pruned_acc*100:.2f}%")
    print(f"✅ Pruned Model Loss: {pruned_loss:.4f}")

    print_section("STAGE 3: Fine-tuning Pruned Model")

    pruned_model = fine_tune_pruned_model(
        pruned_model,
        x_train, y_train,
        x_test, y_test,
        epochs=finetune_epochs,
        batch_size=128,
        learning_rate=0.0001,
        verbose=True
    )

    final_loss, final_acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n✅ Fine-tuned Model Accuracy: {final_acc*100:.2f}%")
    print(f"✅ Fine-tuned Model Loss: {final_loss:.4f}")
    print(f"📈 Accuracy Recovery: {(final_acc - pruned_acc)*100:+.2f}%")

    return pruned_model


def apply_quantization_step(
    pruned_model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray
):
    """Apply post-training quantization (INT8)."""
    print_section("STAGE 4: Applying Post-Training Quantization (INT8)")

    quantized_layers = quantize_model(
        pruned_model,
        symmetric=True,
        verbose=True
    )

    compare_model_sizes(pruned_model, quantized_layers)

    print("Evaluating quantization accuracy impact...")
    evaluate_quantization_accuracy(
        pruned_model,
        quantized_layers,
        x_test,
        y_test,
        verbose=True
    )

    return quantized_layers


def export_to_tflite(
    model: keras.Model,
    x_train: np.ndarray,
    output_path: Path,
    quantize: bool = True
) -> bytes:
    """Export model to TFLite format with optional quantization."""
    print_section(f"STAGE 5: Exporting to TFLite ({'INT8' if quantize else 'FLOAT32'})")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        print("Applying full integer quantization (INT8)...")
        print("This is RECOMMENDED for microcontroller deployment!\n")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        def representative_dataset():
            num_samples = min(100, len(x_train))
            for i in range(num_samples):
                yield [x_train[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset

        print("✅ Representative dataset: 100 samples")
        print("✅ Quantization type: Full INT8 (weights + activations)")
    else:
        print("Exporting without quantization (FLOAT32)...\n")

    print("Converting model to TFLite...")
    tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)

    tflite_size_kb = len(tflite_model) / 1024
    print(f"\n✅ TFLite model saved to: {output_path}")
    print(f"✅ TFLite size: {tflite_size_kb:.2f} KB")

    return tflite_model


def run_pipeline(
    dataset_name: str = "mnist",
    model_name: str = "cnn",
    pruning_ratio: float = 0.5,
    initial_epochs: int = 5,
    finetune_epochs: int = 3,
    output_dir: str = "outputs/pipeline"
):
    """Execute complete TinyML pipeline."""
    print("\n" + "🚀 "*40)
    print(" "*20 + "TINYML PRODUCTION PIPELINE")
    print(" "*15 + "FL/TL → Pruning → Quantization → TFLite")
    print("🚀 "*40)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print_section("STAGE 0: Loading Dataset")
    print(f"Dataset: {dataset_name}")

    x_train, y_train, x_test, y_test = load_dataset(dataset_name)

    print(f"✅ Training samples: {len(x_train)}")
    print(f"✅ Test samples: {len(x_test)}")
    print(f"✅ Input shape: {x_train.shape[1:]}")
    print(f"✅ Number of classes: {len(np.unique(y_train))}")

    # Create model
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    model = get_model(model_name, input_shape, num_classes)
    print(f"\n✅ Model architecture: {model_name}")
    model.summary()

    # Stage 1: Train (or load from FL/TL)
    original_model = train_initial_model(
        model, x_train, y_train, x_test, y_test,
        epochs=initial_epochs
    )

    original_model.save(output_path / "01_original_model.h5")
    print(f"\n💾 Saved: {output_path / '01_original_model.h5'}")

    orig_loss, orig_acc = original_model.evaluate(x_test, y_test, verbose=0)
    orig_params, orig_size = get_model_size(original_model)

    # Stage 2-3: Pruning + Fine-tuning
    pruned_model = apply_pruning_step(
        original_model,
        x_train, y_train,
        x_test, y_test,
        pruning_ratio=pruning_ratio,
        finetune_epochs=finetune_epochs
    )

    pruned_model.save(output_path / "02_pruned_finetuned_model.h5")
    print(f"\n💾 Saved: {output_path / '02_pruned_finetuned_model.h5'}")

    pruned_loss, pruned_acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    pruned_params, pruned_size = get_model_size(pruned_model)

    # Stage 4: Quantization
    quantized_layers = apply_quantization_step(
        pruned_model,
        x_test,
        y_test
    )

    # Stage 5: Export to TFLite

    # Original (float32)
    tflite_orig = export_to_tflite(
        original_model,
        x_train,
        output_path / "03_original_float32.tflite",
        quantize=False
    )

    # Pruned + Quantized (int8) - RECOMMENDED
    tflite_pruned_quant = export_to_tflite(
        pruned_model,
        x_train,
        output_path / "04_pruned_quantized_int8.tflite",
        quantize=True
    )

    # Pruned only (float32)
    tflite_pruned = export_to_tflite(
        pruned_model,
        x_train,
        output_path / "05_pruned_float32.tflite",
        quantize=False
    )

    # Final Summary
    print_section("📊 PIPELINE RESULTS SUMMARY")

    print(f"{'Metric':<40} {'Original':<20} {'Pruned+FT':<20} {'Change':<20}")
    print("-"*100)
    print(f"{'Parameters':<40} {orig_params:>19,} {pruned_params:>19,} {(1-pruned_params/orig_params)*100:>18.1f}%")
    print(f"{'Model Size (KB)':<40} {orig_size:>19.2f} {pruned_size:>19.2f} {(1-pruned_size/orig_size)*100:>18.1f}%")
    print(f"{'Accuracy':<40} {orig_acc:>19.2%} {pruned_acc:>19.2%} {(pruned_acc-orig_acc)*100:>18.2f}%")
    print(f"{'Loss':<40} {orig_loss:>19.4f} {pruned_loss:>19.4f} {(pruned_loss-orig_loss):>19.4f}")

    print(f"\n{'TFLite Export':<40} {'Size (KB)':<20} {'Format':<20}")
    print("-"*100)
    print(f"{'Original (float32)':<40} {len(tflite_orig)/1024:>19.2f} {'FLOAT32':<20}")
    print(f"{'Pruned (float32)':<40} {len(tflite_pruned)/1024:>19.2f} {'FLOAT32':<20}")
    print(f"{'Pruned + Quantized (int8)':<40} {len(tflite_pruned_quant)/1024:>19.2f} {'INT8 ⭐ DEPLOY':<20}")

    final_compression = len(tflite_orig) / len(tflite_pruned_quant)
    print(f"\n{'Total Compression Ratio:':<40} {final_compression:>19.2f}x")
    print(f"{'Total Size Reduction:':<40} {(1 - 1/final_compression)*100:>18.1f}%")

    print("\n" + "="*100)
    print("✅ PIPELINE COMPLETE!")
    print("="*100)

    print(f"\n📁 All outputs saved to: {output_path.absolute()}")
    print("\nGenerated files:")
    print("  1. 01_original_model.h5              - Original trained model")
    print("  2. 02_pruned_finetuned_model.h5      - Pruned and fine-tuned model")
    print("  3. 03_original_float32.tflite        - Original TFLite (float32)")
    print("  4. 04_pruned_quantized_int8.tflite   - ⭐ DEPLOYMENT READY (int8)")
    print("  5. 05_pruned_float32.tflite          - Pruned TFLite (float32)")

    print("\n💡 For microcontroller deployment:")
    print(f"   📱 Use: {output_path / '04_pruned_quantized_int8.tflite'}")
    print(f"   📏 Size: {len(tflite_pruned_quant)/1024:.2f} KB")
    print(f"   🗜️  Compression: {final_compression:.1f}x smaller than original")
    print(f"   🎯 Accuracy: {pruned_acc*100:.2f}% (vs {orig_acc*100:.2f}% original)")

    print("\n")

    return {
        'original_model': original_model,
        'pruned_model': pruned_model,
        'quantized_layers': quantized_layers,
        'metrics': {
            'original_acc': orig_acc,
            'pruned_acc': pruned_acc,
            'compression_ratio': final_compression,
            'tflite_size_kb': len(tflite_pruned_quant) / 1024
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description="TinyML Production Pipeline: FL/TL → Pruning → Quantization → TFLite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default MNIST CNN with 50% pruning
  python main_pipeline.py

  # MNIST MLP with 70% aggressive pruning
  python main_pipeline.py --model mlp --pruning-ratio 0.7

  # Bot-IoT dataset for IoT device deployment
  python main_pipeline.py --dataset bot_iot --model mlp --initial-epochs 10

  # Quick test with minimal training
  python main_pipeline.py --initial-epochs 2 --finetune-epochs 1
        """
    )

    parser.add_argument(
        '--dataset', type=str, default='mnist',
        choices=['mnist', 'bot_iot'],
        help='Dataset to use (default: mnist)'
    )
    parser.add_argument(
        '--model', type=str, default='cnn',
        choices=['cnn', 'mlp'],
        help='Model architecture (default: cnn)'
    )
    parser.add_argument(
        '--pruning-ratio', type=float, default=0.5,
        help='Fraction of neurons to prune, 0.0-1.0 (default: 0.5)'
    )
    parser.add_argument(
        '--initial-epochs', type=int, default=5,
        help='Epochs for initial training (default: 5)'
    )
    parser.add_argument(
        '--finetune-epochs', type=int, default=3,
        help='Epochs for fine-tuning after pruning (default: 3)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='outputs/pipeline',
        help='Directory to save outputs (default: outputs/pipeline)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 <= args.pruning_ratio <= 1:
        parser.error("Pruning ratio must be between 0.0 and 1.0")

    if args.initial_epochs < 1:
        parser.error("Initial epochs must be at least 1")

    if args.finetune_epochs < 1:
        parser.error("Fine-tuning epochs must be at least 1")

    # Run pipeline
    run_pipeline(
        dataset_name=args.dataset,
        model_name=args.model,
        pruning_ratio=args.pruning_ratio,
        initial_epochs=args.initial_epochs,
        finetune_epochs=args.finetune_epochs,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
