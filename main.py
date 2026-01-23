"""
Main Pipeline: Data Loading → Federated Learning → Compression → TFLite Export

This script orchestrates the complete TinyML workflow:
1. Load data and run federated learning using Flower
2. Load the trained global model
3. Compress the model using distillation, pruning, and quantization
4. Export to TFLite for edge deployment

Usage:
    python main.py [--config config/federated.yaml] [--skip-fl] [--output models/compressed_model.tflite]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import project modules
from src.data.loader import load_dataset
from src.compression.distillation import train_with_distillation, create_student_model
from src.compression.pruning import apply_structured_pruning, fine_tune_pruned_model


def load_config(config_path: str = "config/federated.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_data(config: dict):
    """
    Load dataset for federated learning and compression.
    This data will be used by FL (which partitions it internally) and compression.

    Returns:
        tuple: (x_train, y_train, x_test, y_test, num_classes)
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("name", "bot_iot")

    # Pass all config except name and num_clients to data loader
    dataset_kwargs = {
        k: v
        for k, v in data_cfg.items()
        if k not in {"name", "num_clients"}
    }

    print(f"Loading dataset: {dataset_name}")

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Calculate num_classes
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)

    # Ensure labels are 0-indexed
    if np.min(unique_labels) != 0 or np.max(unique_labels) != num_classes - 1:
        print(f"⚠️  Warning: Remapping labels from {unique_labels} to 0-{num_classes-1}")
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])
        unique_labels = np.arange(num_classes)

    if num_classes == 1:
        num_classes = 2
        print(f"⚠️  Warning: Only 1 unique label found, assuming binary classification")

    print(f"✅ Training samples: {len(x_train):,}")
    print(f"✅ Test samples: {len(x_test):,}")
    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Class distribution: {np.bincount(y_train)}\n")

    return x_train, y_train, x_test, y_test, num_classes


def compress_model(
    teacher_model: keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    config: dict
) -> tuple:
    """
    Apply compression pipeline: Distillation → Pruning → Quantization

    Returns:
        tuple: (pruned_model, tflite_model, stats)
    """
    print("\n" + "=" * 70)
    print("STEP 3: MODEL COMPRESSION")
    print("=" * 70)

    # Configuration
    batch_size = config['federated']['batch_size']

    distillation_config = {
        'compression_ratio': 0.5,
        'temperature': 3.0,
        'alpha': 0.1,
        'epochs': 10,
        'batch_size': batch_size
    }

    pruning_config = {
        'target_sparsity': 0.5,
        'epochs': 5,
        'batch_size': batch_size
    }

    # Track compression stats
    stats = {
        'teacher': {
            'params': teacher_model.count_params(),
            'size_kb': (teacher_model.count_params() * 4) / 1024
        }
    }

    # Evaluate teacher
    teacher_loss, teacher_acc = teacher_model.evaluate(x_test, y_test, verbose=0)
    stats['teacher']['loss'] = teacher_loss
    stats['teacher']['accuracy'] = teacher_acc

    print(f"\n🎓 Teacher Model (Global FL Model):")
    print(f"  Parameters: {stats['teacher']['params']:,}")
    print(f"  Size: {stats['teacher']['size_kb']:.2f} KB")
    print(f"  Accuracy: {teacher_acc:.4f}")

    # Phase 1: Knowledge Distillation
    print("\n" + "-" * 70)
    print("Phase 1: Knowledge Distillation")
    print("-" * 70)

    student_model = create_student_model(
        teacher_model,
        compression_ratio=distillation_config['compression_ratio'],
        num_classes=num_classes
    )

    print(f"Student model: {student_model.count_params():,} parameters "
          f"({distillation_config['compression_ratio']*100:.0f}% of teacher)")

    # Split data for distillation
    val_split = int(0.8 * len(x_train))
    x_train_dist = x_train[:val_split]
    y_train_dist = y_train[:val_split]
    x_val_dist = x_train[val_split:]
    y_val_dist = y_train[val_split:]

    print("Training student with knowledge distillation...")
    student_model, _ = train_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        x_train=x_train_dist,
        y_train=y_train_dist,
        x_val=x_val_dist,
        y_val=y_val_dist,
        temperature=distillation_config['temperature'],
        alpha=distillation_config['alpha'],
        epochs=distillation_config['epochs'],
        batch_size=distillation_config['batch_size'],
        verbose=1
    )

    # Evaluate student
    student_loss, student_acc = student_model.evaluate(x_test, y_test, verbose=0)
    stats['student'] = {
        'params': student_model.count_params(),
        'size_kb': (student_model.count_params() * 4) / 1024,
        'loss': student_loss,
        'accuracy': student_acc
    }

    print(f"\n✅ After Distillation:")
    print(f"  Parameters: {stats['student']['params']:,}")
    print(f"  Size: {stats['student']['size_kb']:.2f} KB")
    print(f"  Accuracy: {student_acc:.4f} (vs teacher: {teacher_acc:.4f})")
    print(f"  Compression: {stats['teacher']['params'] / stats['student']['params']:.2f}x")

    # Phase 2: Pruning
    print("\n" + "-" * 70)
    print("Phase 2: Structured Pruning")
    print("-" * 70)

    # Apply structured pruning
    pruned_model = apply_structured_pruning(
        model=student_model,
        pruning_ratio=pruning_config['target_sparsity'],
        skip_last_layer=True,
        verbose=True
    )

    # Fine-tune to recover accuracy
    print("\nFine-tuning pruned model...")
    pruned_model = fine_tune_pruned_model(
        pruned_model=pruned_model,
        x_train=x_train_dist,
        y_train=y_train_dist,
        x_val=x_val_dist,
        y_val=y_val_dist,
        epochs=pruning_config['epochs'],
        batch_size=pruning_config['batch_size'],
        learning_rate=0.0001,
        verbose=True
    )

    # Evaluate pruned model
    pruned_loss, pruned_acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    stats['pruned'] = {
        'params': pruned_model.count_params(),
        'size_kb': (pruned_model.count_params() * 4) / 1024,
        'loss': pruned_loss,
        'accuracy': pruned_acc
    }

    print(f"\n✅ After Pruning:")
    print(f"  Parameters: {stats['pruned']['params']:,}")
    print(f"  Size: {stats['pruned']['size_kb']:.2f} KB")
    print(f"  Accuracy: {pruned_acc:.4f}")
    print(f"  Compression: {stats['teacher']['params'] / stats['pruned']['params']:.2f}x")

    # Summary
    print("\n" + "=" * 70)
    print("COMPRESSION SUMMARY (Before TFLite Export)")
    print("=" * 70)
    print(f"  Original size:     {stats['teacher']['size_kb']:.2f} KB")
    print(f"  Compressed size:   {stats['pruned']['size_kb']:.2f} KB")
    print(f"  Compression ratio: {stats['teacher']['params'] / stats['pruned']['params']:.2f}x")
    print(f"  Accuracy loss:     {(teacher_acc - pruned_acc) * 100:.2f}%")

    return pruned_model, stats


def export_model_to_tflite(
    model: keras.Model,
    x_train: np.ndarray,
    output_path: str = "models/compressed_model.tflite",
    quantize: bool = True
):
    """
    Export model to TFLite with optional quantization.

    Args:
        model: Keras model to export
        x_train: Training data for representative dataset
        output_path: Output path for .tflite file
        quantize: Whether to apply INT8 quantization

    Returns:
        str: Path to saved TFLite model
    """
    print("\n" + "=" * 70)
    print("STEP 4: TFLITE EXPORT & QUANTIZATION")
    print("=" * 70)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare representative data for quantization
    representative_data = x_train[:1000] if len(x_train) > 1000 else x_train

    # Export using the existing export_tflite function
    from src.tinyml import export_tflite as tflite_module

    size_bytes = tflite_module.export_tflite(
        model=model,
        out_path=str(output_file),
        quantize=quantize,
        representative_data=representative_data if quantize else None
    )

    size_kb = size_bytes / 1024

    print(f"\n✅ TFLite model ready for edge deployment!")

    if quantize:
        print(f"✅ INT8 quantization applied")

    return str(output_file), size_kb


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="TinyML Pipeline: FL → Compression → TFLite"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/federated.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--skip-fl',
        action='store_true',
        help='Skip FL and load existing global model'
    )
    parser.add_argument(
        '--fl-model',
        type=str,
        default='models/global_model.h5',
        help='Path to load/save FL global model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/compressed_model.tflite',
        help='Output path for TFLite model'
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("TinyML COMPLETE PIPELINE")
    print("=" * 70)
    print("Pipeline: FL → Compression → TFLite")
    print("=" * 70)

    # Load configuration
    config = load_config(args.config)

    # Step 1: Load data (used for both FL and compression)
    x_train, y_train, x_test, y_test, num_classes = load_data(config)

    # Step 2: Federated Learning
    if args.skip_fl:
        print("\n⚠️  Skipping federated learning, loading existing model...")
        model_path = args.fl_model
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run without --skip-fl to train with FL first."
            )
    else:
        # Run FL using existing client code
        print("\n" + "=" * 70)
        print("STEP 2: FEDERATED LEARNING")
        print("=" * 70)

        from src.federated.client import main as fl_main

        print(f"Starting federated learning with {config['data']['num_clients']} clients...")
        print(f"Rounds: {config['federated']['num_rounds']}")
        print(f"Local epochs: {config['federated']['local_epochs']}")

        # Run FL and save model
        fl_main(save_path=args.fl_model, config_path=args.config)
        model_path = args.fl_model

        print(f"\n✅ Federated learning complete! Model saved to {model_path}")

    # Load global model
    print(f"\n📥 Loading global model from {model_path}...")
    teacher_model = keras.models.load_model(model_path)

    # CRITICAL FIX: Check if model has correct output shape for binary classification
    # Old models may have 1 output for binary, but we need num_classes outputs
    output_units = teacher_model.layers[-1].units
    if num_classes == 2 and output_units == 1:
        print(f"\n⚠️  WARNING: Model has 1 output for binary classification (old format)")
        print(f"⚠️  This model was trained with the old code and is incompatible.")
        print(f"⚠️  Please delete {model_path} and retrain with:")
        print(f"     rm {model_path}")
        print(f"     python main.py")
        raise ValueError(
            f"Incompatible model format. Model has {output_units} outputs but needs {num_classes}. "
            f"Delete the model and retrain with the fixed code."
        )
    elif output_units != num_classes:
        print(f"\n⚠️  WARNING: Model output mismatch!")
        print(f"   Model has {output_units} outputs, but data has {num_classes} classes")
        raise ValueError(f"Model output mismatch: {output_units} != {num_classes}")

    print(f"✅ Global model loaded ({output_units} outputs for {num_classes} classes)")

    # Step 3: Compression pipeline
    compressed_model, _ = compress_model(
        teacher_model,
        x_train, y_train,
        x_test, y_test,
        num_classes,
        config
    )

    # Step 4: Export to TFLite with quantization
    tflite_path, _ = export_model_to_tflite(
        model=compressed_model,
        x_train=x_train,
        output_path=args.output,
        quantize=True
    )

    # Final summary
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - Global FL model: {model_path}")
    print(f"  - TFLite model:    {tflite_path}")
    print(f"\nNext steps:")
    print(f"  1. Deploy {tflite_path} to IoT devices")
    print(f"  2. Use FL for on-device updates")
    print(f"  3. Monitor performance and retrain as needed")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
