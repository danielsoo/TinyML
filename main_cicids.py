"""
CICIDS2017 Pipeline: FL → Compression → TFLite (with architecture comparison)

This script:
1. Loads CICIDS2017 (MachineLearningCVE) dataset
2. Runs FL with the standard MLP (teacher)
3. Compresses the teacher via distillation + pruning
4. Also trains a lightweight model directly via FL (no distillation needed)
5. Compares both approaches: distilled vs lightweight

Usage:
    python main_cicids.py [--config config/federated_cicids.yaml]
    python main_cicids.py --skip-fl   (if models already trained)
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_dataset
from src.compression.distillation import train_with_distillation, create_student_model
from src.compression.pruning import apply_structured_pruning, fine_tune_pruned_model
from src.models.nets import get_model
from src.models.lightweight import get_lightweight_model


def load_config(config_path: str = "config/federated_cicids.yaml") -> dict:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """Load CICIDS2017 dataset."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA LOADING (CICIDS2017)")
    print("=" * 70)

    data_cfg = config.get("data", {})
    dataset_name = data_cfg.get("name", "cicids2017")

    dataset_kwargs = {
        k: v for k, v in data_cfg.items()
        if k not in {"name", "num_clients"}
    }

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = max(2, len(unique_labels))

    # Remap labels to 0-indexed if needed
    if np.min(unique_labels) != 0 or np.max(unique_labels) != num_classes - 1:
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])

    print(f"  Training samples: {len(x_train):,}")
    print(f"  Test samples: {len(x_test):,}")
    print(f"  Features: {x_train.shape[1]}")
    print(f"  Classes: {num_classes}")
    print(f"  Class distribution: {np.bincount(y_train)}\n")

    return x_train, y_train, x_test, y_test, num_classes


def run_fl(config: dict, save_path: str, config_path: str):
    """Run federated learning and save the global model."""
    print("\n" + "=" * 70)
    print("STEP 2: FEDERATED LEARNING")
    print("=" * 70)

    from src.federated.client import main as fl_main

    fed_cfg = config.get("federated", {})
    print(f"  Clients: {config['data']['num_clients']}")
    print(f"  Rounds: {fed_cfg['num_rounds']}")
    print(f"  Local epochs: {fed_cfg['local_epochs']}")

    fl_main(save_path=save_path, config_path=config_path)
    print(f"  Model saved to {save_path}")


def visualize_comparison(models_info: list):
    """
    Print comparison table for multiple models.

    Args:
        models_info: List of dicts with keys: name, model, accuracy, loss
    """
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)

    # Per-model details
    for info in models_info:
        model = info['model']
        print(f"\n{'─' * 70}")
        print(f"  {info['name']}")
        print(f"{'─' * 70}")
        print(f"  {'Layer':<25} {'Type':<10} {'Output Shape':<20} {'Params':>8}")
        print(f"  {'─'*25} {'─'*10} {'─'*20} {'─'*8}")
        for layer in model.layers:
            if isinstance(layer, layers.InputLayer):
                continue
            print(f"  {layer.name:<25} {layer.__class__.__name__:<10} {str(layer.output_shape):<20} {layer.count_params():>8,}")
        print(f"  {'─'*25} {'─'*10} {'─'*20} {'─'*8}")
        print(f"  {'TOTAL':<25} {'':<10} {'':<20} {model.count_params():>8,}")

    # Summary table
    print(f"\n{'═' * 70}")
    print(f"  FINAL COMPARISON")
    print(f"{'═' * 70}")
    print(f"  {'Model':<30} {'Params':>10} {'Size(KB)':>10} {'Accuracy':>10} {'vs Base':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    base_params = models_info[0]['model'].count_params() if models_info else 1

    for info in models_info:
        params = info['model'].count_params()
        size_kb = params * 4 / 1024
        acc = info.get('accuracy', 0)
        ratio = f"{base_params / params:.1f}x" if params < base_params else "1.0x"
        print(f"  {info['name']:<30} {params:>10,} {size_kb:>10.2f} {acc:>9.4f} {ratio:>10}")

    print("=" * 70 + "\n")


def compress_with_distillation(teacher_model, x_train, y_train, x_test, y_test, num_classes, config):
    """Run distillation + pruning pipeline on teacher model."""
    print("\n" + "-" * 70)
    print("  Approach A: Knowledge Distillation + Pruning")
    print("-" * 70)

    batch_size = config['federated']['batch_size']

    # Create student
    student_model = create_student_model(
        teacher_model, compression_ratio=0.5, num_classes=num_classes
    )

    # Split for training
    val_split = int(0.8 * len(x_train))
    x_tr, y_tr = x_train[:val_split], y_train[:val_split]
    x_val, y_val = x_train[val_split:], y_train[val_split:]

    # Distillation
    print(f"  Student params: {student_model.count_params():,}")
    student_model, _ = train_with_distillation(
        teacher_model=teacher_model,
        student_model=student_model,
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        temperature=3.0, alpha=0.1,
        epochs=10, batch_size=batch_size,
        verbose=1
    )

    # Compile student before pruning (distiller trains it without compiling standalone)
    student_model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Pruning
    pruned_model = apply_structured_pruning(
        model=student_model, pruning_ratio=0.5,
        skip_last_layer=True, verbose=True
    )

    pruned_model = fine_tune_pruned_model(
        pruned_model=pruned_model,
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        epochs=5, batch_size=batch_size,
        learning_rate=0.0001, verbose=True
    )

    loss, acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"  Distilled+Pruned: accuracy={acc:.4f}, params={pruned_model.count_params():,}")

    return pruned_model, acc, loss


def train_lightweight_directly(x_train, y_train, x_test, y_test, num_classes, config, arch_name="lightweight"):
    """Train a lightweight model directly (no distillation, standard training)."""
    print(f"\n" + "-" * 70)
    print(f"  Approach B: Direct Lightweight Training ({arch_name})")
    print("-" * 70)

    input_shape = (x_train.shape[1],)
    batch_size = config['federated']['batch_size']

    model = get_lightweight_model(arch_name, input_shape, num_classes)
    print(f"  {arch_name} params: {model.count_params():,}")

    val_split = int(0.8 * len(x_train))
    model.fit(
        x_train[:val_split], y_train[:val_split],
        validation_data=(x_train[val_split:], y_train[val_split:]),
        epochs=15, batch_size=batch_size, verbose=1
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"  {arch_name}: accuracy={acc:.4f}, params={model.count_params():,}")

    return model, acc, loss


def export_tflite(model, x_train, output_path, quantize=True):
    """Export model to TFLite."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    representative_data = x_train[:1000] if len(x_train) > 1000 else x_train

    from src.tinyml import export_tflite as tflite_module
    size_bytes = tflite_module.export_tflite(
        model=model,
        out_path=str(output_file),
        quantize=quantize,
        representative_data=representative_data if quantize else None
    )
    return str(output_file), size_bytes / 1024


def main():
    parser = argparse.ArgumentParser(description="CICIDS2017 TinyML Pipeline")
    parser.add_argument('--config', type=str, default='config/federated_cicids.yaml')
    parser.add_argument('--skip-fl', action='store_true', help='Skip FL, load existing model')
    parser.add_argument('--fl-model', type=str, default='models/cicids_global_model.h5')
    parser.add_argument('--output-dir', type=str, default='models/cicids/')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("CICIDS2017 TinyML PIPELINE")
    print("FL → Compress (Standard vs Lightweight) → TFLite")
    print("=" * 70)

    config = load_config(args.config)

    # Step 1: Load data
    x_train, y_train, x_test, y_test, num_classes = load_data(config)
    input_shape = (x_train.shape[1],)

    # Step 2: FL with standard MLP (teacher)
    if args.skip_fl:
        if not Path(args.fl_model).exists():
            raise FileNotFoundError(f"Model not found: {args.fl_model}. Run without --skip-fl first.")
        print(f"\n  Skipping FL, loading existing model from {args.fl_model}")
    else:
        run_fl(config, save_path=args.fl_model, config_path=args.config)

    teacher_model = keras.models.load_model(args.fl_model)
    teacher_loss, teacher_acc = teacher_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n  Teacher (Standard MLP): accuracy={teacher_acc:.4f}, params={teacher_model.count_params():,}")

    # Step 3: Compression approaches
    print("\n" + "=" * 70)
    print("STEP 3: MODEL COMPRESSION (Two Approaches)")
    print("=" * 70)

    # Approach A: Distillation + Pruning
    distilled_model, distilled_acc, distilled_loss = compress_with_distillation(
        teacher_model, x_train, y_train, x_test, y_test, num_classes, config
    )

    # Approach B: Train lightweight directly
    lightweight_model, lightweight_acc, lightweight_loss = train_lightweight_directly(
        x_train, y_train, x_test, y_test, num_classes, config, arch_name="lightweight"
    )

    # Approach C: Even smaller - bottleneck
    bottleneck_model, bottleneck_acc, bottleneck_loss = train_lightweight_directly(
        x_train, y_train, x_test, y_test, num_classes, config, arch_name="bottleneck"
    )

    # Step 4: Visualization
    models_info = [
        {'name': 'Standard MLP (teacher/FL)', 'model': teacher_model, 'accuracy': teacher_acc, 'loss': teacher_loss},
        {'name': 'Distilled + Pruned', 'model': distilled_model, 'accuracy': distilled_acc, 'loss': distilled_loss},
        {'name': 'Lightweight (direct)', 'model': lightweight_model, 'accuracy': lightweight_acc, 'loss': lightweight_loss},
        {'name': 'Bottleneck (direct)', 'model': bottleneck_model, 'accuracy': bottleneck_acc, 'loss': bottleneck_loss},
    ]
    visualize_comparison(models_info)

    # Step 5: Export best small model to TFLite
    print("\n" + "=" * 70)
    print("STEP 5: TFLITE EXPORT")
    print("=" * 70)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = [
        ("distilled_pruned", distilled_model),
        ("lightweight", lightweight_model),
        ("bottleneck", bottleneck_model),
    ]

    print(f"\n  {'Model':<25} {'TFLite Size(KB)':>15} {'Path'}")
    print(f"  {'─'*25} {'─'*15} {'─'*40}")

    for name, model in exports:
        out_path = str(output_dir / f"{name}.tflite")
        path, size_kb = export_tflite(model, x_train, out_path, quantize=True)
        print(f"  {name:<25} {size_kb:>15.2f} {path}")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n  Dataset: CICIDS2017 ({len(x_train)+len(x_test):,} samples, {num_classes} classes)")
    print(f"  Input features: {input_shape[0]}")
    print(f"\n  Results:")
    print(f"  {'─'*60}")
    for info in models_info:
        params = info['model'].count_params()
        print(f"    {info['name']:<30} acc={info['accuracy']:.4f}  params={params:,}")
    print(f"\n  TFLite models saved to: {output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
