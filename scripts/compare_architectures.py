"""
Compare standard vs lightweight model architectures.

Shows parameter counts, layer details, and estimated sizes
for all available architectures given the same input shape.

Usage:
    python scripts/compare_architectures.py [--config config/federated.yaml]
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import yaml
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tensorflow import keras
from tensorflow.keras import layers
from src.models.nets import get_model
from src.models.lightweight import get_lightweight_model
from src.compression.distillation import create_student_model


def get_layer_info(model):
    """Extract layer details from a model."""
    info = []
    for layer in model.layers:
        if isinstance(layer, layers.InputLayer):
            continue
        info.append({
            'name': layer.name,
            'type': layer.__class__.__name__,
            'output_shape': str(layer.output_shape),
            'params': layer.count_params()
        })
    return info


def print_model_table(model, title):
    """Print a formatted table for a single model."""
    info = get_layer_info(model)
    print(f"\n  {title}")
    print(f"  {'─'*65}")
    print(f"  {'Layer':<25} {'Type':<10} {'Output Shape':<20} {'Params':>8}")
    print(f"  {'─'*25} {'─'*10} {'─'*20} {'─'*8}")
    for item in info:
        print(f"  {item['name']:<25} {item['type']:<10} {item['output_shape']:<20} {item['params']:>8,}")
    print(f"  {'─'*25} {'─'*10} {'─'*20} {'─'*8}")
    print(f"  {'TOTAL':<25} {'':<10} {'':<20} {model.count_params():>8,}")


def main():
    parser = argparse.ArgumentParser(description="Compare model architectures")
    parser.add_argument('--config', type=str, default='config/federated.yaml')
    parser.add_argument('--input-dim', type=int, default=None,
                        help='Override input dimension (default: from data)')
    parser.add_argument('--num-classes', type=int, default=2)
    args = parser.parse_args()

    # Load config to get input shape
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Determine input shape
    if args.input_dim:
        input_dim = args.input_dim
    else:
        # Try to get from data
        try:
            from src.data.loader import load_dataset
            data_cfg = config.get("data", {})
            dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
            x_train, y_train, x_test, y_test = load_dataset(
                data_cfg.get("name", "bot_iot"), **dataset_kwargs
            )
            input_dim = x_train.shape[1]
            unique_labels = np.unique(np.concatenate([y_train, y_test]))
            args.num_classes = max(2, len(unique_labels))
            print(f"Loaded data: input_dim={input_dim}, num_classes={args.num_classes}")
        except Exception as e:
            print(f"Could not load data ({e}), using default input_dim=10")
            input_dim = 10

    input_shape = (input_dim,)
    num_classes = args.num_classes

    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print(f"Input shape: {input_shape}, Output classes: {num_classes}")
    print("=" * 70)

    # Build all models
    models = {}

    # Standard models (from nets.py)
    models["Standard MLP (teacher)"] = get_model("mlp", input_shape, num_classes)

    # Student model (distilled from standard MLP)
    models["Distilled Student (50%)"] = create_student_model(
        models["Standard MLP (teacher)"], compression_ratio=0.5, num_classes=num_classes
    )

    # Lightweight models
    models["Lightweight MLP"] = get_lightweight_model("lightweight", input_shape, num_classes)
    models["Bottleneck MLP"] = get_lightweight_model("bottleneck", input_shape, num_classes)
    models["Tiny MLP"] = get_lightweight_model("tiny", input_shape, num_classes)

    # Print each model
    for title, model in models.items():
        print_model_table(model, title)

    # Summary comparison table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    teacher_params = models["Standard MLP (teacher)"].count_params()

    print(f"\n  {'Architecture':<25} {'Layers':>7} {'Params':>10} {'Size(KB)':>10} {'vs Teacher':>10}")
    print(f"  {'─'*25} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")

    for title, model in models.items():
        params = model.count_params()
        n_layers = len(get_layer_info(model))
        size_kb = params * 4 / 1024
        ratio = f"{teacher_params / params:.1f}x" if params < teacher_params else "1.0x"
        print(f"  {title:<25} {n_layers:>7} {params:>10,} {size_kb:>10.2f} {ratio:>10}")

    print(f"\n{'─' * 70}")
    print(f"  Notes:")
    print(f"  - 'vs Teacher' shows compression ratio relative to Standard MLP")
    print(f"  - Size assumes float32 (4 bytes per parameter)")
    print(f"  - After TFLite INT8 quantization, size is ~4x smaller")
    print(f"  - Lightweight models can be used directly in FL (no distillation needed)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
