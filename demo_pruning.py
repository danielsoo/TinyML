"""
Visual demonstration of structured pruning with detailed metrics.
This script shows the entire pruning process step-by-step with visual feedback.
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import time

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
        # Only loss returned
        return result, 0.0


def create_synthetic_dataset(n_samples=1000, n_features=20, n_classes=2, noise=0.1):
    """Create synthetic dataset for demonstration."""
    np.random.seed(42)

    # Generate data with some structure
    x = np.random.randn(n_samples, n_features)

    # Create labels based on simple rule
    if n_classes == 2:
        y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).astype(int)
    else:
        y = np.random.randint(0, n_classes, n_samples)

    # Add noise
    y = np.where(np.random.rand(n_samples) < noise,
                 1 - y if n_classes == 2 else np.random.randint(0, n_classes, n_samples),
                 y)

    # Split train/test
    split = int(0.8 * n_samples)
    return x[:split], y[:split], x[split:], y[split:]


def visualize_layer_sizes(original_model, pruned_model, save_path=None):
    """Visualize layer-by-layer parameter counts."""
    print("\n📊 Creating layer size comparison plot...")

    orig_layer_params = []
    pruned_layer_params = []
    layer_names = []

    for i, (orig_layer, pruned_layer) in enumerate(zip(original_model.layers, pruned_model.layers)):
        if isinstance(orig_layer, (layers.Dense, layers.Conv2D)):
            orig_params = orig_layer.count_params()
            pruned_params = pruned_layer.count_params()

            orig_layer_params.append(orig_params)
            pruned_layer_params.append(pruned_params)
            layer_names.append(f"{orig_layer.name}\n({orig_layer.__class__.__name__})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar chart
    x = np.arange(len(layer_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, orig_layer_params, width, label='Original', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pruned_layer_params, width, label='Pruned', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    ax1.set_title('Layer-by-Layer Parameter Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=8)

    # Pie chart - overall comparison
    total_orig = sum(orig_layer_params)
    total_pruned = sum(pruned_layer_params)
    total_removed = total_orig - total_pruned

    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)

    ax2.pie([total_pruned, total_removed],
            labels=['Kept Parameters', 'Removed Parameters'],
            autopct='%1.1f%%',
            colors=colors,
            explode=explode,
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax2.set_title(f'Total Parameters: {total_orig:,} → {total_pruned:,}',
                  fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")

    plt.show()


def visualize_accuracy_comparison(metrics_history, save_path=None):
    """Visualize accuracy changes through pruning pipeline."""
    print("\n📊 Creating accuracy comparison plot...")

    stages = list(metrics_history.keys())
    accuracies = [metrics_history[stage]['accuracy'] for stage in stages]
    losses = [metrics_history[stage]['loss'] for stage in stages]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy bar chart
    colors_acc = ['#3498db', '#e67e22', '#2ecc71']
    bars = ax1.bar(stages, accuracies, color=colors_acc, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Through Pruning Pipeline', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Loss bar chart
    colors_loss = ['#3498db', '#e67e22', '#2ecc71']
    bars2 = ax2.bar(stages, losses, color=colors_loss, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Loss Through Pruning Pipeline', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, loss in zip(bars2, losses):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{loss:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved to {save_path}")

    plt.show()


def print_detailed_summary(original_model, pruned_model, metrics_history):
    """Print comprehensive summary table."""
    print("\n" + "="*80)
    print(" "*25 + "🔬 PRUNING SUMMARY REPORT")
    print("="*80)

    # Model size comparison
    orig_params, orig_size = get_model_size(original_model)
    pruned_params, pruned_size = get_model_size(pruned_model)

    print(f"\n{'MODEL SIZE METRICS':<40} {'ORIGINAL':<20} {'PRUNED':<20} {'CHANGE':<20}")
    print("-"*80)
    print(f"{'Total Parameters':<40} {orig_params:>19,} {pruned_params:>19,} {(pruned_params-orig_params):>19,}")
    print(f"{'Model Size (KB)':<40} {orig_size:>19.2f} {pruned_size:>19.2f} {(pruned_size-orig_size):>19.2f}")
    print(f"{'Parameter Reduction':<40} {'':<20} {'':<20} {(1-pruned_params/orig_params)*100:>18.1f}%")
    print(f"{'Size Reduction':<40} {'':<20} {'':<20} {(1-pruned_size/orig_size)*100:>18.1f}%")

    # Performance metrics
    print(f"\n{'PERFORMANCE METRICS':<40} {'ACCURACY':<20} {'LOSS':<20}")
    print("-"*80)
    for stage, metrics in metrics_history.items():
        print(f"{stage:<40} {metrics['accuracy']:>19.2%} {metrics['loss']:>19.4f}")

    # Accuracy changes
    orig_acc = metrics_history['Original']['accuracy']
    pruned_acc = metrics_history['After Pruning']['accuracy']
    final_acc = metrics_history.get('After Fine-tuning', {}).get('accuracy', pruned_acc)

    print(f"\n{'ACCURACY ANALYSIS':<40} {'VALUE':<20}")
    print("-"*80)
    print(f"{'Initial Accuracy':<40} {orig_acc:>19.2%}")
    print(f"{'After Pruning':<40} {pruned_acc:>19.2%}")
    print(f"{'Accuracy Drop from Pruning':<40} {(orig_acc-pruned_acc)*100:>18.2f}%")

    if 'After Fine-tuning' in metrics_history:
        print(f"{'After Fine-tuning':<40} {final_acc:>19.2%}")
        print(f"{'Accuracy Recovery':<40} {(final_acc-pruned_acc)*100:>18.2f}%")
        print(f"{'Final vs Original':<40} {(final_acc-orig_acc)*100:>18.2f}%")

    print("="*80 + "\n")


def demo_pruning_pipeline(pruning_ratio=0.5, finetune_epochs=5):
    """
    Complete demonstration of the pruning pipeline with visual feedback.
    """
    print("\n" + "🎯 "*30)
    print(" "*20 + "STRUCTURED PRUNING DEMONSTRATION")
    print("🎯 "*30 + "\n")

    # Step 1: Create dataset
    print("📦 Step 1: Creating Synthetic Dataset")
    print("-" * 60)
    x_train, y_train, x_test, y_test = create_synthetic_dataset(n_samples=2000, n_features=20)
    print(f"✅ Training samples: {len(x_train)}")
    print(f"✅ Test samples: {len(x_test)}")
    print(f"✅ Features: {x_train.shape[1]}")
    print(f"✅ Classes: {len(np.unique(y_train))}\n")

    # Step 2: Build and train model
    print("🏗️  Step 2: Building and Training Model")
    print("-" * 60)

    model = keras.Sequential([
        keras.Input(shape=(20,)),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dense(32, activation='relu', name='dense3'),
        layers.Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("Model architecture:")
    model.summary()

    print("\n🚀 Training original model...")
    start_time = time.time()

    history = model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"✅ Training completed in {train_time:.2f} seconds\n")

    # Evaluate original
    print("📊 Step 3: Evaluating Original Model")
    print("-" * 60)
    orig_loss, orig_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {orig_acc:.2%}")
    print(f"✅ Test Loss: {orig_loss:.4f}\n")

    metrics_history = {
        'Original': {'accuracy': orig_acc, 'loss': orig_loss}
    }

    # Step 4: Apply pruning
    print(f"✂️  Step 4: Applying Structured Pruning (ratio={pruning_ratio:.0%})")
    print("-" * 60)

    pruned_model = apply_structured_pruning(
        model,
        pruning_ratio=pruning_ratio,
        skip_last_layer=True,
        verbose=True
    )

    # Compare models
    compare_models(model, pruned_model)

    # Evaluate pruned model
    print("📊 Step 5: Evaluating Pruned Model (Before Fine-tuning)")
    print("-" * 60)
    pruned_loss, pruned_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {pruned_acc:.2%}")
    print(f"✅ Test Loss: {pruned_loss:.4f}")
    print(f"⚠️  Accuracy Drop: {(orig_acc - pruned_acc)*100:.2f}%\n")

    metrics_history['After Pruning'] = {'accuracy': pruned_acc, 'loss': pruned_loss}

    # Step 6: Fine-tune
    print(f"🔄 Step 6: Fine-tuning Pruned Model ({finetune_epochs} epochs)")
    print("-" * 60)

    start_time = time.time()
    pruned_model = fine_tune_pruned_model(
        pruned_model,
        x_train, y_train,
        x_test, y_test,
        epochs=finetune_epochs,
        batch_size=32,
        learning_rate=0.0001,
        verbose=True
    )
    finetune_time = time.time() - start_time
    print(f"✅ Fine-tuning completed in {finetune_time:.2f} seconds\n")

    # Final evaluation
    print("📊 Step 7: Final Evaluation")
    print("-" * 60)
    final_loss, final_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {final_acc:.2%}")
    print(f"✅ Test Loss: {final_loss:.4f}")
    print(f"📈 Accuracy Recovery: {(final_acc - pruned_acc)*100:.2f}%")
    print(f"📊 Final vs Original: {(final_acc - orig_acc)*100:+.2f}%\n")

    metrics_history['After Fine-tuning'] = {'accuracy': final_acc, 'loss': final_loss}

    # Step 8: Generate visualizations
    print("📊 Step 8: Generating Visualizations")
    print("-" * 60)

    output_dir = Path("outputs/pruning_demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print detailed summary
    print_detailed_summary(model, pruned_model, metrics_history)

    # Generate plots
    visualize_layer_sizes(model, pruned_model, save_path=output_dir / "layer_comparison.png")
    visualize_accuracy_comparison(metrics_history, save_path=output_dir / "accuracy_comparison.png")

    # Step 9: Save models
    print("\n💾 Step 9: Saving Models")
    print("-" * 60)

    model.save(output_dir / "original_model.h5")
    print(f"✅ Original model saved to {output_dir / 'original_model.h5'}")

    pruned_model.save(output_dir / "pruned_model.h5")
    print(f"✅ Pruned model saved to {output_dir / 'pruned_model.h5'}")

    # Save as TFLite
    try:
        import tensorflow as tf

        # Original model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_orig = converter.convert()
        (output_dir / "original_model.tflite").write_bytes(tflite_orig)

        # Pruned model
        converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
        tflite_pruned = converter.convert()
        (output_dir / "pruned_model.tflite").write_bytes(tflite_pruned)

        print(f"✅ TFLite models saved")
        print(f"   Original: {len(tflite_orig)/1024:.2f} KB")
        print(f"   Pruned: {len(tflite_pruned)/1024:.2f} KB")
        print(f"   Reduction: {(1 - len(tflite_pruned)/len(tflite_orig))*100:.1f}%")

    except Exception as e:
        print(f"⚠️  Could not save TFLite: {e}")

    print("\n" + "="*80)
    print("✅ DEMONSTRATION COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - original_model.h5 / .tflite")
    print("  - pruned_model.h5 / .tflite")
    print("  - layer_comparison.png")
    print("  - accuracy_comparison.png")
    print("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual demonstration of structured pruning")
    parser.add_argument("--pruning-ratio", type=float, default=0.5,
                       help="Fraction of neurons to prune (default: 0.5)")
    parser.add_argument("--finetune-epochs", type=int, default=5,
                       help="Number of fine-tuning epochs (default: 5)")

    args = parser.parse_args()

    demo_pruning_pipeline(
        pruning_ratio=args.pruning_ratio,
        finetune_epochs=args.finetune_epochs
    )
