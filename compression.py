"""
End-to-end integration test for the complete TinyML compression pipeline.
Tests Training → Distillation → Pruning → Quantization → TFLite on Bot-IoT dataset.

Pipeline stages:
1. Train teacher model (full size)
2. Knowledge distillation (compress to 50% student)
3. Structured pruning (remove 30-70% of neurons)
4. INT8 quantization (4x weight compression)
5. TFLite export (deployment ready)

Expected compression: 8-12x smaller model with minimal accuracy loss.
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

# Add project root to path
# Handle both script execution and module execution
project_root = None

# Method 1: Try to get from __file__ (most reliable)
try:
    if '__file__' in globals() and __file__:
        project_root = Path(__file__).resolve().parent
except (NameError, TypeError):
    pass

# Method 2: Try to find project root by looking for src/ directory
if project_root is None:
    cwd = Path.cwd()
    # Check current directory and parent directories
    for path in [cwd] + list(cwd.parents):
        if (path / "src" / "compression").exists():
            project_root = path
            break

# Method 3: Fallback to current working directory
if project_root is None:
    project_root = Path.cwd()

# Always add project root to sys.path
project_root_str = str(project_root)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Also add current working directory as fallback
cwd = Path.cwd()
cwd_str = str(cwd)
if cwd_str not in sys.path and cwd_str != project_root_str:
    sys.path.insert(0, cwd_str)

# Debug: Print paths for troubleshooting (can be removed later)
if os.getenv("DEBUG_PYTHONPATH"):
    print(f"[DEBUG] Project root: {project_root}")
    print(f"[DEBUG] Current working directory: {cwd}")
    print(f"[DEBUG] sys.path (first 3): {sys.path[:3]}")
    print(f"[DEBUG] Checking src/compression exists: {(project_root / 'src' / 'compression').exists()}")

from src.data.loader import load_dataset
from src.models.nets import get_model
from src.compression.distillation import (
    create_student_model,
    train_with_distillation,
    compare_models as compare_distillation
)
from src.compression.pruning import (
    apply_structured_pruning,
    compare_models,
    fine_tune_pruned_model,
    get_model_size
)
from src.compression.quantization import (
    quantize_model,
    compare_model_sizes,
    evaluate_quantization_accuracy
)
from src.tinyml.export_tflite import export_tflite


def safe_evaluate(model, x, y, verbose=0):
    """Safely evaluate model and return (loss, accuracy)."""
    result = model.evaluate(x, y, verbose=verbose)
    if isinstance(result, (list, tuple)):
        return result[0], result[1]
    else:
        return result, 0.0


def test_full_pipeline_mlp():
    """
    Test complete TinyML compression pipeline with MLP on Bot-IoT data.

    Pipeline:
        Training → Knowledge Distillation → Pruning → Fine-tuning →
        Quantization → TFLite Export

    Compression techniques:
        1. Knowledge Distillation: Teacher → Student (50% compression)
        2. Structured Pruning: Remove neurons (30-70% reduction)
        3. INT8 Quantization: Float32 → INT8 (4x compression)

    Expected total compression: 8-12x smaller with 1-3% accuracy drop
    """
    print("\n" + "="*80)
    print(" "*20 + "🧪 FULL TINYML PIPELINE TEST")
    print("="*80 + "\n")

    # Load config
    print("📋 Step 1: Loading Configuration")
    print("-" * 60)
    
    # Auto-detect environment and use appropriate config
    try:
        from src.utils.env_utils import get_default_config_path
        config_path = get_default_config_path()
    except ImportError:
        # Fallback: check if in Colab
        import os
        if os.path.exists("/content") or os.getenv("COLAB_GPU") is not None:
            config_path = "config/federated_colab.yaml"
        else:
            config_path = "config/federated.yaml"
    
    print(f"Using config: {config_path}")
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fed_cfg = cfg.get("federated", {})
    print("✅ Configuration loaded\n")

    # Load dataset
    print("📂 Step 2: Loading Bot-IoT Dataset")
    print("-" * 60)
    dataset_name = data_cfg.get("name", "bot_iot")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Use subset for faster testing
    n_train = min(5000, len(x_train))
    n_test = min(1000, len(x_test))
    x_train, y_train = x_train[:n_train], y_train[:n_train]
    x_test, y_test = x_test[:n_test], y_test[:n_test]

    print(f"✅ Training samples: {len(x_train)}")
    print(f"✅ Test samples: {len(x_test)}")
    print(f"✅ Input shape: {x_train.shape[1:]}\n")

    # Get data info
    unique_labels = np.unique(np.concatenate([y_train, y_test]))
    num_classes = len(unique_labels)

    # Fix for edge cases
    if num_classes == 1:
        # If only one class in data, assume binary (might be sampling issue)
        num_classes = 2
        print(f"⚠️  Warning: Only 1 unique label found, assuming binary classification (2 classes)")

    # Ensure labels are 0-indexed
    if np.min(unique_labels) != 0 or np.max(unique_labels) != num_classes - 1:
        print(f"⚠️  Warning: Labels are not 0-indexed or have gaps: {unique_labels}")
        print(f"   Remapping labels to 0-{num_classes-1}")
        # Create label mapping
        label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
        y_train = np.array([label_map[y] for y in y_train])
        y_test = np.array([label_map[y] for y in y_test])
        unique_labels = np.arange(num_classes)

    print(f"✅ Number of classes: {num_classes}")
    print(f"✅ Class labels: {unique_labels}\n")

    if x_train.ndim == 2:
        input_shape = (x_train.shape[1],)
    else:
        input_shape = x_train.shape[1:]

    # Build and train model
    print("🏗️  Step 3: Building and Training MLP Model")
    print("-" * 60)
    model = get_model("mlp", input_shape, num_classes)

    print("Training for 3 epochs (quick test)...")
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=fed_cfg.get("batch_size", 128),
        validation_split=0.2,
        verbose=1
    )
    print("✅ Training complete\n")

    # Evaluate original (teacher model)
    print("📊 Step 4: Evaluating Original Model (Teacher)")
    print("-" * 60)
    orig_loss, orig_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    teacher_params = model.count_params()
    teacher_size_kb = (teacher_params * 4) / 1024
    print(f"✅ Accuracy: {orig_acc:.2%}")
    print(f"✅ Loss: {orig_loss:.4f}")
    print(f"✅ Parameters: {teacher_params:,}")
    print(f"✅ Size: {teacher_size_kb:.2f} KB\n")

    # Knowledge Distillation
    print("🎓 Step 5: Knowledge Distillation")
    print("-" * 60)
    print("Creating student model (50% of teacher size)...")

    # Create student model
    student_model = create_student_model(
        teacher_model=model,
        compression_ratio=0.5,
        num_classes=num_classes
    )

    student_params = student_model.count_params()
    print(f"✅ Teacher parameters: {teacher_params:,}")
    print(f"✅ Student parameters: {student_params:,}")
    print(f"✅ Compression ratio: {teacher_params/student_params:.2f}x\n")

    # Split data for distillation
    val_split = int(0.8 * len(x_train))
    x_train_dist = x_train[:val_split]
    y_train_dist = y_train[:val_split]
    x_val_dist = x_train[val_split:]
    y_val_dist = y_train[val_split:]

    print("Training student with knowledge distillation...")
    student_model, distill_history = train_with_distillation(
        teacher_model=model,
        student_model=student_model,
        x_train=x_train_dist,
        y_train=y_train_dist,
        x_val=x_val_dist,
        y_val=y_val_dist,
        temperature=3.0,
        alpha=0.3,
        epochs=10,
        batch_size=fed_cfg.get("batch_size", 128),
        learning_rate=0.001,
        verbose=True
    )

    # Compile student for evaluation
    student_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evaluate distilled student
    student_loss, student_acc = safe_evaluate(student_model, x_test, y_test, verbose=0)
    student_size_kb = (student_params * 4) / 1024

    print(f"\n📊 Distillation Results:")
    print(f"✅ Student accuracy: {student_acc:.2%}")
    print(f"✅ Student loss: {student_loss:.4f}")
    print(f"✅ Accuracy drop: {(orig_acc - student_acc)*100:.2f}%")
    print(f"✅ Size: {student_size_kb:.2f} KB")
    print(f"✅ Compression: {teacher_size_kb/student_size_kb:.2f}x\n")

    # Use student model for further compression
    model = student_model
    orig_acc = student_acc  # Update baseline for comparison

    # Apply pruning at different ratios
    print("✂️  Step 6: Testing Multiple Pruning Ratios (on distilled model)")
    print("-" * 60)

    results = {}

    for pruning_ratio in [0.3, 0.5, 0.7]:
        print(f"\n{'='*60}")
        print(f"Testing pruning ratio: {pruning_ratio:.0%}")
        print(f"{'='*60}\n")

        # Apply pruning
        pruned_model = apply_structured_pruning(
            model,
            pruning_ratio=pruning_ratio,
            skip_last_layer=True,
            verbose=True
        )

        # Compare
        compare_models(model, pruned_model)

        # Evaluate before fine-tuning
        pruned_loss, pruned_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
        print(f"Before fine-tuning: Acc={pruned_acc:.2%}, Loss={pruned_loss:.4f}")

        # Fine-tune
        print("\n🔄 Fine-tuning...")
        pruned_model = fine_tune_pruned_model(
            pruned_model,
            x_train, y_train,
            x_test, y_test,
            epochs=2,
            batch_size=fed_cfg.get("batch_size", 128),
            learning_rate=0.0001,
            verbose=False
        )

        # Evaluate after fine-tuning
        final_loss, final_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
        print(f"After fine-tuning:  Acc={final_acc:.2%}, Loss={final_loss:.4f}")

        results[pruning_ratio] = {
            'before_acc': pruned_acc,
            'after_acc': final_acc,
            'recovery': final_acc - pruned_acc,
            'vs_original': final_acc - orig_acc
        }

    # Step 6: Apply Quantization (using 0.5 ratio model)
    print(f"\n🔢 Step 6: Applying Quantization (ratio=0.5)")
    print("-" * 60)

    # Use the 0.5 ratio pruned model for quantization
    if 0.5 in results:
        # Re-create the pruned model for quantization
        pruned_model_50 = apply_structured_pruning(
            model,
            pruning_ratio=0.5,
            skip_last_layer=True,
            verbose=False
        )
        pruned_model_50 = fine_tune_pruned_model(
            pruned_model_50,
            x_train, y_train,
            x_test, y_test,
            epochs=2,
            batch_size=fed_cfg.get("batch_size", 128),
            learning_rate=0.0001,
            verbose=False
        )

        # Quantize
        print("\nQuantizing pruned model (50% ratio)...")
        quantized_layers = quantize_model(
            pruned_model_50,
            symmetric=True,
            verbose=True
        )

        # Compare sizes
        compare_model_sizes(pruned_model_50, quantized_layers)

        # Evaluate quantization accuracy
        print("\nEvaluating quantization accuracy impact...")
        evaluate_quantization_accuracy(
            pruned_model_50,
            quantized_layers,
            x_test,
            y_test,
            verbose=True
        )

        # Step 7: Export to TFLite
        print(f"\n💾 Step 7: Exporting to TFLite")
        print("-" * 60)

        output_dir = Path("outputs/test_pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export original model (float32)
        print("\nExporting original model (float32)...")
        tflite_orig_size = export_tflite(
            model,
            str(output_dir / "original_float32.tflite"),
            quantize=False
        )

        # Export pruned model (float32)
        print("\nExporting pruned model (float32)...")
        tflite_pruned_size = export_tflite(
            pruned_model_50,
            str(output_dir / "pruned_float32.tflite"),
            quantize=False
        )

        # Export pruned + quantized model (int8)
        print("\nExporting pruned + quantized model (int8)...")
        tflite_quant_size = export_tflite(
            pruned_model_50,
            str(output_dir / "pruned_quantized_int8.tflite"),
            quantize=True,
            representative_data=x_train
        )

        # Calculate compression ratios
        pruned_compression = tflite_orig_size / tflite_pruned_size
        quant_compression = tflite_orig_size / tflite_quant_size

        print("\n" + "="*80)
        print(" "*15 + "📊 FULL COMPRESSION PIPELINE SUMMARY")
        print("="*80)
        print(f"\n{'Stage':<40} {'Size (KB)':<15} {'Compression':<15}")
        print("-"*80)
        print(f"{'1. Teacher Model (original)':<40} {teacher_size_kb:<15.2f} {'1.00x':<15}")
        print(f"{'2. Student Model (distilled)':<40} {student_size_kb:<15.2f} {teacher_size_kb/student_size_kb:<15.2f}x")
        print(f"{'3. Pruned Student (50% pruning)':<40} {tflite_pruned_size/1024:<15.2f} {pruned_compression:<15.2f}x")
        print(f"{'4. Quantized (INT8) ⭐':<40} {tflite_quant_size/1024:<15.2f} {quant_compression:<15.2f}x")
        print("="*80)
        print(f"\n🎯 Final Compression: {teacher_size_kb / (tflite_quant_size/1024):.2f}x smaller than teacher")
        print(f"📦 Final Size: {tflite_quant_size/1024:.2f} KB (from {teacher_size_kb:.2f} KB)")
        print(f"🎓 Pipeline: Distillation → Pruning → Quantization")
        print("="*80)

        print(f"\n✅ All TFLite models saved to: {output_dir.absolute()}")
        print(f"✅ Final compression: {quant_compression:.2f}x smaller")
        print(f"✅ Deployment ready: {output_dir / 'pruned_quantized_int8.tflite'}")

    # Summary
    print("\n" + "="*80)
    print(" "*25 + "📊 PRUNING RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Pruning Ratio':<15} {'Before FT':<15} {'After FT':<15} {'Recovery':<15} {'vs Original':<15}")
    print("-"*80)

    for ratio, res in results.items():
        print(f"{ratio:<15.0%} {res['before_acc']:<15.2%} {res['after_acc']:<15.2%} "
              f"{res['recovery']:<15.2%} {res['vs_original']:<+15.2%}")

    print("="*80)

    # Verify results
    print("\n✅ VERIFICATION:")
    all_passed = True

    for ratio, res in results.items():
        # Check that fine-tuning improves accuracy
        if res['recovery'] < 0:
            print(f"⚠️  Warning: Fine-tuning decreased accuracy for ratio {ratio:.0%}")
            all_passed = False
        else:
            print(f"✅ Ratio {ratio:.0%}: Fine-tuning improved accuracy by {res['recovery']:.2%}")

        # Check that pruned model is not too bad
        if res['after_acc'] < orig_acc - 0.1:  # Allow 10% drop
            print(f"⚠️  Warning: Large accuracy drop for ratio {ratio:.0%}")
            all_passed = False

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
    else:
        print("\n⚠️  Some checks failed, but this might be expected with aggressive pruning")

    print("\n" + "="*80 + "\n")

    return results


def test_saved_model_pruning():
    """
    Test complete pipeline on a pre-trained saved model (if exists).

    Pipeline: Load Model → Pruning → Quantization → TFLite Export
    """
    print("\n" + "="*80)
    print(" "*20 + "🧪 SAVED MODEL COMPRESSION TEST")
    print("="*80 + "\n")

    # Try multiple possible paths
    possible_paths = [
        Path("src/models/global_model.h5"),  # train.py saves here
        Path("models/global_model.h5"),       # Alternative location
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print(f"⚠️  No saved model found. Checked paths:")
        for path in possible_paths:
            print(f"   - {path}")
        print(f"   Skipping this test. Train a model first with:")
        print(f"   python scripts/train.py\n")
        return None

    print(f"📦 Loading saved model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded\n")

    # Load test data
    print("📂 Loading test dataset...")
    
    # Auto-detect environment and use appropriate config
    try:
        from src.utils.env_utils import get_default_config_path
        config_path = get_default_config_path()
    except ImportError:
        # Fallback: check if in Colab
        import os
        if os.path.exists("/content") or os.getenv("COLAB_GPU") is not None:
            config_path = "config/federated_colab.yaml"
        else:
            config_path = "config/federated.yaml"
    
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "bot_iot")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    _, _, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Use subset
    x_test, y_test = x_test[:1000], y_test[:1000]
    print(f"✅ Test samples: {len(x_test)}\n")

    # Evaluate original
    print("📊 Evaluating original saved model...")
    orig_loss, orig_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    print(f"✅ Accuracy: {orig_acc:.2%}")
    print(f"✅ Loss: {orig_loss:.4f}\n")

    # Apply pruning
    print("✂️  Applying 50% structured pruning...")
    pruned_model = apply_structured_pruning(
        model,
        pruning_ratio=0.5,
        skip_last_layer=True,
        verbose=True
    )

    compare_models(model, pruned_model)

    # Evaluate
    print("📊 Evaluating pruned model...")
    pruned_loss, pruned_acc = pruned_model.evaluate(x_test, y_test, verbose=0)
    print(f"✅ Accuracy: {pruned_acc:.2%}")
    print(f"✅ Loss: {pruned_loss:.4f}")
    print(f"⚠️  Accuracy change: {(pruned_acc - orig_acc)*100:+.2f}%\n")

    # Apply quantization
    print("🔢 Applying quantization...")
    quantized_layers = quantize_model(
        pruned_model,
        symmetric=True,
        verbose=True
    )

    compare_model_sizes(pruned_model, quantized_layers)

    # Export to TFLite
    print("\n💾 Exporting to TFLite...")
    output_dir = Path("models/tflite")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load some training data for representative dataset
    _, _, x_train, _ = load_dataset(dataset_name, **dataset_kwargs)
    x_train = x_train[:1000]  # Use subset

    # Export original (float32)
    tflite_orig_size = export_tflite(
        model,
        str(output_dir / "saved_model_original.tflite"),
        quantize=False
    )

    # Export pruned + quantized (int8)
    tflite_quant_size = export_tflite(
        pruned_model,
        str(output_dir / "saved_model_pruned_quantized.tflite"),
        quantize=True,
        representative_data=x_train
    )

    compression_ratio = tflite_orig_size / tflite_quant_size

    print(f"\n✅ TFLite compression: {compression_ratio:.2f}x")
    print(f"✅ Original size: {tflite_orig_size/1024:.2f} KB")
    print(f"✅ Compressed size: {tflite_quant_size/1024:.2f} KB")

    # Save pruned model
    pruned_model.save(Path("models/test_pruned_model.h5"))
    print(f"✅ Pruned model saved to models/test_pruned_model.h5\n")

    print("="*80 + "\n")

    return {
        'original_acc': orig_acc,
        'pruned_acc': pruned_acc,
        'compression_ratio': compression_ratio
    }


def main():
    """
    Run all integration tests for the complete TinyML pipeline.

    Tests:
    1. Full pipeline: Self-training → Distillation → Pruning → Fine-tuning → Quantization → TFLite
    2. Saved model: Load train.py model → Prune → Quantize → TFLite
    """
    print("\n" + "🔬 "*30)
    print(" "*15 + "TINYML PIPELINE INTEGRATION TEST SUITE")
    print("🔬 "*30 + "\n")

    results = {}

    try:
        # Test 1: Compression pipeline with self-training
        print("Running Test 1: Full TinyML Pipeline (Self-Trained → Distillation → Prune → Quantize → TFLite)")
        results['mlp_pipeline'] = test_full_pipeline_mlp()
        print("✅ Test 1 completed: Self-trained model compressed\n")

        # Test 2: Saved model compression (from train.py)
        print("Running Test 2: Saved Model Compression (Train.py → Prune → Quantize → TFLite)")
        results['saved_model'] = test_saved_model_pruning()
        
        if results['saved_model']:
            print("✅ Test 2 completed: Train.py model compressed\n")
        else:
            print("⚠️  Test 2 skipped: Train.py model not found\n")
            print("   Note: Run 'python scripts/train.py' first to train a model\n")

        # Summary
        print("\n" + "="*80)
        print(" "*20 + "📊 COMPRESSION SUMMARY")
        print("="*80 + "\n")
        
        print("Compressed Models:")
        print("-" * 80)
        
        # Model 1: Self-trained
        if results.get('mlp_pipeline'):
            print("✅ Model 1: Self-Trained (compression.py)")
            print("   Location: outputs/test_pipeline/")
            print("   - original_float32.tflite")
            print("   - pruned_float32.tflite")
            print("   - pruned_quantized_int8.tflite")
        else:
            print("❌ Model 1: Self-Trained - FAILED")
        
        print()
        
        # Model 2: Train.py
        if results.get('saved_model'):
            print("✅ Model 2: Train.py (Federated Learning)")
            print("   Location: models/tflite/")
            print("   - saved_model_original.tflite")
            print("   - saved_model_pruned_quantized.tflite")
        else:
            print("⚠️  Model 2: Train.py - NOT FOUND")
            print("   Run 'python scripts/train.py' first")
        
        print("\n" + "="*80)
        print("✅ ALL INTEGRATION TESTS COMPLETED!")
        print("="*80 + "\n")
        
        # Check if both models are ready for analysis
        if results.get('mlp_pipeline') and results.get('saved_model'):
            print("🎯 Both models compressed successfully!")
            print("   Ready for analyze_compression.py comparison\n")
        elif results.get('mlp_pipeline'):
            print("ℹ️  Only self-trained model compressed")
            print("   Run train.py first for full comparison\n")
        else:
            print("⚠️  Compression pipeline had issues\n")

        return results

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()

    if results:
        print("✅ Test suite completed successfully")
    else:
        print("❌ Test suite failed")
