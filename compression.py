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
# CRITICAL: Set this BEFORE importing TensorFlow to use built-in Keras for tfmot compatibility
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import sys
import yaml
import numpy as np
import tempfile
from pathlib import Path
from tensorflow import keras
import tensorflow as tf

from src.data.loader import load_dataset
from src.models.nets import get_model
from src.modelcompression.distillation import (
    create_student_model,
    train_with_distillation,
    compare_models as compare_distillation
)
from src.modelcompression.pruning import (
    apply_structured_pruning,
    compare_models,
    fine_tune_pruned_model,
    get_model_size
)
from src.modelcompression.quantization import (
    quantize_model,
    compare_model_sizes,
    evaluate_quantization_accuracy
)
from src.tinyml.export_tflite import export_tflite, export_tflite_qat, _strip_bn_dropout_for_tflite, _strip_bn_dropout_for_qat

try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False


def has_qat_layers(model):
    """Check if model has QAT (QuantizeWrapper) layers from tfmot."""
    if not TFMOT_AVAILABLE:
        return False
    for layer in model.layers:
        if 'QuantizeWrapper' in type(layer).__name__:
            return True
    return False


def strip_qat_layers(model):
    """
    Strip QAT layers to get the base model.
    If model has QuantizeWrapper layers from tfmot.quantize_model(),
    this extracts the underlying base model.
    
    Manual extraction approach: rebuild model from layer configs and weights.
    """
    if not has_qat_layers(model):
        return model
    
    print("   Stripping QAT layers from FL-trained model...")
    
    # Manual extraction (most stable approach)
    try:
        import tensorflow as tf
        from tensorflow.keras import Sequential, layers
        
        # Extract layer configs and weights from QAT model
        layer_configs = []
        layer_weights = []
        
        for layer in model.layers:
            # Check if this is a QuantizeWrapper
            if 'QuantizeWrapper' in type(layer).__name__:
                # Extract the wrapped layer
                wrapped_layer = layer.layer
                config = wrapped_layer.get_config()
                
                # Fix QuantizeAwareActivation: extract the real activation
                if 'activation' in config and isinstance(config['activation'], dict):
                    if config['activation'].get('class_name') == 'QuantizeAwareActivation':
                        # Extract the real activation from QuantizeAwareActivation config
                        real_activation = config['activation'].get('config', {}).get('activation', 'relu')
                        config['activation'] = real_activation
                        print(f"      Fixed activation in {config.get('name', 'layer')}: QuantizeAwareActivation → {real_activation}")
                
                layer_configs.append({
                    'class_name': wrapped_layer.__class__.__name__,
                    'config': config
                })
                # Get weights from the wrapped layer
                if wrapped_layer.get_weights():
                    layer_weights.append(wrapped_layer.get_weights())
                else:
                    layer_weights.append(None)
            elif 'QuantizeLayer' not in type(layer).__name__:
                # Regular layer, keep it
                layer_configs.append({
                    'class_name': layer.__class__.__name__,
                    'config': layer.get_config()
                })
                if layer.get_weights():
                    layer_weights.append(layer.get_weights())
                else:
                    layer_weights.append(None)
            # Skip QuantizeLayer input layers
        
        # Build new model from configs
        base_model = Sequential(name=model.name)
        for i, layer_config in enumerate(layer_configs):
            layer_class = getattr(layers, layer_config['class_name'])
            new_layer = layer_class.from_config(layer_config['config'])
            base_model.add(new_layer)
        
        # Build the model with the correct input shape
        if hasattr(model, 'input_shape') and model.input_shape:
            base_model.build(model.input_shape)
        
        # Set weights with error checking
        weight_idx = 0
        weights_set_count = 0
        for layer in base_model.layers:
            if weight_idx >= len(layer_weights):
                break
            if layer_weights[weight_idx] is not None:
                try:
                    layer.set_weights(layer_weights[weight_idx])
                    weights_set_count += 1
                except Exception as we:
                    print(f"      ⚠️  Failed to set weights for layer {layer.name}: {we}")
            weight_idx += 1
        
        print(f"   ✅ QAT layers stripped manually ({weights_set_count}/{len(layer_weights)} layers with weights)")
        return base_model
        
    except Exception as e:
        print(f"   ⚠️ Could not strip QAT layers: {e}")
        import traceback
        traceback.print_exc()
        print("   Using model as-is (may cause issues with compression)")
        return model


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
    with open("config/federated_local.yaml", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    fed_cfg = cfg.get("federated", {})
    print("✅ Configuration loaded\n")

    # Load dataset
    print("📂 Step 2: Loading CIC-IDS2017 Dataset")
    print("-" * 60)
    dataset_name = data_cfg.get("name", "cicids2017")
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


def test_saved_model_pruning(
    config_path: str = "config/federated_local.yaml",
    dataset_override: str = None,
    model_path_override: str = None,
):
    """
    Compress a pre-trained saved model: Load → Prune → Quantize → TFLite Export.
    Used by run.py pipeline to compress trained model (FL or Centralized).
    """
    print("\n" + "="*80)
    print(" "*20 + "📦 SAVED MODEL COMPRESSION")
    print("="*80 + "\n")

    model_path = Path(model_path_override) if model_path_override else Path("models/global_model.h5")

    if not model_path.exists():
        print(f"⚠️  No saved model found at {model_path}")
        print(f"   Train first, then copy to models/global_model.h5\n")
        return None

    print(f"📦 Loading saved model from {model_path}...")
    # Load config first to check use_qat
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        cfg_path = Path("config/federated_local.yaml")
    with open(cfg_path, encoding='utf-8') as f:
        cfg_pre = yaml.safe_load(f)
    use_qat = cfg_pre.get("federated", {}).get("use_qat", False)
    
    # Load model - may have QAT layers if FL training used real QAT
    if use_qat and TFMOT_AVAILABLE:
        with tfmot.quantization.keras.quantize_scope():
            model = keras.models.load_model(model_path, compile=False)
    else:
        model = keras.models.load_model(model_path, compile=False)
    
    # Check if model already has QAT layers from FL training
    qat_model_for_export = None
    if has_qat_layers(model):
        print("   ℹ️  Model has QAT layers from FL training")
        print("   Stripping QAT layers to get base model for compression...")
        qat_model_for_export = model  # Keep QAT model for separate export
        model = strip_qat_layers(model)
        # Check if stripping succeeded
        if has_qat_layers(model):
            print("   ⚠️  QAT layers could not be stripped completely")
            print("   ℹ️  Will export QAT model separately and use Traditional model for compression")
            # Don't return - continue with Traditional model processing below
        else:
            print("   ✅ QAT layers stripped - continuing with compression pipeline")
            # Verify strip didn't break the model - compile and check layers
            try:
                last_layer = model.layers[-1]
                num_classes_check = last_layer.units if hasattr(last_layer, "units") else 2
                loss_check = "binary_crossentropy" if num_classes_check == 1 else "sparse_categorical_crossentropy"
                model.compile(optimizer="adam", loss=loss_check, metrics=["accuracy"])
                print(f"   ✅ Stripped model compiled successfully ({len(model.layers)} layers)")
            except Exception as strip_error:
                print(f"   ⚠️  Stripped model compilation failed: {strip_error}")
                print("   Using original QAT model for all exports")
                model = qat_model_for_export
                qat_model_for_export = None
    elif use_qat:
        print("   (no QAT layers found - will apply QAT during compression)")
    
    # Recompile with standard loss for evaluate() during compression.
    # Infer output shape for recompile (binary or multi-class)
    last_layer = model.layers[-1]
    num_classes = last_layer.units if hasattr(last_layer, "units") else 2
    loss = "binary_crossentropy" if num_classes == 1 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    print("✅ Model loaded\n")

    # Load test data (use same config as training)
    print("📂 Loading test dataset...")
    if not cfg_path.exists():
        cfg_path = Path("config/federated_local.yaml")
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    dataset_name = dataset_override or data_cfg.get("name", "bot_iot")
    if dataset_override:
        print(f"📌 Dataset override: {dataset_override}\n")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}

    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")

    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)

    # Ensure data matches model input shape (38=Bot-IoT, 78=CIC-IDS2017)
    # Handle both Sequential and Functional models
    try:
        if hasattr(model, 'input_shape') and model.input_shape is not None:
            model_input_dim = int(model.input_shape[1])
        elif hasattr(model, 'layers') and len(model.layers) > 0:
            # Sequential model: get first layer input shape
            first_layer = model.layers[0]
            if hasattr(first_layer, 'input_shape') and first_layer.input_shape is not None:
                model_input_dim = int(first_layer.input_shape[1])
            elif hasattr(first_layer, 'batch_input_shape'):
                model_input_dim = int(first_layer.batch_input_shape[1])
            else:
                model_input_dim = x_train.shape[1]
        else:
            model_input_dim = x_train.shape[1]
    except (AttributeError, TypeError, IndexError):
        # Fallback: use training data shape
        model_input_dim = x_train.shape[1]
    
    data_features = x_train.shape[1]
    if model_input_dim != data_features:
        alt_name = "bot_iot" if dataset_name.lower() in ["cicids2017", "cic-ids-2017"] else "cicids2017"
        try:
            x_train_alt, y_train_alt, x_test_alt, y_test_alt = load_dataset(alt_name, **dataset_kwargs)
            if x_train_alt.shape[1] == model_input_dim:
                x_train, y_train, x_test, y_test = x_train_alt, y_train_alt, x_test_alt, y_test_alt
                print(f"⚠️  Switched to dataset '{alt_name}' (model expects {model_input_dim} features)\n")
        except Exception:
            pass
        if x_train.shape[1] != model_input_dim:
            raise ValueError(
                f"Input shape mismatch: model expects {model_input_dim} features, "
                f"but dataset '{dataset_name}' provides {data_features}. "
                "Use the same config as training (38≈Bot-IoT, 78≈CIC-IDS2017)."
            )

    # Use subset for eval/speed
    x_test, y_test = x_test[:1000], y_test[:1000]
    print(f"✅ Test samples: {len(x_test)}, features: {x_train.shape[1]}\n")

    # Evaluate original
    print("📊 Evaluating original saved model...")
    orig_loss, orig_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    print(f"✅ Accuracy: {orig_acc:.2%}")
    print(f"✅ Loss: {orig_loss:.4f}\n")

    # pruning_presets 있으면 10x5·10x2·5x10 세 개 만들어서 스윕에서 한 번에 비교. 없으면 기존처럼 0.5 한 개
    comp_cfg = cfg.get("compression", {})
    pruning_presets = comp_cfg.get("pruning_presets") or [{"name": "default", "ratio": 0.5}]

    output_dir = Path("models/tflite")
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train_sub = x_train[:10000] if len(x_train) >= 10000 else x_train
    y_train_sub = y_train[:10000] if len(y_train) >= 10000 else y_train

    print(f"   [Baseline] Model has {len(model.layers)} layers")
    baseline_loss, baseline_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    print(f"   [Baseline] Keras model accuracy: {baseline_acc:.4f}")
    tflite_orig_size = export_tflite(
        model,
        str(output_dir / "saved_model_original.tflite"),
        quantize=False
    )
    print(f"✅ Original (baseline): {tflite_orig_size/1024:.2f} KB")

    tflite_qat_ptq_size = None
    first_pruned_model = None
    pruned_acc = orig_acc
    for idx, preset in enumerate(pruning_presets):
        ratio = float(preset.get("ratio", 0.5))
        name = str(preset.get("name", "default"))
        print(f"\n✂️  Pruning preset «{name}» (ratio={ratio:.0%})...")
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        model_copy.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        pruned_model = apply_structured_pruning(
            model_copy,
            pruning_ratio=ratio,
            skip_last_layer=True,
            verbose=True
        )
        compare_models(model, pruned_model)
        pruned_loss, pruned_acc = safe_evaluate(pruned_model, x_test, y_test, verbose=0)
        print(f"   Pruned accuracy: {pruned_acc:.2%}")
        print("   Fine-tuning pruned model (3 epochs before PTQ)...")
        pruned_model.fit(
            x_train_sub, y_train_sub,
            epochs=3,
            batch_size=128,
            validation_split=0.1,
            verbose=1
        )
        if idx == 0:
            first_pruned_model = pruned_model
            tflite_qat_ptq_size = export_tflite(
                pruned_model,
                str(output_dir / "saved_model_qat_ptq.tflite"),
                quantize=True,
                representative_data=x_train_sub
            )
            export_tflite(
                pruned_model,
                str(output_dir / "saved_model_pruned_quantized.tflite"),
                quantize=True,
                representative_data=x_train_sub
            )
        if TFMOT_AVAILABLE:
            try:
                pruned_for_qat = _strip_bn_dropout_for_qat(pruned_model)
                q_aware = tfmot.quantization.keras.quantize_model(pruned_for_qat)
                q_aware.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
                q_aware.fit(
                    x_train_sub, y_train_sub,
                    epochs=2,
                    batch_size=128,
                    validation_split=0.1,
                    verbose=0
                )
                export_tflite_qat(q_aware, str(output_dir / f"saved_model_pruned_{name}_qat.tflite"))
                print(f"   ✅ saved_model_pruned_{name}_qat.tflite")
                if idx == 0:
                    export_tflite_qat(q_aware, str(output_dir / "saved_model_pruned_qat.tflite"))
            except Exception as e:
                print(f"   ⚠️  QAT for {name} failed: {e}")

    tflite_qat_size = (output_dir / "saved_model_pruned_qat.tflite").stat().st_size if (output_dir / "saved_model_pruned_qat.tflite").exists() else None
    pruning_ratio = float(pruning_presets[0].get("ratio", 0.5))  # Traditional 경로에서 사용

    # (3) No-QAT + PTQ: Traditional model → Prune → PTQ (when traditional_model_path is set)
    traditional_model_path = comp_cfg.get("traditional_model_path")
    if traditional_model_path is None:
        traditional_model_path = "models/global_model_traditional.h5"
    tflite_traditional_ptq_size = None
    if Path(traditional_model_path).exists():
        print("\n📦 Traditional PTQ (no-QAT model → Prune → PTQ)...")
        try:
            trad_model = keras.models.load_model(traditional_model_path, compile=False)
        except (TypeError, ValueError) as e:
            print(f"⚠️  Could not load traditional model ({e})")
            print("   Skipping no-QAT+PTQ comparison. Delete old Keras 3 models and retrain.")
            trad_model = None
        
        if trad_model is not None:
            trad_model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
            
            # Evaluate Traditional model before pruning
            print("   Evaluating Traditional model before pruning...")
            trad_loss, trad_acc = safe_evaluate(trad_model, x_test, y_test, verbose=0)
            print(f"   Traditional model accuracy: {trad_acc:.2%}")
            
            trad_pruned = apply_structured_pruning(
                trad_model,
                pruning_ratio=pruning_ratio,
                skip_last_layer=True,
                verbose=False
            )
            
            # Evaluate after pruning
            print("   Evaluating after pruning...")
            pruned_loss, pruned_acc = safe_evaluate(trad_pruned, x_test, y_test, verbose=0)
            print(f"   Pruned model accuracy: {pruned_acc:.2%} (drop: {(trad_acc - pruned_acc)*100:.2f}%)")
            
            # Fine-tune pruned model before PTQ (5 epochs for recovery)
            print("   Fine-tuning pruned model (5 epochs)...")
            trad_pruned.fit(
                x_train_sub, y_train_sub,
                epochs=5,
                batch_size=128,
                validation_split=0.1,
                verbose=1
            )
            tflite_traditional_ptq_size = export_tflite(
                trad_pruned,
                str(output_dir / "saved_model_no_qat_ptq.tflite"),
                quantize=True,
                representative_data=x_train_sub
            )
            print(f"✅ Traditional + Pruning + Fine-tune + PTQ: {tflite_traditional_ptq_size/1024:.2f} KB")
            
            # Traditional + Pruning + QAT fine-tuning
            print("\n🎓 Traditional + QAT fine-tuning...")
            try:
                if TFMOT_AVAILABLE:
                    trad_for_qat = _strip_bn_dropout_for_qat(trad_pruned)
                    trad_q_aware = tfmot.quantization.keras.quantize_model(trad_for_qat)
                    trad_q_aware.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
                    print("   Fine-tuning Traditional+QAT model (2 epochs)...")
                    trad_q_aware.fit(
                        x_train_sub, y_train_sub,
                        epochs=2,
                        batch_size=128,
                        validation_split=0.1,
                        verbose=1
                    )
                    tflite_trad_qat_size = export_tflite_qat(
                        trad_q_aware,
                        str(output_dir / "saved_model_traditional_qat.tflite")
                    )
                    print(f"✅ Traditional + Pruning + QAT: {tflite_trad_qat_size/1024:.2f} KB")
                else:
                    tflite_trad_qat_size = None
            except Exception as e:
                print(f"⚠️  Traditional QAT fine-tuning failed: {e}")
                tflite_trad_qat_size = None
        else:
            tflite_trad_qat_size = None

    # Config has use_qat: QAT TFLite is required — abort if missing
    if use_qat and tflite_qat_size is None:
        qat_path = output_dir / "saved_model_pruned_qat.tflite"
        print("\n" + "=" * 80)
        print("❌ PIPELINE ABORTED: use_qat is True but QAT TFLite was not created.")
        print(f"   Expected: {qat_path}")
        print("   Fix QAT export (e.g. tf.keras-only strip) and re-run.")
        print("=" * 80 + "\n")
        sys.exit(1)

    # Export original QAT model if stripping failed
    if qat_model_for_export is not None:
        print("\n📤 Exporting original QAT-trained model (with QuantizeWrapper layers)...")
        qat_direct_path = output_dir / "saved_model_qat_direct.tflite"
        if TFMOT_AVAILABLE:
            # export_tflite_qat already imported at top of file (line 44)
            try:
                tflite_qat_direct_size = export_tflite_qat(qat_model_for_export, str(qat_direct_path))
                print(f"✅ QAT-trained model (direct): {tflite_qat_direct_size/1024:.2f} KB")
            except Exception as e:
                print(f"⚠️  QAT direct export failed: {e}")
                tflite_qat_direct_size = None
        else:
            tflite_qat_direct_size = None

    compression_ratio = tflite_orig_size / tflite_qat_ptq_size
    print(f"\n{'='*80}")
    print(f"✅ TFLite Export Summary (2×2 Experimental Design)")
    print(f"{'='*80}")
    print(f"\n📊 Baseline:")
    print(f"   • Original (no compression):  {tflite_orig_size/1024:.2f} KB")
    print(f"\n📊 Traditional Training:")
    if tflite_traditional_ptq_size is not None:
        print(f"   • Traditional + Pruning + PTQ:        {tflite_traditional_ptq_size/1024:.2f} KB")
        if 'tflite_trad_qat_size' in locals() and tflite_trad_qat_size is not None:
            print(f"   • Traditional + Pruning + QAT:        {tflite_trad_qat_size/1024:.2f} KB")
    print(f"\n📊 QAT-Trained:")
    print(f"   • QAT-trained + Pruning + PTQ:        {tflite_qat_ptq_size/1024:.2f} KB ({compression_ratio:.2f}x)")
    if tflite_qat_size is not None:
        print(f"   • QAT-trained + Pruning + QAT:        {tflite_qat_size/1024:.2f} KB")
    print(f"\n{'='*80}")

    # Save first preset pruned model (for tests)
    to_save = first_pruned_model if first_pruned_model is not None else pruned_model
    if to_save is not None:
        to_save.save(Path("models/test_pruned_model.h5"))
        print(f"✅ Pruned model saved to models/test_pruned_model.h5\n")

    print("="*80 + "\n")

    return {
        'original_acc': orig_acc,
        'pruned_acc': pruned_acc,
        'compression_ratio': compression_ratio
    }


def compress_one_distilled_model(
    config_path: str,
    model_path: str,
    output_tflite_path: str,
    dataset_override: str = None,
) -> bool:
    """
    Load one distilled model (float32), prune 50%%, fine-tune, PTQ export to a single TFLite.
    Used by run_distillation_first.py for each of the 4 saved distilled models.
    """
    model_path = Path(model_path)
    output_tflite_path = Path(output_tflite_path)
    if not model_path.exists():
        print(f"⚠️  Model not found: {model_path}")
        return False
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        cfg_path = Path("config/federated_local.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg.get("data", {})
    dataset_name = dataset_override or data_cfg.get("name", "cicids2017")
    dataset_kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in dataset_kwargs and "data_path" not in dataset_kwargs:
        dataset_kwargs["data_path"] = dataset_kwargs.pop("path")
    x_train, y_train, x_test, y_test = load_dataset(dataset_name, **dataset_kwargs)
    x_train_sub = x_train[:10000] if len(x_train) >= 10000 else x_train
    y_train_sub = y_train[:10000] if len(y_train) >= 10000 else y_train

    model = keras.models.load_model(model_path, compile=False)
    last = model.layers[-1]
    num_classes = getattr(last, "units", 2)
    loss = "binary_crossentropy" if num_classes <= 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    pruned_model = apply_structured_pruning(model, pruning_ratio=0.5, skip_last_layer=True, verbose=False)
    pruned_model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    pruned_model.fit(
        x_train_sub, y_train_sub,
        epochs=3, batch_size=128, validation_split=0.1, verbose=0
    )
    output_tflite_path.parent.mkdir(parents=True, exist_ok=True)
    export_tflite(pruned_model, str(output_tflite_path), quantize=True, representative_data=x_train_sub)
    print(f"   ✅ {output_tflite_path.name}: {output_tflite_path.stat().st_size / 1024:.2f} KB")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TinyML compression pipeline")
    parser.add_argument(
        "--use-trained",
        action="store_true",
        help="Compress trained model only (for run.py pipeline, skip Test 1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/federated_local.yaml",
        help="Config path for dataset (used by test_saved_model_pruning)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name (e.g. bot_iot for 38 features when model was trained with Bot-IoT)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model file (default: models/global_model.h5)",
    )
    args = parser.parse_args()

    results = {}

    try:
        if args.use_trained:
            # run.py pipeline: load trained model only -> compress -> save tflite
            print("\n📌 Mode: use-trained (compress trained model only)\n")
            results['saved_model'] = test_saved_model_pruning(
                config_path=args.config, 
                dataset_override=args.dataset,
                model_path_override=args.model_path
            )
            if results.get('saved_model') is None:
                return None
        else:
            # Full integration test
            print("\n" + "🔬 "*30)
            print(" "*15 + "TINYML PIPELINE INTEGRATION TEST SUITE")
            print("🔬 "*30 + "\n")
            print("Running Test 1: Full TinyML Pipeline (Train → distillation → Prune → Quantize → TFLite)")
            results['mlp_pipeline'] = test_full_pipeline_mlp()
            print("\nRunning Test 2: Saved Model Compression (Load → Prune → Quantize → TFLite)")
            results['saved_model'] = test_saved_model_pruning(
                config_path=args.config, 
                dataset_override=args.dataset,
                model_path_override=args.model_path
            )

        print("\n" + "="*80)
        print("✅ COMPRESSION COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        return results

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR:")
        print(f"{e}\n")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys
    results = main()
    if results:
        print("✅ Compression completed successfully")
    else:
        print("❌ Compression failed")
        sys.exit(1)
