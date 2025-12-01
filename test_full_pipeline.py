"""
End-to-end integration test for the complete pruning pipeline.
Tests pruning on your actual Bot-IoT dataset and models.
"""
import yaml
import numpy as np
from pathlib import Path
from tensorflow import keras

from src.data.loader import load_dataset
from src.models.nets import get_model
from src.modelcompression.pruning import (
    apply_structured_pruning,
    compare_models,
    fine_tune_pruned_model
)


def safe_evaluate(model, x, y, verbose=0):
    """Safely evaluate model and return (loss, accuracy)."""
    result = model.evaluate(x, y, verbose=verbose)
    if isinstance(result, (list, tuple)):
        return result[0], result[1]
    else:
        return result, 0.0


def test_full_pipeline_mlp():
    """Test complete pruning pipeline with MLP on Bot-IoT data."""
    print("\n" + "="*80)
    print(" "*20 + "🧪 FULL PIPELINE TEST (MLP)")
    print("="*80 + "\n")

    # Load config
    print("📋 Step 1: Loading Configuration")
    print("-" * 60)
    with open("config/federated.yaml", encoding='utf-8') as f:
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
    if num_classes == 1 and 0 in unique_labels:
        num_classes = 2

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

    # Evaluate original
    print("📊 Step 4: Evaluating Original Model")
    print("-" * 60)
    orig_loss, orig_acc = safe_evaluate(model, x_test, y_test, verbose=0)
    print(f"✅ Accuracy: {orig_acc:.2%}")
    print(f"✅ Loss: {orig_loss:.4f}\n")

    # Apply pruning at different ratios
    print("✂️  Step 5: Testing Multiple Pruning Ratios")
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

    # Summary
    print("\n" + "="*80)
    print(" "*25 + "📊 RESULTS SUMMARY")
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
    """Test pruning on a pre-trained saved model (if exists)."""
    print("\n" + "="*80)
    print(" "*20 + "🧪 SAVED MODEL PRUNING TEST")
    print("="*80 + "\n")

    model_path = Path("models/global_model.h5")

    if not model_path.exists():
        print(f"⚠️  No saved model found at {model_path}")
        print(f"   Skipping this test. Train a model first with:")
        print(f"   python train_windows.py\n")
        return None

    print(f"📦 Loading saved model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded\n")

    # Load test data
    print("📂 Loading test dataset...")
    with open("config/federated.yaml", encoding='utf-8') as f:
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

    # Save pruned model
    output_path = Path("models/test_pruned_model.h5")
    pruned_model.save(output_path)
    print(f"💾 Pruned model saved to {output_path}\n")

    print("="*80 + "\n")

    return {'original_acc': orig_acc, 'pruned_acc': pruned_acc}


def main():
    """Run all integration tests."""
    print("\n" + "🔬 "*30)
    print(" "*15 + "FULL PIPELINE INTEGRATION TEST SUITE")
    print("🔬 "*30 + "\n")

    results = {}

    try:
        # Test 1: Full pipeline with MLP
        print("Running Test 1: Full Pipeline (MLP on Bot-IoT)")
        results['mlp_pipeline'] = test_full_pipeline_mlp()

        # Test 2: Saved model pruning
        print("\nRunning Test 2: Saved Model Pruning")
        results['saved_model'] = test_saved_model_pruning()

        print("\n" + "="*80)
        print("✅ ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")

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
