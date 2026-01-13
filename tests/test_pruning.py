"""
Unit tests for structured pruning functionality.
Run with: python -m pytest tests/test_pruning.py -v
or simply: python tests/test_pruning.py
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.compression.pruning import (
    calculate_neuron_importance,
    prune_dense_layer,
    prune_conv2d_layer,
    apply_structured_pruning,
    get_model_size
)


def test_calculate_neuron_importance_dense():
    """Test importance calculation for Dense layers."""
    print("\n" + "="*60)
    print("TEST: Calculate Neuron Importance (Dense)")
    print("="*60)

    # Create simple weight matrix
    weights = np.array([
        [1.0, 0.1, 2.0],  # Input neuron 1
        [0.5, 0.2, 1.5],  # Input neuron 2
        [0.3, 0.1, 1.0],  # Input neuron 3
    ])

    importance = calculate_neuron_importance(weights)

    print(f"Weight matrix shape: {weights.shape}")
    print(f"Importance scores: {importance}")
    print(f"Expected: neuron 0 (index 2) should have highest importance")
    print(f"Actual ranking: {np.argsort(importance)[::-1]}")

    assert len(importance) == 3
    assert importance[2] > importance[0] > importance[1]
    print("✅ PASSED\n")


def test_calculate_neuron_importance_conv2d():
    """Test importance calculation for Conv2D layers."""
    print("="*60)
    print("TEST: Calculate Neuron Importance (Conv2D)")
    print("="*60)

    # Create simple conv weights (3x3 kernel, 2 input channels, 3 output filters)
    weights = np.random.randn(3, 3, 2, 3)
    weights[:, :, :, 0] *= 2  # Make filter 0 more important

    importance = calculate_neuron_importance(weights)

    print(f"Conv2D weight shape: {weights.shape}")
    print(f"Importance scores: {importance}")
    print(f"Filter 0 should be most important (has larger weights)")

    assert len(importance) == 3
    assert importance[0] > importance[1]
    print("✅ PASSED\n")


def test_prune_dense_layer():
    """Test pruning Dense layer neurons."""
    print("="*60)
    print("TEST: Prune Dense Layer")
    print("="*60)

    # Create weight matrix: 4 inputs -> 6 neurons
    weights = np.random.randn(4, 6)
    bias = np.random.randn(6)
    pruning_ratio = 0.5  # Remove 50% = 3 neurons, keep 3

    pruned_w, pruned_b, kept_indices = prune_dense_layer(weights, bias, pruning_ratio)

    print(f"Original shape: {weights.shape}")
    print(f"Pruned shape: {pruned_w.shape}")
    print(f"Pruning ratio: {pruning_ratio:.1%}")
    print(f"Kept {len(kept_indices)} out of {weights.shape[1]} neurons")
    print(f"Kept indices: {kept_indices}")

    assert pruned_w.shape == (4, 3)
    assert pruned_b.shape == (3,)
    assert len(kept_indices) == 3
    print("✅ PASSED\n")


def test_prune_conv2d_layer():
    """Test pruning Conv2D layer filters."""
    print("="*60)
    print("TEST: Prune Conv2D Layer")
    print("="*60)

    # Create conv weights: 3x3 kernel, 16 input channels, 32 output filters
    weights = np.random.randn(3, 3, 16, 32)
    bias = np.random.randn(32)
    pruning_ratio = 0.5  # Remove 50% = 16 filters, keep 16

    pruned_w, pruned_b, kept_indices = prune_conv2d_layer(weights, bias, pruning_ratio)

    print(f"Original shape: {weights.shape}")
    print(f"Pruned shape: {pruned_w.shape}")
    print(f"Pruning ratio: {pruning_ratio:.1%}")
    print(f"Kept {len(kept_indices)} out of {weights.shape[3]} filters")

    assert pruned_w.shape == (3, 3, 16, 16)
    assert pruned_b.shape == (16,)
    assert len(kept_indices) == 16
    print("✅ PASSED\n")


def test_apply_structured_pruning_mlp():
    """Test full structured pruning on MLP model."""
    print("="*60)
    print("TEST: Apply Structured Pruning (MLP)")
    print("="*60)

    # Create simple MLP
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dense(32, activation='relu', name='dense2'),
        layers.Dense(2, activation='softmax', name='output')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    orig_params, orig_size = get_model_size(model)
    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} KB")
    model.summary()

    # Apply pruning
    pruned_model = apply_structured_pruning(model, pruning_ratio=0.5, skip_last_layer=True, verbose=True)

    pruned_params, pruned_size = get_model_size(pruned_model)
    print(f"\nPruned Model:")
    print(f"  Parameters: {pruned_params:,}")
    print(f"  Size: {pruned_size:.2f} KB")
    pruned_model.summary()

    reduction = (1 - pruned_params / orig_params) * 100
    print(f"\nParameter Reduction: {reduction:.1f}%")

    assert pruned_params < orig_params
    assert pruned_model.output_shape == model.output_shape  # Output shape unchanged
    print("✅ PASSED\n")


def test_apply_structured_pruning_cnn():
    """Test full structured pruning on CNN model."""
    print("="*60)
    print("TEST: Apply Structured Pruning (CNN)")
    print("="*60)

    # Create simple CNN
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', name='conv1'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', name='conv2'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dense(10, activation='softmax', name='output')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    orig_params, orig_size = get_model_size(model)
    print(f"\nOriginal Model:")
    print(f"  Parameters: {orig_params:,}")
    print(f"  Size: {orig_size:.2f} KB")

    # Apply pruning
    pruned_model = apply_structured_pruning(model, pruning_ratio=0.5, skip_last_layer=True, verbose=True)

    pruned_params, pruned_size = get_model_size(pruned_model)
    print(f"\nPruned Model:")
    print(f"  Parameters: {pruned_params:,}")
    print(f"  Size: {pruned_size:.2f} KB")

    reduction = (1 - pruned_params / orig_params) * 100
    print(f"\nParameter Reduction: {reduction:.1f}%")

    assert pruned_params < orig_params
    assert pruned_model.output_shape == model.output_shape
    print("✅ PASSED\n")


def test_pruned_model_inference():
    """Test that pruned model can perform inference."""
    print("="*60)
    print("TEST: Pruned Model Inference")
    print("="*60)

    # Create and prune model
    model = keras.Sequential([
        keras.Input(shape=(10,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    pruned_model = apply_structured_pruning(model, pruning_ratio=0.5, verbose=False)

    # Create dummy data
    x_test = np.random.randn(100, 10)

    # Test prediction
    predictions = pruned_model.predict(x_test, verbose=0)

    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Prediction sample: {predictions[0]}")
    print(f"Sum of probabilities: {predictions[0].sum():.4f} (should be ~1.0)")

    assert predictions.shape == (100, 2)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
    print("✅ PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "🔬 "*20)
    print("STRUCTURED PRUNING TEST SUITE")
    print("🔬 "*20 + "\n")

    try:
        test_calculate_neuron_importance_dense()
        test_calculate_neuron_importance_conv2d()
        test_prune_dense_layer()
        test_prune_conv2d_layer()
        test_apply_structured_pruning_mlp()
        test_apply_structured_pruning_cnn()
        test_pruned_model_inference()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60 + "\n")
        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
