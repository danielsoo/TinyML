"""
FGSM Attack Testing Script
Test FGSM attack implementation on Bot-IoT dataset
Automatically detects local vs Colab environment and adjusts settings accordingly
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from src.utils.env_utils import (
        is_colab,
        is_colab_runtime,
        get_default_data_path,
    )
except ImportError:
    # Fallback if env_utils not available
    def is_colab():
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def is_colab_runtime():
        return os.getenv("COLAB_GPU") is not None or os.path.exists("/content")
    
    def get_default_data_path():
        if is_colab() or is_colab_runtime():
            return "/content/drive/MyDrive/TinyML_models"
        else:
            return "data/raw/Bot-IoT"

import numpy as np
import tensorflow as tf
from src.adversarial.fgsm_hook import (
    generate_fgsm_attack,
    evaluate_attack_success,
    tune_epsilon,
    generate_adversarial_dataset,
)
from src.data.loader import load_dataset
from src.models.nets import get_model


def test_fgsm_attack():
    """Test FGSM attack on Bot-IoT dataset"""
    # Detect environment
    in_colab = is_colab() or is_colab_runtime()
    data_path = get_default_data_path()
    
    # Adjust settings based on environment
    if in_colab:
        max_samples = 100000  # Colab: use more samples
        test_subset_size = 500
        tune_subset_size = 1000
        adv_subset_size = 5000
        print("🌐 Running in Colab environment")
    else:
        max_samples = 20000  # Local: use fewer samples to avoid overload
        test_subset_size = 100
        tune_subset_size = 200
        adv_subset_size = 1000
        print("💻 Running in local environment")
    
    print("=" * 60)
    print("FGSM Attack Testing")
    print("=" * 60)
    
    # Load dataset
    print(f"\n1. Loading Bot-IoT dataset...")
    print(f"   Environment: {'Colab' if in_colab else 'Local'}")
    print(f"   Data path: {data_path}")
    print(f"   Max samples: {max_samples}")
    try:
        x_train, y_train, x_test, y_test = load_dataset(
            name="bot_iot",
            data_path=data_path,
            max_samples=max_samples,
        )
        print(f"   ✅ Loaded {len(x_train)} training samples, {len(x_test)} test samples")
        print(f"   Input shape: {x_train.shape[1:]}")
        print(f"   Number of classes: {len(np.unique(y_train))}")
    except Exception as e:
        print(f"   ❌ Error loading dataset: {e}")
        print("   💡 Make sure Bot-IoT dataset is available at data/raw/Bot-IoT/")
        return False
    
    # Load or train model
    print("\n2. Loading/Training model...")
    model_path = "src/models/global_model.h5"
    
    if os.path.exists(model_path):
        print(f"   ✅ Loading existing model from {model_path}")
        print(f"   ℹ️  Using Federated Learning trained model (recommended)")
        model = tf.keras.models.load_model(model_path)
    else:
        print("   ⚠️  Model not found!")
        print("   ⚠️  WARNING: No pre-trained model found.")
        print("   ⚠️  For better results, train a model first using:")
        print("   ⚠️     python scripts/train.py")
        print("   ⚠️  ")
        print("   ⚠️  Training a quick test model (5 epochs only)...")
        print("   ⚠️  This will be less accurate than Federated Learning model.")
        print()
        
        model = get_model("mlp", x_train.shape[1:], len(np.unique(y_train)))
        model.fit(
            x_train, y_train,
            epochs=5,
            batch_size=128,
            validation_split=0.2,
            verbose=1
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"   ✅ Quick test model saved to {model_path}")
        print("   ⚠️  Note: For production use, train with 'python scripts/train.py' first")
    
    # Evaluate original model
    print("\n3. Evaluating original model...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test Loss: {test_loss:.4f}")
    
    # Test FGSM attack with different epsilon values
    print("\n4. Testing FGSM attack with different epsilon values...")
    test_subset_size = min(test_subset_size, len(x_test))
    x_test_subset = x_test[:test_subset_size]
    y_test_subset = y_test[:test_subset_size]
    
    epsilon_values = [0.01, 0.05, 0.1, 0.15, 0.2]
    
    print(f"\n   Testing on {test_subset_size} samples:")
    print(f"   {'Epsilon':<10} {'Original Acc':<15} {'Adv Acc':<15} {'Success Rate':<15} {'Avg Perturb':<15}")
    print("   " + "-" * 70)
    
    results = []
    for eps in epsilon_values:
        # Generate adversarial examples
        x_adv, _ = generate_fgsm_attack(
            model, x_test_subset, y_test_subset, eps=eps
        )
        
        # Evaluate attack
        metrics = evaluate_attack_success(model, x_test_subset, x_adv, y_test_subset)
        results.append(metrics)
        
        print(f"   {eps:<10.2f} {metrics['original_accuracy']:<15.4f} "
              f"{metrics['adversarial_accuracy']:<15.4f} "
              f"{metrics['attack_success_rate']:<15.4f} "
              f"{metrics['avg_perturbation']:<15.6f}")
    
    # Epsilon tuning
    print("\n5. Tuning epsilon parameter...")
    tune_subset_size = min(tune_subset_size, len(x_test))
    x_tune = x_test[:tune_subset_size]
    y_tune = y_test[:tune_subset_size]
    
    tuning_results = tune_epsilon(
        model, x_tune, y_tune,
        target_success_rate=0.5
    )
    
    print(f"   Best epsilon: {tuning_results['best_epsilon']:.4f}")
    print(f"   Target success rate: {tuning_results['target_success_rate']:.2f}")
    
    # Generate full adversarial dataset
    print("\n6. Generating adversarial dataset...")
    adv_subset_size = min(adv_subset_size, len(x_test))
    batch_size = 64 if in_colab else 32  # Larger batch in Colab
    x_adv_full, y_adv_full = generate_adversarial_dataset(
        model, x_test[:adv_subset_size], y_test[:adv_subset_size],
        eps=0.1, batch_size=batch_size
    )
    
    print(f"   ✅ Generated {len(x_adv_full)} adversarial examples")
    
    # Final evaluation
    print("\n7. Final evaluation...")
    final_metrics = evaluate_attack_success(
        model, x_test[:adv_subset_size], x_adv_full, y_adv_full
    )
    
    print("\n" + "=" * 60)
    print("Attack Summary")
    print("=" * 60)
    print(f"Original Accuracy: {final_metrics['original_accuracy']:.4f} ({final_metrics['original_accuracy']*100:.2f}%)")
    print(f"Adversarial Accuracy: {final_metrics['adversarial_accuracy']:.4f} ({final_metrics['adversarial_accuracy']*100:.2f}%)")
    print(f"Attack Success Rate: {final_metrics['attack_success_rate']:.4f} ({final_metrics['attack_success_rate']*100:.2f}%)")
    print(f"Attack Success Count: {final_metrics['attack_success_count']}/{final_metrics['total_samples']}")
    print(f"Average Perturbation: {final_metrics['avg_perturbation']:.6f}")
    print(f"Max Perturbation: {final_metrics['max_perturbation']:.6f}")
    print("=" * 60)
    
    print("\n✅ FGSM attack testing completed!")
    return True


if __name__ == "__main__":
    success = test_fgsm_attack()
    sys.exit(0 if success else 1)

