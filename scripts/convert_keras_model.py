#!/usr/bin/env python3
"""
Convert a Keras 3 saved model to tf_keras format for compatibility with tensorflow_model_optimization.

This script:
1. Loads the model with Keras 3 (extracts weights)
2. Rebuilds it using the project's build_mlp() with tf_keras
3. Saves in tf_keras format

Usage:
    python scripts/convert_keras_model.py models/global_model.h5
    python scripts/convert_keras_model.py src/models/global_model.h5 models/global_model.h5
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def convert_model(input_path: str, output_path: str = None):
    """Load model with Keras 3, rebuild with tf_keras, transfer weights."""
    
    if output_path is None:
        # Default: same path (overwrite)
        output_path = input_path
    
    print(f"📥 Step 1: Loading model with Keras 3 from: {input_path}")
    
    # Load with Keras 3 (no env var yet)
    import keras
    model_keras3 = keras.models.load_model(input_path, compile=False)
    print(f"   ✅ Loaded: {model_keras3.__class__.__module__}.{model_keras3.__class__.__name__}")
    print(f"   📊 Parameters: {model_keras3.count_params():,}")
    
    # Extract info
    weights = model_keras3.get_weights()
    # Get input shape as tuple (e.g., (78,) for 78 features)
    if len(model_keras3.input_shape) > 1:
        input_shape = (model_keras3.input_shape[1],)
    else:
        input_shape = (model_keras3.input_shape[0],)
    print(f"   📐 Input shape: {input_shape}")
    print(f"   📦 Weight arrays: {len(weights)}")
    
    # Step 2: Switch to tf_keras
    print(f"\n🔄 Step 2: Rebuilding with tf_keras...")
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    # Import model builder
    from src.models.nets import make_mlp
    
    # Build fresh model with tf_keras
    model_tf = make_mlp(input_shape=input_shape, num_classes=1)
    print(f"   ✅ Built: {model_tf.__class__.__module__}.{model_tf.__class__.__name__}")
    print(f"   📊 Parameters: {model_tf.count_params():,}")
    
    # Transfer weights
    print(f"\n🔀 Step 3: Transferring weights...")
    model_tf.set_weights(weights)
    print(f"   ✅ Weights transferred successfully")
    
    # Save
    print(f"\n💾 Step 4: Saving in tf_keras format to: {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tf.save(output_path)
    print(f"   ✅ Saved successfully")
    
    # Verify
    print(f"\n✓ Step 5: Verification...")
    from tensorflow import keras as tf_keras
    test_load = tf_keras.models.load_model(output_path, compile=False)
    print(f"   ✅ Model loads correctly: {test_load.__class__.__module__}")
    print(f"   ✅ Parameter count matches: {test_load.count_params() == model_tf.count_params()}")
    
    # File size
    size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"   📦 File size: {size_mb:.2f} MB")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_keras_model.py <input_model.h5> [output_model.h5]")
        print("\nExample:")
        print("  python scripts/convert_keras_model.py models/global_model.h5")
        print("  python scripts/convert_keras_model.py src/models/global_model.h5 models/global_model.h5")
        sys.exit(1)
    
    input_model = sys.argv[1]
    output_model = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_model).exists():
        print(f"❌ Error: Model file not found: {input_model}")
        sys.exit(1)
    
    try:
        result = convert_model(input_model, output_model)
        print(f"\n✅ CONVERSION COMPLETE!")
        print(f"   Converted model: {result}")
        print(f"   This model can now be used with compression.py and tfmot")
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
