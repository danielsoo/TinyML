"""
Create simple test model - for microcontroller deployment validation
"""
import os

import sys
from pathlib import Path

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


import numpy as np
import tensorflow as tf
from src.tinyml.export_tflite import export_tflite


def create_hello_world_model():
    """
    Very simple MLP model (2 inputs, 1 output)
    - Input: [x1, x2] (2 float values)
    - Output: value between 0~1 (sigmoid)
    - Purpose: inference testing on microcontroller
    """
    # Define model structure
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(2,), name='dense1'),
        tf.keras.layers.Dense(2, activation='relu', name='dense2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ])
    
    # Train with simple dummy data
    # Pattern: if x1 + x2 > 1.0 then 1, else 0
    X_train = np.random.rand(100, 2).astype(np.float32)
    y_train = (X_train[:, 0] + X_train[:, 1] > 1.0).astype(np.float32)
    
    # Compile and train model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Training Hello World model...")
    model.fit(X_train, y_train, epochs=20, verbose=1, batch_size=32)
    
    # Simple test
    test_input = np.array([[0.3, 0.3], [0.7, 0.7]], dtype=np.float32)
    predictions = model.predict(test_input, verbose=0)
    print(f"\nTest predictions:")
    print(f"  Input [0.3, 0.3]: {predictions[0][0]:.4f}")
    print(f"  Input [0.7, 0.7]: {predictions[1][0]:.4f}")
    
    return model


if __name__ == "__main__":
    # Create output directory
    output_dir = "data/processed/microcontroller"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Creating Hello World Test Model for Microcontroller")
    print("=" * 60)
    
    # Create model
    model = create_hello_world_model()
    
    # Save as H5 format
    h5_path = os.path.join(output_dir, "hello_world_model.h5")
    model.save(h5_path)
    print(f"\n✅ Saved H5 model: {h5_path}")
    
    # Convert to TFLite
    tflite_path = os.path.join(output_dir, "hello_world_model.tflite")
    export_tflite(model, tflite_path)
    
    # Print model information
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
    # 파일 크기 확인
    tflite_size = os.path.getsize(tflite_path)
    h5_size = os.path.getsize(h5_path)
    
    print(f"\nFile sizes:")
    print(f"  H5: {h5_size:,} bytes ({h5_size/1024:.2f} KB)")
    print(f"  TFLite: {tflite_size:,} bytes ({tflite_size/1024:.2f} KB)")
    if tflite_size > 0:
        print(f"  Compression: {h5_size/tflite_size:.2f}x")
    
    # 모델 파라미터 수
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    print("\n✅ Test model created successfully!")
    print(f"   Ready for deployment: {tflite_path}")

