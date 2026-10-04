# /src/tinyml/export_tflite.py
from __future__ import annotations
import os
import tensorflow as tf
import numpy as np
from typing import Optional
from tensorflow.keras import layers
from src.models.nets import make_small_cnn


def verify_tflite_quantization(tflite_path: str) -> dict:
    """
    Verify TFLite model quantization by inspecting tensor dtypes.
    Returns dict with quantization statistics.
    """
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        
        # Count quantized vs float tensors
        quantized_count = 0
        float_count = 0
        dtypes = {}
        
        for tensor in tensor_details:
            dtype = tensor['dtype']
            dtype_name = str(dtype)
            dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1
            
            if dtype in [np.int8, np.uint8]:
                quantized_count += 1
            elif dtype in [np.float32, np.float16]:
                float_count += 1
        
        total_tensors = len(tensor_details)
        quant_ratio = quantized_count / total_tensors if total_tensors > 0 else 0.0
        
        stats = {
            'total_tensors': total_tensors,
            'quantized_tensors': quantized_count,
            'float_tensors': float_count,
            'quantization_ratio': quant_ratio,
            'dtype_distribution': dtypes,
            'input_dtype': str(input_details[0]['dtype']),
            'output_dtype': str(output_details[0]['dtype']),
        }
        
        return stats
    except Exception as e:
        return {'error': str(e)}


def _is_bn(layer):
    return "BatchNormalization" in type(layer).__name__

def _is_dropout(layer):
    return "Dropout" in type(layer).__name__

def _is_dense(layer):
    return "Dense" in type(layer).__name__


def fold_batchnorm(model) -> list:
    """
    Return [(units, activation, [W, b]), ...] for a Dense/BatchNorm/Dropout MLP with every
    BatchNorm folded away exactly (inference mode; Dropout is identity).

    make_mlp puts BatchNorm *after* the ReLU (Dense(relu) -> BN -> Dropout), so a BN cannot be
    folded into the Dense before it. It is an affine map h -> s*h + t, which folds exactly into
    the next Dense: W' = s[:, None] * W, b' = b + t @ W.
    """
    specs = []
    pending = None  # (s, t) of BatchNorms not yet absorbed by a Dense
    for layer in model.layers:
        if _is_dense(layer):
            weights = layer.get_weights()
            w = weights[0].astype(np.float64)
            b = (weights[1] if len(weights) > 1 else np.zeros(w.shape[1])).astype(np.float64)
            if pending is not None:
                s, t = pending
                b = b + t @ w
                w = w * s[:, None]
                pending = None
            specs.append((layer.units, layer.activation, [w.astype(np.float32), b.astype(np.float32)]))
        elif _is_bn(layer):
            mean = layer.moving_mean.numpy().astype(np.float64)
            var = layer.moving_variance.numpy().astype(np.float64)
            gamma = layer.gamma.numpy().astype(np.float64) if layer.scale else np.ones_like(mean)
            beta = layer.beta.numpy().astype(np.float64) if layer.center else np.zeros_like(mean)
            s = gamma / np.sqrt(var + layer.epsilon)
            t = beta - s * mean
            pending = (s, t) if pending is None else (pending[0] * s, pending[1] * s + t)
        elif _is_dropout(layer) or "InputLayer" in type(layer).__name__:
            continue
        else:
            raise ValueError(f"fold_batchnorm: unsupported layer {type(layer).__name__}")
    if pending is not None:
        raise ValueError("fold_batchnorm: BatchNorm after the last Dense cannot be folded")
    return specs

def _strip_bn_dropout_for_tflite(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build inference-only model: fold BatchNorm into the next Dense, remove Dropout.
    Avoids TFLite NaN from BatchNorm conversion.
    Keras 3 / tf.keras Dense does not accept weights in constructor; use set_weights() after build.
    """
    from tensorflow.keras import Input
    input_shape = model.input_shape[1:]
    new_layers = [Input(shape=input_shape)]
    dense_layers_with_weights = []  # [(Dense layer, [w, b]), ...]
    for units, activation, wb in fold_batchnorm(model):
        d = layers.Dense(units, activation=activation)
        dense_layers_with_weights.append((d, wb))
        new_layers.append(d)

    seq = tf.keras.Sequential(new_layers)
    seq.build((None,) + input_shape)
    for dense_layer, wb in dense_layers_with_weights:
        dense_layer.set_weights(wb)
    return seq


def _strip_bn_dropout_for_qat(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a pure tf.keras Functional model (Dense only) so that
    tfmot.quantization.keras.quantize_model() accepts it.
    Sequential with Input() can be rejected; Functional is more reliable.
    CRITICAL: Must use tensorflow.keras, not standalone keras, for tfmot compatibility.
    Note: TF_USE_LEGACY_KERAS=1 should be set BEFORE tensorflow import (in calling script).
    """
    print(f"   [QAT Strip] Input model type: {type(model).__name__}")
    print(f"   [QAT Strip] Input model class: {model.__class__.__module__}.{model.__class__.__name__}")
    
    # Get input shape - handle both single and multiple inputs
    if isinstance(model.input_shape, list):
        input_shape = model.input_shape[0][1:]
    else:
        input_shape = model.input_shape[1:]
    
    print(f"   [QAT Strip] Creating TF.Keras Functional model with input shape: {input_shape}")
    print(f"   [QAT Strip] TF_USE_LEGACY_KERAS={os.environ.get('TF_USE_LEGACY_KERAS', 'not set')}")
    
    # Dense-only layer specs with BatchNorm folded exactly (see fold_batchnorm)
    dense_layers_info = fold_batchnorm(model)
    print(f"   [QAT Strip] {len(dense_layers_info)} Dense layers after BatchNorm folding")

    # Now build model using ONLY tf.keras (not standalone keras)
    # Import tensorflow first, then use its keras submodule
    import tensorflow
    inputs = tensorflow.keras.Input(shape=input_shape, name='input')
    x = inputs
    
    dense_layers = []
    for idx, (units, activation, weights) in enumerate(dense_layers_info):
        d = tensorflow.keras.layers.Dense(
            units,
            activation=activation,
            name=f'dense_{idx}'
        )
        x = d(x)
        dense_layers.append((d, weights))
    
    out = tensorflow.keras.Model(inputs=inputs, outputs=x, name='qat_stripped_model')
    
    # Set weights after model is built
    for dense_layer, wb in dense_layers:
        dense_layer.set_weights(wb)
    
    print(f"   [QAT Strip] Created model type: {type(out).__name__}")
    print(f"   [QAT Strip] Model class: {out.__class__.__module__}.{out.__class__.__name__}")
    print(f"   [QAT Strip] Total layers: {len(out.layers)}")
    
    # Verify it's a tensorflow.keras model
    is_tf_keras = 'tensorflow' in out.__class__.__module__
    print(f"   [QAT Strip] Is tensorflow.keras model: {is_tf_keras}")
    if not is_tf_keras:
        print(f"   [QAT Strip] WARNING: Model is not tensorflow.keras! QAT may fail.")
    
    return out


def export_tflite(
    model: tf.keras.Model,
    out_path: str,
    quantize: bool = False,
    representative_data: Optional[np.ndarray] = None
):
    """
    Export Keras model to TFLite format.

    Args:
        model: Keras model to convert
        out_path: Output path for .tflite file
        quantize: If True, apply full integer quantization (int8)
        representative_data: Representative dataset for quantization
                           (required if quantize=True)

    Returns:
        Size of the exported model in bytes
    """
    import tempfile
    import os as _os

    # Strip BatchNorm/Dropout for TFLite export (float32 and INT8) to avoid NaN/wrong outputs
    try:
        model = _strip_bn_dropout_for_tflite(model)
        print("✅ BN/Dropout stripped for TFLite export")
    except Exception as e:
        print(f"⚠️ BN strip failed ({e}), using original model")

    # Use from_keras_model directly (more stable with tf_keras)
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception as e:
        print(f"⚠️ from_keras_model failed ({e}), trying SavedModel path...")
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = _os.path.join(tmpdir, "saved_model")
            model.save(saved_path, save_format="tf")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)

    if not quantize:
        converter.optimizations = []

    if quantize:
        if representative_data is not None and len(representative_data) > 0:
            # Full Integer Quantization with proper calibration
            print("Applying full integer quantization (int8 weights + activations with calibration)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Calibration dataset generator
            def representative_dataset():
                # Use subset for calibration (100-500 samples)
                calibration_samples = min(500, len(representative_data))
                for i in range(calibration_samples):
                    # Yield single sample as batch
                    yield [representative_data[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Keep input/output as float32 (better compatibility + accuracy)
            # Only internal ops are int8
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        else:
            # Fallback: Dynamic Range Quantization (weights only)
            print("Applying dynamic range quantization (int8 weights, float I/O)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ Saved: {out_path} ({size_kb:.2f} KB)")
    
    # Verify quantization stats
    stats = verify_tflite_quantization(out_path)
    if 'error' not in stats:
        print(f"   Quantization: {stats['quantized_tensors']}/{stats['total_tensors']} tensors ({stats['quantization_ratio']*100:.1f}%)")
        print(f"   Input: {stats['input_dtype']}, Output: {stats['output_dtype']}")

    return len(tflite_model)


def export_tflite_qat(q_aware_model: tf.keras.Model, out_path: str) -> int:
    """
    Export a QAT (quantization-aware trained) model to TFLite.
    Uses tfmot-recommended conversion: from_keras_model + Optimize.DEFAULT.
    The QAT model already has QuantizeWrapper layers; no BN strip needed.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    size_kb = len(tflite_model) / 1024
    print(f"✅ Saved (QAT): {out_path} ({size_kb:.2f} KB)")
    
    # Verify quantization stats
    stats = verify_tflite_quantization(out_path)
    if 'error' not in stats:
        print(f"   Quantization: {stats['quantized_tensors']}/{stats['total_tensors']} tensors ({stats['quantization_ratio']*100:.1f}%)")
        print(f"   Input: {stats['input_dtype']}, Output: {stats['output_dtype']}")
    return len(tflite_model)


if __name__ == "__main__":
    input_shape = (28, 28, 1)
    num_classes = 2
    model = make_small_cnn(input_shape, num_classes)

    # Export without quantization
    export_tflite(model, "tiny_model_float32.tflite", quantize=False)

    # To export with quantization, you would need representative data:
    # from tensorflow.keras.datasets import mnist
    # (x_train, _), _ = mnist.load_data()
    # x_train = np.expand_dims(x_train[:1000] / 255.0, -1)
    # export_tflite(model, "tiny_model_int8.tflite", quantize=True, representative_data=x_train)
