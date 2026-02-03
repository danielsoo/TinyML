# /src/tinyml/export_tflite.py
from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Optional
from keras import layers
from src.models.nets import make_small_cnn


def _is_bn(layer):
    return "BatchNormalization" in type(layer).__name__

def _is_dropout(layer):
    return "Dropout" in type(layer).__name__

def _is_dense(layer):
    return "Dense" in type(layer).__name__

def _strip_bn_dropout_for_tflite(model: tf.keras.Model) -> tf.keras.Model:
    """
    Build inference-only model: fold BatchNorm into Dense, remove Dropout.
    Avoids TFLite NaN from BatchNorm conversion.
    Keras 3 / tf.keras Dense does not accept weights in constructor; use set_weights() after build.
    """
    from keras import Input
    input_shape = model.input_shape[1:]
    new_layers = [Input(shape=input_shape)]
    dense_layers_with_weights = []  # [(Dense layer, [w, b]), ...]
    i = 0
    while i < len(model.layers):
        layer = model.layers[i]
        if _is_dense(layer):
            w, b = layer.get_weights()
            if i + 1 < len(model.layers) and _is_bn(model.layers[i + 1]):
                bn = model.layers[i + 1]
                gamma, beta, mean, var = bn.get_weights()
                eps = 1e-3
                var_safe = np.maximum(var, eps)
                scale = gamma / np.sqrt(var_safe)
                w_new = w * scale
                b_new = scale * (b - mean) + beta
                d = layers.Dense(layer.units, activation=layer.activation)
                dense_layers_with_weights.append((d, [w_new, b_new]))
                new_layers.append(d)
                i += 2
            else:
                d = layers.Dense(layer.units, activation=layer.activation)
                dense_layers_with_weights.append((d, [w.copy(), b.copy()]))
                new_layers.append(d)
                i += 1
        elif _is_bn(layer) or _is_dropout(layer):
            i += 1
        elif "InputLayer" in type(layer).__name__:
            i += 1
        else:
            new_layers.append(layer)
            i += 1

    seq = tf.keras.Sequential(new_layers)
    seq.build((None,) + input_shape)
    for dense_layer, wb in dense_layers_with_weights:
        dense_layer.set_weights(wb)
    return seq


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

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = _os.path.join(tmpdir, "saved_model")
            model.save(saved_path, save_format="tf")
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_path)
    except Exception:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if not quantize:
        converter.optimizations = []

    if quantize:
        # Dynamic Range Quantization: int8 weights, float32 input/output.
        # Full integer (int8 in/out) causes severe precision loss -> P/R/F1=0.
        # DRQ preserves accuracy while still reducing model size (~4x).
        print("Applying dynamic range quantization (int8 weights, float I/O)...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ Saved: {out_path} ({size_kb:.2f} KB)")

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
