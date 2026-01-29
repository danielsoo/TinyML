# /src/tinyml/export_tflite.py
from __future__ import annotations
import tensorflow as tf
import numpy as np
from typing import Optional
from src.models.nets import make_small_cnn


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
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        if representative_data is None:
            raise ValueError("representative_data is required for quantization")

        print("Applying full integer quantization (INT8)...")

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        def representative_dataset():
            num_samples = min(100, len(representative_data))
            for i in range(num_samples):
                yield [representative_data[i:i+1].astype(np.float32)]

        converter.representative_dataset = representative_dataset
        print(f"✅ Using {min(100, len(representative_data))} representative samples")

    tflite_model = converter.convert()

    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"✅ Saved: {out_path} ({size_kb:.2f} KB)")

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
