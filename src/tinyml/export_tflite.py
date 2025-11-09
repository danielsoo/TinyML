# /src/tinyml/export_tflite.py
from __future__ import annotations
import tensorflow as tf
from src.models.nets import make_small_cnn

def export_tflite(model: tf.keras.Model, out_path: str):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    model = make_small_cnn()
    export_tflite(model, "tiny_model.tflite")
