from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _compile_for_classes(model: keras.Model, num_classes: int) -> keras.Model:
    """Configure final layer and loss based on num_classes.

    IMPORTANT: For distillation compatibility, we ALWAYS use num_classes outputs
    (even for binary: 2 outputs, not 1) with sparse_categorical_crossentropy.
    This ensures teacher and student models have matching output shapes.
    """
    # Ensure num_classes is at least 2 (for binary classification)
    if num_classes <= 1:
        num_classes = 2

    # Always use num_classes outputs with softmax activation
    # This works for both binary (2 outputs) and multi-class (C outputs)
    if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != num_classes:
        model.pop() if hasattr(model, "pop") else None
        model.add(layers.Dense(num_classes, activation="softmax"))

    # Always use sparse_categorical_crossentropy (works for binary and multi-class)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_mlp(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
        ]
    )
    return _compile_for_classes(model, num_classes)


def make_small_cnn(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
        ]
    )
    return _compile_for_classes(model, num_classes)


def get_model(model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    model_name = (model_name or "mlp").lower()
    if model_name in ["cnn", "small_cnn"]:
        return make_small_cnn(input_shape, num_classes)
    else:
        # Default is MLP
        return make_mlp(input_shape, num_classes)