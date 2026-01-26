from typing import Tuple

import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential


def _compile_for_classes(model: Model, num_classes: int) -> Model:
    """Configure final layer and loss based on num_classes."""
    if num_classes <= 2:
        # Binary classification: output 1, sigmoid + binary_crossentropy
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != 1:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        # Multi-class classification: output C, softmax + sparse_categorical_crossentropy
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != num_classes:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def make_mlp(input_shape: Tuple[int, ...], num_classes: int) -> Model:
    from keras import Input
    model = Sequential(
        [
            Input(shape=input_shape),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
        ]
    )
    return _compile_for_classes(model, num_classes)


def make_small_cnn(input_shape: Tuple[int, ...], num_classes: int) -> Model:
    from keras import Input
    model = Sequential(
        [
            Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
        ]
    )
    return _compile_for_classes(model, num_classes)


def get_model(model_name: str, input_shape: Tuple[int, ...], num_classes: int) -> Model:
    model_name = (model_name or "mlp").lower()
    if model_name in ["cnn", "small_cnn"]:
        return make_small_cnn(input_shape, num_classes)
    else:
        # Default is MLP
        return make_mlp(input_shape, num_classes)