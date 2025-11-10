from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _compile_for_classes(model: keras.Model, num_classes: int) -> keras.Model:
    """num_classes에 맞게 마지막 레이어/로스 설정."""
    if num_classes <= 2:
        # 이진 분류: 출력 1, sigmoid + binary_crossentropy
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != 1:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
    else:
        # 다중 분류: 출력 C, softmax + sparse_categorical_crossentropy
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != num_classes:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(num_classes, activation="softmax"))
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
        # 기본은 MLP
        return make_mlp(input_shape, num_classes)