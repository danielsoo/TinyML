"""
Lightweight model architectures designed for edge/TinyML deployment.

These models prioritize small parameter count and low inference cost
while maintaining reasonable accuracy. They can be used directly as
the FL model (instead of the standard MLP) to skip the distillation step,
or compared against the distilled student model.

Standard MLP (nets.py):   Dense(256) → Dense(128) → Dense(64) → Dense(C)
Lightweight MLP:          Dense(64)  → Dense(32)  → Dense(C)
Lightweight Bottleneck:   Dense(64)  → Dense(16)  → Dense(32) → Dense(C)
"""
from typing import Tuple

from tensorflow import keras
from tensorflow.keras import layers


def _compile_model(model: keras.Model, num_classes: int) -> keras.Model:
    """Add output layer and compile with consistent settings."""
    if num_classes <= 1:
        num_classes = 2

    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_lightweight_mlp(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Lightweight MLP with ~3-4x fewer parameters than the standard MLP.

    Architecture: Input → Dense(64, relu) → Dense(32, relu) → Dense(C, softmax)

    Standard MLP params (input=10, C=2):
        10*256 + 256*128 + 128*64 + 64*2 = 43,650
    Lightweight MLP params (input=10, C=2):
        10*64 + 64*32 + 32*2 = 2,752

    ~15x fewer parameters for this input size.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
        ],
        name="lightweight_mlp"
    )
    return _compile_model(model, num_classes)


def make_bottleneck_mlp(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Bottleneck MLP that compresses features through a narrow layer.

    Architecture: Input → Dense(64, relu) → Dense(16, relu) → Dense(32, relu) → Dense(C, softmax)

    The bottleneck (16 units) forces the model to learn compact representations,
    which is beneficial for generalization on edge devices.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(64, activation="relu"),
            layers.Dense(16, activation="relu"),   # bottleneck
            layers.Dense(32, activation="relu"),
        ],
        name="bottleneck_mlp"
    )
    return _compile_model(model, num_classes)


def make_tiny_mlp(input_shape: Tuple[int, ...], num_classes: int) -> keras.Model:
    """
    Minimal MLP for the most constrained devices.

    Architecture: Input → Dense(32, relu) → Dense(C, softmax)

    Single hidden layer — smallest possible model that can still
    learn non-linear decision boundaries.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Dense(32, activation="relu"),
        ],
        name="tiny_mlp"
    )
    return _compile_model(model, num_classes)


def get_lightweight_model(
    model_name: str, input_shape: Tuple[int, ...], num_classes: int
) -> keras.Model:
    """
    Router for lightweight architectures.

    Args:
        model_name: One of "lightweight", "bottleneck", "tiny"
        input_shape: Input feature shape
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    model_name = (model_name or "lightweight").lower()

    if model_name in ["bottleneck", "bottleneck_mlp"]:
        return make_bottleneck_mlp(input_shape, num_classes)
    elif model_name in ["tiny", "tiny_mlp"]:
        return make_tiny_mlp(input_shape, num_classes)
    else:
        return make_lightweight_mlp(input_shape, num_classes)
