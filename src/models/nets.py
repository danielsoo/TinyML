from typing import Tuple

import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential


def _focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """Focal loss for imbalanced binary classification. Focuses more on hard examples."""
    def loss_fn(y_true, y_pred):
        eps = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return alpha_t * (1 - p_t) ** gamma * bce
    return loss_fn


def _compile_for_classes(
    model: Model,
    num_classes: int,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False,
    focal_loss_alpha: float = 0.75,
) -> Model:
    """Configure final layer and loss based on num_classes.

    focal_loss_alpha: weight for positive (minority) class. Use 0.75~0.8 when
    attack=positive is minority to prevent model from predicting all negative.
    """
    import keras
    # Gradient clipping for training stability
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    if num_classes <= 2:
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != 1:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(1, activation="sigmoid"))
        loss = _focal_loss(gamma=2.0, alpha=focal_loss_alpha) if use_focal_loss else "binary_crossentropy"
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"],
        )
    else:
        # Multi-class classification: output C, softmax + sparse_categorical_crossentropy
        if not isinstance(model.layers[-1], layers.Dense) or model.layers[-1].units != num_classes:
            model.pop() if hasattr(model, "pop") else None
            model.add(layers.Dense(num_classes, activation="softmax"))
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model


def make_mlp(
    input_shape: Tuple[int, ...],
    num_classes: int,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.25,
    use_focal_loss: bool = False,
    focal_loss_alpha: float = 0.75,
) -> Model:
    """MLP: 512->256->128, BatchNorm, Dropout. For training stability and accuracy."""
    from keras import Input
    model = Sequential(
        [
            Input(shape=input_shape),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
        ]
    )
    return _compile_for_classes(model, num_classes, learning_rate, use_focal_loss, focal_loss_alpha)


def make_small_cnn(
    input_shape: Tuple[int, ...], num_classes: int, learning_rate: float = 0.001
) -> Model:
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
    return _compile_for_classes(model, num_classes, learning_rate)


def get_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    num_classes: int,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False,
    focal_loss_alpha: float = 0.75,
) -> Model:
    model_name = (model_name or "mlp").lower()
    if model_name in ["cnn", "small_cnn"]:
        return make_small_cnn(input_shape, num_classes, learning_rate)
    return make_mlp(input_shape, num_classes, learning_rate, use_focal_loss=use_focal_loss, focal_loss_alpha=focal_loss_alpha)