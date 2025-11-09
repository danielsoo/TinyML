# /src/models/nets.py
from __future__ import annotations
import tensorflow as tf

def make_small_cnn(input_shape=(28, 28, 1), num_classes=10) -> tf.keras.Model:
    """
    CNN 모델 (이미지 데이터용 - MNIST 등)
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def make_mlp(input_shape: tuple, num_classes: int, hidden_units: list = [64, 32]) -> tf.keras.Model:
    """
    MLP (Multi-Layer Perceptron) 모델 - Tabular 데이터용 (Bot-IoT 등)
    읽는 것: input_shape (예: (35,) - 35개 특징)
    반환하는 것: 학습 가능한 모델
    목적: Bot-IoT 같은 표 형식 데이터를 분류하는 모델 생성
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # Hidden layers
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.2)(x)  # 과적합 방지
    
    # Output layer
    if num_classes == 2:
        # 이진 분류 (정상/공격)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        # 다중 분류
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
