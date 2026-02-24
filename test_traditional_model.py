#!/usr/bin/env python3
"""Quick test to check Traditional model performance"""
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
from src.data.loader import load_dataset

# Load Traditional model
model_path = Path("data/processed/runs/v24/2026-02-24_16-33-10/models/global_model_traditional.h5")
print(f"Loading Traditional model from {model_path}...")
model = keras.models.load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load test data
print("\nLoading test dataset...")
x_train, y_train, x_test, y_test = load_dataset(
    "cicids2017",
    data_path="data/raw/CIC-IDS2017",
    max_samples=2000000,
    balance_ratio=4.0,
    binary=True,
    use_smote=True
)

x_test = x_test[:1000]
y_test = y_test[:1000]

# Evaluate
print(f"\nEvaluating Traditional model...")
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"✅ Traditional Model Accuracy: {acc:.2%}")
print(f"✅ Traditional Model Loss: {loss:.4f}")

# Get predictions
print("\nChecking predictions distribution...")
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = (y_pred_prob > 0.3).astype(int).flatten()

unique, counts = np.unique(y_pred, return_counts=True)
print(f"Prediction distribution: {dict(zip(unique, counts))}")
print(f"Attack samples: {np.sum(y_test == 1)}")
print(f"Normal samples: {np.sum(y_test == 0)}")
print(f"Predicted attack: {np.sum(y_pred == 1)}")
print(f"Predicted normal: {np.sum(y_pred == 0)}")
