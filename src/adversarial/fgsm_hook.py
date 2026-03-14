"""
FGSM (Fast Gradient Sign Method) Attack Implementation
Full implementation with gradient computation, epsilon tuning, and evaluation
"""
from __future__ import annotations
from typing import Tuple, Optional, Dict, List
import numpy as np
import tensorflow as tf
from tensorflow import keras


def fgsm_perturb(x: np.ndarray, grad_sign: np.ndarray, eps: float = 0.05) -> np.ndarray:
    """
    Basic FGSM perturbation function.
    
    Args:
        x: Original input data
        grad_sign: Sign of the gradient
        eps: Perturbation magnitude (epsilon)
    
    Returns:
        Adversarial examples
    """
    x_adv = x + eps * grad_sign
    return np.clip(x_adv, 0.0, 1.0)


def compute_gradients(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    loss_fn: Optional[keras.losses.Loss] = None
) -> np.ndarray:
    """
    Compute gradients of loss with respect to input.
    
    Args:
        model: Trained Keras model
        x: Input data (batch_size, features)
        y: True labels
        loss_fn: Loss function (default: model's loss)
    
    Returns:
        Gradients with respect to input
    """
    if loss_fn is None:
        loss_fn = model.loss
    # Keras may store loss as string (e.g. "binary_crossentropy"); resolve to callable
    if isinstance(loss_fn, str):
        loss_fn = keras.losses.get(loss_fn)
    
    # Convert to TensorFlow tensors
    x_tensor = tf.Variable(x, dtype=tf.float32)
    y_tensor = tf.constant(y, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        predictions = model(x_tensor, training=False)
        # binary_crossentropy requires same rank: target and output (n,) or (n,1)
        # Model often outputs (batch, 1); y is (batch,) -> align by expanding y to (batch, 1)
        pred_rank = len(predictions.shape)
        y_rank = len(y_tensor.shape)
        if pred_rank == 2 and y_rank == 1:
            y_tensor = tf.reshape(y_tensor, [-1, 1])
        loss = loss_fn(y_tensor, predictions)
    
    # Compute gradients
    gradients = tape.gradient(loss, x_tensor)
    grad_np = gradients.numpy()
    # Avoid nan/inf so x_adv and perturbation stats are valid (v18-style report)
    if not np.isfinite(grad_np).all():
        grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)
    return grad_np


def generate_fgsm_attack(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.05,
    clip_min: float = 0.0,
    clip_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate FGSM adversarial examples.
    
    Args:
        model: Trained Keras model
        x: Original input data (batch_size, features)
        y: True labels
        eps: Perturbation magnitude (epsilon)
        clip_min: Minimum value for clipping
        clip_max: Maximum value for clipping
    
    Returns:
        Tuple of (adversarial examples, gradient signs)
    """
    # Compute gradients
    gradients = compute_gradients(model, x, y)
    
    # Get sign of gradients (nan-safe: nan -> 0 so no perturbation)
    grad_sign = np.sign(gradients)
    grad_sign = np.nan_to_num(grad_sign, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Generate adversarial examples
    x_adv = np.asarray(x + eps * grad_sign, dtype=np.float32)
    if not np.isfinite(x_adv).all():
        x_adv = np.where(np.isfinite(x_adv), x_adv, np.asarray(x, dtype=np.float32))
    
    # Clip to valid range
    x_adv = np.clip(x_adv, clip_min, clip_max)
    
    return x_adv, grad_sign


def generate_pgd_attack(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.05,
    steps: int = 10,
    alpha: Optional[float] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate PGD (Projected Gradient Descent) adversarial examples.
    Multi-step attack: iteratively move in gradient direction and project to epsilon-ball.

    Args:
        model: Trained Keras model
        x: Original input data (batch_size, features)
        y: True labels
        eps: Max perturbation magnitude (Linf radius)
        steps: Number of PGD steps
        alpha: Step size per iteration (default: 2.5 * eps / steps, common choice)
        clip_min, clip_max: Clip final x_adv to valid input range
        random_start: If True, start from x + uniform(-eps, eps) (stronger)

    Returns:
        Tuple of (adversarial examples, gradient signs from last step)
    """
    if alpha is None:
        alpha = 2.5 * eps / max(steps, 1)
    x_adv = np.asarray(x, dtype=np.float32)
    if random_start:
        x_adv = x_adv + np.random.uniform(-eps, eps, x_adv.shape).astype(np.float32)
        x_adv = np.clip(x_adv, clip_min, clip_max)
    for _ in range(steps):
        gradients = compute_gradients(model, x_adv, y)
        grad_sign = np.sign(gradients)
        grad_sign = np.nan_to_num(grad_sign, nan=0.0, posinf=0.0, neginf=0.0)
        x_adv = x_adv + alpha * grad_sign
        # Project to Linf epsilon-ball around original x
        x_adv = np.clip(x_adv, x - eps, x + eps)
        x_adv = np.clip(x_adv, clip_min, clip_max).astype(np.float32)
    return x_adv, grad_sign


def generate_adversarial_dataset_pgd(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.05,
    steps: int = 10,
    alpha: Optional[float] = None,
    batch_size: int = 32,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial dataset using PGD (for AT). Same interface as generate_adversarial_dataset.
    """
    x_adv_list = []
    num_batches = (len(x) + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x))
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        x_adv_batch, _ = generate_pgd_attack(
            model, x_batch, y_batch, eps=eps, steps=steps, alpha=alpha,
            clip_min=clip_min, clip_max=clip_max,
        )
        x_adv_list.append(x_adv_batch)
    x_adv = np.concatenate(x_adv_list, axis=0)
    return x_adv, y


def evaluate_attack_success(
    model: keras.Model,
    x_original: np.ndarray,
    x_adversarial: np.ndarray,
    y_true: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate FGSM attack success rate.
    
    Args:
        model: Trained Keras model
        x_original: Original inputs
        x_adversarial: Adversarial inputs
        y_true: True labels
        threshold: Classification threshold for binary classification
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Original predictions
    pred_original = model.predict(x_original, verbose=0)
    
    # Adversarial predictions
    pred_adversarial = model.predict(x_adversarial, verbose=0)
    
    # For binary classification
    if pred_original.shape[1] == 1:
        pred_original_binary = (pred_original.flatten() >= threshold).astype(int)
        pred_adversarial_binary = (pred_adversarial.flatten() >= threshold).astype(int)
        y_true_binary = y_true.flatten().astype(int)
    else:
        # Multi-class classification
        pred_original_binary = np.argmax(pred_original, axis=1)
        pred_adversarial_binary = np.argmax(pred_adversarial, axis=1)
        y_true_binary = y_true.flatten().astype(int)
    
    # Calculate metrics
    original_correct = (pred_original_binary == y_true_binary).sum()
    adversarial_correct = (pred_adversarial_binary == y_true_binary).sum()
    
    total_samples = len(y_true)
    original_accuracy = original_correct / total_samples
    adversarial_accuracy = adversarial_correct / total_samples
    
    # Attack success: samples that were correct but became incorrect
    attack_success = original_correct - adversarial_correct
    attack_success_rate = attack_success / total_samples if total_samples > 0 else 0.0
    
    # Perturbation magnitude (nan-safe so report always shows numbers like v18)
    perturbation = np.abs(np.asarray(x_adversarial, dtype=np.float64) - np.asarray(x_original, dtype=np.float64))
    perturbation = np.nan_to_num(perturbation, nan=0.0, posinf=0.0, neginf=0.0)
    avg_perturbation = float(np.mean(perturbation)) if perturbation.size > 0 else 0.0
    max_perturbation = float(np.max(perturbation)) if perturbation.size > 0 else 0.0
    
    return {
        "original_accuracy": float(original_accuracy),
        "adversarial_accuracy": float(adversarial_accuracy),
        "attack_success_rate": float(attack_success_rate),
        "attack_success_count": int(attack_success),
        "total_samples": int(total_samples),
        "avg_perturbation": float(avg_perturbation),
        "max_perturbation": float(max_perturbation),
    }


def tune_epsilon(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon_range: List[float] = None,
    target_success_rate: float = 0.5,
    clip_min: float = 0.0,
    clip_max: float = 1.0
) -> Dict[str, any]:
    """
    Tune epsilon parameter to achieve target attack success rate.
    
    Args:
        model: Trained Keras model
        x: Input data (subset for tuning)
        y: True labels
        epsilon_range: List of epsilon values to test (default: [0.01, 0.05, 0.1, 0.15, 0.2])
        target_success_rate: Target attack success rate
        clip_min: Minimum value for clipping
        clip_max: Maximum value for clipping
    
    Returns:
        Dictionary with tuning results
    """
    if epsilon_range is None:
        epsilon_range = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    results = []
    
    for eps in epsilon_range:
        # Generate adversarial examples
        x_adv, _ = generate_fgsm_attack(
            model, x, y, eps=eps, clip_min=clip_min, clip_max=clip_max
        )
        
        # Evaluate attack
        metrics = evaluate_attack_success(model, x, x_adv, y)
        metrics["epsilon"] = eps
        results.append(metrics)
    
    # Find best epsilon (closest to target success rate)
    best_eps = None
    best_diff = float('inf')
    
    for result in results:
        diff = abs(result["attack_success_rate"] - target_success_rate)
        if diff < best_diff:
            best_diff = diff
            best_eps = result["epsilon"]
    
    return {
        "best_epsilon": best_eps,
        "target_success_rate": target_success_rate,
        "results": results,
    }


def generate_adversarial_dataset(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    eps: float = 0.05,
    batch_size: int = 32,
    clip_min: float = 0.0,
    clip_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate adversarial dataset for Bot-IoT or other tabular datasets.
    
    Args:
        model: Trained Keras model
        x: Original input data
        y: True labels
        eps: Perturbation magnitude
        batch_size: Batch size for processing
        clip_min: Minimum value for clipping
        clip_max: Maximum value for clipping
    
    Returns:
        Tuple of (adversarial examples, original labels)
    """
    x_adv_list = []
    
    # Process in batches
    num_batches = (len(x) + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(x))
        
        x_batch = x[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Generate adversarial examples for this batch
        x_adv_batch, _ = generate_fgsm_attack(
            model, x_batch, y_batch, eps=eps, clip_min=clip_min, clip_max=clip_max
        )
        
        x_adv_list.append(x_adv_batch)
    
    # Concatenate all batches
    x_adv = np.concatenate(x_adv_list, axis=0)
    
    return x_adv, y
