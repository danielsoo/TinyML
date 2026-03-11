"""
Gradient-Aligned FGSM

Filters perturbation dimensions by alignment between the local client
gradient and the global (server) gradient.  Only features where both
gradients agree on sign contribute to the perturbation — the rest are
zeroed out.  This produces sparser, more targeted adversarial examples
that exploit directions the entire federation agrees are adversarial.

Reference architecture:
    - src/adversarial/fgsm_hook.py      (gradient computation helpers)
    - src/adversarial/pgd_adversarial_training.py (federated AT pattern)
    - src/federated/client.py            (per-client training loop)
    - src/federated/server.py            (FedAvgM aggregation)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.adversarial.fgsm_hook import compute_gradients


# ---------------------------------------------------------------------------
# Gradient-Aligned FGSM Attack
# ---------------------------------------------------------------------------

def gradient_aligned_fgsm(
    local_model: keras.Model,
    global_model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    alignment_threshold: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate adversarial examples using gradient-aligned FGSM.

    The perturbation is masked so that only feature dimensions where
    the local and global gradient signs *agree* are perturbed.

    Alignment for each feature j:
        ``alignment_j = sign(grad_local_j) * sign(grad_global_j)``
    If ``alignment_j > alignment_threshold`` the feature is perturbed;
    otherwise the perturbation for that feature is zero.

    Args:
        local_model:  Client's local Keras model (after local training).
        global_model: Current global/server Keras model.
        x:            Clean inputs (batch_size, features).
        y:            True labels.
        epsilon:      L-inf perturbation budget.
        alignment_threshold: Minimum per-feature alignment score to
                      include in perturbation.  0.0 means both signs
                      must match (product > 0).
        clip_min:     Minimum clip value.
        clip_max:     Maximum clip value.
        loss_fn:      Optional loss function override (used for both
                      local and global gradient computation).

    Returns:
        (x_adv, aligned_mask, alignment_stats)
        - x_adv: adversarial examples
        - aligned_mask: boolean mask of aligned features (batch, features)
        - alignment_stats: dict with alignment ratio, etc.
    """
    # Compute gradient signs from both models
    grad_local = compute_gradients(local_model, x, y, loss_fn=loss_fn)
    sign_local = np.sign(grad_local)
    sign_local = np.nan_to_num(sign_local, nan=0.0, posinf=0.0, neginf=0.0)

    grad_global = compute_gradients(global_model, x, y, loss_fn=loss_fn)
    sign_global = np.sign(grad_global)
    sign_global = np.nan_to_num(sign_global, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-feature alignment: +1 if same sign, -1 if opposite, 0 if either is zero
    alignment = sign_local * sign_global  # (batch, features)

    # Mask: only perturb features where alignment > threshold
    aligned_mask = alignment > alignment_threshold  # (batch, features)

    # Perturbation direction: use local gradient sign, masked by alignment
    direction = sign_local * aligned_mask.astype(np.float32)

    # Perturb
    x_adv = np.asarray(x + epsilon * direction, dtype=np.float32)
    if not np.isfinite(x_adv).all():
        x_adv = np.where(np.isfinite(x_adv), x_adv, np.asarray(x, dtype=np.float32))
    x_adv = np.clip(x_adv, clip_min, clip_max)

    # Alignment statistics
    total_features = alignment.shape[1] if alignment.ndim > 1 else alignment.shape[0]
    aligned_ratio = float(aligned_mask.mean())
    alignment_stats = {
        "aligned_feature_ratio": aligned_ratio,
        "total_features": int(total_features),
        "avg_aligned_per_sample": float(aligned_mask.sum(axis=-1).mean()),
    }

    return x_adv, aligned_mask, alignment_stats


# ---------------------------------------------------------------------------
# Single-Model Convenience Wrapper
# ---------------------------------------------------------------------------

def gradient_aligned_fgsm_single_model(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    global_gradients: np.ndarray,
    epsilon: float = 0.1,
    alignment_threshold: float = 0.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Variant that accepts pre-computed global gradient signs.

    Useful when you only have access to a single model locally but have
    received the global gradient sign vector from the server (e.g. as
    part of aggregation metadata).

    Args:
        model:            Local Keras model.
        x:                Clean inputs.
        y:                True labels.
        global_gradients: Pre-computed global gradient signs (features,)
                          or (1, features).  Will be broadcast.
        epsilon:          L-inf budget.
        alignment_threshold: Same as ``gradient_aligned_fgsm``.
        clip_min/clip_max:  Clip bounds.
        loss_fn:          Optional loss function.

    Returns:
        Same as ``gradient_aligned_fgsm``.
    """
    grad_local = compute_gradients(model, x, y, loss_fn=loss_fn)
    sign_local = np.sign(grad_local)
    sign_local = np.nan_to_num(sign_local, nan=0.0, posinf=0.0, neginf=0.0)

    sign_global = np.sign(global_gradients)
    sign_global = np.nan_to_num(sign_global, nan=0.0, posinf=0.0, neginf=0.0)
    if sign_global.ndim == 1:
        sign_global = sign_global[np.newaxis, :]  # (1, features) for broadcast

    alignment = sign_local * sign_global
    aligned_mask = alignment > alignment_threshold

    direction = sign_local * aligned_mask.astype(np.float32)

    x_adv = np.asarray(x + epsilon * direction, dtype=np.float32)
    if not np.isfinite(x_adv).all():
        x_adv = np.where(np.isfinite(x_adv), x_adv, np.asarray(x, dtype=np.float32))
    x_adv = np.clip(x_adv, clip_min, clip_max)

    total_features = alignment.shape[1] if alignment.ndim > 1 else alignment.shape[0]
    aligned_ratio = float(aligned_mask.mean())
    alignment_stats = {
        "aligned_feature_ratio": aligned_ratio,
        "total_features": int(total_features),
        "avg_aligned_per_sample": float(aligned_mask.sum(axis=-1).mean()),
    }

    return x_adv, aligned_mask, alignment_stats


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_gradient_aligned(
    local_model: keras.Model,
    global_model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    alignment_threshold: float = 0.0,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run gradient-aligned FGSM and return standard attack metrics + alignment stats."""
    from src.adversarial.fgsm_hook import evaluate_attack_success

    x_adv, _, align_stats = gradient_aligned_fgsm(
        local_model, global_model, x, y,
        epsilon=epsilon,
        alignment_threshold=alignment_threshold,
    )
    metrics = evaluate_attack_success(local_model, x, x_adv, y, threshold=threshold)
    metrics.update(align_stats)
    return metrics
