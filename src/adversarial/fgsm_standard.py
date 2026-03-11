"""
Standard (Vanilla) FGSM

A thin wrapper around the existing fgsm_hook.py primitives that presents
the same interface as the other FGSM variants (prior-guided, gradient-
aligned).  This makes it easy to swap techniques via the config selector
without changing calling code.

Reference architecture:
    - src/adversarial/fgsm_hook.py      (core implementation)
    - src/adversarial/pgd_adversarial_training.py (federated AT pattern)
    - src/federated/client.py            (per-client training loop)
    - src/federated/server.py            (FedAvgM aggregation)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from tensorflow import keras

from src.adversarial.fgsm_hook import (
    compute_gradients,
    evaluate_attack_success,
    generate_fgsm_attack,
)


# ---------------------------------------------------------------------------
# Standard FGSM Attack (wraps fgsm_hook.generate_fgsm_attack)
# ---------------------------------------------------------------------------

def standard_fgsm(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate adversarial examples using vanilla FGSM.

    ``x_adv = x + epsilon * sign(grad_x L(model, x, y))``

    Args:
        model:    Trained Keras model.
        x:        Clean inputs (batch_size, features).
        y:        True labels.
        epsilon:  L-inf perturbation budget.
        clip_min: Minimum clip value.
        clip_max: Maximum clip value.
        loss_fn:  Optional loss function override (unused here — kept
                  for interface parity with other variants).

    Returns:
        (x_adv, grad_sign)
    """
    return generate_fgsm_attack(
        model, x, y,
        eps=epsilon,
        clip_min=clip_min,
        clip_max=clip_max,
    )


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate_standard(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run standard FGSM and return attack metrics."""
    x_adv, _ = standard_fgsm(model, x, y, epsilon=epsilon)
    return evaluate_attack_success(model, x, x_adv, y, threshold=threshold)
