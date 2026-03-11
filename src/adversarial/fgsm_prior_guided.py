"""
Prior-Guided FGSM Initialization

Uses the global model's gradient history (momentum) from previous FL rounds
as a prior to initialise the FGSM perturbation direction.  Instead of
starting from zero (vanilla FGSM) or random noise (PGD), the perturbation
is seeded with the exponential moving average of past gradient signs,
then refined with a single FGSM step.

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
# Prior (Gradient Momentum) Tracker
# ---------------------------------------------------------------------------

class GradientPrior:
    """Maintains an exponential moving average of gradient signs across rounds.

    Call ``update()`` each FL round with the current model + data to refresh
    the prior.  The stored ``momentum`` tensor can then be used to seed the
    FGSM perturbation via ``prior_guided_fgsm()``.
    """

    def __init__(self, decay: float = 0.9):
        """
        Args:
            decay: EMA decay factor.  Higher = stronger memory of past
                   gradient directions.
        """
        self.decay = decay
        self.momentum: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def update(
        self,
        model: keras.Model,
        x: np.ndarray,
        y: np.ndarray,
        loss_fn=None,
    ) -> np.ndarray:
        """Compute current gradient sign and fold it into the EMA.

        Args:
            model:  Current global (or local) Keras model.
            x:      Input batch.
            y:      True labels.
            loss_fn: Optional override for the loss function.

        Returns:
            The updated momentum (sign-valued EMA, same shape as x).
        """
        gradients = compute_gradients(model, x, y, loss_fn=loss_fn)
        grad_sign = np.sign(gradients)
        grad_sign = np.nan_to_num(grad_sign, nan=0.0, posinf=0.0, neginf=0.0)

        if self.momentum is None:
            self.momentum = grad_sign
        else:
            # Feature-dimension EMA — broadcast-safe when batch sizes differ
            # between rounds.  We average over the batch dimension first so
            # ``momentum`` is (1, features).
            cur_mean = grad_sign.mean(axis=0, keepdims=True)
            if self.momentum.shape[0] != 1:
                self.momentum = self.momentum.mean(axis=0, keepdims=True)
            self.momentum = self.decay * self.momentum + (1 - self.decay) * cur_mean

        return self.momentum

    # ------------------------------------------------------------------
    def get_prior_direction(self, batch_size: int) -> Optional[np.ndarray]:
        """Return the momentum tiled to ``(batch_size, features)``.

        Returns None if ``update()`` has never been called.
        """
        if self.momentum is None:
            return None
        prior = self.momentum
        if prior.shape[0] == 1:
            prior = np.tile(prior, (batch_size, 1))
        return np.sign(prior)  # keep it in {-1, 0, +1}

    def reset(self):
        """Clear accumulated momentum (e.g. between experiments)."""
        self.momentum = None


# ---------------------------------------------------------------------------
# Prior-Guided FGSM Attack
# ---------------------------------------------------------------------------

def prior_guided_fgsm(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    prior: Optional[GradientPrior] = None,
    prior_weight: float = 0.5,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    loss_fn=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate adversarial examples using prior-guided FGSM.

    The perturbation direction is a weighted combination of:
        1. The prior (EMA of past gradient signs) — captures the *general*
           adversarial direction learned across FL rounds.
        2. The current gradient sign — captures the *sample-specific*
           adversarial direction for this batch.

    ``direction = prior_weight * prior_sign + (1 - prior_weight) * current_sign``

    The final perturbation is ``epsilon * sign(direction)``.

    Args:
        model:        Trained Keras model.
        x:            Clean inputs (batch_size, features).
        y:            True labels.
        epsilon:      L-inf perturbation budget.
        prior:        A ``GradientPrior`` instance (may be None on the first
                      round — falls back to vanilla FGSM).
        prior_weight: Mixing coefficient in [0, 1].  0 = pure current
                      gradient (vanilla FGSM), 1 = pure prior.
        clip_min:     Minimum clip value.
        clip_max:     Maximum clip value.
        loss_fn:      Optional loss function override.

    Returns:
        (x_adv, combined_direction)  — adversarial examples and the
        combined direction tensor (useful for diagnostics).
    """
    # Current gradient sign
    gradients = compute_gradients(model, x, y, loss_fn=loss_fn)
    current_sign = np.sign(gradients)
    current_sign = np.nan_to_num(current_sign, nan=0.0, posinf=0.0, neginf=0.0)

    # Combine with prior
    if prior is not None:
        prior_direction = prior.get_prior_direction(len(x))
    else:
        prior_direction = None

    if prior_direction is not None:
        combined = prior_weight * prior_direction + (1 - prior_weight) * current_sign
        direction = np.sign(combined)
    else:
        # No prior yet — fall back to vanilla FGSM
        direction = current_sign

    # Perturb
    x_adv = np.asarray(x + epsilon * direction, dtype=np.float32)
    if not np.isfinite(x_adv).all():
        x_adv = np.where(np.isfinite(x_adv), x_adv, np.asarray(x, dtype=np.float32))
    x_adv = np.clip(x_adv, clip_min, clip_max)

    return x_adv, direction


# ---------------------------------------------------------------------------
# Evaluate (reuses fgsm_hook metrics but included here for self-containment)
# ---------------------------------------------------------------------------

def evaluate_prior_guided(
    model: keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: float = 0.1,
    prior: Optional[GradientPrior] = None,
    prior_weight: float = 0.5,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run prior-guided FGSM and return standard attack metrics."""
    from src.adversarial.fgsm_hook import evaluate_attack_success

    x_adv, _ = prior_guided_fgsm(
        model, x, y,
        epsilon=epsilon,
        prior=prior,
        prior_weight=prior_weight,
    )
    return evaluate_attack_success(model, x, x_adv, y, threshold=threshold)
