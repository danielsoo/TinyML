# /src/adversarial/fgsm_hook.py
from __future__ import annotations
import numpy as np

def fgsm_perturb(x: np.ndarray, grad_sign: np.ndarray, eps: float = 0.05) -> np.ndarray:
    x_adv = x + eps * grad_sign
    return np.clip(x_adv, 0.0, 1.0)
