"""
Experiment Sweep: FL → Distillation → Pruning → PTQ

Runs all combinations defined in the experiment table:

  FL           | Distillation | Pruning    | PTQ
  -------------|--------------|------------|----
  No-QAT       | Direct       | 10% × 5   | Yes
  Yes-QAT(std) | Progressive  | 10% × 2   | No
               | None         | 5% × 10   |
               |              | None       |

Each combination:
  1. FL training (shared — re-uses pre-trained model if --skip-train given)
  2. Distillation (direct / progressive / none)
  3. Iterative pruning (steps × ratio / none)
  4. PTQ / TFLite export (yes / no)
  5. Evaluate + write per-experiment row to summary CSV/MD

Usage:
    # Full sweep (trains twice: no-QAT and yes-QAT)
    python scripts/run_experiment_sweep.py --config config/federated_local.yaml

    # Skip training (re-use existing models/global_model_noqat.h5 + models/global_model_qat.h5)
    python scripts/run_experiment_sweep.py --skip-train

    # Dry-run: print all experiment configs without running anything
    python scripts/run_experiment_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import shutil
import tempfile
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

# Windows CP949 terminals can't encode emoji characters used in export_tflite.py.
# Reconfigure stdout/stderr to UTF-8 so emoji print statements don't crash.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml


# ══════════════════════════════════════════════════════════════════════════════
# Experiment grid
# ══════════════════════════════════════════════════════════════════════════════

FL_VARIANTS = [
    {"use_qat": False, "tag": "no_qat"},
    {"use_qat": True,  "tag": "yes_qat"},
]

DISTILLATION_VARIANTS = [
    {"mode": "direct",       "tag": "distill_direct"},
    {"mode": "progressive",  "tag": "distill_progressive"},
    {"mode": "none",         "tag": "distill_none"},
]

# (prune_ratio_per_step, num_steps)  — cumulative = 1-(1-ratio)^steps
PRUNING_VARIANTS = [
    {"ratio": 0.10, "steps": 5, "tag": "prune_10x5"},
    {"ratio": 0.10, "steps": 2, "tag": "prune_10x2"},
    {"ratio": 0.05, "steps": 10, "tag": "prune_5x10"},
    {"ratio": 0.0,  "steps": 0, "tag": "prune_none"},
]

PTQ_VARIANTS = [
    {"ptq": True,  "tag": "ptq_yes"},
    {"ptq": False, "tag": "ptq_no"},
]


def build_experiment_list() -> List[Dict[str, Any]]:
    """Cross-product of all variants → list of experiment dicts."""
    exps = []
    for fl, dist, prune, ptq in product(FL_VARIANTS, DISTILLATION_VARIANTS,
                                         PRUNING_VARIANTS, PTQ_VARIANTS):
        tag = "__".join([fl["tag"], dist["tag"], prune["tag"], ptq["tag"]])
        exps.append({
            "tag": tag,
            "fl":    fl,
            "dist":  dist,
            "prune": prune,
            "ptq":   ptq,
        })
    return exps


# ══════════════════════════════════════════════════════════════════════════════
# FL training helpers
# ══════════════════════════════════════════════════════════════════════════════

def _write_temp_config(base_cfg: dict, use_qat: bool) -> str:
    """Write a temp YAML with use_qat overridden; return temp file path."""
    cfg = {**base_cfg}
    cfg["federated"] = {**base_cfg.get("federated", {}), "use_qat": use_qat}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
    tmp.close()
    return tmp.name


def train_fl(base_cfg: dict, use_qat: bool, out_path: Path) -> bool:
    """Run FL training via scripts/train.py (same path as run.py) and save to out_path.

    Uses scripts/train.py so GPU setup, env detection, and the src/models copy
    all happen exactly as in the normal run.py pipeline.
    --skip-gpu-check / --skip-deps / --skip-data-check suppress the interactive
    setup sections that are redundant inside a sweep.
    """
    import subprocess
    tmp_cfg = _write_temp_config(base_cfg, use_qat)
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", tmp_cfg,
        "--save-model", str(out_path),
        "--skip-gpu-check",
        "--skip-deps",
        "--skip-data-check",
    ]
    print(f"\n  [FL] use_qat={use_qat}  →  {out_path.name}")
    try:
        ret = subprocess.run(cmd, check=False)
        return ret.returncode == 0
    finally:
        Path(tmp_cfg).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (cached across experiments)
# ══════════════════════════════════════════════════════════════════════════════

_DATA_CACHE: Optional[Tuple] = None

def get_data(cfg: dict):
    """Load and cache train/test data (called once per sweep)."""
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE

    from src.data.loader import load_dataset
    data_cfg = cfg.get("data", {})
    dataset_name = data_cfg.get("name", "cicids2017")
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    x_tr, y_tr, x_te, y_te = load_dataset(dataset_name, **kwargs)
    _DATA_CACHE = (x_tr, y_tr, x_te, y_te)
    return _DATA_CACHE


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_keras_model(path: Path, cfg: dict):
    """Load a .h5 model saved by client.py and recompile with the correct loss.

    Both QAT and no-QAT models are saved with standalone Keras 3 (from nets.py).
    For QAT models, quantize_scope is added so QuantizeWrapper layers deserialise
    correctly if tfmot was active during training.
    """
    import keras as _keras
    from src.models.nets import _focal_loss

    fed_cfg = cfg.get("federated", {})
    use_qat = fed_cfg.get("use_qat", False)
    use_focal = fed_cfg.get("use_focal_loss", False)
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))
    loss = _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal else "binary_crossentropy"

    if use_qat:
        try:
            import tensorflow_model_optimization as tfmot
            import tf_keras
            with tfmot.quantization.keras.quantize_scope():
                model = tf_keras.models.load_model(str(path), compile=False)
        except Exception:
            import tf_keras
            model = tf_keras.models.load_model(str(path), compile=False)
        model.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=loss,
            metrics=["accuracy"],
        )
    else:
        model = _keras.saving.load_model(str(path), compile=False)
        model.compile(
            optimizer=_keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=loss,
            metrics=["accuracy"],
        )
    return model


def evaluate_keras(model, x_te, y_te) -> Dict[str, float]:
    """Evaluate Keras model; return accuracy, precision, recall, f1."""
    import tensorflow as tf
    y_prob = model.predict(x_te, verbose=0)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.ravel()
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_te.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_tflite(tflite_path: Path, x_te, y_te) -> Dict[str, float]:
    """Evaluate TFLite model; return same metrics as evaluate_keras."""
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    preds = []
    for i in range(len(x_te)):
        sample = x_te[i:i+1].astype(np.float32)
        interp.set_tensor(inp["index"], sample)
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0])

    y_prob = np.array(preds, dtype=np.float32)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.ravel()
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = y_te.astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    size_kb = tflite_path.stat().st_size / 1024
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "size_kb": size_kb}


def model_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024 if path.exists() else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Distillation
# ══════════════════════════════════════════════════════════════════════════════

def _make_student_tfkeras(teacher, compression_ratio: float = 0.5):
    """
    Build a student model (tf_keras Sequential) mirroring the teacher's Dense
    layers at compression_ratio width.  Output is sigmoid [?,1] so it matches
    the binary teacher and works with binary_crossentropy fine-tuning.

    Bypasses create_student_model() from distillation.py, which emits
    Dense(2, softmax) — incompatible with the binary teacher's [?,1] output
    in distillation_loss_fn's tf.cond (both branches are traced at graph-build
    time, causing a shape-mismatch squeeze error).
    """
    import tf_keras
    from tf_keras import layers as tfl

    input_shape = teacher.input_shape[1:]
    dense_layers = [l for l in teacher.layers if type(l).__name__ == "Dense"]

    seq_layers = [tf_keras.Input(shape=input_shape)]
    for layer in dense_layers[:-1]:  # hidden layers
        units = max(1, int(layer.units * compression_ratio))
        seq_layers.append(tfl.Dense(units, activation="relu"))
        seq_layers.append(tfl.BatchNormalization())
        seq_layers.append(tfl.Dropout(0.25))
    # Output: sigmoid [?,1] — same shape as binary teacher
    seq_layers.append(tfl.Dense(1, activation="sigmoid"))

    model = tf_keras.Sequential(seq_layers)
    model.build((None,) + input_shape)
    return model


def _distill_one_pair(teacher, student, x_tr, y_tr, x_te, y_te,
                      temperature: float, alpha: float,
                      epochs: int, batch_size: int, lr: float,
                      use_focal: bool, focal_alpha: float):
    """
    Train student using knowledge distillation from teacher.

    Both teacher and student are expected to output [?,1] sigmoid probabilities
    (binary classification). Loss = alpha * soft_KL + (1-alpha) * hard_BCE.

    The student is compiled with binary_crossentropy for hard labels.
    Soft targets from the teacher are injected via a custom training loop
    to avoid the tf.cond static-graph shape-mismatch in distillation.py.
    """
    import tensorflow as tf
    import tf_keras
    from src.models.nets import _focal_loss

    hard_loss_fn = (_focal_loss(gamma=2.0, alpha=focal_alpha)
                    if use_focal else tf.keras.losses.binary_crossentropy)
    optimizer = tf_keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    # Freeze teacher
    teacher.trainable = False

    # No @tf.function: teacher may be a Keras 3 model while student is tf_keras;
    # mixing frameworks inside a single traced function causes dispatch errors.
    def train_step(x_batch, y_batch):
        x_t = tf.constant(x_batch, dtype=tf.float32)
        y_t = tf.constant(y_batch, dtype=tf.float32)

        teacher_pred = teacher(x_t, training=False)              # [B,1] sigmoid
        t_p = tf.squeeze(teacher_pred, axis=-1)                  # [B]
        eps = 1e-7
        # Soft targets from teacher: 2-class log-odds then softmax at temperature
        teacher_logits = tf.stack(
            [tf.math.log(1 - t_p + eps), tf.math.log(t_p + eps)], axis=-1)  # [B,2]
        teacher_soft = tf.nn.softmax(teacher_logits / temperature)           # [B,2]

        with tf.GradientTape() as tape:
            s_pred = student(x_t, training=True)                 # [B,1] sigmoid
            s_p = tf.squeeze(s_pred, axis=-1)                    # [B]
            # Student soft targets (2-class)
            student_logits = tf.stack(
                [tf.math.log(1 - s_p + eps), tf.math.log(s_p + eps)], axis=-1)  # [B,2]
            student_soft = tf.nn.softmax(student_logits / temperature)           # [B,2]

            # KL / cross-entropy distillation loss (soft)
            soft_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            ) * (temperature ** 2)

            # Hard label loss — reshape y to [B,1] to match s_pred [B,1]
            y_2d = tf.reshape(y_t, [-1, 1])
            hard_loss = tf.reduce_mean(hard_loss_fn(y_2d, s_pred))

            loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
        return loss

    n = len(x_tr)
    for ep in range(epochs):
        idx = np.random.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            bi = idx[start:start + batch_size]
            xb = x_tr[bi].astype(np.float32)
            yb = y_tr[bi].astype(np.float32)
            l = train_step(xb, yb)
            losses.append(float(l.numpy()))
        print(f"      epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}")

    teacher.trainable = True
    return student


def run_distillation(mode: str, teacher, x_tr, y_tr, x_te, y_te,
                     cfg: dict) -> Any:
    """
    Run direct or progressive distillation.
    Returns the final student tf_keras model.

    Uses a self-contained distillation loop (_distill_one_pair) instead of
    distillation.py's train_with_distillation, which has a static-graph
    shape-mismatch when teacher=[?,1] and student=[?,2].
    """
    fed_cfg = cfg.get("federated", {})
    dist_epochs = int(cfg.get("distillation", {}).get("epochs", 10))
    batch_size  = int(fed_cfg.get("batch_size", 128))
    lr          = float(fed_cfg.get("learning_rate", 0.001))
    temperature = float(cfg.get("distillation", {}).get("temperature", 3.0))
    alpha       = float(cfg.get("distillation", {}).get("alpha", 0.3))
    use_focal   = bool(fed_cfg.get("use_focal_loss", False))
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))

    if mode == "direct":
        student = _make_student_tfkeras(teacher, compression_ratio=0.5)
        student = _distill_one_pair(
            teacher, student, x_tr, y_tr, x_te, y_te,
            temperature=temperature, alpha=alpha,
            epochs=dist_epochs, batch_size=batch_size, lr=lr,
            use_focal=use_focal, focal_alpha=focal_alpha,
        )
        return student

    elif mode == "progressive":
        # Stage 1: teacher → intermediate (0.5× width)
        inter = _make_student_tfkeras(teacher, compression_ratio=0.5)
        inter = _distill_one_pair(
            teacher, inter, x_tr, y_tr, x_te, y_te,
            temperature=temperature, alpha=alpha,
            epochs=dist_epochs, batch_size=batch_size, lr=lr,
            use_focal=use_focal, focal_alpha=focal_alpha,
        )
        # Stage 2: intermediate → final student (0.5× again → 0.25× of original)
        student = _make_student_tfkeras(inter, compression_ratio=0.5)
        student = _distill_one_pair(
            inter, student, x_tr, y_tr, x_te, y_te,
            temperature=temperature, alpha=alpha,
            epochs=dist_epochs, batch_size=batch_size, lr=lr,
            use_focal=use_focal, focal_alpha=focal_alpha,
        )
        return student

    else:
        raise ValueError(f"Unknown distillation mode: {mode}")


# ══════════════════════════════════════════════════════════════════════════════
# Iterative pruning
# ══════════════════════════════════════════════════════════════════════════════

def _to_tfkeras_model(model):
    """
    Rebuild an equivalent tf_keras Sequential model from a Keras 3 model's weights.

    pruning.py uses `from tensorflow import keras` (tf_keras in WSL) for isinstance
    checks. A model built with standalone Keras 3 has layers of type
    keras.src.layers.Dense — not tf_keras.layers.Dense — so isinstance returns False
    and no neurons get pruned. Rebuilding with tf_keras makes the checks work.

    Architecture is inferred from the original layer sequence; only Dense,
    BatchNormalization, and Dropout layers are reconstructed (covers our MLP).
    """
    import tf_keras
    from tf_keras import layers as tfl

    weights = model.get_weights()
    input_shape = model.input_shape[1:]

    tf_layers = [tf_keras.Input(shape=input_shape)]
    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "Dense":
            tf_layers.append(
                tfl.Dense(layer.units,
                          activation=layer.activation,
                          name=layer.name)
            )
        elif cls == "BatchNormalization":
            tf_layers.append(tfl.BatchNormalization(name=layer.name))
        elif cls == "Dropout":
            tf_layers.append(tfl.Dropout(layer.rate, name=layer.name))

    tf_model = tf_keras.Sequential(tf_layers)
    tf_model.build((None,) + input_shape)
    tf_model.set_weights(weights)
    return tf_model


def _strip_tfmot(model):
    """Strip tfmot QuantizeWrapper layers before pruning (QAT models only)."""
    try:
        import tensorflow_model_optimization as tfmot
        return tfmot.quantization.keras.strip_quant_dequant(model)
    except Exception:
        pass
    return model


def _prune_one_step(model, prune_ratio: float):
    """
    One step of structured Dense-layer pruning, fully self-contained in tf_keras.

    apply_structured_pruning() from pruning.py mixes `from tensorflow import keras`
    (which resolves to Keras 3 or tf_keras depending on environment) with tf_keras
    layer instances, causing cross-framework tensor errors. This function reimplements
    only the Dense-pruning logic using pure numpy (prune_dense_layer) and rebuilds
    the model exclusively with tf_keras to avoid any cross-framework tensor calls.
    """
    import tf_keras
    from tf_keras import layers as tfl
    from src.modelcompression.pruning import prune_dense_layer

    weights_by_layer = {l.name: l.get_weights() for l in model.layers}
    input_shape = model.input_shape[1:]

    # Identify prunable Dense layers (all except the last Dense = output layer)
    dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
    prunable = set(l.name for l in dense_layers[:-1])  # skip output layer

    # First pass: determine kept indices per Dense layer
    kept: dict = {}
    prev_kept = None
    for layer in model.layers:
        if type(layer).__name__ != "Dense":
            continue
        ws = weights_by_layer[layer.name]
        w, bias = ws[0], (ws[1] if len(ws) > 1 else None)
        if prev_kept is not None:
            w = w[prev_kept, :]
        if layer.name in prunable:
            w, bias, kept_idx = prune_dense_layer(w, bias, prune_ratio)
            kept[layer.name] = kept_idx
            prev_kept = kept_idx
        else:
            kept[layer.name] = None
            prev_kept = None

    # Second pass: build pruned tf_keras Sequential
    new_tf_layers = [tf_keras.Input(shape=input_shape)]
    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "Dense":
            idx = kept.get(layer.name)
            units = len(idx) if idx is not None else layer.units
            new_tf_layers.append(
                tfl.Dense(units, activation=layer.activation,
                          name=layer.name + "_p")
            )
        elif cls == "BatchNormalization":
            new_tf_layers.append(tfl.BatchNormalization(name=layer.name + "_p"))
        elif cls == "Dropout":
            new_tf_layers.append(tfl.Dropout(layer.rate, name=layer.name + "_p"))

    new_model = tf_keras.Sequential(new_tf_layers)
    new_model.build((None,) + input_shape)

    # Third pass: copy pruned weights
    prev_kept = None
    for layer in model.layers:
        cls = type(layer).__name__
        new_name = layer.name + "_p"
        new_layer = next((l for l in new_model.layers if l.name == new_name), None)
        if new_layer is None:
            continue
        if cls == "Dense":
            ws = weights_by_layer[layer.name]
            w, bias = ws[0], (ws[1] if len(ws) > 1 else None)
            if prev_kept is not None:
                w = w[prev_kept, :]
            idx = kept.get(layer.name)
            if idx is not None:
                w = w[:, idx]
                bias = bias[idx] if bias is not None else None
                prev_kept = idx
            else:
                prev_kept = None
            new_layer.set_weights([w, bias] if bias is not None else [w])
        elif cls == "BatchNormalization":
            # BN has 4 weight arrays: gamma, beta, moving_mean, moving_var
            # all shaped (units,) matching the preceding Dense output.
            # Slice them to the kept indices of the last pruned Dense.
            bn_ws = weights_by_layer[layer.name]
            if prev_kept is not None and len(bn_ws) > 0 and len(bn_ws[0]) != new_layer.get_weights()[0].shape[0]:
                bn_ws = [w[prev_kept] for w in bn_ws]
            new_layer.set_weights(bn_ws)

    return new_model


def run_iterative_pruning(model, prune_ratio: float, steps: int,
                          x_tr, y_tr, x_te, y_te,
                          cfg: dict):
    """
    Apply structured pruning iteratively, fine-tuning after each step.
    Returns the final pruned tf_keras model.
    """
    import tf_keras
    from src.models.nets import _focal_loss

    fed_cfg = cfg.get("federated", {})
    ft_epochs  = int(cfg.get("pruning", {}).get("finetune_epochs", 2))
    batch_size = int(fed_cfg.get("batch_size", 128))

    use_focal  = fed_cfg.get("use_focal_loss", False)
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))
    loss_fn    = _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal else "binary_crossentropy"

    def _opt():
        return tf_keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

    # Strip QAT wrappers (no-op for non-QAT), then convert to tf_keras
    current = _strip_tfmot(model)
    current = _to_tfkeras_model(current)

    # Focal loss does arithmetic on y_true; cast to float32 to avoid int64 type errors
    y_tr_f = y_tr.astype(np.float32)

    for step in range(steps):
        print(f"    [Prune] Step {step+1}/{steps} - ratio per step={prune_ratio:.0%}")
        current = _prune_one_step(current, prune_ratio)
        current.compile(optimizer=_opt(), loss=loss_fn, metrics=["accuracy"])
        current.fit(x_tr, y_tr_f, epochs=ft_epochs, batch_size=batch_size, verbose=0)

    return current


# ══════════════════════════════════════════════════════════════════════════════
# PTQ / TFLite export
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize_model_weights(model):
    """Replace any NaN/Inf weights with 0 to prevent NaN scales in TFLite INT8 export."""
    fixed = False
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            continue
        new_ws = []
        for w in ws:
            w_arr = np.array(w)
            if np.any(~np.isfinite(w_arr)):
                w_arr = np.nan_to_num(w_arr, nan=0.0, posinf=0.0, neginf=0.0)
                fixed = True
            new_ws.append(w_arr)
        if fixed:
            layer.set_weights(new_ws)
    if fixed:
        print("  ⚠️  NaN/Inf weights found and replaced with 0 before PTQ export")


def run_ptq(model, out_path: Path, x_tr) -> Path:
    """Export model to TFLite with dynamic-range quantization (int8 weights).

    Sanitizes NaN/Inf weights before conversion to prevent 'unsupported scale
    value (-nan)' errors that occur when zero-range or degenerate activations
    result in undefined INT8 scale values.
    """
    import tensorflow as tf
    from src.tinyml.export_tflite import export_tflite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _sanitize_model_weights(model)
    rep_data = x_tr[:500].astype(np.float32)
    try:
        export_tflite(model, str(out_path), quantize=True, representative_data=rep_data)
    except Exception as e:
        if "nan" in str(e).lower() or "scale" in str(e).lower():
            # NaN scale: fall back to float32 TFLite (no quantization)
            print(f"  ⚠️  PTQ INT8 failed ({e}), falling back to float32 TFLite")
            fallback_path = out_path.parent / (out_path.stem + "_float32.tflite")
            export_tflite(model, str(fallback_path), quantize=False)
            return fallback_path
        raise
    return out_path


def run_float_export(model, out_path: Path) -> Path:
    """Export model to TFLite float32 (no quantization)."""
    from src.tinyml.export_tflite import export_tflite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_tflite(model, str(out_path), quantize=False)
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Single experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(exp: Dict[str, Any], fl_model_path: Path, cfg: dict,
                   out_dir: Path) -> Dict[str, Any]:
    """
    Run one experiment:
      (optionally distill) → (optionally prune) → (optionally ptq) → evaluate

    Returns a result dict ready for CSV/MD output.
    """
    tag = exp["tag"]
    dist_cfg  = exp["dist"]
    prune_cfg = exp["prune"]
    ptq_cfg   = exp["ptq"]

    exp_dir = out_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*70}")
    print(f"  Experiment: {tag}")
    print(f"{'─'*70}")

    x_tr, y_tr, x_te, y_te = get_data(cfg)

    # ── load FL model ────────────────────────────────────────────────────────
    model = load_keras_model(fl_model_path, cfg)
    fl_metrics = evaluate_keras(model, x_te, y_te)
    fl_size_kb = model_size_kb(fl_model_path)
    print(f"  FL model   acc={fl_metrics['accuracy']:.4f}  f1={fl_metrics['f1']:.4f}  "
          f"size={fl_size_kb:.1f} KB")

    # ── distillation ─────────────────────────────────────────────────────────
    if dist_cfg["mode"] != "none":
        # Cache distilled model by (fl_tag, dist_mode) so it is only trained once
        fl_tag = exp["fl"]["tag"]
        dist_cache_dir = Path("models/distilled")
        dist_cache_dir.mkdir(parents=True, exist_ok=True)
        dist_cache_path = dist_cache_dir / f"{fl_tag}_{dist_cfg['mode']}.h5"

        if dist_cache_path.exists() and dist_cache_path.stat().st_size > 10_000:
            print(f"  Distillation: {dist_cfg['mode']} [cached] ...")
            model = load_keras_model(dist_cache_path, cfg)
        else:
            print(f"  Distillation: {dist_cfg['mode']} ...")
            try:
                model = run_distillation(dist_cfg["mode"], model,
                                         x_tr, y_tr, x_te, y_te, cfg)
                model.save(str(dist_cache_path))
                print(f"  Saved distilled model -> {dist_cache_path}")
            except Exception as e:
                print(f"  Distillation failed: {e}")

        dist_metrics = evaluate_keras(model, x_te, y_te)
        print(f"  After distill  acc={dist_metrics['accuracy']:.4f}  "
              f"f1={dist_metrics['f1']:.4f}")
    else:
        dist_metrics = None

    # ── iterative pruning ────────────────────────────────────────────────────
    if prune_cfg["steps"] > 0:
        print(f"  Pruning: {prune_cfg['ratio']:.0%} × {prune_cfg['steps']} steps ...")
        try:
            model = run_iterative_pruning(
                model, prune_cfg["ratio"], prune_cfg["steps"],
                x_tr, y_tr, x_te, y_te, cfg,
            )
            prune_path = exp_dir / "pruned_model.h5"
            model.save(str(prune_path))
        except Exception as e:
            print(f"  ⚠️  Pruning failed: {e}")
        prune_metrics = evaluate_keras(model, x_te, y_te)
        print(f"  After prune    acc={prune_metrics['accuracy']:.4f}  "
              f"f1={prune_metrics['f1']:.4f}")
    else:
        prune_metrics = None

    # ── PTQ / TFLite export ──────────────────────────────────────────────────
    tflite_metrics = None
    tflite_size_kb = None
    if ptq_cfg["ptq"]:
        print(f"  PTQ export ...")
        tflite_path = exp_dir / "model_ptq.tflite"
        try:
            actual_tflite_path = run_ptq(model, tflite_path, x_tr)
            tflite_metrics = evaluate_tflite(actual_tflite_path, x_te, y_te)
            tflite_size_kb = tflite_metrics.pop("size_kb", None)
            print(f"  After PTQ      acc={tflite_metrics['accuracy']:.4f}  "
                  f"f1={tflite_metrics['f1']:.4f}  "
                  f"size={tflite_size_kb:.1f} KB")
        except Exception as e:
            print(f"  ⚠️  PTQ failed: {e}")
    else:
        # Export float32 TFLite anyway (for size reference)
        tflite_path = exp_dir / "model_float.tflite"
        try:
            run_float_export(model, tflite_path)
            tflite_size_kb = model_size_kb(tflite_path)
        except Exception as e:
            print(f"  ⚠️  Float TFLite export failed: {e}")

    # Final Keras model size (after all compression)
    final_keras_path = exp_dir / "final_model.h5"
    try:
        model.save(str(final_keras_path))
        final_size_kb = model_size_kb(final_keras_path)
    except Exception as e:
        print(f"  ⚠️  Could not save final model: {e}")
        final_size_kb = None

    # ── result dict ──────────────────────────────────────────────────────────
    def _m(d, k, fmt=".4f"):
        return f"{d[k]:{fmt}}" if d and k in d else "-"

    result = {
        "tag":           tag,
        "fl_qat":        exp["fl"]["use_qat"],
        "distillation":  dist_cfg["mode"],
        "pruning":       exp["prune"]["tag"],
        "ptq":           ptq_cfg["ptq"],
        # FL baseline
        "fl_acc":        fl_metrics["accuracy"],
        "fl_f1":         fl_metrics["f1"],
        "fl_prec":       fl_metrics["precision"],
        "fl_rec":        fl_metrics["recall"],
        "fl_size_kb":    fl_size_kb,
        # After distillation
        "dist_acc":      dist_metrics["accuracy"] if dist_metrics else None,
        "dist_f1":       dist_metrics["f1"] if dist_metrics else None,
        # After pruning
        "prune_acc":     prune_metrics["accuracy"] if prune_metrics else None,
        "prune_f1":      prune_metrics["f1"] if prune_metrics else None,
        # Final (TFLite or Keras)
        "final_acc":     tflite_metrics["accuracy"] if tflite_metrics else (
                             evaluate_keras(model, x_te, y_te)["accuracy"] if model else None),
        "final_f1":      tflite_metrics["f1"] if tflite_metrics else (
                             evaluate_keras(model, x_te, y_te)["f1"] if model else None),
        "final_prec":    tflite_metrics["precision"] if tflite_metrics else None,
        "final_rec":     tflite_metrics["recall"] if tflite_metrics else None,
        "tflite_size_kb": tflite_size_kb,
        "final_size_kb": final_size_kb,
    }

    # Save per-experiment JSON
    with open(exp_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Report writers
# ══════════════════════════════════════════════════════════════════════════════

_CSV_FIELDS = [
    "tag", "fl_qat", "distillation", "pruning", "ptq",
    "fl_acc", "fl_f1", "fl_prec", "fl_rec", "fl_size_kb",
    "dist_acc", "dist_f1",
    "prune_acc", "prune_f1",
    "final_acc", "final_f1", "final_prec", "final_rec",
    "tflite_size_kb", "final_size_kb",
]


def write_csv(results: List[Dict], path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in _CSV_FIELDS})
    print(f"  CSV  → {path}")


def _pct(v) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v)*100:.2f}%"
    except (TypeError, ValueError):
        return str(v)


def _kb(v) -> str:
    if v is None or v == "":
        return "—"
    try:
        return f"{float(v):.1f} KB"
    except (TypeError, ValueError):
        return str(v)


def write_markdown(results: List[Dict], path: Path):
    """Write a comparison table as Markdown."""
    lines = [
        "# Experiment Sweep — Results\n",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "FGSM step is planned as a separate phase (not included here).\n",
        "",
        "## Summary Table\n",
        "| Tag | FL QAT | Distillation | Pruning | PTQ "
        "| FL Acc | FL F1 | FL size "
        "| Dist Acc | Dist F1 "
        "| Prune Acc | Prune F1 "
        "| Final Acc | Final F1 | Final Prec | Final Rec "
        "| TFLite size | Final size |",
        "|-----|--------|--------------|---------|-----|"
        "--------|-------|---------|"
        "----------|---------|"
        "-----------|----------|"
        "-----------|----------|------------|-----------|"
        "------------|------------|",
    ]
    for r in results:
        row = " | ".join([
            r["tag"],
            "✓" if r["fl_qat"] else "✗",
            r["distillation"],
            r["pruning"].replace("prune_", ""),
            "✓" if r["ptq"] else "✗",
            _pct(r.get("fl_acc")),
            _pct(r.get("fl_f1")),
            _kb(r.get("fl_size_kb")),
            _pct(r.get("dist_acc")),
            _pct(r.get("dist_f1")),
            _pct(r.get("prune_acc")),
            _pct(r.get("prune_f1")),
            _pct(r.get("final_acc")),
            _pct(r.get("final_f1")),
            _pct(r.get("final_prec")),
            _pct(r.get("final_rec")),
            _kb(r.get("tflite_size_kb")),
            _kb(r.get("final_size_kb")),
        ])
        lines.append(f"| {row} |")

    lines += [
        "",
        "## Column Legend",
        "- **FL Acc / F1** — metrics immediately after FL training",
        "- **Dist Acc / F1** — metrics after distillation (if applied)",
        "- **Prune Acc / F1** — metrics after iterative pruning (if applied)",
        "- **Final Acc / F1 / Prec / Rec** — metrics on the deployed model "
          "(TFLite if PTQ=✓, Keras otherwise)",
        "- **TFLite size** — exported .tflite size",
        "- **Final size** — Keras .h5 size after all compression steps",
        "",
        "## Pruning Key",
        "- `10x5` — 10% pruned per step × 5 steps  (cumulative ≈ 41% removed)",
        "- `10x2` — 10% pruned per step × 2 steps  (cumulative ≈ 19% removed)",
        "- `5x10` — 5% pruned per step × 10 steps  (cumulative ≈ 40% removed)",
        "- `none` — no pruning",
        "",
        "## Distillation Key",
        "- `direct` — teacher → student in one step (0.5× width)",
        "- `progressive` — teacher → intermediate → student (0.25× width)",
        "- `none` — no distillation",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  MD   → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment sweep (FL → Distill → Prune → PTQ)"
    )
    parser.add_argument(
        "--config", default="config/federated_local.yaml",
        help="Base FL config (data, model, federated settings)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip FL training — reuse models/global_model_noqat.h5 and "
             "models/global_model_qat.h5 if they exist",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sweep",
        help="Root output directory for all experiment results",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all experiment configs without running anything",
    )
    parser.add_argument(
        "--filter", default="",
        help="Only run experiments whose tag contains this string",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # ── output dir ────────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  TinyML Experiment Sweep")
    print(f"{'='*70}")
    print(f"  Config:     {args.config}")
    print(f"  Output dir: {out_dir}")
    print(f"  Run ID:     {run_id}")

    experiments = build_experiment_list()
    if args.filter:
        experiments = [e for e in experiments if args.filter in e["tag"]]
    print(f"  Experiments: {len(experiments)}")

    if args.dry_run:
        print("\n[DRY RUN] Experiments that would run:\n")
        for i, e in enumerate(experiments, 1):
            print(f"  {i:3d}. {e['tag']}")
        return 0

    # ── FL training (once per QAT variant) ───────────────────────────────────
    fl_model_paths: Dict[str, Path] = {}
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    for fl_var in FL_VARIANTS:
        qat_tag  = fl_var["tag"]          # "no_qat" or "yes_qat"
        use_qat  = fl_var["use_qat"]
        model_path = models_dir / f"global_model_{qat_tag}.h5"

        if model_path.exists() and model_path.stat().st_size > 100_000:
            print(f"\n  [FL] Reusing existing {model_path} ({model_path.stat().st_size // 1024}KB)")
        else:
            print(f"\n{'='*70}")
            print(f"  STEP 1: FL Training — {qat_tag}  (use_qat={use_qat})")
            print(f"{'='*70}")
            ok = train_fl(base_cfg, use_qat, model_path)
            if not ok:
                print(f"  ❌ FL training failed for {qat_tag}. Aborting sweep.")
                return 1
            print(f"  ✅ Saved FL model → {model_path}")

        fl_model_paths[qat_tag] = model_path

    # ── Run experiments ───────────────────────────────────────────────────────
    results: List[Dict] = []
    n_total = len(experiments)

    for idx, exp in enumerate(experiments, 1):
        fl_tag   = exp["fl"]["tag"]
        fl_path  = fl_model_paths[fl_tag]

        # Build a per-experiment config with use_qat matching the FL variant.
        # base_cfg may have use_qat=True/False from the YAML; override it so
        # load_keras_model uses the correct loader (keras3 vs tf_keras+tfmot).
        exp_cfg = {**base_cfg,
                   "federated": {**base_cfg.get("federated", {}),
                                 "use_qat": exp["fl"]["use_qat"]}}

        print(f"\n[{idx}/{n_total}] {exp['tag']}")
        try:
            result = run_experiment(exp, fl_path, exp_cfg, out_dir)
            results.append(result)
        except Exception as e:
            import traceback
            print(f"  ❌ Experiment failed: {e}")
            traceback.print_exc()
            results.append({"tag": exp["tag"], "error": str(e)})

        # Write intermediate results after each experiment
        csv_path = out_dir / "sweep_results.csv"
        md_path  = out_dir / "sweep_results.md"
        valid = [r for r in results if "error" not in r]
        if valid:
            write_csv(valid, csv_path)
            write_markdown(valid, md_path)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SWEEP COMPLETE  ({len(results)} experiments)")
    print(f"{'='*70}")
    failed = [r["tag"] for r in results if "error" in r]
    if failed:
        print(f"  ⚠️  Failed ({len(failed)}): {', '.join(failed)}")

    csv_path = out_dir / "sweep_results.csv"
    md_path  = out_dir / "sweep_results.md"
    valid = [r for r in results if "error" not in r]
    write_csv(valid, csv_path)
    write_markdown(valid, md_path)

    print(f"\n  Results → {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
