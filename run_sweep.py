"""
Focused sweep: FL → Distill → Prune → PTQ for two target models only.

Target models (from prior full sweep analysis):
  1. most_compressed : progressive distillation → prune 5x10 → PTQ
  2. most_accurate   : progressive distillation → prune 10x5 → no PTQ

Pipeline per model:
  1. FL training  (no-QAT) — skipped if models/global_model_no_qat.h5 exists
                             and --skip-train is passed
  2. Distillation (progressive)
  3. Iterative pruning
  4. PTQ / TFLite export (model 1 only)
  5. Evaluate → sweep_results.csv
  6. PGD adversarial training (AT) on each final_model.h5
  7. Evaluate AT model → pgd_at_results.csv

Usage:
    python run_sweep.py                 # full pipeline (trains FL)
    python run_sweep.py --skip-train    # reuse existing global_model_no_qat.h5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Two target experiments
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    {
        "label": "most_compressed",
        "tag":   "no_qat__distill_progressive__prune_5x10__ptq_yes",
        "fl":    {"use_qat": False, "tag": "no_qat"},
        "dist":  {"mode": "progressive"},
        "prune": {"ratio": 0.05, "steps": 10},
        "ptq":   True,
    },
    {
        "label": "most_accurate",
        "tag":   "no_qat__distill_progressive__prune_10x5__ptq_no",
        "fl":    {"use_qat": False, "tag": "no_qat"},
        "dist":  {"mode": "progressive"},
        "prune": {"ratio": 0.10, "steps": 5},
        "ptq":   False,
    },
]

# ---------------------------------------------------------------------------
# PGD-AT hyperparams (matches pgd_at_run.py)
# ---------------------------------------------------------------------------
EPSILON    = 0.01
PGD_ALPHA  = 0.001
PGD_STEPS  = 10
ADV_RATIO  = 0.5
AT_EPOCHS  = 10
BATCH_SIZE = 128
AT_LR      = 0.0001
THRESHOLD  = 0.5

# ---------------------------------------------------------------------------
# Config / data loading
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_DATA_CACHE: Optional[Tuple] = None

def get_data(cfg: dict):
    global _DATA_CACHE
    if _DATA_CACHE is not None:
        return _DATA_CACHE
    from src.data.loader import load_dataset
    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")
    x_tr, y_tr, x_te, y_te = load_dataset(data_cfg["name"], **kwargs)
    _DATA_CACHE = (
        x_tr.astype("float32"), y_tr.astype("float32"),
        x_te.astype("float32"), y_te.astype("float32"),
    )
    return _DATA_CACHE


# ---------------------------------------------------------------------------
# FL training
# ---------------------------------------------------------------------------

def train_fl(base_cfg: dict, out_path: Path) -> bool:
    import subprocess, tempfile
    cfg = {**base_cfg, "federated": {**base_cfg.get("federated", {}), "use_qat": False}}
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True, sort_keys=False)
    tmp.close()
    cmd = [
        sys.executable, "scripts/train.py",
        "--config", tmp.name,
        "--save-model", str(out_path),
        "--skip-gpu-check", "--skip-deps", "--skip-data-check",
    ]
    try:
        ret = subprocess.run(cmd, check=False)
        return ret.returncode == 0
    finally:
        Path(tmp.name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Model loading / evaluation
# ---------------------------------------------------------------------------

def load_model(path: Path, cfg: dict):
    import keras as _keras
    from src.models.nets import _focal_loss
    fed_cfg = cfg.get("federated", {})
    use_focal = fed_cfg.get("use_focal_loss", False)
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))
    loss = _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal else "binary_crossentropy"
    model = _keras.saving.load_model(str(path), compile=False)
    model.compile(
        optimizer=_keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
        loss=loss, metrics=["accuracy"],
    )
    return model


def _metrics(model, x, y) -> Dict[str, float]:
    y_prob = model.predict(x, verbose=0).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)
    y_true = y.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec}


def _tflite_metrics(tflite_path: Path, x, y) -> Dict[str, float]:
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    preds = []
    for i in range(len(x)):
        interp.set_tensor(inp["index"], x[i:i+1])
        interp.invoke()
        preds.append(interp.get_tensor(out["index"])[0])
    y_prob = np.array(preds, dtype=np.float32).ravel()
    y_pred = (y_prob >= THRESHOLD).astype(int)
    y_true = y.astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    acc  = (tp + tn) / max(len(y_true), 1)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    size_kb = tflite_path.stat().st_size / 1024
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec, "size_kb": size_kb}


# ---------------------------------------------------------------------------
# Distillation (progressive)
# ---------------------------------------------------------------------------

def _make_student(teacher, ratio: float = 0.5):
    import tf_keras
    from tf_keras import layers as tfl
    input_shape = teacher.input_shape[1:]
    dense_layers = [l for l in teacher.layers if type(l).__name__ == "Dense"]
    seq = [tf_keras.Input(shape=input_shape)]
    for layer in dense_layers[:-1]:
        units = max(1, int(layer.units * ratio))
        seq.append(tfl.Dense(units, activation="relu"))
        seq.append(tfl.BatchNormalization())
        seq.append(tfl.Dropout(0.25))
    seq.append(tfl.Dense(1, activation="sigmoid"))
    m = tf_keras.Sequential(seq)
    m.build((None,) + input_shape)
    return m


def _distill(teacher, student, x_tr, y_tr, cfg: dict):
    import tensorflow as tf
    import tf_keras
    from src.models.nets import _focal_loss
    fed_cfg = cfg.get("federated", {})
    dist_cfg = cfg.get("distillation", {})
    epochs     = int(dist_cfg.get("epochs", 10))
    batch_size = int(fed_cfg.get("batch_size", 128))
    lr         = float(fed_cfg.get("learning_rate", 0.001))
    temperature = float(dist_cfg.get("temperature", 3.0))
    alpha      = float(dist_cfg.get("alpha", 0.3))
    use_focal  = bool(fed_cfg.get("use_focal_loss", False))
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))

    hard_loss_fn = (
        _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal
        else tf.keras.losses.binary_crossentropy
    )
    optimizer = tf_keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    teacher.trainable = False

    def train_step(xb, yb):
        x_t = tf.constant(xb, dtype=tf.float32)
        y_t = tf.constant(yb, dtype=tf.float32)
        tp = tf.squeeze(teacher(x_t, training=False), axis=-1)
        eps = 1e-7
        t_logits = tf.stack([tf.math.log(1 - tp + eps), tf.math.log(tp + eps)], axis=-1)
        t_soft = tf.nn.softmax(t_logits / temperature)
        with tf.GradientTape() as tape:
            sp = tf.squeeze(student(x_t, training=True), axis=-1)
            s_logits = tf.stack([tf.math.log(1 - sp + eps), tf.math.log(sp + eps)], axis=-1)
            s_soft = tf.nn.softmax(s_logits / temperature)
            soft_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(t_soft, s_soft)
            ) * (temperature ** 2)
            hard_loss = tf.reduce_mean(hard_loss_fn(tf.reshape(y_t, [-1, 1]),
                                                     tf.reshape(sp, [-1, 1])))
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
            l = train_step(x_tr[bi].astype(np.float32), y_tr[bi].astype(np.float32))
            losses.append(float(l.numpy()))
        print(f"      epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}")

    teacher.trainable = True
    return student


def run_progressive_distillation(teacher, x_tr, y_tr, cfg: dict):
    print("  [Distill] Stage 1: teacher → intermediate (0.5x) ...")
    inter = _make_student(teacher, ratio=0.5)
    inter = _distill(teacher, inter, x_tr, y_tr, cfg)
    print("  [Distill] Stage 2: intermediate → student (0.25x) ...")
    student = _make_student(inter, ratio=0.5)
    student = _distill(inter, student, x_tr, y_tr, cfg)
    return student


# ---------------------------------------------------------------------------
# Iterative pruning
# ---------------------------------------------------------------------------

def _to_tfkeras(model):
    import tf_keras
    from tf_keras import layers as tfl
    weights = model.get_weights()
    input_shape = model.input_shape[1:]
    seq = [tf_keras.Input(shape=input_shape)]
    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "Dense":
            seq.append(tfl.Dense(layer.units, activation=layer.activation, name=layer.name))
        elif cls == "BatchNormalization":
            seq.append(tfl.BatchNormalization(name=layer.name))
        elif cls == "Dropout":
            seq.append(tfl.Dropout(layer.rate, name=layer.name))
    m = tf_keras.Sequential(seq)
    m.build((None,) + input_shape)
    m.set_weights(weights)
    return m


def _prune_step(model, ratio: float):
    import tf_keras
    from tf_keras import layers as tfl
    from src.modelcompression.pruning import prune_dense_layer

    weights_by_layer = {l.name: l.get_weights() for l in model.layers}
    input_shape = model.input_shape[1:]
    dense_layers = [l for l in model.layers if type(l).__name__ == "Dense"]
    prunable = {l.name for l in dense_layers[:-1]}

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
            w, bias, kept_idx = prune_dense_layer(w, bias, ratio)
            kept[layer.name] = kept_idx
            prev_kept = kept_idx
        else:
            kept[layer.name] = None
            prev_kept = None

    new_seq = [tf_keras.Input(shape=input_shape)]
    for layer in model.layers:
        cls = type(layer).__name__
        if cls == "Dense":
            idx = kept.get(layer.name)
            units = len(idx) if idx is not None else layer.units
            new_seq.append(tfl.Dense(units, activation=layer.activation, name=layer.name + "_p"))
        elif cls == "BatchNormalization":
            new_seq.append(tfl.BatchNormalization(name=layer.name + "_p"))
        elif cls == "Dropout":
            new_seq.append(tfl.Dropout(layer.rate, name=layer.name + "_p"))

    new_model = tf_keras.Sequential(new_seq)
    new_model.build((None,) + input_shape)

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
            bn_ws = weights_by_layer[layer.name]
            if (prev_kept is not None and len(bn_ws) > 0
                    and len(bn_ws[0]) != new_layer.get_weights()[0].shape[0]):
                bn_ws = [w[prev_kept] for w in bn_ws]
            new_layer.set_weights(bn_ws)

    return new_model


def run_pruning(model, ratio: float, steps: int, x_tr, y_tr, cfg: dict):
    import tf_keras
    from src.models.nets import _focal_loss
    fed_cfg = cfg.get("federated", {})
    ft_epochs  = int(cfg.get("pruning", {}).get("finetune_epochs", 2))
    batch_size = int(fed_cfg.get("batch_size", 128))
    use_focal  = fed_cfg.get("use_focal_loss", False)
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))
    loss_fn = _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal else "binary_crossentropy"

    current = _to_tfkeras(model)
    y_tr_f = y_tr.astype(np.float32)
    for step in range(steps):
        print(f"  [Prune] Step {step+1}/{steps} (ratio={ratio:.0%}) ...")
        current = _prune_step(current, ratio)
        current.compile(
            optimizer=tf_keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0),
            loss=loss_fn, metrics=["accuracy"],
        )
        current.fit(x_tr, y_tr_f, epochs=ft_epochs, batch_size=batch_size, verbose=0)
    return current


# ---------------------------------------------------------------------------
# PTQ / TFLite export
# ---------------------------------------------------------------------------

def _sanitize(model):
    for layer in model.layers:
        ws = layer.get_weights()
        if not ws:
            continue
        new_ws = [np.nan_to_num(np.array(w), nan=0., posinf=0., neginf=0.) for w in ws]
        if any(not np.array_equal(a, b) for a, b in zip(ws, new_ws)):
            layer.set_weights(new_ws)


def export_ptq(model, out_path: Path, x_tr) -> Path:
    from src.tinyml.export_tflite import export_tflite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _sanitize(model)
    rep_data = x_tr[:500].astype(np.float32)
    try:
        export_tflite(model, str(out_path), quantize=True, representative_data=rep_data)
    except Exception as e:
        if "nan" in str(e).lower() or "scale" in str(e).lower():
            fallback = out_path.parent / (out_path.stem + "_float32.tflite")
            export_tflite(model, str(fallback), quantize=False)
            return fallback
        raise
    return out_path


def export_float(model, out_path: Path) -> Path:
    from src.tinyml.export_tflite import export_tflite
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_tflite(model, str(out_path), quantize=False)
    return out_path


# ---------------------------------------------------------------------------
# PGD-AT helpers
# ---------------------------------------------------------------------------

def pgd_attack(model, x, y, loss_fn=None):
    import tensorflow as tf
    if loss_fn is None:
        loss_fn = model.loss
    if isinstance(loss_fn, str):
        from tensorflow import keras
        loss_fn = keras.losses.get(loss_fn)
    x_orig = tf.constant(x, dtype=tf.float32)
    y_t    = tf.constant(y, dtype=tf.float32)
    x_adv  = x_orig + tf.random.uniform(x_orig.shape, -EPSILON, EPSILON)
    for _ in range(PGD_STEPS):
        x_var = tf.Variable(x_adv, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_var)
            preds = model(x_var, training=False)
            y_loss = tf.reshape(y_t, [-1, 1]) if (
                len(preds.shape) == 2 and len(y_t.shape) == 1) else y_t
            loss = loss_fn(y_loss, preds)
        grads = tape.gradient(loss, x_var).numpy()
        grads = np.nan_to_num(grads, nan=0., posinf=0., neginf=0.)
        x_adv = x_adv + PGD_ALPHA * tf.sign(tf.constant(grads, dtype=tf.float32))
        x_adv = tf.clip_by_value(x_adv, x_orig - EPSILON, x_orig + EPSILON)
    return x_adv.numpy()


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    raw = len(y) / (len(classes) * counts)
    smoothed = np.sqrt(raw)
    smoothed /= np.mean(smoothed)
    return {int(c): float(w) for c, w in zip(classes, smoothed)}


def at_finetune(model, x_train, y_train) -> float:
    import tensorflow as tf
    model.optimizer.learning_rate.assign(AT_LR)
    class_weights = compute_class_weights(y_train)
    loss_fn = model.loss
    n = len(x_train)
    t0 = time.perf_counter()
    for epoch in range(AT_EPOCHS):
        perm = np.random.default_rng(epoch).permutation(n)
        x_sh = x_train[perm]
        y_sh = y_train[perm]
        n_batch = (n + BATCH_SIZE - 1) // BATCH_SIZE
        epoch_loss = 0.0
        for b in range(n_batch):
            xb = x_sh[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            yb = y_sh[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            n_adv   = max(1, int(len(xb) * ADV_RATIO))
            n_clean = len(xb) - n_adv
            x_adv = pgd_attack(model, xb[n_clean:], yb[n_clean:], loss_fn=loss_fn)
            if n_clean > 0:
                x_combined = np.concatenate([xb[:n_clean], x_adv])
                y_combined = np.concatenate([yb[:n_clean], yb[n_clean:]])
            else:
                x_combined, y_combined = x_adv, yb
            bl = model.train_on_batch(x_combined, y_combined, class_weight=class_weights)
            epoch_loss += float(bl[0] if isinstance(bl, (list, tuple)) else bl)
        print(f"    Epoch {epoch+1}/{AT_EPOCHS}  loss={epoch_loss/n_batch:.4f}")
    return time.perf_counter() - t0


def load_model_for_at(path: Path, cfg: dict):
    """Load final_model.h5 and recompile for AT (uses keras.models.load_model)."""
    import tensorflow as tf
    from tensorflow import keras
    from src.models.nets import _focal_loss
    fed_cfg = cfg.get("federated", {})
    use_focal = fed_cfg.get("use_focal_loss", False)
    focal_alpha = float(fed_cfg.get("focal_loss_alpha", 0.25))
    loss = _focal_loss(gamma=2.0, alpha=focal_alpha) if use_focal else "binary_crossentropy"
    model = keras.models.load_model(str(path), compile=False)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=AT_LR, clipnorm=1.0),
        loss=loss, metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(exp: dict, fl_model_path: Path, cfg: dict, out_dir: Path) -> dict:
    label = exp["label"]
    tag   = exp["tag"]
    exp_dir = out_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  [{label}]  {tag}")
    print(f"{'='*70}")

    x_tr, y_tr, x_te, y_te = get_data(cfg)

    # ── FL model ──────────────────────────────────────────────────────────────
    # Build exp-specific cfg with use_qat=False
    exp_cfg = {**cfg, "federated": {**cfg.get("federated", {}), "use_qat": False}}
    model = load_model(fl_model_path, exp_cfg)
    fl_m  = _metrics(model, x_te, y_te)
    fl_kb = fl_model_path.stat().st_size / 1024
    print(f"  FL  acc={fl_m['acc']:.4f}  f1={fl_m['f1']:.4f}  size={fl_kb:.1f} KB")

    # ── Progressive distillation ───────────────────────────────────────────
    model = run_progressive_distillation(model, x_tr, y_tr, exp_cfg)
    dist_m = _metrics(model, x_te, y_te)
    model.save(str(exp_dir / "distilled_model.h5"))
    print(f"  Dist acc={dist_m['acc']:.4f}  f1={dist_m['f1']:.4f}")

    # ── Iterative pruning ─────────────────────────────────────────────────
    ratio = exp["prune"]["ratio"]
    steps = exp["prune"]["steps"]
    model = run_pruning(model, ratio, steps, x_tr, y_tr, exp_cfg)
    prune_m = _metrics(model, x_te, y_te)
    model.save(str(exp_dir / "pruned_model.h5"))
    print(f"  Prune acc={prune_m['acc']:.4f}  f1={prune_m['f1']:.4f}")

    # ── PTQ / TFLite export ───────────────────────────────────────────────
    tflite_size_kb = None
    final_m = prune_m  # default: Keras metrics
    if exp["ptq"]:
        tflite_path = exp_dir / "model_ptq.tflite"
        actual = export_ptq(model, tflite_path, x_tr)
        tm = _tflite_metrics(actual, x_te, y_te)
        tflite_size_kb = tm.pop("size_kb")
        final_m = tm
        print(f"  PTQ  acc={final_m['acc']:.4f}  f1={final_m['f1']:.4f}  "
              f"size={tflite_size_kb:.1f} KB")
    else:
        tflite_path = exp_dir / "model_float.tflite"
        actual = export_float(model, tflite_path)
        tflite_size_kb = actual.stat().st_size / 1024
        print(f"  Float TFLite  size={tflite_size_kb:.1f} KB")

    # Save final Keras model (needed for AT)
    final_keras = exp_dir / "final_model.h5"
    model.save(str(final_keras))
    final_kb = final_keras.stat().st_size / 1024
    print(f"  Saved final_model.h5  ({final_kb:.1f} KB)")

    result = {
        "label":          label,
        "tag":            tag,
        "fl_qat":         False,
        "distillation":   "progressive",
        "pruning":        f"prune_{int(ratio*100)}x{steps}",
        "ptq":            exp["ptq"],
        "fl_acc":         fl_m["acc"],    "fl_f1":    fl_m["f1"],
        "dist_acc":       dist_m["acc"],  "dist_f1":  dist_m["f1"],
        "prune_acc":      prune_m["acc"], "prune_f1": prune_m["f1"],
        "final_acc":      final_m["acc"], "final_f1": final_m["f1"],
        "final_prec":     final_m.get("prec"), "final_rec": final_m.get("rec"),
        "tflite_size_kb": tflite_size_kb,
        "final_size_kb":  final_kb,
    }

    with open(exp_dir / "result.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# PGD-AT on saved final_model.h5
# ---------------------------------------------------------------------------

def run_at(label: str, final_keras: Path, cfg: dict) -> dict:
    print(f"\n{'─'*70}")
    print(f"  [AT] {label}")
    print(f"{'─'*70}")

    x_tr, y_tr, x_te, y_te = get_data(cfg)

    # Subsample test set for speed
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_te), min(5000, len(x_te)), replace=False)
    x_ev, y_ev = x_te[idx], y_te[idx]

    exp_cfg = {**cfg, "federated": {**cfg.get("federated", {}), "use_qat": False}}
    model = load_model_for_at(final_keras, exp_cfg)

    # Before AT
    clean_f1_b, clean_acc_b = _metrics(model, x_ev, y_ev)["f1"], _metrics(model, x_ev, y_ev)["acc"]
    x_adv_b = pgd_attack(model, x_ev, y_ev, loss_fn=model.loss)
    adv_f1_b, adv_acc_b = _metrics(model, x_adv_b, y_ev)["f1"], _metrics(model, x_adv_b, y_ev)["acc"]
    print(f"    BEFORE  clean F1={clean_f1_b:.4f} acc={clean_acc_b:.4f}  "
          f"adv F1={adv_f1_b:.4f} acc={adv_acc_b:.4f}")

    # AT fine-tune
    at_time = at_finetune(model, x_tr, y_tr)

    # After AT
    clean_f1_a, clean_acc_a = _metrics(model, x_ev, y_ev)["f1"], _metrics(model, x_ev, y_ev)["acc"]
    x_adv_a = pgd_attack(model, x_ev, y_ev, loss_fn=model.loss)
    adv_f1_a, adv_acc_a = _metrics(model, x_adv_a, y_ev)["f1"], _metrics(model, x_adv_a, y_ev)["acc"]
    print(f"    AFTER   clean F1={clean_f1_a:.4f} acc={clean_acc_a:.4f}  "
          f"adv F1={adv_f1_a:.4f} acc={adv_acc_a:.4f}")

    # Save AT model
    at_path = final_keras.parent / "final_model_at.h5"
    model.save(str(at_path))
    print(f"    Saved: {at_path}")

    def pct(after, before):
        return (after - before) / before * 100 if before != 0 else float("nan")

    return {
        "label":              label,
        "clean_f1_before":    clean_f1_b,
        "clean_acc_before":   clean_acc_b,
        "adv_f1_before":      adv_f1_b,
        "adv_acc_before":     adv_acc_b,
        "clean_f1_after":     clean_f1_a,
        "clean_acc_after":    clean_acc_a,
        "adv_f1_after":       adv_f1_a,
        "adv_acc_after":      adv_acc_a,
        "delta_clean_f1_pct": pct(clean_f1_a, clean_f1_b),
        "delta_clean_acc_pct":pct(clean_acc_a, clean_acc_b),
        "delta_adv_f1_pct":   pct(adv_f1_a, adv_f1_b),
        "delta_adv_acc_pct":  pct(adv_acc_a, adv_acc_b),
        "at_time_s":          at_time,
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

_SWEEP_FIELDS = [
    "label", "tag", "fl_qat", "distillation", "pruning", "ptq",
    "fl_acc", "fl_f1", "dist_acc", "dist_f1", "prune_acc", "prune_f1",
    "final_acc", "final_f1", "final_prec", "final_rec",
    "tflite_size_kb", "final_size_kb",
]

_AT_FIELDS = [
    "label",
    "clean_f1_before", "clean_acc_before", "adv_f1_before", "adv_acc_before",
    "clean_f1_after",  "clean_acc_after",  "adv_f1_after",  "adv_acc_after",
    "delta_clean_f1_pct", "delta_clean_acc_pct",
    "delta_adv_f1_pct",   "delta_adv_acc_pct",
    "at_time_s",
]


def write_csv(rows, path: Path, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in fields})
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Focused sweep: FL → Distill → Prune → PTQ → AT (2 models)"
    )
    parser.add_argument("--config", default="config/federated_local.yaml")
    parser.add_argument("--skip-train", action="store_true",
                        help="Reuse models/global_model_no_qat.h5 if it exists")
    parser.add_argument("--output-dir", default="data/processed/sweep")
    parser.add_argument("--skip-at", action="store_true",
                        help="Skip adversarial training phase")
    args = parser.parse_args()

    cfg = load_cfg(args.config)

    run_id  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Focused TinyML Sweep (2 models)")
    print(f"{'='*70}")
    print(f"  Config:     {args.config}")
    print(f"  Output dir: {out_dir}")

    # ── FL training ───────────────────────────────────────────────────────────
    fl_model_path = Path("models/global_model_no_qat.h5")
    if args.skip_train and fl_model_path.exists():
        print(f"\n  [FL] Reusing {fl_model_path}")
    else:
        print(f"\n{'='*70}")
        print(f"  STEP 1: FL Training (no-QAT)")
        print(f"{'='*70}")
        ok = train_fl(cfg, fl_model_path)
        if not ok:
            print("  FL training failed. Aborting.")
            return 1
        print(f"  Saved FL model → {fl_model_path}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    sweep_rows = []
    for exp in EXPERIMENTS:
        result = run_experiment(exp, fl_model_path, cfg, out_dir)
        sweep_rows.append(result)
        write_csv(sweep_rows, out_dir / "sweep_results.csv", _SWEEP_FIELDS)

    print(f"\n  Sweep results → {out_dir / 'sweep_results.csv'}")

    # ── PGD-AT ────────────────────────────────────────────────────────────────
    if not args.skip_at:
        print(f"\n{'='*70}")
        print(f"  PGD Adversarial Training")
        print(f"{'='*70}")
        at_rows = []
        for exp in EXPERIMENTS:
            final_keras = out_dir / exp["tag"] / "final_model.h5"
            if not final_keras.exists():
                print(f"  ⚠️  {final_keras} not found, skipping AT")
                continue
            at_row = run_at(exp["label"], final_keras, cfg)
            at_rows.append(at_row)
            write_csv(at_rows, out_dir / "pgd_at_results.csv", _AT_FIELDS)

        print(f"\n  AT results → {out_dir / 'pgd_at_results.csv'}")

    print(f"\n{'='*70}")
    print(f"  DONE  —  results in {out_dir}/")
    print(f"{'='*70}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
