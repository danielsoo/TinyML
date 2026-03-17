#!/usr/bin/env python3
"""
Distillation-first pipeline: run distillation 4 times (no_QAT+direct, no_QAT+progressive,
QAT+direct, QAT+progressive), save 4 models, then run prune+PTQ on each (reuse saved models).

Usage:
  python scripts/run_distillation_first.py --config config/federated_local_sky.yaml
  python scripts/run_distillation_first.py --config config/federated.yaml --skip-compress  # only build 4 distilled
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set before TF
import os
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import numpy as np
import yaml
import tensorflow as tf
from tensorflow import keras
from src.data.loader import load_dataset
from src.modelcompression.distillation import create_student_model, Distiller

try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False

DISTILLED_DIR = project_root / "models" / "distilled"
DISTILL_EPOCHS = 8
DISTILL_EPOCHS_PROGRESSIVE = 5  # per stage


def _scalar_distill_loss(y_true, student_preds, teacher_preds, temperature, alpha):
    """Distillation loss that always returns a scalar. Handles both binary (sigmoid)
    and multi-class (softmax) teachers/students without tf.cond branch tracing issues."""
    # Soft targets: teacher probabilities softened by temperature
    t_shape = tf.shape(teacher_preds)[-1]
    # If teacher outputs 1 value (binary sigmoid), convert to 2-class probs
    teacher_2class = tf.cond(
        tf.equal(t_shape, 1),
        lambda: tf.concat([1.0 - teacher_preds, teacher_preds], axis=-1),
        lambda: teacher_preds,
    )
    teacher_soft = tf.nn.softmax(
        tf.math.log(tf.clip_by_value(teacher_2class, 1e-7, 1.0)) / temperature
    )
    student_soft = tf.nn.softmax(student_preds / temperature)
    distill_loss = tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft) * (temperature ** 2)
    student_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, student_preds, from_logits=True)
    total = alpha * distill_loss + (1.0 - alpha) * student_loss
    return tf.reduce_mean(total)


def _to_sigmoid_binary(model: keras.Model) -> keras.Model:
    """Convert a 2-class softmax student to a 1-output sigmoid model.
    The rest of the pipeline (sweep_compression_grid.py) compiles with
    binary_crossentropy which requires a single sigmoid output."""
    last = model.layers[-1]
    if getattr(last, "units", None) != 2:
        return model  # already correct shape
    # Rebuild: same hidden layers, replace final Dense(2,softmax) with Dense(1,sigmoid)
    inp = keras.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers:
        if layer is last:
            continue
        if hasattr(layer, '__call__') and not isinstance(layer, keras.layers.InputLayer):
            x = layer(x)
    x = keras.layers.Dense(1, activation="sigmoid", name="output_sigmoid")(x)
    new_model = keras.Model(inputs=inp, outputs=x)
    # Copy weights from all hidden layers
    for old_layer, new_layer in zip(model.layers, new_model.layers):
        if old_layer is last:
            break
        try:
            if old_layer.get_weights():
                new_layer.set_weights(old_layer.get_weights())
        except Exception:
            pass
    new_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return new_model


def _train_one(teacher, student, x_train, y_train, x_val, y_val, epochs):
    """Train student via distillation with scalar loss. Returns trained student."""
    distiller = Distiller(student=student, teacher=teacher, temperature=3.0, alpha=0.3)
    distiller.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[],
        distillation_loss_fn=_scalar_distill_loss,
    )
    distiller.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=1,
    )
    return student


def _strip_qat_if_needed(model: keras.Model):
    """Strip QAT wrappers so we have a plain float32 model for distillation."""
    if not TFMOT_AVAILABLE:
        return model
    for layer in model.layers:
        if "QuantizeWrapper" in type(layer).__name__:
            import compression as comp
            return comp.strip_qat_layers(model)
    return model


def load_teacher(path: str, use_qat_scope: bool = False) -> keras.Model:
    path = Path(path)
    if not path.exists():
        return None
    if use_qat_scope and TFMOT_AVAILABLE:
        with tfmot.quantization.keras.quantize_scope():
            model = keras.models.load_model(path, compile=False)
    else:
        model = keras.models.load_model(path, compile=False)
    model = _strip_qat_if_needed(model)
    last = model.layers[-1]
    num_classes = getattr(last, "units", 2)
    loss = "binary_crossentropy" if num_classes <= 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def run_direct(teacher: keras.Model, x_train, y_train, x_val, y_val, num_classes: int) -> keras.Model:
    student = create_student_model(teacher, compression_ratio=0.5, num_classes=num_classes)
    student = _train_one(teacher, student, x_train, y_train, x_val, y_val, DISTILL_EPOCHS)
    return _to_sigmoid_binary(student)


def run_progressive(teacher: keras.Model, x_train, y_train, x_val, y_val, num_classes: int) -> keras.Model:
    intermediate = create_student_model(teacher, compression_ratio=0.5, num_classes=num_classes)
    intermediate = _train_one(teacher, intermediate, x_train, y_train, x_val, y_val, DISTILL_EPOCHS_PROGRESSIVE)
    student = create_student_model(intermediate, compression_ratio=0.5, num_classes=num_classes)
    student = _train_one(intermediate, student, x_train, y_train, x_val, y_val, DISTILL_EPOCHS_PROGRESSIVE)
    return _to_sigmoid_binary(student)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/federated_local_sky.yaml")
    parser.add_argument("--skip-compress", action="store_true", help="Only build 4 distilled models, do not run prune+PTQ")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    name = data_cfg.get("name", "cicids2017")
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    print("\n📂 Loading dataset...")
    x_train, y_train, x_test, y_test = load_dataset(name, **kwargs)
    val_split = int(0.85 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]
    num_classes = 2 if np.max(y_train) <= 1 and len(np.unique(y_train)) <= 2 else int(np.max(y_train)) + 1
    print(f"   Train: {len(x_train):,}, Val: {len(x_val):,}, num_classes: {num_classes}\n")

    comp_cfg = cfg.get("compression", {})
    trad_path = comp_cfg.get("traditional_model_path") or str(project_root / "models" / "global_model_traditional.h5")
    qat_path = str(project_root / "models" / "global_model.h5")
    use_qat = cfg.get("federated", {}).get("use_qat", False)

    DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    # 1) no_QAT + direct
    teacher_no = load_teacher(trad_path, use_qat_scope=False)
    if teacher_no is not None:
        print("\n" + "="*60 + "\n  no_QAT + direct (0.5x)\n" + "="*60)
        student = run_direct(teacher_no, x_train, y_train, x_val, y_val, num_classes)
        p = DISTILLED_DIR / "no_qat_direct.h5"
        student.save(p)
        saved_paths.append(("no_qat_direct", str(p)))
    else:
        print(f"   ⚠️  Skip no_QAT (no teacher at {trad_path})")

    # 2) no_QAT + progressive
    if teacher_no is not None:
        print("\n" + "="*60 + "\n  no_QAT + progressive (0.25x)\n" + "="*60)
        student = run_progressive(teacher_no, x_train, y_train, x_val, y_val, num_classes)
        p = DISTILLED_DIR / "no_qat_progressive.h5"
        student.save(p)
        saved_paths.append(("no_qat_progressive", str(p)))

    # 3) QAT + direct
    teacher_qat = load_teacher(qat_path, use_qat_scope=use_qat)
    if teacher_qat is not None:
        print("\n" + "="*60 + "\n  QAT + direct (0.5x)\n" + "="*60)
        student = run_direct(teacher_qat, x_train, y_train, x_val, y_val, num_classes)
        p = DISTILLED_DIR / "qat_direct.h5"
        student.save(p)
        saved_paths.append(("qat_direct", str(p)))

    # 4) QAT + progressive
    if teacher_qat is not None:
        print("\n" + "="*60 + "\n  QAT + progressive (0.25x)\n" + "="*60)
        student = run_progressive(teacher_qat, x_train, y_train, x_val, y_val, num_classes)
        p = DISTILLED_DIR / "qat_progressive.h5"
        student.save(p)
        saved_paths.append(("qat_progressive", str(p)))

    print(f"\n✅ Distillation phase done. Saved {len(saved_paths)} models under {DISTILLED_DIR}\n")

    if args.skip_compress or not saved_paths:
        return 0

    # Prune + PTQ for each saved distilled model (one TFLite per tag)
    print("\n" + "="*60 + "\n  Prune + PTQ for each distilled model\n" + "="*60 + "\n")
    import compression as comp
    out_dir = Path("models/tflite")
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, model_path in saved_paths:
        print(f"\n--- {tag} ---")
        out_path = out_dir / f"saved_model_{tag}.tflite"
        ok = comp.compress_one_distilled_model(args.config, model_path, str(out_path))
        if not ok:
            print(f"   ⚠️  Compression failed for {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
