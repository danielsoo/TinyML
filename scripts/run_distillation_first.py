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
from src.modelcompression.distillation import create_student_model, train_with_distillation

try:
    import tensorflow_model_optimization as tfmot
    TFMOT_AVAILABLE = True
except ImportError:
    TFMOT_AVAILABLE = False

DISTILLED_DIR = Path("models/distilled")
DISTILL_EPOCHS = 8
DISTILL_EPOCHS_PROGRESSIVE = 5  # per stage


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
    student, _ = train_with_distillation(
        teacher, student, x_train, y_train, x_val, y_val,
        temperature=3.0, alpha=0.3, epochs=DISTILL_EPOCHS, batch_size=128, learning_rate=0.001, verbose=True
    )
    return student


def run_progressive(teacher: keras.Model, x_train, y_train, x_val, y_val, num_classes: int) -> keras.Model:
    intermediate = create_student_model(teacher, compression_ratio=0.5, num_classes=num_classes)
    intermediate, _ = train_with_distillation(
        teacher, intermediate, x_train, y_train, x_val, y_val,
        temperature=3.0, alpha=0.3, epochs=DISTILL_EPOCHS_PROGRESSIVE, batch_size=128, learning_rate=0.001, verbose=True
    )
    student = create_student_model(intermediate, compression_ratio=0.5, num_classes=num_classes)
    student, _ = train_with_distillation(
        intermediate, student, x_train, y_train, x_val, y_val,
        temperature=3.0, alpha=0.3, epochs=DISTILL_EPOCHS_PROGRESSIVE, batch_size=128, learning_rate=0.001, verbose=True
    )
    return student


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
    trad_path = comp_cfg.get("traditional_model_path") or "models/global_model_traditional.h5"
    qat_path = "models/global_model.h5"
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
