#!/usr/bin/env python3
"""
Run 48 compression grid sweep, then PGD on all 48 models, then merge results.
Creates a run directory with sweep_compression_grid.csv, pgd/, and sweep_compression_grid_with_pgd.csv.

Prerequisites:
  - models/global_model.h5, models/global_model_traditional.h5
  - models/distilled/*.h5 (from run_distillation_first.py)

Usage:
  python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml
  python scripts/run_sweep_and_pgd.py --config config/federated_local_sky.yaml --version v1_PGD --quick
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.experiment_record import write_experiment_record

# ---------------------------------------------------------------------------
# Inline distillation (Step 0)
# Replicates run_distillation_first.py but with a scalar loss wrapper so
# tf_keras's progress-bar aggregator doesn't crash on variable batch sizes.
# Only runs if models/distilled/*.h5 are missing.
# ---------------------------------------------------------------------------

_DISTILLED_DIR = ROOT / "models" / "distilled"
_DISTILL_EPOCHS = 8
_DISTILL_EPOCHS_PROG = 5
_DISTILLED_NEED = [
    "no_qat_direct.h5", "no_qat_progressive.h5",
    "qat_direct.h5",    "qat_progressive.h5",
]


def _has_qat_layers(model):
    return any("QuantizeWrapper" in type(l).__name__ or "QuantizeLayer" in type(l).__name__
               or "QuantizeWrapperV2" in type(l).__name__
               for l in model.layers)


def _manual_strip_qat(model):
    """Strip QAT wrappers without tfmot API (handles old tfmot with no strip_quant_dequant).

    QuantizeWrapper stores its inner layer as .layer (public) or ._layer (private).
    We build a new tf_keras Sequential by unwrapping each layer, then copy weights.
    The resulting model has zero tfmot references in memory or on disk.
    """
    import tf_keras
    from tf_keras import layers as tfl
    import numpy as np

    input_shape = model.input_shape[1:]
    new_layers = [tf_keras.Input(shape=input_shape)]
    src_layers = []  # (original_layer, inner_layer) pairs for weight copying

    for layer in model.layers:
        cls = type(layer).__name__
        # Unwrap QuantizeWrapper / QuantizeWrapperV2 → inner layer
        if "QuantizeWrapper" in cls:
            inner = getattr(layer, "layer", None) or getattr(layer, "_layer", None)
            if inner is None:
                continue
            layer = inner
            cls = type(layer).__name__
        # Skip QuantizeLayer / QuantizeLayerV2 (input quant/dequant — no plain equivalent)
        if "QuantizeLayer" in cls:
            continue
        # Match by class name substring to handle tf_keras vs keras naming differences
        if "Dense" in cls and hasattr(layer, "units"):
            act = layer.activation
            act_name = getattr(act, "__name__", None) or getattr(act, "name", "linear")
            new_layers.append(tfl.Dense(layer.units, activation=act_name, name=layer.name))
            src_layers.append(layer)
        elif "BatchNormalization" in cls:
            new_layers.append(tfl.BatchNormalization(name=layer.name))
            src_layers.append(layer)
        elif "Dropout" in cls and hasattr(layer, "rate"):
            new_layers.append(tfl.Dropout(layer.rate, name=layer.name))
            src_layers.append(None)  # no weights

    if len(new_layers) <= 1:  # only Input
        raise ValueError(f"No layers extracted from QAT model (layers: {[type(l).__name__ for l in model.layers]})")

    clean = tf_keras.Sequential(new_layers)
    clean.build((None,) + input_shape)

    # Copy weights layer by layer (positional fallback if name match fails)
    weighted_new = [l for l in clean.layers if hasattr(l, "get_weights") and l.get_weights()]
    wi = 0
    for src in src_layers:
        if src is None:
            continue
        ws = src.get_weights()
        if not ws:
            continue
        # Try name match first
        tgt = next((l for l in clean.layers if l.name == src.name), None)
        if tgt is None and wi < len(weighted_new):
            tgt = weighted_new[wi]
            wi += 1
        if tgt is not None:
            try:
                tgt.set_weights(ws)
            except ValueError:
                pass
    return clean


def _strip_qat(model):
    """Strip QAT wrappers. Tries tfmot API first, falls back to manual unwrap."""
    if not _has_qat_layers(model):
        return model
    try:
        import tensorflow_model_optimization as tfmot
        return tfmot.quantization.keras.strip_quant_dequant(model)
    except Exception:
        pass
    # tfmot API unavailable or failed — unwrap manually
    try:
        return _manual_strip_qat(model)
    except Exception:
        pass
    return model


def _load_teacher(path, use_qat_scope=False):
    """Load a teacher model.

    QAT models (use_qat_scope=True) must be loaded with tf_keras (tf.keras)
    inside quantize_scope — tfmot registers QuantizeLayer/QuantizeWrapper into
    tf.keras's custom-object table, not into standalone Keras3's table, so
    keras.saving.load_model always raises 'Unknown layer: QuantizeLayer'.

    Non-QAT models use standalone Keras3 (same framework they were saved with).
    """
    p = Path(path)
    if not p.exists():
        return None

    if use_qat_scope:
        # Use tf_keras + quantize_scope so QuantizeLayer deserialises correctly
        try:
            import tensorflow_model_optimization as tfmot
            import tf_keras
            with tfmot.quantization.keras.quantize_scope():
                model = tf_keras.models.load_model(str(p), compile=False)
        except Exception:
            # Last resort: try without scope (will work if QAT was already stripped)
            import tf_keras
            model = tf_keras.models.load_model(str(p), compile=False)
    else:
        import keras as _keras
        model = _keras.saving.load_model(str(p), compile=False)

    model = _strip_qat(model)
    last = model.layers[-1]
    num_out = getattr(last, "units", 1)
    loss = "binary_crossentropy" if num_out <= 2 else "sparse_categorical_crossentropy"
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def _make_student(teacher, compression_ratio: float = 0.5):
    """Build a tf_keras student with sigmoid [?,1] output — same as run_experiment_sweep.py.
    Using tf_keras avoids cross-framework tensor dispatch errors during training."""
    import tf_keras
    from tf_keras import layers as tfl
    import numpy as np

    input_shape = teacher.input_shape[1:]
    dense_layers = [l for l in teacher.layers if type(l).__name__ == "Dense"]

    seq_layers = [tf_keras.Input(shape=input_shape)]
    for layer in dense_layers[:-1]:
        units = max(1, int(layer.units * compression_ratio))
        seq_layers.append(tfl.Dense(units, activation="relu"))
        seq_layers.append(tfl.BatchNormalization())
        seq_layers.append(tfl.Dropout(0.25))
    seq_layers.append(tfl.Dense(1, activation="sigmoid"))

    model = tf_keras.Sequential(seq_layers)
    model.build((None,) + input_shape)
    return model


def _distill_one_pair(teacher, student, x_tr, y_tr, epochs, batch_size=128,
                      temperature=3.0, alpha=0.3, lr=0.001):
    """Custom distillation training loop — no Distiller/tf.cond, no cross-framework issues.
    Identical to run_experiment_sweep.py's _distill_one_pair."""
    import tensorflow as tf
    import tf_keras
    import numpy as np

    optimizer = tf_keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)
    teacher.trainable = False

    def train_step(x_batch, y_batch):
        x_t = tf.constant(x_batch, dtype=tf.float32)
        y_t = tf.constant(y_batch, dtype=tf.float32)
        teacher_pred = teacher(x_t, training=False)
        t_p = tf.squeeze(teacher_pred, axis=-1)
        eps = 1e-7
        teacher_logits = tf.stack(
            [tf.math.log(1 - t_p + eps), tf.math.log(t_p + eps)], axis=-1)
        teacher_soft = tf.nn.softmax(teacher_logits / temperature)
        with tf.GradientTape() as tape:
            s_pred = student(x_t, training=True)
            s_p = tf.squeeze(s_pred, axis=-1)
            student_logits = tf.stack(
                [tf.math.log(1 - s_p + eps), tf.math.log(s_p + eps)], axis=-1)
            student_soft = tf.nn.softmax(student_logits / temperature)
            soft_loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(teacher_soft, student_soft)
            ) * (temperature ** 2)
            y_2d = tf.reshape(y_t, [-1, 1])
            hard_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(y_2d, s_pred))
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
            losses.append(float(train_step(xb, yb).numpy()))
        print(f"    epoch {ep+1}/{epochs}  loss={np.mean(losses):.4f}")

    teacher.trainable = True
    return student


def _rebuild_clean_model(qat_model):
    """Rebuild a pure tf_keras Sequential from a stripped model's weights.

    strip_quant_dequant removes QuantizeWrapper layers but the saved .h5 can still
    contain tfmot config (AllValuesQuantizer etc.) in its model JSON, which causes
    clone_model to crash. Rebuilding from scratch with plain tf_keras layers guarantees
    the serialised .h5 has zero tfmot references.
    """
    import tf_keras
    from tf_keras import layers as tfl

    weights = qat_model.get_weights()
    input_shape = qat_model.input_shape[1:]
    new_layers = [tf_keras.Input(shape=input_shape)]
    for layer in qat_model.layers:
        cls = type(layer).__name__
        if cls == "Dense":
            act = layer.activation
            act_name = getattr(act, "__name__", None) or getattr(act, "name", "linear")
            new_layers.append(tfl.Dense(layer.units, activation=act_name))
        elif cls == "BatchNormalization":
            new_layers.append(tfl.BatchNormalization())
        elif cls == "Dropout":
            new_layers.append(tfl.Dropout(layer.rate))
    clean = tf_keras.Sequential(new_layers)
    clean.build((None,) + input_shape)
    clean.set_weights(weights)
    return clean


def _strip_global_model_if_qat():
    """Ensure models/global_model.h5 is a clean tf_keras model with no QAT metadata.

    Strategy: always load from the QAT backup (original) if it exists, strip, rebuild
    from scratch, and overwrite global_model.h5. This handles the case where a previous
    broken strip left an .h5 that looks clean in memory but still has AllValuesQuantizer
    in its JSON config, causing clone_model to crash.

    If no backup exists, load current file; if it has QAT layers, strip+rebuild+save.
    If it has no QAT layers, do nothing.
    """
    import shutil
    import tf_keras
    import tensorflow_model_optimization as tfmot

    p = ROOT / "models" / "global_model.h5"
    backup = p.with_suffix(".qat_backup.h5")

    if not p.exists():
        return

    # Prefer loading from the original QAT backup so we always start clean
    src = backup if backup.exists() else p

    try:
        with tfmot.quantization.keras.quantize_scope():
            model = tf_keras.models.load_model(str(src), compile=False)
    except ImportError:
        return
    except Exception:
        try:
            model = tf_keras.models.load_model(str(src), compile=False)
        except Exception as e:
            print(f"  [Pre-sweep] ⚠️  Could not load {src.name}: {e} — skipping strip.")
            return

    if not _has_qat_layers(model):
        # Source has no QAT layers — nothing to do
        return

    print(f"  [Pre-sweep] Stripping QAT layers from {src.name} → {p.name} ...")

    # Try tfmot API first; fall back to manual unwrap (old tfmot has no strip_quant_dequant)
    clean = None
    try:
        stripped = tfmot.quantization.keras.strip_quant_dequant(model)
        if not _has_qat_layers(stripped):
            clean = _rebuild_clean_model(stripped)
            print("  [Pre-sweep] Stripped via tfmot API + rebuilt clean model.")
    except Exception as e:
        print(f"  [Pre-sweep] tfmot strip_quant_dequant unavailable ({e}), using manual unwrap.")

    if clean is None:
        try:
            clean = _manual_strip_qat(model)
            if _has_qat_layers(clean):
                print("  [Pre-sweep] ⚠️  QAT layers remain after manual strip — skipping overwrite.")
                return
            print("  [Pre-sweep] Rebuilt clean tf_keras model via manual unwrap.")
        except Exception as e:
            print(f"  [Pre-sweep] ⚠️  Manual strip failed: {e} — skipping overwrite.")
            return

    # Find last Dense layer (skip InputLayer etc.)
    dense_layers = [l for l in clean.layers if getattr(l, "units", None) is not None]
    if not dense_layers:
        print("  [Pre-sweep] ⚠️  No Dense layers found in cleaned model — skipping overwrite.")
        return
    nout = dense_layers[-1].units
    loss = "binary_crossentropy" if nout <= 2 else "sparse_categorical_crossentropy"
    clean.compile(optimizer="adam", loss=loss, metrics=["accuracy"])

    # Save backup of original only once (before first overwrite)
    if not backup.exists():
        shutil.copy2(str(p), str(backup))
        print(f"  [Pre-sweep] Backed up original to {backup.name}")

    clean.save(str(p))
    print(f"  [Pre-sweep] Saved clean model → {p.name}\n")


def _run_distillation(config_path: str):
    """Build models/distilled/*.h5 if any are missing. Skip if all present.
    Uses the proven custom training loop from run_experiment_sweep.py."""
    _DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in _DISTILLED_NEED if not (_DISTILLED_DIR / n).exists()]
    if not missing:
        print("  [Step 0] All distilled models present — skipping distillation.\n")
        return True

    print(f"  [Step 0] Building distilled models (missing: {missing}) ...")

    import numpy as np
    from src.data.loader import load_dataset

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg.get("data", {})
    kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
    if "path" in kwargs and "data_path" not in kwargs:
        kwargs["data_path"] = kwargs.pop("path")

    print("  Loading dataset ...")
    x_tr, y_tr, x_te, y_te = load_dataset(data_cfg.get("name", "cicids2017"), **kwargs)
    x_tr = np.asarray(x_tr, dtype=np.float32)
    y_tr = np.asarray(y_tr, dtype=np.float32)
    print(f"  Train: {len(x_tr):,}  Test: {len(x_te):,}")

    comp_cfg  = cfg.get("compression", {})
    trad_path = comp_cfg.get("traditional_model_path") or str(ROOT / "models" / "global_model_traditional.h5")
    qat_path  = str(ROOT / "models" / "global_model.h5")
    use_qat   = cfg.get("federated", {}).get("use_qat", False)

    teacher_no  = _load_teacher(trad_path, use_qat_scope=False)
    teacher_qat = _load_teacher(qat_path,  use_qat_scope=use_qat)

    pairs = [
        ("no_qat_direct.h5",      teacher_no,  "direct"),
        ("no_qat_progressive.h5", teacher_no,  "progressive"),
        ("qat_direct.h5",         teacher_qat, "direct"),
        ("qat_progressive.h5",    teacher_qat, "progressive"),
    ]

    for fname, teacher, mode in pairs:
        if (_DISTILLED_DIR / fname).exists():
            print(f"  Already exists: {fname}")
            continue
        if teacher is None:
            print(f"  ⚠️  No teacher for {fname}, skipping")
            continue

        print(f"\n  {'='*50}\n  Distilling ({mode}) → {fname}\n  {'='*50}")
        if mode == "progressive":
            inter = _make_student(teacher, compression_ratio=0.5)
            inter = _distill_one_pair(teacher, inter, x_tr, y_tr, _DISTILL_EPOCHS_PROG)
            student = _make_student(inter, compression_ratio=0.5)
            student = _distill_one_pair(inter, student, x_tr, y_tr, _DISTILL_EPOCHS_PROG)
        else:
            student = _make_student(teacher, compression_ratio=0.5)
            student = _distill_one_pair(teacher, student, x_tr, y_tr, _DISTILL_EPOCHS)

        out = _DISTILLED_DIR / fname
        student.save(str(out))
        print(f"  Saved {out}")

    still_missing = [n for n in _DISTILLED_NEED if not (_DISTILLED_DIR / n).exists()]
    if still_missing:
        print(f"  ⚠️  Some distilled models still missing: {still_missing}", file=sys.stderr)
        return False
    return True


def run_cmd(cmd: list, desc: str) -> bool:
    print(f"\n{'='*60}\n  {desc}\n{'='*60}\n")
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n  {desc} done.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Failed: {e}\n", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="48 sweep + PGD on all + merge")
    parser.add_argument("--config", default="config/federated_local_sky.yaml")
    parser.add_argument("--version", default="sweep_pgd", help="Version prefix for run dir, e.g. v1_PGD")
    parser.add_argument("--quick", action="store_true", help="Run only 4 combinations (sweep --quick)")
    parser.add_argument("--skip-sweep", action="store_true", help="Use existing sweep CSV in run-dir (run PGD + merge only)")
    parser.add_argument("--skip-distill", action="store_true", help="Skip distillation step (assume models/distilled/*.h5 already exist)")
    parser.add_argument("--run-dir", default=None, help="Use this run dir (default: data/processed/runs/<version>/<timestamp>)")
    parser.add_argument("--resume-csv", default=None,
                        help="Path to partial sweep_compression_grid.csv from a crashed run; "
                             "completed tags are skipped and merged into final output")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        base = Path("data/processed/runs") / args.version
        base.mkdir(parents=True, exist_ok=True)
        run_dir = base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir.resolve()}")

    # Step 0: Save run config and experiment record (so pgd_report etc. can refer to them)
    with open(args.config, encoding="utf-8") as f:
        run_cfg = yaml.safe_load(f)
    run_config_path = run_dir / "run_config.yaml"
    with open(run_config_path, "w", encoding="utf-8") as f:
        yaml.dump(run_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"   📋 Saved run config: {run_config_path}\n")
    write_experiment_record(run_cfg, run_dir)

    # Step 0: build distilled models if missing (uses scalar loss to avoid tf_keras crash)
    if args.skip_distill:
        print("  [Step 0] --skip-distill set — assuming models/distilled/*.h5 exist.\n")
    elif not _run_distillation(args.config):
        print("  Distillation failed. Aborting.", file=sys.stderr)
        return 1

    # Step 0b: strip QAT from global model so sweep can clone_model() it
    _strip_global_model_if_qat()

    if not args.skip_sweep:
        sweep_cmd = [
            sys.executable,
            str(project_root / "scripts" / "sweep_compression_grid.py"),
            "--config", args.config,
            "--run-dir", str(run_dir),
        ]
        if args.quick:
            sweep_cmd.append("--quick")
        if args.resume_csv:
            sweep_cmd += ["--resume-csv", args.resume_csv]
        elif (run_dir / "sweep_compression_grid.csv").exists():
            # Auto-resume: if a partial CSV already exists in the run-dir, use it
            sweep_cmd += ["--resume-csv", str(run_dir / "sweep_compression_grid.csv")]
            print(f"  [Auto-resume] Found partial sweep CSV in run-dir — resuming from it.\n")
        if not run_cmd(sweep_cmd, "Step 1: 48 (or 4) compression grid sweep"):
            return 1

    list_file = run_dir / "pgd_model_list.txt"
    if not list_file.exists():
        print(f"  pgd_model_list.txt not found. Run sweep without --skip-sweep first.", file=sys.stderr)
        return 1

    # Verify Keras attack model exists (first .h5 line in pgd_model_list.txt)
    with open(list_file, encoding="utf-8") as _lf:
        _attack_path = next((ln.strip() for ln in _lf if ln.strip().endswith(".h5") and not ln.startswith("#")), None)
    if not _attack_path or not Path(_attack_path).exists():
        print(f"  Attack model not found: {_attack_path!r}. PGD requires models/global_model.h5.", file=sys.stderr)
        return 1

    pgd_out = run_dir / "pgd"
    pgd_cmd = [
        sys.executable,
        str(project_root / "scripts" / "run_pgd.py"),
        "--models-file", str(list_file),
        "--config", args.config,
        "--output-dir", str(pgd_out),
    ]
    if not run_cmd(pgd_cmd, "Step 2: PGD on all models"):
        return 1

    merge_cmd = [
        sys.executable,
        str(project_root / "scripts" / "merge_sweep_pgd.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(merge_cmd, "Step 3: Merge PGD into sweep CSV"):
        return 1

    # Step 4: Sweep summary (sweep_summary.md)
    summary_cmd = [
        sys.executable,
        str(project_root / "scripts" / "write_sweep_summary.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(summary_cmd, "Step 4: Write sweep summary"):
        pass  # non-fatal

    # Step 5: Heatmaps (sweep_heatmap_*.png)
    heatmap_cmd = [
        sys.executable,
        str(project_root / "scripts" / "plot_sweep_heatmaps.py"),
        "--run-dir", str(run_dir),
    ]
    if not run_cmd(heatmap_cmd, "Step 5: Plot sweep heatmaps"):
        pass  # non-fatal

    out_csv = run_dir / "sweep_compression_grid_with_pgd.csv"
    print(f"\n  Result: {out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
