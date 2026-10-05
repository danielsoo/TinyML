#!/usr/bin/env python3
"""
Quantization methods from the literature, applied to our federated IDS models
(docs/QUANT_LITERATURE.md, experiments A-D). The spec's `part` selects one experiment.

  calib     (A) PTQ range estimators on the float federated model: min/max, percentile
                (99.9 / 99.99), MSE-optimal clipping (ACIQ's objective, grid search) and KL /
                entropy calibration (TensorRT style), against our +-5 SD input clipping.
                Same calibration draws as ptq_calibration_check.py / job l and m.
  ptq_qat   (B) Order: QAT fine-tuning that starts from PTQ-calibrated ranges (Wu et al. 2020)
                vs tfmot's default QAT, whose ranges start at +-6 and move by an EMA with decay
                0.999 (~140 steps here, so ~87% of the +-6 start remains). Ranges either keep
                updating ("init") or stay fixed at the calibrated values ("fixed"). Note that
                tfmot's input QuantizeLayer tracks the running min/max of all training inputs
                (AllValuesQuantizer), so only "fixed" keeps a clipped input range.
  datafree  (C) Server-side calibration with no client data: synthetic N(0,1) inputs (inputs are
                standardized) or ranges computed from the federated BatchNorm statistics,
                with and without cross-layer equalization (DFQ, Nagel et al. 2019).
  learned   (D) QAT with learned activation clipping (PACT-style: the clip value is a trained
                parameter, gradients from tf.quantization.fake_quant_with_min_max_vars).

Range estimators are applied through a tfmot fake-quant model whose range variables are set
directly (prefix "fq_"); this keeps weight quantization identical across estimators
(per-tensor symmetric, as in our QAT exports). "tflite_" rows use TFLite's own PTQ calibration
(min/max over the representative set), which is what jobs l and m measured.

Spec (YAML):
  part: calib | ptq_qat | datafree | learned
  calib_n: 2000            # calibration samples
  calib_draws: 3           # A, C
  ft_samples: 10000        # B, D (client-0 fine-tuning draws, seeds 100+draw as in job m)
  ft_draws: 5
  estimators: [max, pct999, pct9999, mse, kl, clip5_max]   # clip<c>_<est>: inputs clipped first
  clip: 5.0
  bn_k: [4, 6]             # C: range = mean + k*std of the BN statistics
  qat_inits: [max, pct9999, mse, kl, clip5_max]   # B: calibrated start, then tfmot's EMA
  qat_fixed: [max, pct9999, mse, clip5_max]       # B: ranges held at the calibrated values
  learned_inits: [max, pct9999, clip5_max]        # D
  learned_lr_mult: 10.0                # D: clip = exp(lr_mult * a); larger = faster-moving clip
  models:
    - {name: cic_near_iid, model: <b_fl.h5>, config: <b_fl.yaml>, ft_client: 0}

Usage:
  python scripts/quant_lit_methods.py --spec <spec.yaml> --output-dir <dir>
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow_model_optimization as tfmot

from scripts.ablation_utils import load_yaml, rows_to_csv, rows_to_markdown, save_json
from scripts.compression_ablation import (
    PRUNE_FT_EPOCHS,
    PRUNE_RATIO,
    QAT_FT_EPOCHS,
    _clone,
    _fit,
    _load_fl_model,
    _qat,
)
from scripts.ptq_calibration_check import convert_int8
from src.data.loader import load_dataset, partition_data
from src.evaluation.metrics import evaluate_tflite_model
from src.modelcompression.pruning import apply_structured_pruning
from src.tinyml.export_tflite import _strip_bn_dropout_for_qat, export_tflite_qat

METRICS = ["accuracy", "precision", "attack_recall", "f1", "false_alarm_rate", "fn", "fp"]
Ranges = Dict[str, Tuple[float, float]]  # "input", "d0", "d1", ... -> (min, max)


# --------------------------------------------------------------------------- float model helpers
def dense_layers(model: keras.Model) -> List[keras.layers.Dense]:
    dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)]
    others = [l for l in model.layers if not isinstance(l, (keras.layers.Dense, keras.layers.InputLayer))]
    if others:
        raise ValueError(f"expected a folded Dense-only model, found {[type(l).__name__ for l in others]}")
    return dense


def tensors(model: keras.Model, x: np.ndarray) -> Dict[str, np.ndarray]:
    """Values at every point tfmot quantizes: input, hidden Dense outputs (post-ReLU) and the
    last Dense's pre-activation (the sigmoid output has a fixed scale in TFLite)."""
    dense = dense_layers(model)
    out = {"input": x.astype(np.float32)}
    h = x.astype(np.float32)
    for i, layer in enumerate(dense):
        w, b = layer.get_weights()[:2]
        z = h @ w + b
        if i == len(dense) - 1:
            out[f"d{i}"] = z
        else:
            h = layer.activation(z).numpy()
            out[f"d{i}"] = h
    return out


# --------------------------------------------------------------------------- range estimators
def _with_zero(lo: float, hi: float) -> Tuple[float, float]:
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def _quant_mse(v: np.ndarray, lo: float, hi: float, levels: int = 256) -> float:
    scale = (hi - lo) / (levels - 1)
    q = np.clip(np.round((v - lo) / scale), 0, levels - 1) * scale + lo
    return float(np.mean((v - q) ** 2))


def _kl_threshold(a: np.ndarray, levels: int, bins: int = 2048, stride: int = 4) -> float:
    """TensorRT-style entropy calibration on |values|: threshold minimizing KL(P || Q)."""
    amax = float(a.max())
    if amax <= 0:
        return 1e-6
    hist, edges = np.histogram(a, bins=bins, range=(0.0, amax))
    hist = hist.astype(np.float64)
    best_kl, best_t = np.inf, amax
    for i in range(levels, bins + 1, stride):
        p = hist[:i].copy()
        p[-1] += hist[i:].sum()
        sliced = hist[:i]
        nz = sliced > 0
        bounds = np.linspace(0, i, levels + 1).astype(int)
        q = np.zeros(i)
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e <= s:
                continue
            cnt = nz[s:e].sum()
            if cnt:
                q[s:e] = np.where(nz[s:e], sliced[s:e].sum() / cnt, 0.0)
        ps, qs = p.sum(), q.sum()
        if ps == 0 or qs == 0:
            continue
        p, q = p / ps, q / qs
        mask = p > 0
        kl = float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))
        if kl < best_kl:
            best_kl, best_t = kl, float(edges[i])
    return best_t


def estimate(v: np.ndarray, method: str, rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    v = np.asarray(v, dtype=np.float64).ravel()
    if len(v) > 400_000:
        v = (rng or np.random.default_rng(0)).choice(v, 400_000, replace=False)
    vmin, vmax = float(v.min()), float(v.max())
    if method == "max":
        return _with_zero(vmin, vmax)
    if method.startswith("pct"):
        p = {"pct999": 99.9, "pct9999": 99.99}[method]
        lo = np.percentile(v, 100 - p) if vmin < 0 else 0.0
        return _with_zero(lo, np.percentile(v, p))
    if method == "mse":  # ACIQ's objective (clipping + rounding MSE), solved by grid search
        lo0, hi0 = _with_zero(vmin, vmax)
        best = min((_quant_mse(v, lo0 * f, hi0 * f), f) for f in np.linspace(0.02, 1.0, 50))
        return _with_zero(lo0 * best[1], hi0 * best[1])
    if method == "kl":
        if vmin >= 0:  # post-ReLU: asymmetric int8 uses all 256 levels on [0, max]
            return _with_zero(0.0, _kl_threshold(v, 256))
        t = _kl_threshold(np.abs(v), 128)
        return _with_zero(max(-t, vmin), min(t, vmax))
    raise ValueError(method)


def estimate_ranges(model: keras.Model, x: np.ndarray, method: str) -> Ranges:
    """method: an estimate() name, or "clip<c>_<name>" = inputs clipped to +-c SD first."""
    if method.startswith("clip"):
        c, method = method[4:].split("_", 1)
        x = np.clip(x, -float(c), float(c))
    return {k: estimate(v, method) for k, v in tensors(model, x).items()}


def fmt_ranges(r: Ranges) -> str:
    return " ".join(f"{k}:[{lo:.2f},{hi:.2f}]" for k, (lo, hi) in r.items())


def convert_int8_per_tensor(model: keras.Model, rep: np.ndarray, out_path: Path) -> None:
    """TFLite full-integer PTQ with per-tensor weight scales. TFLite's default PTQ gives Dense
    weights one scale per output channel, while tfmot QAT exports (and the fq_ models) use one
    scale per tensor; this row separates the two effects."""
    from src.tinyml.export_tflite import _strip_bn_dropout_for_tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(_strip_bn_dropout_for_tflite(model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_disable_per_channel = True

    def representative_dataset():
        for i in range(len(rep)):
            yield [rep[i:i + 1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    out_path.write_bytes(converter.convert())


# --------------------------------------------------------------------------- fake-quant models
def qat_model(model: keras.Model) -> keras.Model:
    return tfmot.quantization.keras.quantize_model(_strip_bn_dropout_for_qat(model))


def _range_vars(q: keras.Model) -> Dict[str, Tuple[tf.Variable, tf.Variable]]:
    """Activation range variables keyed like tensors(): "input", "d0", "d1", ..."""
    out: Dict[str, Tuple[tf.Variable, tf.Variable]] = {}
    i = 0
    for layer in q.layers:
        names = {w.name.split("/")[-1].split(":")[0]: w for w in layer.weights}
        if type(layer).__name__ == "QuantizeLayer":
            lo = [w for n, w in names.items() if n.endswith("_min")]
            hi = [w for n, w in names.items() if n.endswith("_max")]
            out["input"] = (lo[0], hi[0])
        elif hasattr(layer, "quantize_config"):  # one wrapped Dense
            for pre in ("post_activation", "pre_activation"):
                if f"{pre}_min" in names:
                    out[f"d{i}"] = (names[f"{pre}_min"], names[f"{pre}_max"])
            i += 1
    return out


def set_ranges(q: keras.Model, ranges: Ranges) -> None:
    """Set activation ranges and the per-tensor symmetric kernel ranges (LastValueQuantizer)."""
    rv = _range_vars(q)
    missing = set(ranges) ^ set(rv)
    if missing:
        raise KeyError(f"range keys do not match the QAT model: {sorted(missing)}")
    for k, (lo, hi) in ranges.items():
        rv[k][0].assign(lo)
        rv[k][1].assign(hi)
    for layer in q.layers:
        names = {w.name.split("/")[-1].split(":")[0]: w for w in layer.weights}
        if "kernel_min" in names:
            amax = float(np.abs(names["kernel"].numpy()).max())
            names["kernel_min"].assign(-amax)
            names["kernel_max"].assign(amax)


def get_ranges(q: keras.Model) -> Ranges:
    return {k: (float(a.numpy()), float(b.numpy())) for k, (a, b) in _range_vars(q).items()}


class FreezeRanges(keras.callbacks.Callback):
    """Keep activation ranges at their calibrated values during QAT fine-tuning."""

    def __init__(self, ranges: Ranges):
        super().__init__()
        self.ranges = ranges

    def on_train_batch_end(self, batch, logs=None):
        for k, (a, b) in _range_vars(self.model).items():
            a.assign(self.ranges[k][0])
            b.assign(self.ranges[k][1])


def fq_export(model: keras.Model, ranges: Ranges, path: Path) -> None:
    q = qat_model(model)
    set_ranges(q, ranges)
    export_tflite_qat(q, str(path))


def qat_from_ranges(model: keras.Model, ranges: Ranges, x, y, fixed: bool) -> keras.Model:
    q = qat_model(model)
    set_ranges(q, ranges)
    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    q.fit(x, y, epochs=QAT_FT_EPOCHS, batch_size=128, validation_split=0.1, verbose=0,
          callbacks=[FreezeRanges(ranges)] if fixed else [])
    if fixed:
        set_ranges(q, ranges)
    return q


# --------------------------------------------------------------------------- (D) learned clipping
class LearnedRangeQuantizer(tfmot.quantization.keras.quantizers.Quantizer):
    """Activation fake-quant with a trained clip (PACT-style). clip = exp(lr_mult * a), so the
    clip moves multiplicatively and fast enough to matter in a short fine-tune."""

    def __init__(self, init_min: float, init_max: float, learn_min: bool, lr_mult: float = 10.0):
        self.init_min, self.init_max = float(init_min), float(init_max)
        self.learn_min, self.lr_mult = bool(learn_min), float(lr_mult)

    def build(self, tensor_shape, name, layer):
        w = {"a_max": layer.add_weight(
            name + "_log_max", shape=(), trainable=True,
            initializer=keras.initializers.Constant(np.log(max(self.init_max, 1e-3)) / self.lr_mult))}
        if self.learn_min:
            w["a_min"] = layer.add_weight(
                name + "_log_negmin", shape=(), trainable=True,
                initializer=keras.initializers.Constant(np.log(max(-self.init_min, 1e-3)) / self.lr_mult))
        return w

    def __call__(self, inputs, training, weights, **kwargs):
        mx = tf.exp(weights["a_max"] * self.lr_mult)
        mn = -tf.exp(weights["a_min"] * self.lr_mult) if self.learn_min else tf.constant(0.0)
        return tf.quantization.fake_quant_with_min_max_vars(inputs, mn, mx, num_bits=8, narrow_range=False)

    def get_config(self):
        return {"init_min": self.init_min, "init_max": self.init_max,
                "learn_min": self.learn_min, "lr_mult": self.lr_mult}


class LearnedActDenseConfig(tfmot.quantization.keras.QuantizeConfig):
    """Hidden Dense+ReLU: default 8-bit weights, learned output clip on the ReLU output."""

    def __init__(self, init_max: float, lr_mult: float = 10.0):
        self.init_max, self.lr_mult = float(init_max), float(lr_mult)

    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(
            num_bits=8, symmetric=True, narrow_range=True, per_axis=False))]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return [LearnedRangeQuantizer(0.0, self.init_max, learn_min=False, lr_mult=self.lr_mult)]

    def get_config(self):
        return {"init_max": self.init_max, "lr_mult": self.lr_mult}


SCOPE = {"LearnedRangeQuantizer": LearnedRangeQuantizer, "LearnedActDenseConfig": LearnedActDenseConfig}


def learned_qat(model: keras.Model, ranges: Ranges, x, y, lr_mult: float) -> Tuple[keras.Model, Ranges]:
    base = _strip_bn_dropout_for_qat(model)
    dense = dense_layers(base)
    hidden = {l.name: i for i, l in enumerate(dense[:-1])}

    def annotate(layer):
        if layer.name in hidden:
            hi = ranges[f"d{hidden[layer.name]}"][1]
            return tfmot.quantization.keras.quantize_annotate_layer(
                layer, quantize_config=LearnedActDenseConfig(hi, lr_mult))
        if isinstance(layer, keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer)
        return layer

    annotated = keras.models.clone_model(base, clone_function=annotate)
    annotated.set_weights(base.get_weights())
    with tfmot.quantization.keras.quantize_scope(SCOPE):
        q = tfmot.quantization.keras.quantize_apply(annotated)
    # input and output-logit ranges: calibrated start, EMA as usual
    rv = _range_vars(q)
    for k, (a, b) in rv.items():
        a.assign(ranges[k][0])
        b.assign(ranges[k][1])
    for layer in q.layers:
        names = {w.name.split("/")[-1].split(":")[0]: w for w in layer.weights}
        if "kernel_min" in names:
            amax = float(np.abs(names["kernel"].numpy()).max())
            names["kernel_min"].assign(-amax)
            names["kernel_max"].assign(amax)
    q.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    _fit(q, x, y, QAT_FT_EPOCHS)
    learned: Ranges = dict(get_ranges(q))
    i = 0
    for layer in q.layers:
        if not hasattr(layer, "quantize_config"):
            continue
        for w in layer.weights:
            if "_log_max" in w.name:
                learned[f"d{i}"] = (0.0, float(np.exp(w.numpy() * lr_mult)))
        i += 1
    return q, dict(sorted(learned.items(), key=lambda kv: (kv[0] != "input", kv[0])))


# --------------------------------------------------------------------------- (C) data-free helpers
def bn_stats(raw_path: Path) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(mean, std) of each post-ReLU BatchNorm of the unfolded federated model, in layer order."""
    with tfmot.quantization.keras.quantize_scope():
        raw = keras.models.load_model(str(raw_path), compile=False)
    out = []
    for layer in raw.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            _, _, mu, var = layer.get_weights()
            out.append((mu.astype(np.float64), np.sqrt(var).astype(np.float64)))
    return out


def bn_ranges(model: keras.Model, stats, k: float, scales: Optional[List[np.ndarray]] = None) -> Ranges:
    dense = dense_layers(model)
    if len(stats) != len(dense) - 1:
        raise ValueError(f"{len(stats)} BN layers for {len(dense)} Dense layers")
    scales = scales or [np.ones_like(s[0]) for s in stats]
    r: Ranges = {"input": (-float(k), float(k))}  # inputs are standardized
    for i, ((mu, sd), s) in enumerate(zip(stats, scales)):
        r[f"d{i}"] = _with_zero(0.0, float(np.max((mu + k * sd) / s)))
    w, b = dense[-1].get_weights()[:2]
    mu, sd = stats[-1]
    s = scales[-1]
    mean = float((mu / s) @ w[:, 0] + b[0])
    std = float(np.sqrt(((sd / s) ** 2) @ (w[:, 0] ** 2)))
    r[f"d{len(dense) - 1}"] = _with_zero(mean - k * std, mean + k * std)
    return r


def cross_layer_equalize(model: keras.Model, iters: int = 20) -> Tuple[keras.Model, List[np.ndarray]]:
    """DFQ cross-layer equalization for a ReLU MLP (exact in float). Returns the model and the
    cumulative per-channel scale s of each hidden output (new activation = old / s)."""
    m = _clone(model)
    dense = dense_layers(m)
    ws = [l.get_weights() for l in dense]
    cum = [np.ones(w[0].shape[1]) for w in ws[:-1]]
    for _ in range(iters):
        for i in range(len(ws) - 1):
            w1, b1 = ws[i][0], ws[i][1]
            w2 = ws[i + 1][0]
            r1, r2 = np.abs(w1).max(axis=0), np.abs(w2).max(axis=1)
            s = np.ones_like(r1)
            ok = (r1 > 0) & (r2 > 0)
            s[ok] = np.sqrt(r1[ok] * r2[ok]) / r2[ok]
            ws[i][0], ws[i][1] = w1 / s, b1 / s
            ws[i + 1][0] = w2 * s[:, None]
            cum[i] = cum[i] * s
    for l, w in zip(dense, ws):
        l.set_weights(w)
    return m, cum


# --------------------------------------------------------------------------- experiment driver
class Runner:
    def __init__(self, entry: dict, spec: dict, out_dir: Path):
        self.name = entry["name"]
        self.entry, self.spec = entry, spec
        cfg = load_yaml(ROOT / entry["config"])
        data_cfg = dict(cfg.get("data", {}))
        kwargs = {k: v for k, v in data_cfg.items() if k not in {"name", "num_clients"}}
        if "path" in kwargs:
            kwargs["data_path"] = kwargs.pop("path")
        self.x_train, self.y_train, self.x_test, self.y_test = load_dataset(
            data_cfg.get("name", "cicids2017"), **kwargs)
        self.parts = partition_data(
            self.x_train, self.y_train, int(data_cfg.get("num_clients", 4)),
            strategy=data_cfg.get("partition_strategy", "label_balanced"),
            dirichlet_alpha=float(data_cfg.get("dirichlet_alpha", 0.3)),
        )
        self.threshold = float(cfg.get("evaluation", {}).get("prediction_threshold", 0.3))
        self.attacks = int(np.sum(self.y_test))
        self.base = _load_fl_model(str(ROOT / entry["model"]))
        self.tdir = out_dir / self.name
        self.tdir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.rows: List[Dict[str, Any]] = []

    def record(self, row: Dict[str, Any], build: Callable[[Path], Optional[Ranges]]):
        row = {"model": self.name, **row}
        p = self.tdir / f"{row['method']}_{row.get('source', '')}_d{row.get('draw', 0)}.tflite"
        try:
            ranges = build(p)
            m = evaluate_tflite_model(p, self.x_test, self.y_test, threshold=self.threshold, latency_runs=0)
            row.update({k: m[k] for k in METRICS})
            row["missed"] = f"{int(m['fn'])}/{self.attacks}"
            row["size_kb"] = round(p.stat().st_size / 1024, 2)
            if ranges:
                row["ranges"] = fmt_ranges(ranges)
            print(f"  [{self.name}] {row['method']:22s} {row.get('source', ''):12s} d{row.get('draw', 0)} "
                  f"f1={row['f1']:.4f} far={row['false_alarm_rate']:.4f} missed={row['missed']}", flush=True)
            p.unlink()
        except Exception as err:
            traceback.print_exc()
            row["error"] = f"{type(err).__name__}: {str(err)[:200]}"
            print(f"  ⚠️ [{self.name}] {row['method']} failed: {row['error']}", flush=True)
        self.rows.append(row)
        rows_to_csv(self.rows, self.out_dir / f"{self.name}_partial.csv")

    def calib_sets(self, draw: int, n: int) -> List[Tuple[str, np.ndarray]]:
        rng = np.random.default_rng(1000 + draw)  # same order of draws as ptq_calibration_check.py
        sets = [("pooled_rand", self.x_train[rng.choice(len(self.y_train), size=n, replace=False)])]
        for cid, part in enumerate(self.parts):
            sets.append((f"client{cid}", part["x"][rng.choice(len(part["y"]), size=min(n, len(part["y"])),
                                                             replace=False)]))
        return sets

    def ft_draw(self, draw: int):
        k = int(self.entry.get("ft_client", 0))
        cx, cy = self.parts[k]["x"], self.parts[k]["y"].astype(np.float32)
        n_ft = int(self.spec.get("ft_samples", 10000))
        tf.keras.utils.set_random_seed(100 + draw)
        idx = np.random.default_rng(100 + draw).permutation(len(cy))[:n_ft]
        xf, yf = cx[idx], cy[idx]
        pr = apply_structured_pruning(_clone(self.base), pruning_ratio=PRUNE_RATIO, skip_last_layer=True,
                                      verbose=False)
        _fit(pr, xf, yf, PRUNE_FT_EPOCHS)
        return xf, yf, pr

    # ---- (A) calibration estimators ----
    def part_calib(self):
        n, clip = int(self.spec.get("calib_n", 2000)), float(self.spec.get("clip", 5.0))
        ests = self.spec.get("estimators", ["max", "pct999", "pct9999", "mse", "kl", "clip5_max"])
        for draw in range(int(self.spec.get("calib_draws", 3))):
            for source, xc in self.calib_sets(draw, n):
                xcl = np.clip(xc, -clip, clip)
                meta = {"source": source, "draw": draw}
                self.record({"method": "tflite_max", **meta},
                            lambda p: convert_int8(self.base, xc, "builtins_int8", p))
                self.record({"method": f"tflite_clip{clip:g}", **meta},
                            lambda p: convert_int8(self.base, xcl, "builtins_int8", p))
                self.record({"method": "tflite_max_per_tensor", **meta},
                            lambda p: convert_int8_per_tensor(self.base, xc, p))
                for est in ests:
                    r = estimate_ranges(self.base, xc, est)
                    self.record({"method": f"fq_{est}", **meta}, lambda p, r=r: (fq_export(self.base, r, p), r)[1])

    # ---- (B) PTQ-initialized QAT ----
    def part_ptq_qat(self):
        n = int(self.spec.get("calib_n", 2000))
        inits = self.spec.get("qat_inits", ["max", "pct9999", "mse", "kl", "clip5_max"])
        fixed = self.spec.get("qat_fixed", ["max", "pct9999", "mse", "clip5_max"])
        for draw in range(int(self.spec.get("ft_draws", 5))):
            xf, yf, pr = self.ft_draw(draw)
            xc = xf[:n]
            meta = {"source": "client0_ft", "draw": draw}
            self.record({"method": "ptq_tflite_max", **meta}, lambda p: convert_int8(pr, xf[:500], "builtins_int8", p))
            est = {e: estimate_ranges(pr, xc, e) for e in sorted(set(inits) | set(fixed))}
            for e in inits:
                self.record({"method": f"ptq_fq_{e}", **meta}, lambda p, e=e: (fq_export(pr, est[e], p), est[e])[1])

            def default(p):
                q = _qat(pr, xf, yf)
                export_tflite_qat(q, str(p))
                return get_ranges(q)
            self.record({"method": "qat_default", **meta}, default)
            for e in inits:
                def init(p, e=e):
                    q = qat_from_ranges(pr, est[e], xf, yf, fixed=False)
                    export_tflite_qat(q, str(p))
                    return get_ranges(q)
                self.record({"method": f"qat_init_{e}", **meta}, init)
            for e in fixed:
                def frozen(p, e=e):
                    q = qat_from_ranges(pr, est[e], xf, yf, fixed=True)
                    export_tflite_qat(q, str(p))
                    return get_ranges(q)
                self.record({"method": f"qat_fixed_{e}", **meta}, frozen)

    # ---- (C) data-free calibration ----
    def part_datafree(self):
        n, clip = int(self.spec.get("calib_n", 2000)), float(self.spec.get("clip", 5.0))
        ks = [float(k) for k in self.spec.get("bn_k", [4, 6])]
        stats = bn_stats(ROOT / self.entry["model"])
        cle, scales = cross_layer_equalize(self.base)
        probe = self.x_test[:2000]
        diff = float(np.abs(cle.predict(probe, verbose=0) - self.base.predict(probe, verbose=0)).max())
        print(f"  CLE float output max |diff| = {diff:.2e}")
        for tag, model, sc in [("", self.base, None), ("cle_", cle, scales)]:
            for k in ks:
                r = bn_ranges(model, stats, k, sc)
                self.record({"method": f"{tag}df_bn_k{k:g}", "source": "bn_stats", "draw": 0, "cle_diff": diff},
                            lambda p, r=r, model=model: (fq_export(model, r, p), r)[1])
            for draw in range(int(self.spec.get("calib_draws", 3))):
                xg = np.random.default_rng(2000 + draw).standard_normal(
                    (n, self.base.input_shape[1])).astype(np.float32)
                meta = {"source": "gauss", "draw": draw}
                self.record({"method": f"{tag}df_gauss_tflite", **meta},
                            lambda p, model=model: convert_int8(model, xg, "builtins_int8", p))
                r = estimate_ranges(model, xg, "max")
                self.record({"method": f"{tag}df_gauss_fq_max", **meta},
                            lambda p, r=r, model=model: (fq_export(model, r, p), r)[1])
                xc = self.calib_sets(draw, n)[1][1]  # client 0, same draw as part A
                xcl = np.clip(xc, -clip, clip)
                meta = {"source": "client0", "draw": draw}
                self.record({"method": f"{tag}real_tflite_max", **meta},
                            lambda p, model=model: convert_int8(model, xc, "builtins_int8", p))
                self.record({"method": f"{tag}real_tflite_clip{clip:g}", **meta},
                            lambda p, model=model: convert_int8(model, xcl, "builtins_int8", p))
                for src, data in [("max", xc), (f"clip{clip:g}_max", xcl)]:
                    r = estimate_ranges(model, data, "max")
                    self.record({"method": f"{tag}real_fq_{src}", **meta},
                                lambda p, r=r, model=model: (fq_export(model, r, p), r)[1])

    # ---- (D) learned clipping QAT ----
    def part_learned(self):
        n = int(self.spec.get("calib_n", 2000))
        inits = self.spec.get("learned_inits", ["max", "pct9999", "clip5_max"])
        lr_mult = float(self.spec.get("learned_lr_mult", 10.0))
        for draw in range(int(self.spec.get("ft_draws", 5))):
            xf, yf, pr = self.ft_draw(draw)
            meta = {"source": "client0_ft", "draw": draw}

            def default(p):
                q = _qat(pr, xf, yf)
                export_tflite_qat(q, str(p))
                return get_ranges(q)
            self.record({"method": "qat_default", **meta}, default)
            for e in inits:
                r0 = estimate_ranges(pr, xf[:n], e)

                def learned(p, r0=r0):
                    q, r = learned_qat(pr, r0, xf, yf, lr_mult)
                    export_tflite_qat(q, str(p))
                    return r
                self.record({"method": f"qat_learned_{e}", **meta, "init_ranges": fmt_ranges(r0)}, learned)


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys: List[Tuple[str, str]] = []
    for r in rows:
        if (r["model"], r["method"]) not in keys:
            keys.append((r["model"], r["method"]))
    out = []
    for model, method in keys:
        ok = [r for r in rows if r["model"] == model and r["method"] == method and "f1" in r]
        bad = [r for r in rows if r["model"] == model and r["method"] == method and "error" in r]
        row: Dict[str, Any] = {"model": model, "method": method, "runs_ok": len(ok), "runs_failed": len(bad)}
        if ok:
            f1 = [r["f1"] * 100 for r in ok]
            far = [r["false_alarm_rate"] * 100 for r in ok]
            fn = [r["fn"] for r in ok]
            total = ok[0]["missed"].split("/")[1]
            row.update({"f1_mean": round(statistics.mean(f1), 2), "f1_min": round(min(f1), 2),
                        "f1_max": round(max(f1), 2), "f1_sd": round(statistics.pstdev(f1), 2),
                        "far_mean": round(statistics.mean(far), 2),
                        "missed_mean": f"{statistics.mean(fn):.0f}/{total}",
                        "missed_range": f"{min(fn)}-{max(fn)}"})
        out.append(row)
    return out


PARTS = {"calib": "part_calib", "ptq_qat": "part_ptq_qat", "datafree": "part_datafree", "learned": "part_learned"}
TITLES = {"calib": "(A) PTQ calibration estimators", "ptq_qat": "(B) PTQ-initialized QAT",
          "datafree": "(C) Data-free calibration", "learned": "(D) Learned-clipping QAT"}


def main():
    parser = argparse.ArgumentParser(description="Literature quantization methods (A-D)")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = load_yaml(args.spec)
    part = spec["part"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for entry in spec["models"]:
        runner = Runner(entry, spec, out_dir)
        getattr(runner, PARTS[part])()
        rows.extend(runner.rows)
        rows_to_csv(rows, out_dir / "quant_lit.csv")
        rows_to_csv(summarize(rows), out_dir / "quant_lit_summary.csv")
    rows_to_markdown(summarize(rows), out_dir / "quant_lit.md",
                     f"{TITLES[part]} (mean / min / max over draws and calibration sets)")
    save_json({"part": part, "rows": rows, "summary": summarize(rows)}, out_dir / "quant_lit.json")
    print(f"✅ saved to {out_dir}")


if __name__ == "__main__":
    main()
