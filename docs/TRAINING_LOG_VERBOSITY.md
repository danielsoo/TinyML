# Training log verbosity

When you run `python run.py` or `python scripts/train.py`, some of the output comes from our code and some from dependencies (Ray, Flower, TensorFlow).

## Our code (already reduced)

- **QAT**: We log only once (`[Client 0] QAT enabled - model quantization-aware`) instead of once per client, to avoid repeated lines when using Ray simulation.
- **Class weights**: If there are more than 8 classes, we print a one-line summary (e.g. `Class weights: 80 classes, min=0.xxx, max=2.xxx`) instead of a long `{0: ..., 1: ..., ...}` dump.

## From dependencies (not from our code)

- **Long `INFO : (0, 0.0), (1, 0.0), ... (80, 0.0)` lines**: Usually from Ray or Flower logging a large object (e.g. parameters or metrics). We set `RAY_BACKEND_LOG_LEVEL=warning` by default to reduce backend noise; if it still appears, it may be Python-level logging inside the simulation.
- **`Failed to establish connection to the metrics exporter agent`**: From Ray’s C++ core. Harmless for training; you can ignore it or set `RAY_BACKEND_LOG_LEVEL=error` to hide it.
- **`(ClientAppActor pid=...) [Client 2] Real QAT enabled - tfmot.quantize_model ...`**: Ray wraps our client process; the “Real QAT” / “tfmot.quantize_model” part can come from TensorFlow Model Optimization or the runtime. Our own message is the shorter “QAT enabled - model quantization-aware” (only from Client 0).

## Optional: quieter run

To try to reduce log volume further:

```bash
RAY_BACKEND_LOG_LEVEL=error python run.py --config config/federated.yaml
```

Or in the same shell before running:

```bash
export RAY_BACKEND_LOG_LEVEL=error
```

Training behavior and results are unchanged; only log verbosity is affected.
