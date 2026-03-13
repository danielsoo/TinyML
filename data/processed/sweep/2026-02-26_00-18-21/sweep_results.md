# Experiment Sweep — Results

Generated: 2026-02-26 02:32:14

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 95.09% | 87.67% | 864.1 KB | 97.73% | 93.89% | 82.41% | 0.00% | 17.59% | 29.92% | 17.59% | 100.00% | 31.5 KB | 342.5 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 95.09% | 87.67% | 864.1 KB | 97.48% | 93.27% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | 102.5 KB | 342.5 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 95.09% | 87.67% | 864.1 KB | 97.68% | 93.77% | 17.59% | 29.92% | 17.59% | 29.92% | 17.59% | 100.00% | 50.3 KB | 547.7 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 95.09% | 87.67% | 864.1 KB | 97.68% | 93.76% | 17.59% | 29.92% | 17.59% | 29.92% | — | — | 171.2 KB | 547.7 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 95.09% | 87.67% | 864.1 KB | 97.98% | 94.49% | 82.41% | 0.00% | 17.59% | 29.92% | 17.59% | 100.00% | 31.3 KB | 340.1 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 95.09% | 87.67% | 864.1 KB | 97.77% | 93.99% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | 101.5 KB | 340.1 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 95.09% | 87.67% | 864.1 KB | 97.68% | 93.78% | — | — | 97.53% | 93.40% | 88.25% | 99.17% | 69.5 KB | 266.8 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 95.09% | 87.67% | 864.1 KB | 97.92% | 94.39% | — | — | 97.92% | 94.39% | — | — | 242.7 KB | 266.8 KB |

## Column Legend
- **FL Acc / F1** — metrics immediately after FL training
- **Dist Acc / F1** — metrics after distillation (if applied)
- **Prune Acc / F1** — metrics after iterative pruning (if applied)
- **Final Acc / F1 / Prec / Rec** — metrics on the deployed model (TFLite if PTQ=✓, Keras otherwise)
- **TFLite size** — exported .tflite size
- **Final size** — Keras .h5 size after all compression steps

## Pruning Key
- `10x5` — 10% pruned per step × 5 steps  (cumulative ≈ 41% removed)
- `10x2` — 10% pruned per step × 2 steps  (cumulative ≈ 19% removed)
- `5x10` — 5% pruned per step × 10 steps  (cumulative ≈ 40% removed)
- `none` — no pruning

## Distillation Key
- `direct` — teacher → student in one step (0.5× width)
- `progressive` — teacher → intermediate → student (0.25× width)
- `none` — no distillation