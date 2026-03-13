# Experiment Sweep — Results

Generated: 2026-02-26 13:22:53

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | — | 342.5 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | 102.5 KB | 342.5 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | — | 547.7 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | 171.2 KB | 547.7 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | — | 340.1 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 82.41% | 0.00% | 82.41% | 0.00% | — | — | 101.5 KB | 340.1 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | — | — | 82.41% | 0.00% | — | — | — | 266.8 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | — | — | 82.41% | 0.00% | — | — | 242.2 KB | 266.8 KB |
| no_qat__distill_progressive__prune_10x5__ptq_yes | ✗ | progressive | 10x5 | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 97.41% | 92.48% | 94.39% | 86.21% | 75.93% | 99.72% | 88.5 KB | 1022.1 KB |
| no_qat__distill_progressive__prune_10x5__ptq_no | ✗ | progressive | 10x5 | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 99.05% | 97.36% | 99.05% | 97.36% | — | — | 314.7 KB | 1022.1 KB |
| no_qat__distill_progressive__prune_10x2__ptq_yes | ✗ | progressive | 10x2 | ✓ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 97.64% | 93.68% | 97.52% | 93.17% | 90.38% | 96.15% | 150.3 KB | 1734.0 KB |
| no_qat__distill_progressive__prune_10x2__ptq_no | ✗ | progressive | 10x2 | ✗ | 82.41% | 0.00% | 864.1 KB | 82.41% | 0.00% | 95.61% | 88.87% | 95.61% | 88.87% | — | — | 549.9 KB | 1734.0 KB |

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