# Experiment Sweep — Results

Generated: 2026-04-15 16:20:44

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 95.09% | 83.25% | 94.89% | 82.42% | 99.60% | 70.30% | 34.9 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 96.77% | 90.33% | 96.77% | 90.33% | — | — | 102.3 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.19% | 94.87% | 95.33% | 85.66% | 89.87% | 81.83% | 54.8 KB | 585.3 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 95.85% | 86.26% | 95.85% | 86.26% | — | — | 170.9 KB | 585.3 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 97.38% | 92.30% | 95.58% | 86.28% | 91.65% | 81.51% | 34.7 KB | 371.7 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 96.94% | 90.90% | 96.94% | 90.90% | — | — | 101.3 KB | 371.7 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | — | — | 94.32% | 85.66% | 75.20% | 99.51% | 75.0 KB | 297.1 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | — | — | 96.94% | 91.68% | — | — | 242.4 KB | 297.1 KB |
| no_qat__distill_progressive__prune_10x5__ptq_yes | ✗ | progressive | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.07% | 94.54% | 94.12% | 79.19% | 99.68% | 65.68% | 15.4 KB | 176.9 KB |
| no_qat__distill_progressive__prune_10x5__ptq_no | ✗ | progressive | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.14% | 94.72% | 98.14% | 94.72% | — | — | 37.9 KB | 176.9 KB |
| no_qat__distill_progressive__prune_10x2__ptq_yes | ✗ | progressive | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.17% | 94.83% | 95.79% | 87.06% | 91.42% | 83.10% | 22.3 KB | 246.9 KB |
| no_qat__distill_progressive__prune_10x2__ptq_no | ✗ | progressive | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.23% | 94.98% | 98.23% | 94.98% | — | — | 60.1 KB | 246.9 KB |
| no_qat__distill_progressive__prune_5x10__ptq_yes | ✗ | progressive | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.27% | 94.84% | 96.56% | 89.35% | 94.55% | 84.68% | 15.1 KB | 174.3 KB |
| no_qat__distill_progressive__prune_5x10__ptq_no | ✗ | progressive | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.51% | 95.47% | 98.51% | 95.47% | — | — | 37.0 KB | 174.3 KB |
| no_qat__distill_progressive__prune_none__ptq_yes | ✗ | progressive | none | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | — | — | 95.57% | 88.37% | 79.96% | 98.77% | 29.1 KB | 133.9 KB |
| no_qat__distill_progressive__prune_none__ptq_no | ✗ | progressive | none | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | — | — | 97.04% | 91.90% | — | — | 82.4 KB | 133.9 KB |
| no_qat__distill_none__prune_10x5__ptq_yes | ✗ | none | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 96.50% | 88.63% | 65.32% | 49.51% | 32.92% | 99.79% | 95.1 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x5__ptq_no | ✗ | none | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 98.00% | 93.79% | 98.00% | 93.79% | — | — | 314.7 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x2__ptq_yes | ✗ | none | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 99.17% | 97.58% | 72.97% | 55.47% | 38.56% | 98.80% | 159.1 KB | 1734.0 KB |
| no_qat__distill_none__prune_10x2__ptq_no | ✗ | none | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 99.54% | 98.65% | 99.54% | 98.65% | — | — | 549.9 KB | 1734.0 KB |
| no_qat__distill_none__prune_5x10__ptq_yes | ✗ | none | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 98.79% | 96.47% | 65.85% | 49.89% | 33.26% | 99.79% | 95.9 KB | 1029.3 KB |
| no_qat__distill_none__prune_5x10__ptq_no | ✗ | none | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 98.91% | 96.76% | 98.91% | 96.76% | — | — | 317.7 KB | 1029.3 KB |
| no_qat__distill_none__prune_none__ptq_yes | ✗ | none | none | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | — | — | 94.61% | 86.06% | 76.94% | 97.64% | 226.8 KB | 864.1 KB |
| no_qat__distill_none__prune_none__ptq_no | ✗ | none | none | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | — | — | 96.32% | 90.11% | — | — | 802.4 KB | 864.1 KB |

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