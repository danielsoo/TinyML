# Experiment Sweep — Results

Generated: 2026-04-16 09:24:44

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 96.20% | 87.48% | 94.96% | 82.80% | 98.80% | 71.26% | 34.9 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.22% | 94.90% | 98.22% | 94.90% | — | — | 102.3 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.21% | 94.92% | 95.23% | 85.15% | 90.66% | 80.28% | 54.8 KB | 585.3 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.00% | 94.36% | 98.00% | 94.36% | — | — | 170.9 KB | 585.3 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.38% | 95.39% | 96.70% | 90.24% | 91.01% | 89.48% | 34.7 KB | 371.7 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | 98.49% | 95.48% | 98.49% | 95.48% | — | — | 101.3 KB | 371.7 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | — | — | 94.32% | 85.66% | 75.20% | 99.51% | 75.0 KB | 297.1 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 96.32% | 90.11% | 850.8 KB | 96.94% | 91.68% | — | — | 96.94% | 91.68% | — | — | 242.4 KB | 297.1 KB |
| no_qat__distill_progressive__prune_10x5__ptq_yes | ✗ | progressive | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.00% | 94.36% | 93.00% | 74.21% | 99.64% | 59.12% | 15.4 KB | 176.9 KB |
| no_qat__distill_progressive__prune_10x5__ptq_no | ✗ | progressive | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 97.68% | 93.49% | 97.68% | 93.49% | — | — | 37.9 KB | 176.9 KB |
| no_qat__distill_progressive__prune_10x2__ptq_yes | ✗ | progressive | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.10% | 94.64% | 95.67% | 86.59% | 91.54% | 82.15% | 22.3 KB | 246.9 KB |
| no_qat__distill_progressive__prune_10x2__ptq_no | ✗ | progressive | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.18% | 94.82% | 98.18% | 94.82% | — | — | 60.1 KB | 246.9 KB |
| no_qat__distill_progressive__prune_5x10__ptq_yes | ✗ | progressive | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 98.22% | 94.97% | 96.23% | 87.60% | 99.57% | 78.19% | 15.1 KB | 174.3 KB |
| no_qat__distill_progressive__prune_5x10__ptq_no | ✗ | progressive | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | 97.87% | 93.76% | 97.87% | 93.76% | — | — | 37.0 KB | 174.3 KB |
| no_qat__distill_progressive__prune_none__ptq_yes | ✗ | progressive | none | ✓ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | — | — | 95.57% | 88.37% | 79.96% | 98.77% | 29.1 KB | 133.9 KB |
| no_qat__distill_progressive__prune_none__ptq_no | ✗ | progressive | none | ✗ | 96.32% | 90.11% | 850.8 KB | 97.04% | 91.90% | — | — | 97.04% | 91.90% | — | — | 82.4 KB | 133.9 KB |
| no_qat__distill_none__prune_10x5__ptq_yes | ✗ | none | 10x5 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 97.83% | 93.43% | 64.29% | 48.74% | 32.26% | 99.66% | 95.1 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x5__ptq_no | ✗ | none | 10x5 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 97.57% | 93.01% | 97.57% | 93.01% | — | — | 314.7 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x2__ptq_yes | ✗ | none | 10x2 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 97.68% | 93.08% | 82.29% | 65.72% | 49.03% | 99.63% | 159.1 KB | 1734.0 KB |
| no_qat__distill_none__prune_10x2__ptq_no | ✗ | none | 10x2 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 98.24% | 94.69% | 98.24% | 94.69% | — | — | 549.9 KB | 1734.0 KB |
| no_qat__distill_none__prune_5x10__ptq_yes | ✗ | none | 5x10 | ✓ | 96.32% | 90.11% | 850.8 KB | — | — | 91.77% | 68.22% | 80.05% | 63.04% | 46.05% | 99.85% | 95.9 KB | 1029.3 KB |
| no_qat__distill_none__prune_5x10__ptq_no | ✗ | none | 5x10 | ✗ | 96.32% | 90.11% | 850.8 KB | — | — | 98.53% | 95.56% | 98.53% | 95.56% | — | — | 317.7 KB | 1029.3 KB |
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