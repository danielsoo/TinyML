# Experiment Sweep — Results

Generated: 2026-03-06 09:14:40

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 93.83% | 85.03% | 864.1 KB | 95.89% | 89.48% | 97.28% | 92.71% | 94.36% | 83.00% | 88.27% | 78.32% | 31.3 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 93.83% | 85.03% | 864.1 KB | 96.69% | 91.35% | 98.91% | 96.97% | 98.91% | 96.97% | — | — | 102.3 KB | 374.5 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 93.83% | 85.03% | 864.1 KB | 96.24% | 90.25% | 99.34% | 98.12% | 97.22% | 92.51% | 87.97% | 97.55% | 50.1 KB | 585.9 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 93.83% | 85.03% | 864.1 KB | 96.04% | 89.79% | 98.58% | 95.98% | 98.58% | 95.98% | — | — | 170.9 KB | 585.9 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 93.83% | 85.03% | 864.1 KB | 98.78% | 96.55% | 98.27% | 95.27% | 97.41% | 93.00% | 88.54% | 97.94% | 31.1 KB | 371.8 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 93.83% | 85.03% | 864.1 KB | 97.54% | 93.39% | 97.41% | 92.75% | 97.41% | 92.75% | — | — | 101.3 KB | 371.8 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 93.83% | 85.03% | 864.1 KB | 96.21% | 90.16% | — | — | 94.85% | 87.06% | 77.98% | 98.53% | 69.3 KB | 283.1 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 93.83% | 85.03% | 864.1 KB | 97.57% | 93.50% | — | — | 97.57% | 93.50% | — | — | 242.4 KB | 283.1 KB |
| no_qat__distill_progressive__prune_10x5__ptq_yes | ✗ | progressive | 10x5 | ✓ | 93.83% | 85.03% | 864.1 KB | 97.35% | 92.55% | 97.22% | 92.15% | 94.60% | 83.71% | 89.20% | 78.86% | 14.9 KB | 176.8 KB |
| no_qat__distill_progressive__prune_10x5__ptq_no | ✗ | progressive | 10x5 | ✗ | 93.83% | 85.03% | 864.1 KB | 97.60% | 93.51% | 99.16% | 97.60% | 99.16% | 97.60% | — | — | 37.9 KB | 176.8 KB |
| no_qat__distill_progressive__prune_10x2__ptq_yes | ✗ | progressive | 10x2 | ✓ | 93.83% | 85.03% | 864.1 KB | 97.67% | 93.75% | 97.63% | 93.65% | 97.17% | 92.44% | 87.25% | 98.28% | 19.6 KB | 246.9 KB |
| no_qat__distill_progressive__prune_10x2__ptq_no | ✗ | progressive | 10x2 | ✗ | 93.83% | 85.03% | 864.1 KB | 95.87% | 89.39% | 97.28% | 92.59% | 97.28% | 92.59% | — | — | 60.1 KB | 246.9 KB |
| no_qat__distill_progressive__prune_5x10__ptq_yes | ✗ | progressive | 5x10 | ✓ | 93.83% | 85.03% | 864.1 KB | 97.89% | 94.15% | 98.40% | 95.35% | 93.54% | 77.63% | 99.35% | 63.71% | 14.4 KB | 174.4 KB |
| no_qat__distill_progressive__prune_5x10__ptq_no | ✗ | progressive | 5x10 | ✗ | 93.83% | 85.03% | 864.1 KB | 97.79% | 94.02% | 97.72% | 93.70% | 97.72% | 93.70% | — | — | 37.0 KB | 174.4 KB |
| no_qat__distill_progressive__prune_none__ptq_yes | ✗ | progressive | none | ✓ | 93.83% | 85.03% | 864.1 KB | 97.59% | 93.47% | — | — | 97.65% | 93.59% | 89.97% | 97.51% | 25.9 KB | 120.1 KB |
| no_qat__distill_progressive__prune_none__ptq_no | ✗ | progressive | none | ✗ | 93.83% | 85.03% | 864.1 KB | 97.51% | 93.22% | — | — | 97.51% | 93.22% | — | — | 82.4 KB | 120.1 KB |
| no_qat__distill_none__prune_10x5__ptq_yes | ✗ | none | 10x5 | ✓ | 93.83% | 85.03% | 864.1 KB | — | — | 99.12% | 97.53% | 87.57% | 66.50% | 63.24% | 70.12% | 88.5 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x5__ptq_no | ✗ | none | 10x5 | ✗ | 93.83% | 85.03% | 864.1 KB | — | — | 96.89% | 91.25% | 96.89% | 91.25% | — | — | 314.7 KB | 1022.1 KB |
| no_qat__distill_none__prune_10x2__ptq_yes | ✗ | none | 10x2 | ✓ | 93.83% | 85.03% | 864.1 KB | — | — | 98.93% | 97.03% | 93.91% | 79.43% | 97.82% | 66.86% | 150.3 KB | 1734.0 KB |
| no_qat__distill_none__prune_10x2__ptq_no | ✗ | none | 10x2 | ✗ | 93.83% | 85.03% | 864.1 KB | — | — | 97.71% | 93.81% | 97.71% | 93.81% | — | — | 549.9 KB | 1734.0 KB |
| no_qat__distill_none__prune_5x10__ptq_yes | ✗ | none | 5x10 | ✓ | 93.83% | 85.03% | 864.1 KB | — | — | 98.49% | 95.86% | 93.03% | 77.17% | 90.95% | 67.01% | 89.3 KB | 1029.3 KB |
| no_qat__distill_none__prune_5x10__ptq_no | ✗ | none | 5x10 | ✗ | 93.83% | 85.03% | 864.1 KB | — | — | 98.29% | 95.27% | 98.29% | 95.27% | — | — | 317.7 KB | 1029.3 KB |
| no_qat__distill_none__prune_none__ptq_yes | ✗ | none | none | ✓ | 93.83% | 85.03% | 864.1 KB | — | — | — | — | 77.22% | 59.78% | 43.35% | 96.25% | 216.0 KB | 864.1 KB |
| no_qat__distill_none__prune_none__ptq_no | ✗ | none | none | ✗ | 93.83% | 85.03% | 864.1 KB | — | — | — | — | 93.83% | 85.03% | — | — | 802.4 KB | 864.1 KB |
| yes_qat__distill_direct__prune_10x5__ptq_yes | ✓ | direct | 10x5 | ✓ | 84.64% | 32.71% | 864.1 KB | 84.68% | 32.78% | 97.68% | 93.38% | 90.82% | 64.72% | 99.94% | 47.86% | 31.4 KB | 374.4 KB |
| yes_qat__distill_direct__prune_10x5__ptq_no | ✓ | direct | 10x5 | ✗ | 84.64% | 32.71% | 864.1 KB | 84.64% | 32.30% | 94.37% | 81.71% | 94.37% | 81.71% | — | — | 102.3 KB | 374.4 KB |
| yes_qat__distill_direct__prune_10x2__ptq_yes | ✓ | direct | 10x2 | ✓ | 84.64% | 32.71% | 864.1 KB | 84.61% | 32.02% | 99.13% | 97.56% | 93.81% | 78.96% | 98.24% | 66.01% | 50.1 KB | 585.9 KB |
| yes_qat__distill_direct__prune_10x2__ptq_no | ✓ | direct | 10x2 | ✗ | 84.64% | 32.71% | 864.1 KB | 84.69% | 32.83% | 96.49% | 90.86% | 96.49% | 90.86% | — | — | 170.9 KB | 585.9 KB |
| yes_qat__distill_direct__prune_5x10__ptq_yes | ✓ | direct | 5x10 | ✓ | 84.64% | 32.71% | 864.1 KB | 84.65% | 32.63% | 97.60% | 93.56% | 94.62% | 84.30% | 86.51% | 82.20% | 31.1 KB | 371.8 KB |
| yes_qat__distill_direct__prune_5x10__ptq_no | ✓ | direct | 5x10 | ✗ | 84.64% | 32.71% | 864.1 KB | 84.69% | 32.89% | 97.68% | 93.73% | 97.68% | 93.73% | — | — | 101.3 KB | 371.8 KB |
| yes_qat__distill_direct__prune_none__ptq_yes | ✓ | direct | none | ✓ | 84.64% | 32.71% | 864.1 KB | 84.63% | 32.45% | — | — | 85.19% | 27.36% | 99.50% | 15.86% | 69.3 KB | 283.1 KB |
| yes_qat__distill_direct__prune_none__ptq_no | ✓ | direct | none | ✗ | 84.64% | 32.71% | 864.1 KB | 84.72% | 33.00% | — | — | 84.72% | 33.00% | — | — | 242.5 KB | 283.1 KB |
| yes_qat__distill_progressive__prune_10x5__ptq_yes | ✓ | progressive | 10x5 | ✓ | 84.64% | 32.71% | 864.1 KB | 86.15% | 35.09% | 97.70% | 93.65% | 90.80% | 64.62% | 99.90% | 47.75% | 14.9 KB | 176.8 KB |
| yes_qat__distill_progressive__prune_10x5__ptq_no | ✓ | progressive | 10x5 | ✗ | 84.64% | 32.71% | 864.1 KB | 86.06% | 34.41% | 97.91% | 94.33% | 97.91% | 94.33% | — | — | 37.9 KB | 176.8 KB |
| yes_qat__distill_progressive__prune_10x2__ptq_yes | ✓ | progressive | 10x2 | ✓ | 84.64% | 32.71% | 864.1 KB | 86.08% | 34.55% | 97.76% | 93.86% | 93.35% | 76.73% | 99.87% | 62.30% | 19.6 KB | 246.9 KB |
| yes_qat__distill_progressive__prune_10x2__ptq_no | ✓ | progressive | 10x2 | ✗ | 84.64% | 32.71% | 864.1 KB | 86.06% | 34.44% | 97.44% | 93.03% | 97.44% | 93.03% | — | — | 60.1 KB | 246.9 KB |
| yes_qat__distill_progressive__prune_5x10__ptq_yes | ✓ | progressive | 5x10 | ✓ | 84.64% | 32.71% | 864.1 KB | 86.14% | 35.05% | 97.76% | 93.77% | 95.81% | 86.66% | 98.55% | 77.33% | 14.4 KB | 174.4 KB |
| yes_qat__distill_progressive__prune_5x10__ptq_no | ✓ | progressive | 5x10 | ✗ | 84.64% | 32.71% | 864.1 KB | 86.14% | 35.05% | 97.82% | 94.10% | 97.82% | 94.10% | — | — | 37.0 KB | 174.4 KB |
| yes_qat__distill_progressive__prune_none__ptq_yes | ✓ | progressive | none | ✓ | 84.64% | 32.71% | 864.1 KB | 86.07% | 34.50% | — | — | 85.93% | 33.35% | 99.80% | 20.02% | 25.9 KB | 120.1 KB |
| yes_qat__distill_progressive__prune_none__ptq_no | ✓ | progressive | none | ✗ | 84.64% | 32.71% | 864.1 KB | 86.11% | 34.84% | — | — | 86.11% | 34.84% | — | — | 82.5 KB | 120.1 KB |
| yes_qat__distill_none__prune_10x5__ptq_yes | ✓ | none | 10x5 | ✓ | 84.64% | 32.71% | 864.1 KB | — | — | 45.26% | 39.07% | 73.05% | 55.40% | 39.07% | 95.19% | 88.5 KB | 1022.1 KB |
| yes_qat__distill_none__prune_10x5__ptq_no | ✓ | none | 10x5 | ✗ | 84.64% | 32.71% | 864.1 KB | — | — | 92.83% | 82.98% | 92.83% | 82.98% | — | — | 314.7 KB | 1022.1 KB |
| yes_qat__distill_none__prune_10x2__ptq_yes | ✓ | none | 10x2 | ✓ | 84.64% | 32.71% | 864.1 KB | — | — | 88.27% | 74.92% | 83.99% | 68.27% | 52.39% | 97.98% | 150.3 KB | 1734.0 KB |
| yes_qat__distill_none__prune_10x2__ptq_no | ✓ | none | 10x2 | ✗ | 84.64% | 32.71% | 864.1 KB | — | — | 90.63% | 78.89% | 90.63% | 78.89% | — | — | 549.9 KB | 1734.0 KB |
| yes_qat__distill_none__prune_5x10__ptq_yes | ✓ | none | 5x10 | ✓ | 84.64% | 32.71% | 864.1 KB | — | — | 90.75% | 79.09% | 88.99% | 73.64% | 63.60% | 87.43% | 89.3 KB | 1029.3 KB |
| yes_qat__distill_none__prune_5x10__ptq_no | ✓ | none | 5x10 | ✗ | 84.64% | 32.71% | 864.1 KB | — | — | 92.27% | 81.91% | 92.27% | 81.91% | — | — | 317.7 KB | 1029.3 KB |
| yes_qat__distill_none__prune_none__ptq_yes | ✓ | none | none | ✓ | 84.64% | 32.71% | 864.1 KB | — | — | — | — | 72.33% | 50.04% | 36.66% | 78.77% | 216.0 KB | 864.1 KB |
| yes_qat__distill_none__prune_none__ptq_no | ✓ | none | none | ✗ | 84.64% | 32.71% | 864.1 KB | — | — | — | — | 84.64% | 32.71% | — | — | 802.5 KB | 864.1 KB |

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