# Experiment Sweep — Results

Generated: 2026-04-16 18:23:14

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| yes_qat__distill_direct__prune_10x5__ptq_yes | ✓ | direct | 10x5 | ✓ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 95.34% | 86.66% | 93.83% | 83.06% | 77.98% | 88.86% | 1.8 KB | 19.8 KB |
| yes_qat__distill_direct__prune_10x5__ptq_no | ✓ | direct | 10x5 | ✗ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 95.48% | 86.94% | 95.48% | 86.94% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_direct__prune_10x2__ptq_yes | ✓ | direct | 10x2 | ✓ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 94.45% | 84.69% | 93.80% | 79.67% | 90.18% | 71.35% | 1.8 KB | 19.8 KB |
| yes_qat__distill_direct__prune_10x2__ptq_no | ✓ | direct | 10x2 | ✗ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 94.45% | 84.76% | 94.45% | 84.76% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_direct__prune_5x10__ptq_yes | ✓ | direct | 5x10 | ✓ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 95.68% | 87.36% | 92.08% | 80.54% | 69.26% | 96.21% | 1.8 KB | 19.8 KB |
| yes_qat__distill_direct__prune_5x10__ptq_no | ✓ | direct | 5x10 | ✗ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | 95.75% | 87.67% | 95.75% | 87.67% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_direct__prune_none__ptq_yes | ✓ | direct | none | ✓ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | — | — | 92.26% | 79.08% | 73.31% | 85.84% | 1.8 KB | 13.6 KB |
| yes_qat__distill_direct__prune_none__ptq_no | ✓ | direct | none | ✗ | 92.84% | 80.47% | 847.7 KB | 92.86% | 80.41% | — | — | 92.86% | 80.41% | — | — | 1.6 KB | 13.6 KB |
| yes_qat__distill_progressive__prune_10x5__ptq_yes | ✓ | progressive | 10x5 | ✓ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.53% | 86.99% | 93.49% | 83.51% | 73.48% | 96.71% | 1.8 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_10x5__ptq_no | ✓ | progressive | 10x5 | ✗ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.61% | 87.30% | 95.61% | 87.30% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_10x2__ptq_yes | ✓ | progressive | 10x2 | ✓ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.42% | 86.69% | 94.27% | 82.11% | 87.78% | 77.13% | 1.8 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_10x2__ptq_no | ✓ | progressive | 10x2 | ✗ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.33% | 86.73% | 95.33% | 86.73% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_5x10__ptq_yes | ✓ | progressive | 5x10 | ✓ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.66% | 87.31% | 92.98% | 82.20% | 72.39% | 95.08% | 1.8 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_5x10__ptq_no | ✓ | progressive | 5x10 | ✗ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | 95.51% | 87.09% | 95.51% | 87.09% | — | — | 1.6 KB | 19.8 KB |
| yes_qat__distill_progressive__prune_none__ptq_yes | ✓ | progressive | none | ✓ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | — | — | 93.68% | 81.78% | 80.33% | 83.28% | 1.8 KB | 13.6 KB |
| yes_qat__distill_progressive__prune_none__ptq_no | ✓ | progressive | none | ✗ | 92.84% | 80.47% | 847.7 KB | 93.26% | 81.29% | — | — | 93.26% | 81.29% | — | — | 1.6 KB | 13.6 KB |
| yes_qat__distill_none__prune_10x5__ptq_yes | ✓ | none | 10x5 | ✓ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | 75.18% | 86.57% | 206.2 KB | 843.7 KB |
| yes_qat__distill_none__prune_10x5__ptq_no | ✓ | none | 10x5 | ✗ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | — | — | 806.0 KB | 843.7 KB |
| yes_qat__distill_none__prune_10x2__ptq_yes | ✓ | none | 10x2 | ✓ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | 75.18% | 86.57% | 206.2 KB | 843.7 KB |
| yes_qat__distill_none__prune_10x2__ptq_no | ✓ | none | 10x2 | ✗ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | — | — | 806.0 KB | 843.7 KB |
| yes_qat__distill_none__prune_5x10__ptq_yes | ✓ | none | 5x10 | ✓ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | 75.18% | 86.57% | 206.2 KB | 843.7 KB |
| yes_qat__distill_none__prune_5x10__ptq_no | ✓ | none | 5x10 | ✗ | 92.84% | 80.47% | 847.7 KB | — | — | 92.84% | 80.47% | 92.84% | 80.47% | — | — | 806.0 KB | 843.7 KB |
| yes_qat__distill_none__prune_none__ptq_yes | ✓ | none | none | ✓ | 92.84% | 80.47% | 847.7 KB | — | — | — | — | 92.84% | 80.47% | 75.18% | 86.57% | 206.2 KB | 843.7 KB |
| yes_qat__distill_none__prune_none__ptq_no | ✓ | none | none | ✗ | 92.84% | 80.47% | 847.7 KB | — | — | — | — | 92.84% | 80.47% | — | — | 806.0 KB | 843.7 KB |

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