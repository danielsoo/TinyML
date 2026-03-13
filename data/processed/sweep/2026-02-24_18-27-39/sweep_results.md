# Experiment Sweep — Results

Generated: 2026-02-25 00:43:06

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 95.01% | 87.55% | 864.1 KB | 97.64% | 93.67% | 97.64% | 93.67% | 97.22% | 92.62% | 86.88% | 99.17% | 69.5 KB | 266.8 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 95.01% | 87.55% | 864.1 KB | 97.44% | 93.19% | 97.44% | 93.19% | 97.44% | 93.19% | — | — | 242.7 KB | 266.8 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 95.01% | 87.55% | 864.1 KB | 97.65% | 93.69% | 97.65% | 93.69% | 97.69% | 93.68% | 90.30% | 97.33% | 69.5 KB | 266.8 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 95.01% | 87.55% | 864.1 KB | 97.68% | 93.77% | 97.68% | 93.77% | 97.68% | 93.77% | — | — | 242.7 KB | 266.8 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 95.01% | 87.55% | 864.1 KB | 96.76% | 91.53% | 96.76% | 91.53% | 97.43% | 93.15% | 87.60% | 99.45% | 69.5 KB | 266.8 KB |
| no_qat__distill_direct__prune_5x10__ptq_no | ✗ | direct | 5x10 | ✗ | 95.01% | 87.55% | 864.1 KB | 97.67% | 93.74% | 97.67% | 93.74% | 97.67% | 93.74% | — | — | 242.7 KB | 266.8 KB |
| no_qat__distill_direct__prune_none__ptq_yes | ✗ | direct | none | ✓ | 95.01% | 87.55% | 864.1 KB | 97.51% | 93.35% | — | — | 97.63% | 93.59% | 89.24% | 98.39% | 69.5 KB | 266.8 KB |
| no_qat__distill_direct__prune_none__ptq_no | ✗ | direct | none | ✗ | 95.01% | 87.55% | 864.1 KB | 97.54% | 93.43% | — | — | 97.54% | 93.43% | — | — | 242.7 KB | 266.8 KB |
| no_qat__distill_progressive__prune_10x5__ptq_yes | ✗ | progressive | 10x5 | ✓ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_progressive__prune_10x5__ptq_no | ✗ | progressive | 10x5 | ✗ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_progressive__prune_10x2__ptq_yes | ✗ | progressive | 10x2 | ✓ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_progressive__prune_10x2__ptq_no | ✗ | progressive | 10x2 | ✗ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_progressive__prune_5x10__ptq_yes | ✗ | progressive | 5x10 | ✓ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_progressive__prune_5x10__ptq_no | ✗ | progressive | 5x10 | ✗ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_progressive__prune_none__ptq_yes | ✗ | progressive | none | ✓ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | — | — | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_progressive__prune_none__ptq_no | ✗ | progressive | none | ✗ | 95.01% | 87.55% | 864.1 KB | 95.01% | 87.55% | — | — | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_none__prune_10x5__ptq_yes | ✗ | none | 10x5 | ✓ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_none__prune_10x5__ptq_no | ✗ | none | 10x5 | ✗ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_none__prune_10x2__ptq_yes | ✗ | none | 10x2 | ✓ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_none__prune_10x2__ptq_no | ✗ | none | 10x2 | ✗ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_none__prune_5x10__ptq_yes | ✗ | none | 5x10 | ✓ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_none__prune_5x10__ptq_no | ✗ | none | 5x10 | ✗ | 95.01% | 87.55% | 864.1 KB | — | — | 95.01% | 87.55% | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| no_qat__distill_none__prune_none__ptq_yes | ✗ | none | none | ✓ | 95.01% | 87.55% | 864.1 KB | — | — | — | — | 94.77% | 86.36% | 79.73% | 94.19% | 216.0 KB | 864.1 KB |
| no_qat__distill_none__prune_none__ptq_no | ✗ | none | none | ✗ | 95.01% | 87.55% | 864.1 KB | — | — | — | — | 95.01% | 87.55% | — | — | 802.4 KB | 864.1 KB |
| yes_qat__distill_direct__prune_10x5__ptq_yes | ✓ | direct | 10x5 | ✓ | 21.23% | 30.86% | 864.1 KB | 96.92% | 91.90% | 96.92% | 91.90% | 96.81% | 91.65% | 84.95% | 99.51% | 69.6 KB | 266.8 KB |

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