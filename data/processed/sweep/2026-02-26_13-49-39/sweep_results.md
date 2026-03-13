# Experiment Sweep — Results

Generated: 2026-02-26 14:08:16

FGSM step is planned as a separate phase (not included here).


## Summary Table

| Tag | FL QAT | Distillation | Pruning | PTQ | FL Acc | FL F1 | FL size | Dist Acc | Dist F1 | Prune Acc | Prune F1 | Final Acc | Final F1 | Final Prec | Final Rec | TFLite size | Final size |
|-----|--------|--------------|---------|-----|--------|-------|---------|----------|---------|-----------|----------|-----------|----------|------------|-----------|------------|------------|
| no_qat__distill_direct__prune_10x5__ptq_yes | ✗ | direct | 10x5 | ✓ | 81.31% | 0.00% | 864.1 KB | 81.31% | 0.00% | 81.31% | 0.00% | 18.69% | 31.49% | 18.69% | 100.00% | 29.9 KB | 374.4 KB |
| no_qat__distill_direct__prune_10x5__ptq_no | ✗ | direct | 10x5 | ✗ | 81.31% | 0.00% | 864.1 KB | 81.31% | 0.00% | 81.31% | 0.00% | 81.31% | 0.00% | — | — | 102.2 KB | 374.5 KB |
| no_qat__distill_direct__prune_10x2__ptq_yes | ✗ | direct | 10x2 | ✓ | 81.31% | 0.00% | 864.1 KB | 81.31% | 0.00% | 81.31% | 0.00% | 18.69% | 31.49% | 18.69% | 100.00% | 48.2 KB | 585.9 KB |
| no_qat__distill_direct__prune_10x2__ptq_no | ✗ | direct | 10x2 | ✗ | 81.31% | 0.00% | 864.1 KB | 81.31% | 0.00% | 81.31% | 0.00% | 81.31% | 0.00% | — | — | 170.7 KB | 585.9 KB |
| no_qat__distill_direct__prune_5x10__ptq_yes | ✗ | direct | 5x10 | ✓ | 81.31% | 0.00% | 864.1 KB | 81.31% | 0.00% | 81.31% | 0.00% | 18.69% | 31.49% | 18.69% | 100.00% | 29.7 KB | 371.8 KB |

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