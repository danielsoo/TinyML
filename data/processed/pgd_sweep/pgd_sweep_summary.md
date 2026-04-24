# PGD Sweep Results — Final 6 for FGSM

- Surrogate attacker: `global_model.h5`
- Epsilon: 0.1  PGD steps: 10
- Eval samples: 5000

## Final 6 Models

| Role | Tag | PTQ | F1 | PGD Adv Acc | Size KB |
|------|-----|-----|----|-------------|---------|
| Best no-PTQ (PGD) | `yes_qat__distill_direct__prune_none__ptq_no` | No | 0.330 | 0.916 | 242.5 |
| Best yes-PTQ (PGD) | `yes_qat__distill_direct__prune_5x10__ptq_yes` | Yes | 0.843 | 0.915 | 31.1 |
| Fixed: extreme compression | `yes_qat__distill_progressive__prune_5x10__ptq_yes` | Yes | 0.867 | 0.914 | 14.4 |
| Fixed: moderate accuracy | `no_qat__distill_progressive__prune_10x5__ptq_no` | No | 0.935 | 0.176 | 37.9 |
| Fixed: compact+accurate | `no_qat__distill_progressive__prune_none__ptq_yes` | Yes | 0.884 | 0.287 | 29.1 |
| Additional pick | `yes_qat__distill_direct__prune_5x10__ptq_no` | No | 0.937 | 0.916 | 101.3 |

## Top 5 No-PTQ by PGD Adv Acc

| Tag | F1 | PGD Adv Acc | Size KB |
|-----|----|-------------|---------|
| `yes_qat__distill_direct__prune_none__ptq_no` | 0.330 | 0.916 | 242.5 |
| `yes_qat__distill_direct__prune_5x10__ptq_no` | 0.937 | 0.916 | 101.3 |
| `yes_qat__distill_direct__prune_10x2__ptq_no` | 0.909 | 0.916 | 170.9 |
| `yes_qat__distill_direct__prune_10x5__ptq_no` | 0.817 | 0.916 | 102.3 |
| `yes_qat__distill_progressive__prune_none__ptq_no` | 0.348 | 0.915 | 82.5 |

## Top 5 Yes-PTQ by PGD Adv Acc

| Tag | F1 | PGD Adv Acc | Size KB |
|-----|----|-------------|---------|
| `yes_qat__distill_direct__prune_5x10__ptq_yes` | 0.843 | 0.915 | 31.1 |
| `yes_qat__distill_direct__prune_10x2__ptq_yes` | 0.790 | 0.915 | 50.1 |
| `yes_qat__distill_direct__prune_10x5__ptq_yes` | 0.647 | 0.915 | 31.4 |
| `yes_qat__distill_direct__prune_none__ptq_yes` | 0.274 | 0.915 | 69.3 |
| `yes_qat__distill_progressive__prune_10x2__ptq_yes` | 0.767 | 0.914 | 19.6 |