# PGD AT + PTQ Evaluation Report

- Model: `global_model.h5`
- AT: eps=0.1, steps=10, epochs=3
- Eval epsilons: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3]  |  Eval samples: 5000

## Design
- **Phase A (pre_AT_h5)**: h5 evaluated directly, PGD from clean h5
- **Phase B (post_AT_h5)**: h5 evaluated directly after PGD AT
- **Phase C (float_tflite)**: float32 TFLite exported pre-AT, attacked via transfer from AT'd h5
- **Phase D (ptq_tflite)**: PTQ TFLite exported pre-AT, attacked via transfer from AT'd h5

**A→B**: Does AT improve h5 robustness?
**A→C**: Does float32 TFLite conversion hurt robustness under transfer attack?
**A→D**: Does PTQ further degrade robustness vs float TFLite?
**B vs D**: AT'd h5 vs PTQ TFLite under same adversarial pressure.

## Results by Phase

| Phase | eps | Orig Acc | Adv Acc | Adv F1 | Adv Rec | Success |
|-------|-----|----------|---------|--------|---------|---------|
| pre_AT_h5 | 0.01 | 0.757 | 0.809 | 0.353 | 0.296 | -0.053 |
| pre_AT_h5 | 0.05 | 0.757 | 0.736 | 0.015 | 0.011 | +0.021 |
| pre_AT_h5 | 0.1 | 0.757 | 0.708 | 0.007 | 0.006 | +0.049 |
| pre_AT_h5 | 0.15 | 0.757 | 0.674 | 0.005 | 0.005 | +0.083 |
| pre_AT_h5 | 0.2 | 0.757 | 0.627 | 0.000 | 0.000 | +0.130 |
| pre_AT_h5 | 0.3 | 0.757 | 0.518 | 0.000 | 0.000 | +0.239 |
| post_AT_h5 | 0.01 | 0.918 | 0.913 | 0.712 | 0.611 | +0.005 |
| post_AT_h5 | 0.05 | 0.918 | 0.910 | 0.702 | 0.604 | +0.008 |
| post_AT_h5 | 0.1 | 0.918 | 0.906 | 0.691 | 0.600 | +0.013 |
| post_AT_h5 | 0.15 | 0.918 | 0.896 | 0.662 | 0.581 | +0.022 |
| post_AT_h5 | 0.2 | 0.918 | 0.882 | 0.627 | 0.564 | +0.036 |
| post_AT_h5 | 0.3 | 0.918 | 0.823 | 0.492 | 0.487 | +0.095 |
| float_tflite | 0.01 | 0.757 | 0.820 | 0.383 | 0.319 | -0.063 |
| float_tflite | 0.05 | 0.757 | 0.819 | 0.360 | 0.290 | -0.062 |
| float_tflite | 0.1 | 0.757 | 0.809 | 0.294 | 0.227 | -0.052 |
| float_tflite | 0.15 | 0.757 | 0.800 | 0.231 | 0.171 | -0.043 |
| float_tflite | 0.2 | 0.757 | 0.794 | 0.173 | 0.123 | -0.037 |
| float_tflite | 0.3 | 0.757 | 0.786 | 0.117 | 0.081 | -0.029 |
| ptq_tflite | 0.01 | 0.753 | 0.795 | 0.323 | 0.278 | -0.042 |
| ptq_tflite | 0.05 | 0.753 | 0.816 | 0.341 | 0.271 | -0.063 |
| ptq_tflite | 0.1 | 0.753 | 0.808 | 0.273 | 0.205 | -0.055 |
| ptq_tflite | 0.15 | 0.753 | 0.806 | 0.227 | 0.162 | -0.053 |
| ptq_tflite | 0.2 | 0.753 | 0.781 | 0.170 | 0.128 | -0.028 |
| ptq_tflite | 0.3 | 0.753 | 0.781 | 0.129 | 0.092 | -0.027 |

## Summary (mean over epsilon sweep)

| Phase | Mean Adv Acc | Mean Adv F1 | Mean Adv Rec | Min Adv Acc |
|-------|-------------|------------|-------------|------------|
| pre_AT_h5 | 0.679 | 0.063 | 0.053 | 0.518 |
| post_AT_h5 | 0.888 | 0.648 | 0.574 | 0.823 |
| float_tflite | 0.804 | 0.260 | 0.202 | 0.786 |
| ptq_tflite | 0.798 | 0.244 | 0.189 | 0.781 |