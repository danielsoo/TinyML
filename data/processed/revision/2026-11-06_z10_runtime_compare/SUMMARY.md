# Revision experiments (full)

- commit: 55ce31cbbffae0693ca1357033a8ab4e26754ef3
- configs: config/jobs/2026-11-06_z10_runtime_compare (eval_split: )
- host: Linux 6.18.54.1-microsoft-standard-WSL2 x86_64, 16 cores
- device_agree: 6 min

# E7 device agreement

## esp32_runtime_arduino_tflm_1_0_0: ok

# Device agreement (?, ? MHz)

| model | inputs | exact output match | max diff (INT8 steps) | decisions FAR 1% | decisions FAR 5% | flash B | arena B | quantize µs | invoke µs (median / p95) |
|---|---|---|---|---|---|---|---|---|---|
| ton_S80 | 4462 | 170 | 170.00 | 2209/4462 | 4269/4462 | 67936 | 1440 | 11.0 | 13272.0 / 13285 |
| ton_PQ80_int16 | 4462 | 425 | 255.04 | 3379/4462 | 4346/4462 | 73456 | 1824 | 11.0 | 13866.0 / 13913 |
| ton_S80_pt | 4462 | 4099 | 4.00 | 4462/4462 | 4458/4462 | 57184 | 1440 | 11.0 | 13389.0 / 13403 |


## esp32_runtime_esp_tflite_micro: build/upload failed

