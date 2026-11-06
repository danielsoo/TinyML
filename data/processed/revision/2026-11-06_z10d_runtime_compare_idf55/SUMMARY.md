# Revision experiments (full)

- commit: b558097fad41075a8f32a945855b7dd10529e47e
- configs: config/jobs/2026-11-06_z10d_runtime_compare_idf55 (eval_split: )
- host: Linux 6.18.54.1-microsoft-standard-WSL2 x86_64, 16 cores
- device_agree: 6 min

# E7 device agreement

## esp32_runtime_esp_tflite_micro: ok

# Device agreement (ESP32, 240 MHz)

| model | inputs | exact output match | max diff (INT8 steps) | decisions FAR 1% | decisions FAR 5% | flash B | arena B | quantize µs | invoke µs (median / p95) |
|---|---|---|---|---|---|---|---|---|---|
| ton_S80 | 4462 | 4377 | 3.00 | 4459/4462 | 4462/4462 | 67936 | 5528 | 13.0 | 10425.5 / 10435 |
| ton_PQ80_int16 | 4462 | 0 | 141.95 | 1894/4462 | 3149/4462 | 73456 | 5912 | 13.0 | 12098.0 / 12118 |
| ton_S80_pt | 4462 | 4099 | 4.00 | 4462/4462 | 4458/4462 | 57184 | 1944 | 13.0 | 10295.0 / 10299 |


