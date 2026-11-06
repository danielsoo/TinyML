# Runtime comparison on the classic ESP32 (same files, same inputs)

Inputs: 4462 (2722 attacks, 1330 benign, 410 synthetic probes without a label). Reference: host TFLite interpreter on the same file. Decision changes at the frozen thresholds of the bundle; misses / false alarms count labelled records only.

## Runtimes

| runtime | device banner | build |
|---|---|---|
| TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | runtime_name=TensorFlowLite_ESP32-1.0.0, runtime_kernels=reference, memory_free_heap=291168, memory_min_free_heap=285568, memory_arena_static=49152 | library.properties, piopm, platformio.ini, platformio_version.txt |
| esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | chip=ESP32, cores=2, cpu_mhz=240, flash_bytes=4194304, sdk=5.5.0, runtime_name=esp-tflite-micro, runtime_kernels=esp-nn, memory_free_heap=215024, memory_min_free_heap=215024, memory_arena_static=49152 | component_hash, dependencies.lock, idf_component.yml, idf_component.yml, platformio.ini, platformio_version.txt, sdkconfig.defaults, sdkconfig.esp32dev |
| esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | chip=ESP32, cores=2, cpu_mhz=240, flash_bytes=4194304, sdk=5.5.0, runtime_name=esp-tflite-micro, runtime_kernels=ansi-c, memory_free_heap=215024, memory_min_free_heap=215024, memory_arena_static=49152 | component_hash, dependencies.lock, idf_component.yml, idf_component.yml, platformio.ini, platformio_version.txt, sdkconfig.defaults, sdkconfig.esp32dev-ansi-c, sdkconfig_ansi_c.defaults |

## Models

| model | sha256 (first 16) | bytes | ops (versions) | graph | per-channel tensors |
|---|---|---|---|---|---|
| ton_S80 | bdbdedee6af55657 | 67936 | DEQUANTIZE v2, FULLY_CONNECTED v4, LOGISTIC v2, QUANTIZE v1 | QUANTIZE(FLOAT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> LOGISTIC(INT8) -> DEQUANTIZE(INT8) | 6 |
| ton_S80_pt | 766e6832ba407c41 | 57184 | DEQUANTIZE v2, FULLY_CONNECTED v4, LOGISTIC v2, QUANTIZE v1 | QUANTIZE(FLOAT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> FULLY_CONNECTED(INT8,INT8,INT32) -> LOGISTIC(INT8) -> DEQUANTIZE(INT8) | 0 |
| ton_PQ80_int16 | 8d33d042d7a87733 | 73456 | DEQUANTIZE v2, FULLY_CONNECTED v4, LOGISTIC v2, QUANTIZE v1 | QUANTIZE(FLOAT32) -> FULLY_CONNECTED(INT16,INT8,INT64) -> FULLY_CONNECTED(INT16,INT8,INT64) -> FULLY_CONNECTED(INT16,INT8,INT64) -> FULLY_CONNECTED(INT16,INT8,INT64) -> LOGISTIC(INT16) -> DEQUANTIZE(INT16) | 6 |

## Outputs and decisions vs host

| model | runtime | exact output = host | max abs diff | output std (host / device) | arena used B | free heap B | invoke µs median | FAR | decisions = host | extra misses | extra false alarms | fewer false alarms | extra detections |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ton_S80 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 170/4462 | 0.6641 | 0.315 / 0.232 | 1440 | 291168 | 13272 | 0.01 | 2209/4462 | 2069 | 0 | 23 | 0 |
| ton_S80 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 170/4462 | 0.6641 | 0.315 / 0.232 | 1440 | 291168 | 13272 | 0.05 | 4269/4462 | 49 | 0 | 89 | 0 |
| ton_S80 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 4377/4462 | 0.0117 | 0.315 / 0.315 | 5528 | 215024 | 10426 | 0.01 | 4459/4462 | 0 | 0 | 0 | 3 |
| ton_S80 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 4377/4462 | 0.0117 | 0.315 / 0.315 | 5528 | 215024 | 10426 | 0.05 | 4462/4462 | 0 | 0 | 0 | 0 |
| ton_S80 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4377/4462 | 0.0117 | 0.315 / 0.315 | 5528 | 215024 | 10405 | 0.01 | 4459/4462 | 0 | 0 | 0 | 3 |
| ton_S80 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4377/4462 | 0.0117 | 0.315 / 0.315 | 5528 | 215024 | 10405 | 0.05 | 4462/4462 | 0 | 0 | 0 | 0 |
| ton_S80_pt | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1440 | 291136 | 13389 | 0.01 | 4462/4462 | 0 | 0 | 0 | 0 |
| ton_S80_pt | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1440 | 291136 | 13389 | 0.05 | 4458/4462 | 0 | 0 | 4 | 0 |
| ton_S80_pt | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1944 | 215004 | 10295 | 0.01 | 4462/4462 | 0 | 0 | 0 | 0 |
| ton_S80_pt | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1944 | 215004 | 10295 | 0.05 | 4458/4462 | 0 | 0 | 4 | 0 |
| ton_S80_pt | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1944 | 215004 | 10283 | 0.01 | 4462/4462 | 0 | 0 | 0 | 0 |
| ton_S80_pt | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4099/4462 | 0.0156 | 0.315 / 0.315 | 1944 | 215004 | 10283 | 0.05 | 4458/4462 | 0 | 0 | 4 | 0 |
| ton_PQ80_int16 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 425/4462 | 0.9962 | 0.441 / 0.416 | 1824 | 291136 | 13866 | 0.01 | 3379/4462 | 933 | 0 | 12 | 4 |
| ton_PQ80_int16 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | 425/4462 | 0.9962 | 0.441 / 0.416 | 1824 | 291136 | 13866 | 0.05 | 4346/4462 | 23 | 0 | 26 | 0 |
| ton_PQ80_int16 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 0/4462 | 0.5545 | 0.441 / 0.000 | 5912 | 215004 | 12098 | 0.01 | 1894/4462 | 2278 | 0 | 23 | 0 |
| ton_PQ80_int16 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 0/4462 | 0.5545 | 0.441 / 0.000 | 5912 | 215004 | 12098 | 0.05 | 3149/4462 | 0 | 1211 | 0 | 37 |
| ton_PQ80_int16 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 0/4462 | 0.5545 | 0.441 / 0.000 | 5912 | 215004 | 12070 | 0.01 | 1894/4462 | 2278 | 0 | 23 | 0 |
| ton_PQ80_int16 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 0/4462 | 0.5545 | 0.441 / 0.000 | 5912 | 215004 | 12070 | 0.05 | 3149/4462 | 0 | 1211 | 0 | 37 |

## Device vs device (bit-identical outputs)

| model | runtime A | runtime B | identical outputs |
|---|---|---|---|
| ton_S80 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 160/4462 |
| ton_S80 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 160/4462 |
| ton_S80 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4462/4462 |
| ton_S80_pt | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 4462/4462 |
| ton_S80_pt | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4462/4462 |
| ton_S80_pt | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4462/4462 |
| ton_PQ80_int16 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | 0/4462 |
| ton_PQ80_int16 | TFLM 1.0.0 (Arduino TensorFlowLite_ESP32), reference kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 0/4462 |
| ton_PQ80_int16 | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), esp-nn kernels | esp-tflite-micro 1.4.2 (ESP-IDF 5.5), reference (ANSI C) kernels | 4462/4462 |
