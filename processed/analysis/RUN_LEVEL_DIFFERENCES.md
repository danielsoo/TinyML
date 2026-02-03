# Run-Level Differences (per-run comparison by timestamp)

Detailed comparison of multiple runs (timestamps) within each version. For reference in papers/reports.

## v11 (runs: 5)

| Run | Data | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig (MB) | Comp (MB) | Params | Estimated Changes |
|-----|------|----------|----------|----------|----------|-----------|-----------|--------|-------------------|
| 2026-02-02_16-13-48 | cicids2017_max1500k_bal1. | 0.8242 | 0.00/0.00 | 0.8386 | 1.00/0.08 | 0.784 | 0.073 | 205,777 | Orig TFLite NaN/collapse; large 78feat; Comp~73KB(Full INT8) |
| 2026-02-02_18-17-38 | cicids2017_max1500k_bal1. | 0.9490 | 0.80/0.95 | 0.8242 | 0.00/0.00 | 0.784 | 0.073 | 205,777 | Comp Full INT8 collapse; large 78feat; Comp~73KB(Full INT8) |
| 2026-02-02_19-11-40 | cicids2017_max1500k_bal1. | 0.9490 | 0.80/0.95 | 0.8242 | 0.00/0.00 | 0.784 | 0.073 | 205,777 | Comp Full INT8 collapse; large 78feat; Comp~73KB(Full INT8) |
| 2026-02-02_19-16-09 | cicids2017_max1500k_bal1. | 0.9490 | 0.80/0.95 | 0.8242 | 0.00/0.00 | 0.784 | 0.073 | 205,777 | Comp Full INT8 collapse; large 78feat; Comp~73KB(Full INT8) |
| 2026-02-02_23-28-45 | cicids2017_max1500k_bal1. | 0.9373 | 0.74/0.98 | 0.9020 | 0.95/0.47 | 0.784 | 0.068 | 205,777 | DRQ applied, both OK; large 78feat; Comp~73KB(Full INT8) |

## v8 (runs: 2)

| Run | Data | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig (MB) | Comp (MB) | Params | Estimated Changes |
|-----|------|----------|----------|----------|----------|-----------|-----------|--------|-------------------|
| 2026-02-01_20-53-20 | cicids2017_max1500k | 0.3468 | 0.22/0.83 | 0.2162 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | DRQ applied, both OK; small 38feat; Comp~21KB |
| 2026-02-02_01-38-52 | cicids2017_max1500k | 0.7853 | 0.00/0.00 | 0.2147 | 0.21/1.00 | 0.784 | 0.073 | 205,777 | Orig TFLite NaN/collapse; large 78feat; Comp~73KB(Full INT8) |

## v8_centralized (runs: 2)

| Run | Data | Orig Acc | Orig P/R | Comp Acc | Comp P/R | Orig (MB) | Comp (MB) | Params | Estimated Changes |
|-----|------|----------|----------|----------|----------|-----------|-----------|--------|-------------------|
| 2026-02-01_22-48-06 | cicids2017_max1500k | 0.3468 | 0.22/0.83 | 0.2162 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | DRQ applied, both OK; small 38feat; Comp~21KB |
| 2026-02-02_00-48-48 | cicids2017_max1500k | 0.3468 | 0.22/0.83 | 0.2162 | 0.21/1.00 | 0.206 | 0.021 | 53,844 | DRQ applied, both OK; small 38feat; Comp~21KB |

---

### Notes on Estimated Changes

- **Original TFLite NaN/prediction collapse**: BN strip not applied; TFLite conversion yields NaN or single-class predictions
- **Compressed Full INT8 collapse**: Full INT8 (int8 in/out) yields P/R/F1=0
- **BN strip applied but PTQ not applied**: Original OK, Compressed still Full INT8
- **DRQ applied**: Dynamic Range Quantization (int8 weights, float32 I/O), both OK
- **Small/large model**: 38 feat vs 78 feat, Bot-IoT vs CIC-IDS2017
