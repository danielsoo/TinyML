# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-15 04:22:55 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2190 | 0.2938 | 0.3697 | 0.4462 | 0.5227 | 0.5969 | 0.6756 | 0.7505 | 0.8266 | 0.9024 | 0.9785 |
| QAT+Prune only | 0.9399 | 0.9358 | 0.9310 | 0.9265 | 0.9221 | 0.9165 | 0.9124 | 0.9069 | 0.9021 | 0.8977 | 0.8930 |
| QAT+PTQ | 0.9396 | 0.9355 | 0.9307 | 0.9263 | 0.9219 | 0.9163 | 0.9124 | 0.9069 | 0.9022 | 0.8979 | 0.8933 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9396 | 0.9355 | 0.9307 | 0.9263 | 0.9219 | 0.9163 | 0.9124 | 0.9069 | 0.9022 | 0.8979 | 0.8933 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2170 | 0.3831 | 0.5146 | 0.6212 | 0.7083 | 0.7835 | 0.8460 | 0.9003 | 0.9475 | 0.9891 |
| QAT+Prune only | 0.0000 | 0.7356 | 0.8381 | 0.8794 | 0.9017 | 0.9145 | 0.9244 | 0.9307 | 0.9359 | 0.9402 | 0.9435 |
| QAT+PTQ | 0.0000 | 0.7347 | 0.8376 | 0.8791 | 0.9015 | 0.9143 | 0.9244 | 0.9307 | 0.9360 | 0.9403 | 0.9436 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7347 | 0.8376 | 0.8791 | 0.9015 | 0.9143 | 0.9244 | 0.9307 | 0.9360 | 0.9403 | 0.9436 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2190 | 0.2177 | 0.2175 | 0.2181 | 0.2188 | 0.2154 | 0.2213 | 0.2186 | 0.2191 | 0.2178 | 0.0000 |
| QAT+Prune only | 0.9399 | 0.9406 | 0.9405 | 0.9408 | 0.9414 | 0.9399 | 0.9414 | 0.9392 | 0.9384 | 0.9396 | 0.0000 |
| QAT+PTQ | 0.9396 | 0.9402 | 0.9401 | 0.9404 | 0.9411 | 0.9394 | 0.9410 | 0.9388 | 0.9379 | 0.9399 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9396 | 0.9402 | 0.9401 | 0.9404 | 0.9411 | 0.9394 | 0.9410 | 0.9388 | 0.9379 | 0.9399 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2190 | 0.0000 | 0.0000 | 0.0000 | 0.2190 | 1.0000 |
| 90 | 10 | 299,940 | 0.2938 | 0.1220 | 0.9787 | 0.2170 | 0.2177 | 0.9892 |
| 80 | 20 | 291,350 | 0.3697 | 0.2382 | 0.9785 | 0.3831 | 0.2175 | 0.9759 |
| 70 | 30 | 194,230 | 0.4462 | 0.3491 | 0.9785 | 0.5146 | 0.2181 | 0.9595 |
| 60 | 40 | 145,675 | 0.5227 | 0.4550 | 0.9785 | 0.6212 | 0.2188 | 0.9385 |
| 50 | 50 | 116,540 | 0.5969 | 0.5550 | 0.9785 | 0.7083 | 0.2154 | 0.9092 |
| 40 | 60 | 97,115 | 0.6756 | 0.6534 | 0.9785 | 0.7835 | 0.2213 | 0.8728 |
| 30 | 70 | 83,240 | 0.7505 | 0.7450 | 0.9785 | 0.8460 | 0.2186 | 0.8133 |
| 20 | 80 | 72,835 | 0.8266 | 0.8337 | 0.9785 | 0.9003 | 0.2191 | 0.7180 |
| 10 | 90 | 64,740 | 0.9024 | 0.9184 | 0.9785 | 0.9475 | 0.2178 | 0.5295 |
| 0 | 100 | 58,270 | 0.9785 | 1.0000 | 0.9785 | 0.9891 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9399 | 0.0000 | 0.0000 | 0.0000 | 0.9399 | 1.0000 |
| 90 | 10 | 299,940 | 0.9358 | 0.6255 | 0.8928 | 0.7356 | 0.9406 | 0.9875 |
| 80 | 20 | 291,350 | 0.9310 | 0.7895 | 0.8930 | 0.8381 | 0.9405 | 0.9724 |
| 70 | 30 | 194,230 | 0.9265 | 0.8661 | 0.8930 | 0.8794 | 0.9408 | 0.9535 |
| 60 | 40 | 145,675 | 0.9221 | 0.9105 | 0.8930 | 0.9017 | 0.9414 | 0.9296 |
| 50 | 50 | 116,540 | 0.9165 | 0.9369 | 0.8930 | 0.9145 | 0.9399 | 0.8978 |
| 40 | 60 | 97,115 | 0.9124 | 0.9581 | 0.8930 | 0.9244 | 0.9414 | 0.8544 |
| 30 | 70 | 83,240 | 0.9069 | 0.9717 | 0.8930 | 0.9307 | 0.9392 | 0.7901 |
| 20 | 80 | 72,835 | 0.9021 | 0.9830 | 0.8931 | 0.9359 | 0.9384 | 0.6869 |
| 10 | 90 | 64,740 | 0.8977 | 0.9925 | 0.8930 | 0.9402 | 0.9396 | 0.4940 |
| 0 | 100 | 58,270 | 0.8930 | 1.0000 | 0.8930 | 0.9435 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9396 | 0.0000 | 0.0000 | 0.0000 | 0.9396 | 1.0000 |
| 90 | 10 | 299,940 | 0.9355 | 0.6240 | 0.8930 | 0.7347 | 0.9402 | 0.9875 |
| 80 | 20 | 291,350 | 0.9307 | 0.7885 | 0.8933 | 0.8376 | 0.9401 | 0.9724 |
| 70 | 30 | 194,230 | 0.9263 | 0.8654 | 0.8933 | 0.8791 | 0.9404 | 0.9536 |
| 60 | 40 | 145,675 | 0.9219 | 0.9100 | 0.8933 | 0.9015 | 0.9411 | 0.9297 |
| 50 | 50 | 116,540 | 0.9163 | 0.9364 | 0.8933 | 0.9143 | 0.9394 | 0.8980 |
| 40 | 60 | 97,115 | 0.9124 | 0.9578 | 0.8933 | 0.9244 | 0.9410 | 0.8546 |
| 30 | 70 | 83,240 | 0.9069 | 0.9715 | 0.8933 | 0.9307 | 0.9388 | 0.7903 |
| 20 | 80 | 72,835 | 0.9022 | 0.9829 | 0.8933 | 0.9360 | 0.9379 | 0.6872 |
| 10 | 90 | 64,740 | 0.8979 | 0.9926 | 0.8932 | 0.9403 | 0.9399 | 0.4945 |
| 0 | 100 | 58,270 | 0.8933 | 1.0000 | 0.8933 | 0.9436 | 0.0000 | 0.0000 |

### noQAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| 90 | 10 | 299,940 | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.9000 |
| 80 | 20 | 291,350 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.8000 |
| 70 | 30 | 194,230 | 0.7000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7000 |
| 60 | 40 | 145,675 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6000 |
| 50 | 50 | 116,540 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5000 |
| 40 | 60 | 97,115 | 0.4000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4000 |
| 30 | 70 | 83,240 | 0.3000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.3000 |
| 20 | 80 | 72,835 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.2000 |
| 10 | 90 | 64,740 | 0.1000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1000 |
| 0 | 100 | 58,270 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Compressed (PTQ)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9396 | 0.0000 | 0.0000 | 0.0000 | 0.9396 | 1.0000 |
| 90 | 10 | 299,940 | 0.9355 | 0.6240 | 0.8930 | 0.7347 | 0.9402 | 0.9875 |
| 80 | 20 | 291,350 | 0.9307 | 0.7885 | 0.8933 | 0.8376 | 0.9401 | 0.9724 |
| 70 | 30 | 194,230 | 0.9263 | 0.8654 | 0.8933 | 0.8791 | 0.9404 | 0.9536 |
| 60 | 40 | 145,675 | 0.9219 | 0.9100 | 0.8933 | 0.9015 | 0.9411 | 0.9297 |
| 50 | 50 | 116,540 | 0.9163 | 0.9364 | 0.8933 | 0.9143 | 0.9394 | 0.8980 |
| 40 | 60 | 97,115 | 0.9124 | 0.9578 | 0.8933 | 0.9244 | 0.9410 | 0.8546 |
| 30 | 70 | 83,240 | 0.9069 | 0.9715 | 0.8933 | 0.9307 | 0.9388 | 0.7903 |
| 20 | 80 | 72,835 | 0.9022 | 0.9829 | 0.8933 | 0.9360 | 0.9379 | 0.6872 |
| 10 | 90 | 64,740 | 0.8979 | 0.9926 | 0.8932 | 0.9403 | 0.9399 | 0.4945 |
| 0 | 100 | 58,270 | 0.8933 | 1.0000 | 0.8933 | 0.9436 | 0.0000 | 0.0000 |


## Threshold Tuning (Original)

Model: `models/tflite/saved_model_original.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221   <--
0.20       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.25       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.30       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.35       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.40       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.45       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.50       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.55       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.60       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.65       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.70       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.75       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
0.80       0.2938   0.2171   0.2176   0.9895   0.9792   0.1221  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2938, F1=0.2171, Normal Recall=0.2176, Normal Precision=0.9895, Attack Recall=0.9792, Attack Precision=0.1221

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382   <--
0.20       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.25       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.30       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.35       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.40       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.45       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.50       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.55       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.60       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.65       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.70       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.75       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
0.80       0.3697   0.3831   0.2175   0.9759   0.9785   0.2382  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3697, F1=0.3831, Normal Recall=0.2175, Normal Precision=0.9759, Attack Recall=0.9785, Attack Precision=0.2382

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493   <--
0.20       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.25       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.30       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.35       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.40       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.45       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.50       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.55       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.60       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.65       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.70       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.75       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
0.80       0.4467   0.5148   0.2188   0.9596   0.9785   0.3493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4467, F1=0.5148, Normal Recall=0.2188, Normal Precision=0.9596, Attack Recall=0.9785, Attack Precision=0.3493

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552   <--
0.20       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.25       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.30       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.35       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.40       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.45       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.50       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.55       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.60       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.65       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.70       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.75       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
0.80       0.5229   0.6213   0.2191   0.9386   0.9785   0.4552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5229, F1=0.6213, Normal Recall=0.2191, Normal Precision=0.9386, Attack Recall=0.9785, Attack Precision=0.4552

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_original.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560   <--
0.20       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.25       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.30       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.35       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.40       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.45       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.50       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.55       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.60       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.65       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.70       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.75       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
0.80       0.5986   0.7091   0.2187   0.9105   0.9785   0.5560  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5986, F1=0.7091, Normal Recall=0.2187, Normal Precision=0.9105, Attack Recall=0.9785, Attack Precision=0.5560

```


## Threshold Tuning (QAT+Prune only)

Model: `models/tflite/saved_model_qat_pruned_float32.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259   <--
0.20       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.25       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.30       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.35       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.40       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.45       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.50       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.55       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.60       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.65       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.70       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.75       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
0.80       0.9360   0.7366   0.9406   0.9877   0.8947   0.6259  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9360, F1=0.7366, Normal Recall=0.9406, Normal Precision=0.9877, Attack Recall=0.8947, Attack Precision=0.6259

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898   <--
0.20       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.25       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.30       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.35       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.40       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.45       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.50       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.55       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.60       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.65       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.70       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.75       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
0.80       0.9311   0.8382   0.9406   0.9724   0.8930   0.7898  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9311, F1=0.8382, Normal Recall=0.9406, Normal Precision=0.9724, Attack Recall=0.8930, Attack Precision=0.7898

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645   <--
0.20       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.25       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.30       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.35       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.40       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.45       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.50       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.55       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.60       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.65       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.70       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.75       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
0.80       0.9259   0.8785   0.9400   0.9535   0.8930   0.8645  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9259, F1=0.8785, Normal Recall=0.9400, Normal Precision=0.9535, Attack Recall=0.8930, Attack Precision=0.8645

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093   <--
0.20       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.25       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.30       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.35       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.40       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.45       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.50       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.55       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.60       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.65       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.70       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.75       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
0.80       0.9216   0.9011   0.9406   0.9295   0.8930   0.9093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9216, F1=0.9011, Normal Recall=0.9406, Normal Precision=0.9295, Attack Recall=0.8930, Attack Precision=0.9093

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_pruned_float32.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376   <--
0.20       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.25       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.30       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.35       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.40       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.45       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.50       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.55       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.60       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.65       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.70       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.75       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
0.80       0.9168   0.9148   0.9405   0.8979   0.8930   0.9376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9168, F1=0.9148, Normal Recall=0.9405, Normal Precision=0.8979, Attack Recall=0.8930, Attack Precision=0.9376

```


## Threshold Tuning (QAT+PTQ)

Model: `models/tflite/saved_model_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245   <--
0.20       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.25       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.30       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.35       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.40       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.45       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.50       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.55       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.60       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.65       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.70       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.75       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.80       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9357, F1=0.7356, Normal Recall=0.9402, Normal Precision=0.9877, Attack Recall=0.8949, Attack Precision=0.6245

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888   <--
0.20       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.25       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.30       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.35       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.40       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.45       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.50       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.55       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.60       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.65       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.70       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.75       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.80       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9308, F1=0.8378, Normal Recall=0.9402, Normal Precision=0.9724, Attack Recall=0.8933, Attack Precision=0.7888

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638   <--
0.20       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.25       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.30       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.35       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.40       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.45       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.50       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.55       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.60       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.65       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.70       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.75       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.80       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9257, F1=0.8783, Normal Recall=0.9396, Normal Precision=0.9536, Attack Recall=0.8933, Attack Precision=0.8638

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089   <--
0.20       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.25       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.30       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.35       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.40       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.45       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.50       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.55       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.60       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.65       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.70       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.75       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.80       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9215, F1=0.9010, Normal Recall=0.9403, Normal Precision=0.9296, Attack Recall=0.8933, Attack Precision=0.9089

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372   <--
0.20       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.25       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.30       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.35       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.40       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.45       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.50       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.55       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.60       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.65       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.70       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.75       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.80       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9167, F1=0.9147, Normal Recall=0.9402, Normal Precision=0.8980, Attack Recall=0.8933, Attack Precision=0.9372

```


## Threshold Tuning (noQAT+PTQ)

Model: `models/tflite/saved_model_no_qat_ptq.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000   <--
0.20       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.25       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.30       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.35       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.40       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.45       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.50       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.55       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.60       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.65       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.70       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.75       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
0.80       0.9000   0.0000   1.0000   0.9000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.9000, Attack Recall=0.0000, Attack Precision=0.0000

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000   <--
0.20       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.25       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.30       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.35       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.40       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.45       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.50       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.55       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.60       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.65       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.70       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.75       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
0.80       0.8000   0.0000   1.0000   0.8000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.8000, Attack Recall=0.0000, Attack Precision=0.0000

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000   <--
0.20       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.25       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.30       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.35       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.40       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.45       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.50       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.55       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.60       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.65       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.70       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.75       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
0.80       0.7000   0.0000   1.0000   0.7000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.7000, Attack Recall=0.0000, Attack Precision=0.0000

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000   <--
0.20       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.25       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.30       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.35       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.40       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.45       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.50       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.55       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.60       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.65       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.70       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.75       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
0.80       0.6000   0.0000   1.0000   0.6000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.6000, Attack Recall=0.0000, Attack Precision=0.0000

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_no_qat_ptq.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000   <--
0.20       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.25       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.30       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.35       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.40       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.45       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.50       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.55       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.60       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.65       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.70       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.75       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
0.80       0.5000   0.0000   1.0000   0.5000   0.0000   0.0000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5000, F1=0.0000, Normal Recall=1.0000, Normal Precision=0.5000, Attack Recall=0.0000, Attack Precision=0.0000

```


## Threshold Tuning (Compressed (PTQ))

Model: `models/tflite/saved_model_pruned_quantized.tflite`

### Normal 90% : Attack 10%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 90% : Attack 10%  (n=299,940, N=269,946, A=29,994)
Getting predictions...

--- Threshold sweep (Normal 90% : Attack 10%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245   <--
0.20       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.25       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.30       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.35       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.40       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.45       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.50       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.55       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.60       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.65       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.70       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.75       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
0.80       0.9357   0.7356   0.9402   0.9877   0.8949   0.6245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9357, F1=0.7356, Normal Recall=0.9402, Normal Precision=0.9877, Attack Recall=0.8949, Attack Precision=0.6245

```

### Normal 80% : Attack 20%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 80% : Attack 20%  (n=291,350, N=233,080, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 80% : Attack 20%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888   <--
0.20       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.25       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.30       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.35       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.40       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.45       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.50       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.55       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.60       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.65       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.70       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.75       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
0.80       0.9308   0.8378   0.9402   0.9724   0.8933   0.7888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9308, F1=0.8378, Normal Recall=0.9402, Normal Precision=0.9724, Attack Recall=0.8933, Attack Precision=0.7888

```

### Normal 70% : Attack 30%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 70% : Attack 30%  (n=194,230, N=135,961, A=58,269)
Getting predictions...

--- Threshold sweep (Normal 70% : Attack 30%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638   <--
0.20       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.25       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.30       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.35       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.40       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.45       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.50       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.55       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.60       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.65       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.70       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.75       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
0.80       0.9257   0.8783   0.9396   0.9536   0.8933   0.8638  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9257, F1=0.8783, Normal Recall=0.9396, Normal Precision=0.9536, Attack Recall=0.8933, Attack Precision=0.8638

```

### Normal 60% : Attack 40%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 60% : Attack 40%  (n=145,675, N=87,405, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 60% : Attack 40%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089   <--
0.20       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.25       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.30       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.35       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.40       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.45       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.50       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.55       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.60       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.65       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.70       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.75       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
0.80       0.9215   0.9010   0.9403   0.9296   0.8933   0.9089  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9215, F1=0.9010, Normal Recall=0.9403, Normal Precision=0.9296, Attack Recall=0.8933, Attack Precision=0.9089

```

### Normal 50% : Attack 50%

```
Loading dataset...
[load_cicids2017] data_path=data/raw/CIC-IDS2017
[load_cicids2017] Found 8 CSV files
[load_cicids2017] Target 250000 samples/file (random from full file)
[load_cicids2017] Loaded 225,745 rows from Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
[load_cicids2017] Loaded 191,033 rows from Friday-WorkingHours-Morning.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Monday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
[load_cicids2017] Loaded 170,366 rows from Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Tuesday-WorkingHours.pcap_ISCX.csv
[load_cicids2017] Loaded 250,000 rows from Wednesday-workingHours.pcap_ISCX.csv
[load_cicids2017] Total samples loaded: 1,837,144
[load_cicids2017] Removed 196,026 duplicates (1,641,118 unique)
[load_cicids2017] Found 15 unique labels
[load_cicids2017] Binary mode: BENIGN=0, ATTACK=1
[load_cicids2017] Label distribution: BENIGN=1349766, ATTACK=291352
[load_cicids2017] Shuffled full pool (random_state=42)
[load_cicids2017] Balanced training set: majority 1079812 -> 932328 (ratio<=4.0), total=1,165,410
[load_cicids2017] StandardScaler applied (fit on train).
[load_cicids2017] SMOTE applied: train -> 1,864,656 (BENIGN=932,328, ATTACK=932,328)
[load_cicids2017] Final feature shape: 78 features
[load_cicids2017] Train samples: 1,864,656, Test samples: 328,224
  Test: 328,224 (Normal=269,954, Attack=58,270)
Loading model: models/tflite/saved_model_pruned_quantized.tflite

Test set: Normal 50% : Attack 50%  (n=116,540, N=58,270, A=58,270)
Getting predictions...

--- Threshold sweep (Normal 50% : Attack 50%) ---
Threshold  Accuracy F1       NormRec  NormPrec AttackRec AttackPrec
--------------------------------------------------------------------------
0.15       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372   <--
0.20       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.25       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.30       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.35       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.40       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.45       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.50       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.55       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.60       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.65       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.70       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.75       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
0.80       0.9167   0.9147   0.9402   0.8980   0.8933   0.9372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9167, F1=0.9147, Normal Recall=0.9402, Normal Precision=0.8980, Attack Recall=0.8933, Attack Precision=0.9372

```

