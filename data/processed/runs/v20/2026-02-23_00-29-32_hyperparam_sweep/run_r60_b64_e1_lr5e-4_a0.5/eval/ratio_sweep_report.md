# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-23 05:08:30 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6139 | 0.6472 | 0.6798 | 0.7132 | 0.7452 | 0.7776 | 0.8125 | 0.8457 | 0.8772 | 0.9102 | 0.9441 |
| QAT+Prune only | 0.6233 | 0.6610 | 0.6980 | 0.7368 | 0.7735 | 0.8113 | 0.8493 | 0.8877 | 0.9246 | 0.9622 | 0.9998 |
| QAT+PTQ | 0.6202 | 0.6579 | 0.6952 | 0.7344 | 0.7715 | 0.8096 | 0.8482 | 0.8867 | 0.9239 | 0.9618 | 0.9998 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6202 | 0.6579 | 0.6952 | 0.7344 | 0.7715 | 0.8096 | 0.8482 | 0.8867 | 0.9239 | 0.9618 | 0.9998 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3489 | 0.5412 | 0.6639 | 0.7478 | 0.8094 | 0.8580 | 0.8955 | 0.9248 | 0.9498 | 0.9713 |
| QAT+Prune only | 0.0000 | 0.3710 | 0.5698 | 0.6951 | 0.7793 | 0.8412 | 0.8884 | 0.9257 | 0.9550 | 0.9794 | 0.9999 |
| QAT+PTQ | 0.0000 | 0.3689 | 0.5675 | 0.6931 | 0.7778 | 0.8400 | 0.8877 | 0.9251 | 0.9546 | 0.9792 | 0.9999 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3689 | 0.5675 | 0.6931 | 0.7778 | 0.8400 | 0.8877 | 0.9251 | 0.9546 | 0.9792 | 0.9999 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6139 | 0.6141 | 0.6137 | 0.6142 | 0.6126 | 0.6111 | 0.6150 | 0.6162 | 0.6095 | 0.6052 | 0.0000 |
| QAT+Prune only | 0.6233 | 0.6234 | 0.6226 | 0.6241 | 0.6226 | 0.6229 | 0.6235 | 0.6262 | 0.6237 | 0.6237 | 0.0000 |
| QAT+PTQ | 0.6202 | 0.6199 | 0.6190 | 0.6207 | 0.6192 | 0.6194 | 0.6207 | 0.6227 | 0.6202 | 0.6200 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6202 | 0.6199 | 0.6190 | 0.6207 | 0.6192 | 0.6194 | 0.6207 | 0.6227 | 0.6202 | 0.6200 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6139 | 0.0000 | 0.0000 | 0.0000 | 0.6139 | 1.0000 |
| 90 | 10 | 299,940 | 0.6472 | 0.2140 | 0.9453 | 0.3489 | 0.6141 | 0.9902 |
| 80 | 20 | 291,350 | 0.6798 | 0.3793 | 0.9441 | 0.5412 | 0.6137 | 0.9778 |
| 70 | 30 | 194,230 | 0.7132 | 0.5119 | 0.9441 | 0.6639 | 0.6142 | 0.9625 |
| 60 | 40 | 145,675 | 0.7452 | 0.6190 | 0.9441 | 0.7478 | 0.6126 | 0.9427 |
| 50 | 50 | 116,540 | 0.7776 | 0.7083 | 0.9441 | 0.8094 | 0.6111 | 0.9162 |
| 40 | 60 | 97,115 | 0.8125 | 0.7863 | 0.9441 | 0.8580 | 0.6150 | 0.8801 |
| 30 | 70 | 83,240 | 0.8457 | 0.8516 | 0.9441 | 0.8955 | 0.6162 | 0.8254 |
| 20 | 80 | 72,835 | 0.8772 | 0.9063 | 0.9441 | 0.9248 | 0.6095 | 0.7317 |
| 10 | 90 | 64,740 | 0.9102 | 0.9556 | 0.9441 | 0.9498 | 0.6052 | 0.5462 |
| 0 | 100 | 58,270 | 0.9441 | 1.0000 | 0.9441 | 0.9713 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6233 | 0.0000 | 0.0000 | 0.0000 | 0.6233 | 1.0000 |
| 90 | 10 | 299,940 | 0.6610 | 0.2278 | 0.9998 | 0.3710 | 0.6234 | 1.0000 |
| 80 | 20 | 291,350 | 0.6980 | 0.3984 | 0.9998 | 0.5698 | 0.6226 | 0.9999 |
| 70 | 30 | 194,230 | 0.7368 | 0.5327 | 0.9998 | 0.6951 | 0.6241 | 0.9999 |
| 60 | 40 | 145,675 | 0.7735 | 0.6385 | 0.9998 | 0.7793 | 0.6226 | 0.9998 |
| 50 | 50 | 116,540 | 0.8113 | 0.7261 | 0.9998 | 0.8412 | 0.6229 | 0.9997 |
| 40 | 60 | 97,115 | 0.8493 | 0.7993 | 0.9998 | 0.8884 | 0.6235 | 0.9995 |
| 30 | 70 | 83,240 | 0.8877 | 0.8619 | 0.9998 | 0.9257 | 0.6262 | 0.9992 |
| 20 | 80 | 72,835 | 0.9246 | 0.9140 | 0.9998 | 0.9550 | 0.6237 | 0.9987 |
| 10 | 90 | 64,740 | 0.9622 | 0.9599 | 0.9998 | 0.9794 | 0.6237 | 0.9970 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6202 | 0.0000 | 0.0000 | 0.0000 | 0.6202 | 1.0000 |
| 90 | 10 | 299,940 | 0.6579 | 0.2261 | 0.9998 | 0.3689 | 0.6199 | 1.0000 |
| 80 | 20 | 291,350 | 0.6952 | 0.3962 | 0.9998 | 0.5675 | 0.6190 | 0.9999 |
| 70 | 30 | 194,230 | 0.7344 | 0.5304 | 0.9998 | 0.6931 | 0.6207 | 0.9999 |
| 60 | 40 | 145,675 | 0.7715 | 0.6364 | 0.9998 | 0.7778 | 0.6192 | 0.9998 |
| 50 | 50 | 116,540 | 0.8096 | 0.7243 | 0.9998 | 0.8400 | 0.6194 | 0.9997 |
| 40 | 60 | 97,115 | 0.8482 | 0.7982 | 0.9998 | 0.8877 | 0.6207 | 0.9995 |
| 30 | 70 | 83,240 | 0.8867 | 0.8608 | 0.9998 | 0.9251 | 0.6227 | 0.9992 |
| 20 | 80 | 72,835 | 0.9239 | 0.9133 | 0.9998 | 0.9546 | 0.6202 | 0.9987 |
| 10 | 90 | 64,740 | 0.9618 | 0.9595 | 0.9998 | 0.9792 | 0.6200 | 0.9970 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6202 | 0.0000 | 0.0000 | 0.0000 | 0.6202 | 1.0000 |
| 90 | 10 | 299,940 | 0.6579 | 0.2261 | 0.9998 | 0.3689 | 0.6199 | 1.0000 |
| 80 | 20 | 291,350 | 0.6952 | 0.3962 | 0.9998 | 0.5675 | 0.6190 | 0.9999 |
| 70 | 30 | 194,230 | 0.7344 | 0.5304 | 0.9998 | 0.6931 | 0.6207 | 0.9999 |
| 60 | 40 | 145,675 | 0.7715 | 0.6364 | 0.9998 | 0.7778 | 0.6192 | 0.9998 |
| 50 | 50 | 116,540 | 0.8096 | 0.7243 | 0.9998 | 0.8400 | 0.6194 | 0.9997 |
| 40 | 60 | 97,115 | 0.8482 | 0.7982 | 0.9998 | 0.8877 | 0.6207 | 0.9995 |
| 30 | 70 | 83,240 | 0.8867 | 0.8608 | 0.9998 | 0.9251 | 0.6227 | 0.9992 |
| 20 | 80 | 72,835 | 0.9239 | 0.9133 | 0.9998 | 0.9546 | 0.6202 | 0.9987 |
| 10 | 90 | 64,740 | 0.9618 | 0.9595 | 0.9998 | 0.9792 | 0.6200 | 0.9970 |
| 0 | 100 | 58,270 | 0.9998 | 1.0000 | 0.9998 | 0.9999 | 0.0000 | 0.0000 |


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
0.15       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138   <--
0.20       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.25       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.30       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.35       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.40       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.45       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.50       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.55       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.60       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.65       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.70       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.75       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
0.80       0.6472   0.3487   0.6141   0.9900   0.9444   0.2138  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6472, F1=0.3487, Normal Recall=0.6141, Normal Precision=0.9900, Attack Recall=0.9444, Attack Precision=0.2138

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
0.15       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797   <--
0.20       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.25       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.30       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.35       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.40       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.45       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.50       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.55       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.60       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.65       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.70       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.75       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
0.80       0.6803   0.5416   0.6144   0.9778   0.9441   0.3797  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6803, F1=0.5416, Normal Recall=0.6144, Normal Precision=0.9778, Attack Recall=0.9441, Attack Precision=0.3797

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
0.15       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118   <--
0.20       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.25       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.30       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.35       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.40       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.45       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.50       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.55       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.60       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.65       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.70       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.75       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
0.80       0.7130   0.6637   0.6140   0.9625   0.9441   0.5118  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7130, F1=0.6637, Normal Recall=0.6140, Normal Precision=0.9625, Attack Recall=0.9441, Attack Precision=0.5118

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
0.15       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200   <--
0.20       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.25       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.30       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.35       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.40       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.45       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.50       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.55       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.60       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.65       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.70       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.75       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
0.80       0.7461   0.7485   0.6142   0.9428   0.9441   0.6200  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7461, F1=0.7485, Normal Recall=0.6142, Normal Precision=0.9428, Attack Recall=0.9441, Attack Precision=0.6200

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
0.15       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099   <--
0.20       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.25       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.30       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.35       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.40       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.45       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.50       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.55       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.60       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.65       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.70       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.75       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
0.80       0.7792   0.8105   0.6142   0.9166   0.9441   0.7099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7792, F1=0.8105, Normal Recall=0.6142, Normal Precision=0.9166, Attack Recall=0.9441, Attack Precision=0.7099

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
0.15       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278   <--
0.20       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.25       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.30       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.35       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.40       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.45       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.50       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.55       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.60       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.65       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.70       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.75       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
0.80       0.6610   0.3710   0.6234   1.0000   0.9998   0.2278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6610, F1=0.3710, Normal Recall=0.6234, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2278

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
0.15       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992   <--
0.20       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.25       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.30       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.35       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.40       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.45       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.50       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.55       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.60       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.65       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.70       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.75       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
0.80       0.6990   0.5706   0.6238   0.9999   0.9998   0.3992  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6990, F1=0.5706, Normal Recall=0.6238, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3992

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
0.15       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321   <--
0.20       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.25       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.30       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.35       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.40       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.45       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.50       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.55       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.60       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.65       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.70       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.75       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
0.80       0.7362   0.6945   0.6232   0.9999   0.9998   0.5321  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7362, F1=0.6945, Normal Recall=0.6232, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.5321

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
0.15       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388   <--
0.20       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.25       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.30       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.35       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.40       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.45       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.50       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.55       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.60       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.65       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.70       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.75       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
0.80       0.7738   0.7795   0.6231   0.9998   0.9998   0.6388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7738, F1=0.7795, Normal Recall=0.6231, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.6388

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
0.15       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261   <--
0.20       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.25       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.30       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.35       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.40       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.45       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.50       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.55       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.60       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.65       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.70       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.75       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
0.80       0.8113   0.8412   0.6229   0.9997   0.9998   0.7261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8113, F1=0.8412, Normal Recall=0.6229, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.7261

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
0.15       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262   <--
0.20       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.25       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.30       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.35       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.40       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.45       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.50       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.55       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.60       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.65       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.70       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.75       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.80       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6579, F1=0.3689, Normal Recall=0.6199, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2262

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
0.15       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970   <--
0.20       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.25       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.30       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.35       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.40       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.45       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.50       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.55       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.60       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.65       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.70       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.75       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.80       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6963, F1=0.5684, Normal Recall=0.6204, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3970

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
0.15       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299   <--
0.20       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.25       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.30       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.35       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.40       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.45       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.50       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.55       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.60       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.65       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.70       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.75       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.80       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7339, F1=0.6927, Normal Recall=0.6199, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.5299

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
0.15       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369   <--
0.20       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.25       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.30       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.35       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.40       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.45       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.50       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.55       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.60       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.65       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.70       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.75       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.80       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7720, F1=0.7781, Normal Recall=0.6201, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.6369

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
0.15       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245   <--
0.20       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.25       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.30       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.35       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.40       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.45       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.50       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.55       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.60       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.65       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.70       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.75       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.80       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8098, F1=0.8402, Normal Recall=0.6199, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.7245

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
0.15       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262   <--
0.20       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.25       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.30       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.35       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.40       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.45       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.50       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.55       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.60       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.65       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.70       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.75       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
0.80       0.6579   0.3689   0.6199   1.0000   0.9998   0.2262  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6579, F1=0.3689, Normal Recall=0.6199, Normal Precision=1.0000, Attack Recall=0.9998, Attack Precision=0.2262

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
0.15       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970   <--
0.20       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.25       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.30       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.35       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.40       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.45       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.50       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.55       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.60       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.65       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.70       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.75       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
0.80       0.6963   0.5684   0.6204   0.9999   0.9998   0.3970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6963, F1=0.5684, Normal Recall=0.6204, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.3970

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
0.15       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299   <--
0.20       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.25       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.30       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.35       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.40       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.45       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.50       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.55       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.60       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.65       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.70       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.75       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
0.80       0.7339   0.6927   0.6199   0.9999   0.9998   0.5299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7339, F1=0.6927, Normal Recall=0.6199, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.5299

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
0.15       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369   <--
0.20       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.25       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.30       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.35       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.40       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.45       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.50       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.55       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.60       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.65       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.70       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.75       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
0.80       0.7720   0.7781   0.6201   0.9998   0.9998   0.6369  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7720, F1=0.7781, Normal Recall=0.6201, Normal Precision=0.9998, Attack Recall=0.9998, Attack Precision=0.6369

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
0.15       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245   <--
0.20       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.25       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.30       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.35       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.40       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.45       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.50       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.55       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.60       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.65       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.70       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.75       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
0.80       0.8098   0.8402   0.6199   0.9997   0.9998   0.7245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8098, F1=0.8402, Normal Recall=0.6199, Normal Precision=0.9997, Attack Recall=0.9998, Attack Precision=0.7245

```

