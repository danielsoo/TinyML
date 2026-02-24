# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-14 11:59:58 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4168 | 0.4743 | 0.5318 | 0.5897 | 0.6482 | 0.7039 | 0.7615 | 0.8195 | 0.8775 | 0.9353 | 0.9921 |
| QAT+Prune only | 0.2260 | 0.3042 | 0.3810 | 0.4591 | 0.5366 | 0.6125 | 0.6911 | 0.7672 | 0.8430 | 0.9224 | 0.9985 |
| QAT+PTQ | 0.2253 | 0.3035 | 0.3804 | 0.4585 | 0.5362 | 0.6122 | 0.6909 | 0.7670 | 0.8428 | 0.9223 | 0.9985 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.2253 | 0.3035 | 0.3804 | 0.4585 | 0.5362 | 0.6122 | 0.6909 | 0.7670 | 0.8428 | 0.9223 | 0.9985 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2740 | 0.4587 | 0.5920 | 0.6929 | 0.7702 | 0.8331 | 0.8850 | 0.9283 | 0.9650 | 0.9960 |
| QAT+Prune only | 0.0000 | 0.2230 | 0.3922 | 0.5255 | 0.6329 | 0.7204 | 0.7950 | 0.8572 | 0.9105 | 0.9586 | 0.9993 |
| QAT+PTQ | 0.0000 | 0.2228 | 0.3920 | 0.5253 | 0.6327 | 0.7203 | 0.7949 | 0.8572 | 0.9104 | 0.9585 | 0.9992 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2228 | 0.3920 | 0.5253 | 0.6327 | 0.7203 | 0.7949 | 0.8572 | 0.9104 | 0.9585 | 0.9992 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4168 | 0.4168 | 0.4167 | 0.4172 | 0.4189 | 0.4158 | 0.4157 | 0.4169 | 0.4190 | 0.4246 | 0.0000 |
| QAT+Prune only | 0.2260 | 0.2270 | 0.2266 | 0.2279 | 0.2287 | 0.2266 | 0.2300 | 0.2275 | 0.2210 | 0.2373 | 0.0000 |
| QAT+PTQ | 0.2253 | 0.2263 | 0.2259 | 0.2271 | 0.2281 | 0.2260 | 0.2294 | 0.2270 | 0.2202 | 0.2362 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.2253 | 0.2263 | 0.2259 | 0.2271 | 0.2281 | 0.2260 | 0.2294 | 0.2270 | 0.2202 | 0.2362 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4168 | 0.0000 | 0.0000 | 0.0000 | 0.4168 | 1.0000 |
| 90 | 10 | 299,940 | 0.4743 | 0.1589 | 0.9918 | 0.2740 | 0.4168 | 0.9978 |
| 80 | 20 | 291,350 | 0.5318 | 0.2984 | 0.9921 | 0.4587 | 0.4167 | 0.9953 |
| 70 | 30 | 194,230 | 0.5897 | 0.4218 | 0.9921 | 0.5920 | 0.4172 | 0.9919 |
| 60 | 40 | 145,675 | 0.6482 | 0.5323 | 0.9921 | 0.6929 | 0.4189 | 0.9875 |
| 50 | 50 | 116,540 | 0.7039 | 0.6294 | 0.9921 | 0.7702 | 0.4158 | 0.9813 |
| 40 | 60 | 97,115 | 0.7615 | 0.7181 | 0.9921 | 0.8331 | 0.4157 | 0.9722 |
| 30 | 70 | 83,240 | 0.8195 | 0.7988 | 0.9921 | 0.8850 | 0.4169 | 0.9575 |
| 20 | 80 | 72,835 | 0.8775 | 0.8723 | 0.9921 | 0.9283 | 0.4190 | 0.9296 |
| 10 | 90 | 64,740 | 0.9353 | 0.9395 | 0.9921 | 0.9650 | 0.4246 | 0.8561 |
| 0 | 100 | 58,270 | 0.9921 | 1.0000 | 0.9921 | 0.9960 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2260 | 0.0000 | 0.0000 | 0.0000 | 0.2260 | 1.0000 |
| 90 | 10 | 299,940 | 0.3042 | 0.1255 | 0.9985 | 0.2230 | 0.2270 | 0.9992 |
| 80 | 20 | 291,350 | 0.3810 | 0.2440 | 0.9985 | 0.3922 | 0.2266 | 0.9984 |
| 70 | 30 | 194,230 | 0.4591 | 0.3566 | 0.9985 | 0.5255 | 0.2279 | 0.9972 |
| 60 | 40 | 145,675 | 0.5366 | 0.4633 | 0.9985 | 0.6329 | 0.2287 | 0.9957 |
| 50 | 50 | 116,540 | 0.6125 | 0.5635 | 0.9985 | 0.7204 | 0.2266 | 0.9935 |
| 40 | 60 | 97,115 | 0.6911 | 0.6605 | 0.9985 | 0.7950 | 0.2300 | 0.9904 |
| 30 | 70 | 83,240 | 0.7672 | 0.7510 | 0.9985 | 0.8572 | 0.2275 | 0.9849 |
| 20 | 80 | 72,835 | 0.8430 | 0.8368 | 0.9985 | 0.9105 | 0.2210 | 0.9737 |
| 10 | 90 | 64,740 | 0.9224 | 0.9218 | 0.9985 | 0.9586 | 0.2373 | 0.9464 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9993 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2253 | 0.0000 | 0.0000 | 0.0000 | 0.2253 | 1.0000 |
| 90 | 10 | 299,940 | 0.3035 | 0.1254 | 0.9985 | 0.2228 | 0.2263 | 0.9992 |
| 80 | 20 | 291,350 | 0.3804 | 0.2438 | 0.9985 | 0.3920 | 0.2259 | 0.9983 |
| 70 | 30 | 194,230 | 0.4585 | 0.3564 | 0.9985 | 0.5253 | 0.2271 | 0.9972 |
| 60 | 40 | 145,675 | 0.5362 | 0.4630 | 0.9985 | 0.6327 | 0.2281 | 0.9956 |
| 50 | 50 | 116,540 | 0.6122 | 0.5633 | 0.9985 | 0.7203 | 0.2260 | 0.9934 |
| 40 | 60 | 97,115 | 0.6909 | 0.6603 | 0.9985 | 0.7949 | 0.2294 | 0.9902 |
| 30 | 70 | 83,240 | 0.7670 | 0.7509 | 0.9985 | 0.8572 | 0.2270 | 0.9847 |
| 20 | 80 | 72,835 | 0.8428 | 0.8367 | 0.9985 | 0.9104 | 0.2202 | 0.9733 |
| 10 | 90 | 64,740 | 0.9223 | 0.9217 | 0.9985 | 0.9585 | 0.2362 | 0.9456 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9992 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.2253 | 0.0000 | 0.0000 | 0.0000 | 0.2253 | 1.0000 |
| 90 | 10 | 299,940 | 0.3035 | 0.1254 | 0.9985 | 0.2228 | 0.2263 | 0.9992 |
| 80 | 20 | 291,350 | 0.3804 | 0.2438 | 0.9985 | 0.3920 | 0.2259 | 0.9983 |
| 70 | 30 | 194,230 | 0.4585 | 0.3564 | 0.9985 | 0.5253 | 0.2271 | 0.9972 |
| 60 | 40 | 145,675 | 0.5362 | 0.4630 | 0.9985 | 0.6327 | 0.2281 | 0.9956 |
| 50 | 50 | 116,540 | 0.6122 | 0.5633 | 0.9985 | 0.7203 | 0.2260 | 0.9934 |
| 40 | 60 | 97,115 | 0.6909 | 0.6603 | 0.9985 | 0.7949 | 0.2294 | 0.9902 |
| 30 | 70 | 83,240 | 0.7670 | 0.7509 | 0.9985 | 0.8572 | 0.2270 | 0.9847 |
| 20 | 80 | 72,835 | 0.8428 | 0.8367 | 0.9985 | 0.9104 | 0.2202 | 0.9733 |
| 10 | 90 | 64,740 | 0.9223 | 0.9217 | 0.9985 | 0.9585 | 0.2362 | 0.9456 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9992 | 0.0000 | 0.0000 |


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
0.15       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591   <--
0.20       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.25       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.30       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.35       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.40       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.45       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.50       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.55       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.60       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.65       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.70       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.75       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
0.80       0.4744   0.2742   0.4168   0.9981   0.9927   0.1591  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4744, F1=0.2742, Normal Recall=0.4168, Normal Precision=0.9981, Attack Recall=0.9927, Attack Precision=0.1591

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
0.15       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984   <--
0.20       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.25       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.30       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.35       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.40       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.45       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.50       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.55       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.60       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.65       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.70       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.75       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
0.80       0.5319   0.4588   0.4169   0.9953   0.9921   0.2984  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5319, F1=0.4588, Normal Recall=0.4169, Normal Precision=0.9953, Attack Recall=0.9921, Attack Precision=0.2984

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
0.15       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219   <--
0.20       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.25       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.30       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.35       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.40       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.45       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.50       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.55       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.60       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.65       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.70       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.75       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
0.80       0.5898   0.5920   0.4174   0.9919   0.9921   0.4219  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5898, F1=0.5920, Normal Recall=0.4174, Normal Precision=0.9919, Attack Recall=0.9921, Attack Precision=0.4219

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
0.15       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319   <--
0.20       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.25       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.30       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.35       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.40       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.45       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.50       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.55       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.60       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.65       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.70       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.75       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
0.80       0.6476   0.6925   0.4180   0.9875   0.9921   0.5319  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6476, F1=0.6925, Normal Recall=0.4180, Normal Precision=0.9875, Attack Recall=0.9921, Attack Precision=0.5319

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
0.15       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298   <--
0.20       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.25       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.30       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.35       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.40       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.45       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.50       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.55       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.60       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.65       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.70       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.75       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
0.80       0.7045   0.7705   0.4169   0.9813   0.9921   0.6298  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7045, F1=0.7705, Normal Recall=0.4169, Normal Precision=0.9813, Attack Recall=0.9921, Attack Precision=0.6298

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
0.15       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255   <--
0.20       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.25       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.30       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.35       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.40       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.45       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.50       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.55       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.60       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.65       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.70       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.75       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
0.80       0.3042   0.2230   0.2270   0.9994   0.9987   0.1255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3042, F1=0.2230, Normal Recall=0.2270, Normal Precision=0.9994, Attack Recall=0.9987, Attack Precision=0.1255

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
0.15       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441   <--
0.20       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.25       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.30       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.35       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.40       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.45       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.50       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.55       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.60       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.65       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.70       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.75       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
0.80       0.3812   0.3923   0.2268   0.9984   0.9985   0.2441  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3812, F1=0.3923, Normal Recall=0.2268, Normal Precision=0.9984, Attack Recall=0.9985, Attack Precision=0.2441

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
0.15       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561   <--
0.20       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.25       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.30       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.35       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.40       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.45       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.50       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.55       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.60       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.65       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.70       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.75       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
0.80       0.4578   0.5249   0.2261   0.9972   0.9985   0.3561  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4578, F1=0.5249, Normal Recall=0.2261, Normal Precision=0.9972, Attack Recall=0.9985, Attack Precision=0.3561

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
0.15       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623   <--
0.20       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.25       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.30       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.35       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.40       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.45       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.50       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.55       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.60       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.65       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.70       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.75       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
0.80       0.5349   0.6320   0.2258   0.9956   0.9985   0.4623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5349, F1=0.6320, Normal Recall=0.2258, Normal Precision=0.9956, Attack Recall=0.9985, Attack Precision=0.4623

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
0.15       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632   <--
0.20       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.25       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.30       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.35       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.40       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.45       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.50       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.55       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.60       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.65       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.70       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.75       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
0.80       0.6120   0.7202   0.2255   0.9934   0.9985   0.5632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6120, F1=0.7202, Normal Recall=0.2255, Normal Precision=0.9934, Attack Recall=0.9985, Attack Precision=0.5632

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
0.15       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254   <--
0.20       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.25       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.30       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.35       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.40       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.45       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.50       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.55       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.60       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.65       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.70       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.75       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.80       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3035, F1=0.2229, Normal Recall=0.2263, Normal Precision=0.9994, Attack Recall=0.9987, Attack Precision=0.1254

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
0.15       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439   <--
0.20       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.25       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.30       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.35       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.40       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.45       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.50       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.55       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.60       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.65       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.70       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.75       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.80       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3806, F1=0.3920, Normal Recall=0.2261, Normal Precision=0.9983, Attack Recall=0.9985, Attack Precision=0.2439

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
0.15       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558   <--
0.20       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.25       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.30       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.35       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.40       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.45       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.50       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.55       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.60       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.65       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.70       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.75       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.80       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4573, F1=0.5247, Normal Recall=0.2254, Normal Precision=0.9971, Attack Recall=0.9985, Attack Precision=0.3558

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
0.15       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621   <--
0.20       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.25       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.30       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.35       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.40       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.45       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.50       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.55       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.60       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.65       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.70       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.75       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.80       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5345, F1=0.6318, Normal Recall=0.2251, Normal Precision=0.9955, Attack Recall=0.9985, Attack Precision=0.4621

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
0.15       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629   <--
0.20       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.25       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.30       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.35       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.40       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.45       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.50       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.55       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.60       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.65       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.70       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.75       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.80       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6116, F1=0.7199, Normal Recall=0.2246, Normal Precision=0.9933, Attack Recall=0.9985, Attack Precision=0.5629

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
0.15       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254   <--
0.20       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.25       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.30       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.35       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.40       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.45       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.50       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.55       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.60       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.65       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.70       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.75       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
0.80       0.3035   0.2229   0.2263   0.9994   0.9987   0.1254  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3035, F1=0.2229, Normal Recall=0.2263, Normal Precision=0.9994, Attack Recall=0.9987, Attack Precision=0.1254

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
0.15       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439   <--
0.20       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.25       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.30       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.35       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.40       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.45       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.50       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.55       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.60       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.65       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.70       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.75       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
0.80       0.3806   0.3920   0.2261   0.9983   0.9985   0.2439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3806, F1=0.3920, Normal Recall=0.2261, Normal Precision=0.9983, Attack Recall=0.9985, Attack Precision=0.2439

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
0.15       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558   <--
0.20       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.25       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.30       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.35       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.40       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.45       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.50       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.55       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.60       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.65       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.70       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.75       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
0.80       0.4573   0.5247   0.2254   0.9971   0.9985   0.3558  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4573, F1=0.5247, Normal Recall=0.2254, Normal Precision=0.9971, Attack Recall=0.9985, Attack Precision=0.3558

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
0.15       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621   <--
0.20       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.25       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.30       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.35       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.40       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.45       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.50       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.55       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.60       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.65       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.70       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.75       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
0.80       0.5345   0.6318   0.2251   0.9955   0.9985   0.4621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5345, F1=0.6318, Normal Recall=0.2251, Normal Precision=0.9955, Attack Recall=0.9985, Attack Precision=0.4621

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
0.15       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629   <--
0.20       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.25       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.30       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.35       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.40       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.45       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.50       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.55       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.60       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.65       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.70       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.75       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
0.80       0.6116   0.7199   0.2246   0.9933   0.9985   0.5629  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6116, F1=0.7199, Normal Recall=0.2246, Normal Precision=0.9933, Attack Recall=0.9985, Attack Precision=0.5629

```

