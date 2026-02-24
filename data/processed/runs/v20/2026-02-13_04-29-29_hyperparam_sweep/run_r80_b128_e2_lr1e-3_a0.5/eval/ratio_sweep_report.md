# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-18 02:43:53 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5241 | 0.5547 | 0.5851 | 0.6163 | 0.6465 | 0.6752 | 0.7066 | 0.7372 | 0.7666 | 0.7976 | 0.8274 |
| QAT+Prune only | 0.8230 | 0.8407 | 0.8579 | 0.8756 | 0.8923 | 0.9098 | 0.9271 | 0.9439 | 0.9618 | 0.9791 | 0.9963 |
| QAT+PTQ | 0.8236 | 0.8414 | 0.8584 | 0.8762 | 0.8926 | 0.9100 | 0.9273 | 0.9440 | 0.9617 | 0.9789 | 0.9959 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8236 | 0.8414 | 0.8584 | 0.8762 | 0.8926 | 0.9100 | 0.9273 | 0.9440 | 0.9617 | 0.9789 | 0.9959 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2706 | 0.4437 | 0.5640 | 0.6519 | 0.7181 | 0.7719 | 0.8151 | 0.8501 | 0.8803 | 0.9055 |
| QAT+Prune only | 0.0000 | 0.5558 | 0.7371 | 0.8277 | 0.8810 | 0.9170 | 0.9425 | 0.9613 | 0.9766 | 0.9885 | 0.9982 |
| QAT+PTQ | 0.0000 | 0.5567 | 0.7378 | 0.8283 | 0.8812 | 0.9171 | 0.9427 | 0.9614 | 0.9765 | 0.9883 | 0.9979 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5567 | 0.7378 | 0.8283 | 0.8812 | 0.9171 | 0.9427 | 0.9614 | 0.9765 | 0.9883 | 0.9979 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5241 | 0.5246 | 0.5245 | 0.5258 | 0.5259 | 0.5231 | 0.5254 | 0.5268 | 0.5236 | 0.5297 | 0.0000 |
| QAT+Prune only | 0.8230 | 0.8234 | 0.8233 | 0.8239 | 0.8230 | 0.8233 | 0.8233 | 0.8215 | 0.8238 | 0.8242 | 0.0000 |
| QAT+PTQ | 0.8236 | 0.8242 | 0.8240 | 0.8248 | 0.8237 | 0.8241 | 0.8244 | 0.8228 | 0.8247 | 0.8256 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8236 | 0.8242 | 0.8240 | 0.8248 | 0.8237 | 0.8241 | 0.8244 | 0.8228 | 0.8247 | 0.8256 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5241 | 0.0000 | 0.0000 | 0.0000 | 0.5241 | 1.0000 |
| 90 | 10 | 299,940 | 0.5547 | 0.1618 | 0.8262 | 0.2706 | 0.5246 | 0.9645 |
| 80 | 20 | 291,350 | 0.5851 | 0.3031 | 0.8274 | 0.4437 | 0.5245 | 0.9240 |
| 70 | 30 | 194,230 | 0.6163 | 0.4278 | 0.8274 | 0.5640 | 0.5258 | 0.8766 |
| 60 | 40 | 145,675 | 0.6465 | 0.5378 | 0.8274 | 0.6519 | 0.5259 | 0.8205 |
| 50 | 50 | 116,540 | 0.6752 | 0.6344 | 0.8274 | 0.7181 | 0.5231 | 0.7519 |
| 40 | 60 | 97,115 | 0.7066 | 0.7234 | 0.8274 | 0.7719 | 0.5254 | 0.6699 |
| 30 | 70 | 83,240 | 0.7372 | 0.8031 | 0.8274 | 0.8151 | 0.5268 | 0.5667 |
| 20 | 80 | 72,835 | 0.7666 | 0.8742 | 0.8274 | 0.8501 | 0.5236 | 0.4313 |
| 10 | 90 | 64,740 | 0.7976 | 0.9406 | 0.8274 | 0.8803 | 0.5297 | 0.2542 |
| 0 | 100 | 58,270 | 0.8274 | 1.0000 | 0.8274 | 0.9055 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8230 | 0.0000 | 0.0000 | 0.0000 | 0.8230 | 1.0000 |
| 90 | 10 | 299,940 | 0.8407 | 0.3854 | 0.9964 | 0.5558 | 0.8234 | 0.9995 |
| 80 | 20 | 291,350 | 0.8579 | 0.5850 | 0.9963 | 0.7371 | 0.8233 | 0.9989 |
| 70 | 30 | 194,230 | 0.8756 | 0.7080 | 0.9963 | 0.8277 | 0.8239 | 0.9981 |
| 60 | 40 | 145,675 | 0.8923 | 0.7896 | 0.9963 | 0.8810 | 0.8230 | 0.9970 |
| 50 | 50 | 116,540 | 0.9098 | 0.8494 | 0.9963 | 0.9170 | 0.8233 | 0.9956 |
| 40 | 60 | 97,115 | 0.9271 | 0.8943 | 0.9963 | 0.9425 | 0.8233 | 0.9934 |
| 30 | 70 | 83,240 | 0.9439 | 0.9287 | 0.9963 | 0.9613 | 0.8215 | 0.9897 |
| 20 | 80 | 72,835 | 0.9618 | 0.9577 | 0.9963 | 0.9766 | 0.8238 | 0.9825 |
| 10 | 90 | 64,740 | 0.9791 | 0.9808 | 0.9963 | 0.9885 | 0.8242 | 0.9614 |
| 0 | 100 | 58,270 | 0.9963 | 1.0000 | 0.9963 | 0.9982 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8236 | 0.0000 | 0.0000 | 0.0000 | 0.8236 | 1.0000 |
| 90 | 10 | 299,940 | 0.8414 | 0.3864 | 0.9959 | 0.5567 | 0.8242 | 0.9995 |
| 80 | 20 | 291,350 | 0.8584 | 0.5859 | 0.9959 | 0.7378 | 0.8240 | 0.9988 |
| 70 | 30 | 194,230 | 0.8762 | 0.7090 | 0.9959 | 0.8283 | 0.8248 | 0.9979 |
| 60 | 40 | 145,675 | 0.8926 | 0.7901 | 0.9959 | 0.8812 | 0.8237 | 0.9967 |
| 50 | 50 | 116,540 | 0.9100 | 0.8499 | 0.9959 | 0.9171 | 0.8241 | 0.9950 |
| 40 | 60 | 97,115 | 0.9273 | 0.8948 | 0.9959 | 0.9427 | 0.8244 | 0.9926 |
| 30 | 70 | 83,240 | 0.9440 | 0.9292 | 0.9959 | 0.9614 | 0.8228 | 0.9885 |
| 20 | 80 | 72,835 | 0.9617 | 0.9578 | 0.9959 | 0.9765 | 0.8247 | 0.9805 |
| 10 | 90 | 64,740 | 0.9789 | 0.9809 | 0.9959 | 0.9883 | 0.8256 | 0.9572 |
| 0 | 100 | 58,270 | 0.9959 | 1.0000 | 0.9959 | 0.9979 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8236 | 0.0000 | 0.0000 | 0.0000 | 0.8236 | 1.0000 |
| 90 | 10 | 299,940 | 0.8414 | 0.3864 | 0.9959 | 0.5567 | 0.8242 | 0.9995 |
| 80 | 20 | 291,350 | 0.8584 | 0.5859 | 0.9959 | 0.7378 | 0.8240 | 0.9988 |
| 70 | 30 | 194,230 | 0.8762 | 0.7090 | 0.9959 | 0.8283 | 0.8248 | 0.9979 |
| 60 | 40 | 145,675 | 0.8926 | 0.7901 | 0.9959 | 0.8812 | 0.8237 | 0.9967 |
| 50 | 50 | 116,540 | 0.9100 | 0.8499 | 0.9959 | 0.9171 | 0.8241 | 0.9950 |
| 40 | 60 | 97,115 | 0.9273 | 0.8948 | 0.9959 | 0.9427 | 0.8244 | 0.9926 |
| 30 | 70 | 83,240 | 0.9440 | 0.9292 | 0.9959 | 0.9614 | 0.8228 | 0.9885 |
| 20 | 80 | 72,835 | 0.9617 | 0.9578 | 0.9959 | 0.9765 | 0.8247 | 0.9805 |
| 10 | 90 | 64,740 | 0.9789 | 0.9809 | 0.9959 | 0.9883 | 0.8256 | 0.9572 |
| 0 | 100 | 58,270 | 0.9959 | 1.0000 | 0.9959 | 0.9979 | 0.0000 | 0.0000 |


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
0.15       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619   <--
0.20       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.25       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.30       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.35       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.40       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.45       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.50       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.55       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.60       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.65       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.70       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.75       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
0.80       0.5548   0.2708   0.5245   0.9646   0.8267   0.1619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5548, F1=0.2708, Normal Recall=0.5245, Normal Precision=0.9646, Attack Recall=0.8267, Attack Precision=0.1619

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
0.15       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029   <--
0.20       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.25       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.30       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.35       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.40       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.45       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.50       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.55       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.60       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.65       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.70       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.75       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
0.80       0.5847   0.4435   0.5240   0.9239   0.8274   0.3029  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5847, F1=0.4435, Normal Recall=0.5240, Normal Precision=0.9239, Attack Recall=0.8274, Attack Precision=0.3029

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
0.15       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274   <--
0.20       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.25       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.30       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.35       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.40       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.45       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.50       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.55       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.60       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.65       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.70       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.75       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
0.80       0.6157   0.5636   0.5250   0.8765   0.8274   0.4274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6157, F1=0.5636, Normal Recall=0.5250, Normal Precision=0.8765, Attack Recall=0.8274, Attack Precision=0.4274

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
0.15       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368   <--
0.20       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.25       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.30       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.35       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.40       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.45       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.50       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.55       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.60       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.65       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.70       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.75       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
0.80       0.6454   0.6512   0.5241   0.8200   0.8274   0.5368  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6454, F1=0.6512, Normal Recall=0.5241, Normal Precision=0.8200, Attack Recall=0.8274, Attack Precision=0.5368

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
0.15       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345   <--
0.20       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.25       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.30       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.35       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.40       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.45       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.50       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.55       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.60       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.65       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.70       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.75       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
0.80       0.6754   0.7182   0.5233   0.7520   0.8274   0.6345  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6754, F1=0.7182, Normal Recall=0.5233, Normal Precision=0.7520, Attack Recall=0.8274, Attack Precision=0.6345

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
0.15       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855   <--
0.20       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.25       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.30       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.35       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.40       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.45       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.50       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.55       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.60       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.65       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.70       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.75       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
0.80       0.8408   0.5559   0.8234   0.9996   0.9967   0.3855  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8408, F1=0.5559, Normal Recall=0.8234, Normal Precision=0.9996, Attack Recall=0.9967, Attack Precision=0.3855

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
0.15       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856   <--
0.20       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.25       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.30       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.35       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.40       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.45       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.50       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.55       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.60       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.65       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.70       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.75       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
0.80       0.8583   0.7377   0.8238   0.9989   0.9963   0.5856  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8583, F1=0.7377, Normal Recall=0.8238, Normal Precision=0.9989, Attack Recall=0.9963, Attack Precision=0.5856

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
0.15       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082   <--
0.20       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.25       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.30       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.35       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.40       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.45       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.50       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.55       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.60       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.65       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.70       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.75       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
0.80       0.8758   0.8279   0.8241   0.9981   0.9963   0.7082  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8758, F1=0.8279, Normal Recall=0.8241, Normal Precision=0.9981, Attack Recall=0.9963, Attack Precision=0.7082

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
0.15       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896   <--
0.20       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.25       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.30       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.35       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.40       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.45       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.50       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.55       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.60       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.65       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.70       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.75       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
0.80       0.8924   0.8810   0.8230   0.9970   0.9963   0.7896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8924, F1=0.8810, Normal Recall=0.8230, Normal Precision=0.9970, Attack Recall=0.9963, Attack Precision=0.7896

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
0.15       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491   <--
0.20       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.25       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.30       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.35       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.40       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.45       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.50       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.55       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.60       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.65       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.70       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.75       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
0.80       0.9096   0.9168   0.8229   0.9956   0.9963   0.8491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9096, F1=0.9168, Normal Recall=0.8229, Normal Precision=0.9956, Attack Recall=0.9963, Attack Precision=0.8491

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
0.15       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865   <--
0.20       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.25       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.30       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.35       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.40       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.45       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.50       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.55       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.60       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.65       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.70       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.75       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.80       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8415, F1=0.5570, Normal Recall=0.8242, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3865

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
0.15       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866   <--
0.20       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.25       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.30       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.35       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.40       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.45       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.50       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.55       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.60       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.65       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.70       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.75       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.80       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8588, F1=0.7383, Normal Recall=0.8245, Normal Precision=0.9988, Attack Recall=0.9959, Attack Precision=0.5866

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
0.15       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089   <--
0.20       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.25       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.30       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.35       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.40       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.45       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.50       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.55       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.60       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.65       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.70       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.75       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.80       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8761, F1=0.8283, Normal Recall=0.8248, Normal Precision=0.9979, Attack Recall=0.9959, Attack Precision=0.7089

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
0.15       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902   <--
0.20       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.25       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.30       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.35       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.40       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.45       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.50       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.55       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.60       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.65       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.70       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.75       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.80       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8926, F1=0.8812, Normal Recall=0.8238, Normal Precision=0.9967, Attack Recall=0.9959, Attack Precision=0.7902

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
0.15       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496   <--
0.20       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.25       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.30       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.35       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.40       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.45       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.50       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.55       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.60       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.65       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.70       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.75       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.80       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9098, F1=0.9169, Normal Recall=0.8237, Normal Precision=0.9950, Attack Recall=0.9959, Attack Precision=0.8496

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
0.15       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865   <--
0.20       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.25       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.30       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.35       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.40       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.45       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.50       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.55       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.60       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.65       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.70       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.75       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
0.80       0.8415   0.5570   0.8242   0.9995   0.9965   0.3865  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8415, F1=0.5570, Normal Recall=0.8242, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3865

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
0.15       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866   <--
0.20       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.25       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.30       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.35       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.40       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.45       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.50       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.55       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.60       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.65       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.70       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.75       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
0.80       0.8588   0.7383   0.8245   0.9988   0.9959   0.5866  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8588, F1=0.7383, Normal Recall=0.8245, Normal Precision=0.9988, Attack Recall=0.9959, Attack Precision=0.5866

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
0.15       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089   <--
0.20       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.25       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.30       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.35       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.40       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.45       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.50       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.55       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.60       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.65       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.70       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.75       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
0.80       0.8761   0.8283   0.8248   0.9979   0.9959   0.7089  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8761, F1=0.8283, Normal Recall=0.8248, Normal Precision=0.9979, Attack Recall=0.9959, Attack Precision=0.7089

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
0.15       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902   <--
0.20       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.25       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.30       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.35       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.40       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.45       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.50       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.55       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.60       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.65       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.70       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.75       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
0.80       0.8926   0.8812   0.8238   0.9967   0.9959   0.7902  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8926, F1=0.8812, Normal Recall=0.8238, Normal Precision=0.9967, Attack Recall=0.9959, Attack Precision=0.7902

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
0.15       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496   <--
0.20       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.25       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.30       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.35       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.40       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.45       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.50       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.55       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.60       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.65       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.70       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.75       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
0.80       0.9098   0.9169   0.8237   0.9950   0.9959   0.8496  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9098, F1=0.9169, Normal Recall=0.8237, Normal Precision=0.9950, Attack Recall=0.9959, Attack Precision=0.8496

```

