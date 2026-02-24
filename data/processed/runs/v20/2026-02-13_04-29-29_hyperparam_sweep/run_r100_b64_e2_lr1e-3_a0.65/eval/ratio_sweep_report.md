# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-20 17:57:38 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8793 | 0.8723 | 0.8658 | 0.8601 | 0.8530 | 0.8472 | 0.8411 | 0.8349 | 0.8284 | 0.8220 | 0.8159 |
| QAT+Prune only | 0.7982 | 0.8190 | 0.8383 | 0.8585 | 0.8789 | 0.8971 | 0.9188 | 0.9384 | 0.9582 | 0.9776 | 0.9980 |
| QAT+PTQ | 0.7984 | 0.8192 | 0.8385 | 0.8587 | 0.8790 | 0.8970 | 0.9188 | 0.9385 | 0.9582 | 0.9777 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7984 | 0.8192 | 0.8385 | 0.8587 | 0.8790 | 0.8970 | 0.9188 | 0.9385 | 0.9582 | 0.9777 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5614 | 0.7086 | 0.7777 | 0.8162 | 0.8423 | 0.8604 | 0.8737 | 0.8838 | 0.8919 | 0.8986 |
| QAT+Prune only | 0.0000 | 0.5244 | 0.7117 | 0.8089 | 0.8683 | 0.9065 | 0.9365 | 0.9577 | 0.9745 | 0.9877 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.5247 | 0.7119 | 0.8091 | 0.8683 | 0.9065 | 0.9365 | 0.9578 | 0.9745 | 0.9877 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5247 | 0.7119 | 0.8091 | 0.8683 | 0.9065 | 0.9365 | 0.9578 | 0.9745 | 0.9877 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8793 | 0.8783 | 0.8782 | 0.8790 | 0.8777 | 0.8785 | 0.8789 | 0.8791 | 0.8781 | 0.8763 | 0.0000 |
| QAT+Prune only | 0.7982 | 0.7990 | 0.7984 | 0.7987 | 0.7995 | 0.7962 | 0.8000 | 0.7993 | 0.7992 | 0.7946 | 0.0000 |
| QAT+PTQ | 0.7984 | 0.7993 | 0.7986 | 0.7990 | 0.7996 | 0.7961 | 0.8001 | 0.7997 | 0.7993 | 0.7949 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7984 | 0.7993 | 0.7986 | 0.7990 | 0.7996 | 0.7961 | 0.8001 | 0.7997 | 0.7993 | 0.7949 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8793 | 0.0000 | 0.0000 | 0.0000 | 0.8793 | 1.0000 |
| 90 | 10 | 299,940 | 0.8723 | 0.4275 | 0.8176 | 0.5614 | 0.8783 | 0.9774 |
| 80 | 20 | 291,350 | 0.8658 | 0.6262 | 0.8159 | 0.7086 | 0.8782 | 0.9502 |
| 70 | 30 | 194,230 | 0.8601 | 0.7429 | 0.8159 | 0.7777 | 0.8790 | 0.9176 |
| 60 | 40 | 145,675 | 0.8530 | 0.8164 | 0.8159 | 0.8162 | 0.8777 | 0.8773 |
| 50 | 50 | 116,540 | 0.8472 | 0.8704 | 0.8159 | 0.8423 | 0.8785 | 0.8268 |
| 40 | 60 | 97,115 | 0.8411 | 0.9099 | 0.8159 | 0.8604 | 0.8789 | 0.7609 |
| 30 | 70 | 83,240 | 0.8349 | 0.9403 | 0.8159 | 0.8737 | 0.8791 | 0.6718 |
| 20 | 80 | 72,835 | 0.8284 | 0.9640 | 0.8159 | 0.8838 | 0.8781 | 0.5439 |
| 10 | 90 | 64,740 | 0.8220 | 0.9834 | 0.8159 | 0.8919 | 0.8763 | 0.3460 |
| 0 | 100 | 58,270 | 0.8159 | 1.0000 | 0.8159 | 0.8986 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7982 | 0.0000 | 0.0000 | 0.0000 | 0.7982 | 1.0000 |
| 90 | 10 | 299,940 | 0.8190 | 0.3556 | 0.9981 | 0.5244 | 0.7990 | 0.9997 |
| 80 | 20 | 291,350 | 0.8383 | 0.5531 | 0.9980 | 0.7117 | 0.7984 | 0.9994 |
| 70 | 30 | 194,230 | 0.8585 | 0.6800 | 0.9980 | 0.8089 | 0.7987 | 0.9989 |
| 60 | 40 | 145,675 | 0.8789 | 0.7684 | 0.9980 | 0.8683 | 0.7995 | 0.9983 |
| 50 | 50 | 116,540 | 0.8971 | 0.8304 | 0.9980 | 0.9065 | 0.7962 | 0.9975 |
| 40 | 60 | 97,115 | 0.9188 | 0.8822 | 0.9980 | 0.9365 | 0.8000 | 0.9962 |
| 30 | 70 | 83,240 | 0.9384 | 0.9206 | 0.9980 | 0.9577 | 0.7993 | 0.9941 |
| 20 | 80 | 72,835 | 0.9582 | 0.9521 | 0.9980 | 0.9745 | 0.7992 | 0.9900 |
| 10 | 90 | 64,740 | 0.9776 | 0.9776 | 0.9980 | 0.9877 | 0.7946 | 0.9776 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7984 | 0.0000 | 0.0000 | 0.0000 | 0.7984 | 1.0000 |
| 90 | 10 | 299,940 | 0.8192 | 0.3559 | 0.9981 | 0.5247 | 0.7993 | 0.9997 |
| 80 | 20 | 291,350 | 0.8385 | 0.5533 | 0.9980 | 0.7119 | 0.7986 | 0.9994 |
| 70 | 30 | 194,230 | 0.8587 | 0.6803 | 0.9980 | 0.8091 | 0.7990 | 0.9989 |
| 60 | 40 | 145,675 | 0.8790 | 0.7685 | 0.9980 | 0.8683 | 0.7996 | 0.9983 |
| 50 | 50 | 116,540 | 0.8970 | 0.8304 | 0.9980 | 0.9065 | 0.7961 | 0.9975 |
| 40 | 60 | 97,115 | 0.9188 | 0.8822 | 0.9980 | 0.9365 | 0.8001 | 0.9962 |
| 30 | 70 | 83,240 | 0.9385 | 0.9208 | 0.9980 | 0.9578 | 0.7997 | 0.9941 |
| 20 | 80 | 72,835 | 0.9582 | 0.9521 | 0.9980 | 0.9745 | 0.7993 | 0.9900 |
| 10 | 90 | 64,740 | 0.9777 | 0.9777 | 0.9980 | 0.9877 | 0.7949 | 0.9776 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7984 | 0.0000 | 0.0000 | 0.0000 | 0.7984 | 1.0000 |
| 90 | 10 | 299,940 | 0.8192 | 0.3559 | 0.9981 | 0.5247 | 0.7993 | 0.9997 |
| 80 | 20 | 291,350 | 0.8385 | 0.5533 | 0.9980 | 0.7119 | 0.7986 | 0.9994 |
| 70 | 30 | 194,230 | 0.8587 | 0.6803 | 0.9980 | 0.8091 | 0.7990 | 0.9989 |
| 60 | 40 | 145,675 | 0.8790 | 0.7685 | 0.9980 | 0.8683 | 0.7996 | 0.9983 |
| 50 | 50 | 116,540 | 0.8970 | 0.8304 | 0.9980 | 0.9065 | 0.7961 | 0.9975 |
| 40 | 60 | 97,115 | 0.9188 | 0.8822 | 0.9980 | 0.9365 | 0.8001 | 0.9962 |
| 30 | 70 | 83,240 | 0.9385 | 0.9208 | 0.9980 | 0.9578 | 0.7997 | 0.9941 |
| 20 | 80 | 72,835 | 0.9582 | 0.9521 | 0.9980 | 0.9745 | 0.7993 | 0.9900 |
| 10 | 90 | 64,740 | 0.9777 | 0.9777 | 0.9980 | 0.9877 | 0.7949 | 0.9776 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274   <--
0.20       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.25       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.30       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.35       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.40       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.45       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.50       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.55       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.60       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.65       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.70       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.75       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
0.80       0.8722   0.5614   0.8783   0.9774   0.8175   0.4274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8722, F1=0.5614, Normal Recall=0.8783, Normal Precision=0.9774, Attack Recall=0.8175, Attack Precision=0.4274

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
0.15       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272   <--
0.20       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.25       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.30       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.35       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.40       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.45       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.50       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.55       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.60       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.65       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.70       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.75       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
0.80       0.8662   0.7093   0.8788   0.9502   0.8159   0.6272  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8662, F1=0.7093, Normal Recall=0.8788, Normal Precision=0.9502, Attack Recall=0.8159, Attack Precision=0.6272

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
0.15       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436   <--
0.20       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.25       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.30       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.35       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.40       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.45       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.50       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.55       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.60       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.65       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.70       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.75       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
0.80       0.8604   0.7781   0.8794   0.9177   0.8159   0.7436  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8604, F1=0.7781, Normal Recall=0.8794, Normal Precision=0.9177, Attack Recall=0.8159, Attack Precision=0.7436

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
0.15       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173   <--
0.20       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.25       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.30       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.35       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.40       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.45       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.50       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.55       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.60       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.65       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.70       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.75       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
0.80       0.8534   0.8166   0.8784   0.8774   0.8159   0.8173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8534, F1=0.8166, Normal Recall=0.8784, Normal Precision=0.8774, Attack Recall=0.8159, Attack Precision=0.8173

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
0.15       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690   <--
0.20       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.25       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.30       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.35       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.40       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.45       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.50       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.55       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.60       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.65       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.70       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.75       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
0.80       0.8465   0.8416   0.8770   0.8265   0.8159   0.8690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8465, F1=0.8416, Normal Recall=0.8770, Normal Precision=0.8265, Attack Recall=0.8159, Attack Precision=0.8690

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
0.15       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556   <--
0.20       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.25       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.30       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.35       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.40       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.45       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.50       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.55       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.60       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.65       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.70       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.75       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
0.80       0.8190   0.5244   0.7990   0.9997   0.9981   0.3556  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8190, F1=0.5244, Normal Recall=0.7990, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.3556

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
0.15       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545   <--
0.20       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.25       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.30       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.35       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.40       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.45       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.50       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.55       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.60       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.65       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.70       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.75       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
0.80       0.8392   0.7129   0.7995   0.9994   0.9980   0.5545  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8392, F1=0.7129, Normal Recall=0.7995, Normal Precision=0.9994, Attack Recall=0.9980, Attack Precision=0.5545

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
0.15       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800   <--
0.20       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.25       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.30       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.35       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.40       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.45       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.50       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.55       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.60       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.65       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.70       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.75       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
0.80       0.8585   0.8088   0.7987   0.9989   0.9980   0.6800  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8585, F1=0.8088, Normal Recall=0.7987, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6800

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
0.15       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673   <--
0.20       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.25       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.30       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.35       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.40       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.45       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.50       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.55       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.60       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.65       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.70       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.75       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
0.80       0.8781   0.8676   0.7982   0.9983   0.9980   0.7673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8781, F1=0.8676, Normal Recall=0.7982, Normal Precision=0.9983, Attack Recall=0.9980, Attack Precision=0.7673

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
0.15       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310   <--
0.20       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.25       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.30       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.35       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.40       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.45       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.50       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.55       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.60       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.65       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.70       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.75       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
0.80       0.8975   0.9068   0.7970   0.9975   0.9980   0.8310  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8975, F1=0.9068, Normal Recall=0.7970, Normal Precision=0.9975, Attack Recall=0.9980, Attack Precision=0.8310

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
0.15       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559   <--
0.20       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.25       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.30       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.35       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.40       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.45       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.50       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.55       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.60       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.65       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.70       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.75       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.80       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8192, F1=0.5247, Normal Recall=0.7993, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.3559

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
0.15       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548   <--
0.20       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.25       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.30       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.35       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.40       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.45       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.50       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.55       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.60       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.65       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.70       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.75       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.80       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8394, F1=0.7132, Normal Recall=0.7998, Normal Precision=0.9994, Attack Recall=0.9980, Attack Precision=0.5548

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
0.15       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803   <--
0.20       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.25       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.30       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.35       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.40       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.45       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.50       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.55       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.60       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.65       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.70       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.75       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.80       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8587, F1=0.8091, Normal Recall=0.7990, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6803

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
0.15       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675   <--
0.20       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.25       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.30       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.35       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.40       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.45       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.50       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.55       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.60       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.65       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.70       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.75       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.80       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8783, F1=0.8677, Normal Recall=0.7985, Normal Precision=0.9983, Attack Recall=0.9980, Attack Precision=0.7675

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
0.15       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311   <--
0.20       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.25       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.30       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.35       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.40       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.45       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.50       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.55       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.60       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.65       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.70       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.75       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.80       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.9069, Normal Recall=0.7972, Normal Precision=0.9975, Attack Recall=0.9980, Attack Precision=0.8311

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
0.15       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559   <--
0.20       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.25       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.30       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.35       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.40       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.45       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.50       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.55       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.60       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.65       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.70       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.75       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
0.80       0.8192   0.5247   0.7993   0.9997   0.9981   0.3559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8192, F1=0.5247, Normal Recall=0.7993, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.3559

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
0.15       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548   <--
0.20       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.25       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.30       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.35       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.40       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.45       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.50       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.55       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.60       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.65       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.70       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.75       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
0.80       0.8394   0.7132   0.7998   0.9994   0.9980   0.5548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8394, F1=0.7132, Normal Recall=0.7998, Normal Precision=0.9994, Attack Recall=0.9980, Attack Precision=0.5548

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
0.15       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803   <--
0.20       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.25       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.30       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.35       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.40       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.45       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.50       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.55       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.60       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.65       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.70       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.75       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
0.80       0.8587   0.8091   0.7990   0.9989   0.9980   0.6803  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8587, F1=0.8091, Normal Recall=0.7990, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6803

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
0.15       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675   <--
0.20       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.25       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.30       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.35       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.40       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.45       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.50       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.55       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.60       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.65       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.70       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.75       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
0.80       0.8783   0.8677   0.7985   0.9983   0.9980   0.7675  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8783, F1=0.8677, Normal Recall=0.7985, Normal Precision=0.9983, Attack Recall=0.9980, Attack Precision=0.7675

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
0.15       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311   <--
0.20       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.25       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.30       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.35       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.40       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.45       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.50       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.55       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.60       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.65       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.70       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.75       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
0.80       0.8976   0.9069   0.7972   0.9975   0.9980   0.8311  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.9069, Normal Recall=0.7972, Normal Precision=0.9975, Attack Recall=0.9980, Attack Precision=0.8311

```

