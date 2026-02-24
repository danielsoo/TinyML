# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-20 15:16:37 |

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
| Original (TFLite) | 0.6134 | 0.6264 | 0.6387 | 0.6512 | 0.6625 | 0.6736 | 0.6891 | 0.6994 | 0.7116 | 0.7245 | 0.7365 |
| QAT+Prune only | 0.8569 | 0.8699 | 0.8821 | 0.8956 | 0.9078 | 0.9203 | 0.9336 | 0.9459 | 0.9586 | 0.9712 | 0.9841 |
| QAT+PTQ | 0.8576 | 0.8704 | 0.8825 | 0.8960 | 0.9079 | 0.9207 | 0.9338 | 0.9459 | 0.9586 | 0.9711 | 0.9841 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8576 | 0.8704 | 0.8825 | 0.8960 | 0.9079 | 0.9207 | 0.9338 | 0.9459 | 0.9586 | 0.9711 | 0.9841 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2825 | 0.4491 | 0.5588 | 0.6358 | 0.6929 | 0.7398 | 0.7743 | 0.8034 | 0.8279 | 0.8482 |
| QAT+Prune only | 0.0000 | 0.6023 | 0.7695 | 0.8498 | 0.8952 | 0.9250 | 0.9468 | 0.9622 | 0.9744 | 0.9840 | 0.9920 |
| QAT+PTQ | 0.0000 | 0.6032 | 0.7701 | 0.8503 | 0.8953 | 0.9254 | 0.9469 | 0.9622 | 0.9744 | 0.9840 | 0.9920 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6032 | 0.7701 | 0.8503 | 0.8953 | 0.9254 | 0.9469 | 0.9622 | 0.9744 | 0.9840 | 0.9920 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6134 | 0.6143 | 0.6142 | 0.6146 | 0.6132 | 0.6108 | 0.6181 | 0.6130 | 0.6124 | 0.6165 | 0.0000 |
| QAT+Prune only | 0.8569 | 0.8572 | 0.8566 | 0.8577 | 0.8570 | 0.8564 | 0.8578 | 0.8567 | 0.8566 | 0.8546 | 0.0000 |
| QAT+PTQ | 0.8576 | 0.8577 | 0.8571 | 0.8583 | 0.8572 | 0.8573 | 0.8584 | 0.8568 | 0.8569 | 0.8550 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8576 | 0.8577 | 0.8571 | 0.8583 | 0.8572 | 0.8573 | 0.8584 | 0.8568 | 0.8569 | 0.8550 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6134 | 0.0000 | 0.0000 | 0.0000 | 0.6134 | 1.0000 |
| 90 | 10 | 299,940 | 0.6264 | 0.1748 | 0.7353 | 0.2825 | 0.6143 | 0.9543 |
| 80 | 20 | 291,350 | 0.6387 | 0.3231 | 0.7365 | 0.4491 | 0.6142 | 0.9031 |
| 70 | 30 | 194,230 | 0.6512 | 0.4502 | 0.7365 | 0.5588 | 0.6146 | 0.8448 |
| 60 | 40 | 145,675 | 0.6625 | 0.5593 | 0.7365 | 0.6358 | 0.6132 | 0.7773 |
| 50 | 50 | 116,540 | 0.6736 | 0.6543 | 0.7365 | 0.6929 | 0.6108 | 0.6986 |
| 40 | 60 | 97,115 | 0.6891 | 0.7431 | 0.7365 | 0.7398 | 0.6181 | 0.6099 |
| 30 | 70 | 83,240 | 0.6994 | 0.8162 | 0.7365 | 0.7743 | 0.6130 | 0.4992 |
| 20 | 80 | 72,835 | 0.7116 | 0.8837 | 0.7365 | 0.8034 | 0.6124 | 0.3675 |
| 10 | 90 | 64,740 | 0.7245 | 0.9453 | 0.7365 | 0.8279 | 0.6165 | 0.2063 |
| 0 | 100 | 58,270 | 0.7365 | 1.0000 | 0.7365 | 0.8482 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8569 | 0.0000 | 0.0000 | 0.0000 | 0.8569 | 1.0000 |
| 90 | 10 | 299,940 | 0.8699 | 0.4338 | 0.9848 | 0.6023 | 0.8572 | 0.9980 |
| 80 | 20 | 291,350 | 0.8821 | 0.6317 | 0.9841 | 0.7695 | 0.8566 | 0.9954 |
| 70 | 30 | 194,230 | 0.8956 | 0.7477 | 0.9841 | 0.8498 | 0.8577 | 0.9921 |
| 60 | 40 | 145,675 | 0.9078 | 0.8210 | 0.9841 | 0.8952 | 0.8570 | 0.9878 |
| 50 | 50 | 116,540 | 0.9203 | 0.8727 | 0.9841 | 0.9250 | 0.8564 | 0.9818 |
| 40 | 60 | 97,115 | 0.9336 | 0.9121 | 0.9841 | 0.9468 | 0.8578 | 0.9730 |
| 30 | 70 | 83,240 | 0.9459 | 0.9413 | 0.9841 | 0.9622 | 0.8567 | 0.9585 |
| 20 | 80 | 72,835 | 0.9586 | 0.9648 | 0.9841 | 0.9744 | 0.8566 | 0.9309 |
| 10 | 90 | 64,740 | 0.9712 | 0.9839 | 0.9841 | 0.9840 | 0.8546 | 0.8566 |
| 0 | 100 | 58,270 | 0.9841 | 1.0000 | 0.9841 | 0.9920 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8576 | 0.0000 | 0.0000 | 0.0000 | 0.8576 | 1.0000 |
| 90 | 10 | 299,940 | 0.8704 | 0.4347 | 0.9848 | 0.6032 | 0.8577 | 0.9980 |
| 80 | 20 | 291,350 | 0.8825 | 0.6325 | 0.9841 | 0.7701 | 0.8571 | 0.9954 |
| 70 | 30 | 194,230 | 0.8960 | 0.7485 | 0.9841 | 0.8503 | 0.8583 | 0.9921 |
| 60 | 40 | 145,675 | 0.9079 | 0.8212 | 0.9841 | 0.8953 | 0.8572 | 0.9878 |
| 50 | 50 | 116,540 | 0.9207 | 0.8734 | 0.9841 | 0.9254 | 0.8573 | 0.9817 |
| 40 | 60 | 97,115 | 0.9338 | 0.9124 | 0.9841 | 0.9469 | 0.8584 | 0.9729 |
| 30 | 70 | 83,240 | 0.9459 | 0.9413 | 0.9841 | 0.9622 | 0.8568 | 0.9584 |
| 20 | 80 | 72,835 | 0.9586 | 0.9649 | 0.9841 | 0.9744 | 0.8569 | 0.9307 |
| 10 | 90 | 64,740 | 0.9711 | 0.9839 | 0.9841 | 0.9840 | 0.8550 | 0.8563 |
| 0 | 100 | 58,270 | 0.9841 | 1.0000 | 0.9841 | 0.9920 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8576 | 0.0000 | 0.0000 | 0.0000 | 0.8576 | 1.0000 |
| 90 | 10 | 299,940 | 0.8704 | 0.4347 | 0.9848 | 0.6032 | 0.8577 | 0.9980 |
| 80 | 20 | 291,350 | 0.8825 | 0.6325 | 0.9841 | 0.7701 | 0.8571 | 0.9954 |
| 70 | 30 | 194,230 | 0.8960 | 0.7485 | 0.9841 | 0.8503 | 0.8583 | 0.9921 |
| 60 | 40 | 145,675 | 0.9079 | 0.8212 | 0.9841 | 0.8953 | 0.8572 | 0.9878 |
| 50 | 50 | 116,540 | 0.9207 | 0.8734 | 0.9841 | 0.9254 | 0.8573 | 0.9817 |
| 40 | 60 | 97,115 | 0.9338 | 0.9124 | 0.9841 | 0.9469 | 0.8584 | 0.9729 |
| 30 | 70 | 83,240 | 0.9459 | 0.9413 | 0.9841 | 0.9622 | 0.8568 | 0.9584 |
| 20 | 80 | 72,835 | 0.9586 | 0.9649 | 0.9841 | 0.9744 | 0.8569 | 0.9307 |
| 10 | 90 | 64,740 | 0.9711 | 0.9839 | 0.9841 | 0.9840 | 0.8550 | 0.8563 |
| 0 | 100 | 58,270 | 0.9841 | 1.0000 | 0.9841 | 0.9920 | 0.0000 | 0.0000 |


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
0.15       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746   <--
0.20       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.25       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.30       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.35       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.40       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.45       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.50       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.55       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.60       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.65       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.70       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.75       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
0.80       0.6263   0.2822   0.6143   0.9542   0.7344   0.1746  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6263, F1=0.2822, Normal Recall=0.6143, Normal Precision=0.9542, Attack Recall=0.7344, Attack Precision=0.1746

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
0.15       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231   <--
0.20       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.25       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.30       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.35       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.40       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.45       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.50       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.55       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.60       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.65       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.70       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.75       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
0.80       0.6387   0.4491   0.6143   0.9031   0.7365   0.3231  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6387, F1=0.4491, Normal Recall=0.6143, Normal Precision=0.9031, Attack Recall=0.7365, Attack Precision=0.3231

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
0.15       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499   <--
0.20       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.25       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.30       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.35       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.40       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.45       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.50       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.55       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.60       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.65       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.70       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.75       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
0.80       0.6508   0.5586   0.6141   0.8447   0.7365   0.4499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6508, F1=0.5586, Normal Recall=0.6141, Normal Precision=0.8447, Attack Recall=0.7365, Attack Precision=0.4499

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
0.15       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594   <--
0.20       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.25       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.30       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.35       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.40       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.45       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.50       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.55       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.60       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.65       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.70       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.75       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
0.80       0.6626   0.6359   0.6133   0.7773   0.7365   0.5594  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6626, F1=0.6359, Normal Recall=0.6133, Normal Precision=0.7773, Attack Recall=0.7365, Attack Precision=0.5594

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
0.15       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553   <--
0.20       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.25       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.30       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.35       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.40       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.45       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.50       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.55       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.60       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.65       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.70       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.75       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
0.80       0.6745   0.6935   0.6126   0.6992   0.7365   0.6553  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6745, F1=0.6935, Normal Recall=0.6126, Normal Precision=0.6992, Attack Recall=0.7365, Attack Precision=0.6553

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
0.15       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338   <--
0.20       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.25       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.30       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.35       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.40       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.45       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.50       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.55       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.60       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.65       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.70       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.75       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
0.80       0.8699   0.6022   0.8572   0.9980   0.9847   0.4338  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8699, F1=0.6022, Normal Recall=0.8572, Normal Precision=0.9980, Attack Recall=0.9847, Attack Precision=0.4338

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
0.15       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334   <--
0.20       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.25       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.30       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.35       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.40       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.45       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.50       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.55       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.60       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.65       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.70       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.75       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
0.80       0.8829   0.7707   0.8576   0.9954   0.9841   0.6334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8829, F1=0.7707, Normal Recall=0.8576, Normal Precision=0.9954, Attack Recall=0.9841, Attack Precision=0.6334

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
0.15       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477   <--
0.20       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.25       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.30       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.35       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.40       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.45       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.50       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.55       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.60       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.65       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.70       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.75       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
0.80       0.8956   0.8498   0.8577   0.9921   0.9841   0.7477  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8956, F1=0.8498, Normal Recall=0.8577, Normal Precision=0.9921, Attack Recall=0.9841, Attack Precision=0.7477

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
0.15       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208   <--
0.20       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.25       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.30       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.35       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.40       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.45       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.50       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.55       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.60       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.65       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.70       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.75       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
0.80       0.9077   0.8951   0.8568   0.9878   0.9841   0.8208  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9077, F1=0.8951, Normal Recall=0.8568, Normal Precision=0.9878, Attack Recall=0.9841, Attack Precision=0.8208

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
0.15       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723   <--
0.20       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.25       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.30       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.35       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.40       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.45       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.50       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.55       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.60       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.65       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.70       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.75       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
0.80       0.9200   0.9248   0.8559   0.9818   0.9841   0.8723  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9200, F1=0.9248, Normal Recall=0.8559, Normal Precision=0.9818, Attack Recall=0.9841, Attack Precision=0.8723

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
0.15       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347   <--
0.20       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.25       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.30       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.35       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.40       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.45       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.50       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.55       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.60       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.65       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.70       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.75       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.80       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.6031, Normal Recall=0.8577, Normal Precision=0.9980, Attack Recall=0.9845, Attack Precision=0.4347

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
0.15       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344   <--
0.20       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.25       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.30       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.35       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.40       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.45       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.50       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.55       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.60       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.65       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.70       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.75       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.80       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8834, F1=0.7714, Normal Recall=0.8582, Normal Precision=0.9954, Attack Recall=0.9841, Attack Precision=0.6344

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
0.15       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487   <--
0.20       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.25       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.30       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.35       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.40       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.45       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.50       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.55       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.60       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.65       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.70       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.75       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.80       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8961, F1=0.8504, Normal Recall=0.8584, Normal Precision=0.9921, Attack Recall=0.9841, Attack Precision=0.7487

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
0.15       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215   <--
0.20       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.25       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.30       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.35       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.40       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.45       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.50       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.55       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.60       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.65       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.70       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.75       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.80       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.8954, Normal Recall=0.8574, Normal Precision=0.9878, Attack Recall=0.9841, Attack Precision=0.8215

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
0.15       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727   <--
0.20       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.25       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.30       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.35       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.40       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.45       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.50       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.55       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.60       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.65       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.70       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.75       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.80       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9203, F1=0.9250, Normal Recall=0.8565, Normal Precision=0.9817, Attack Recall=0.9841, Attack Precision=0.8727

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
0.15       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347   <--
0.20       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.25       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.30       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.35       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.40       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.45       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.50       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.55       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.60       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.65       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.70       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.75       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
0.80       0.8704   0.6031   0.8577   0.9980   0.9845   0.4347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.6031, Normal Recall=0.8577, Normal Precision=0.9980, Attack Recall=0.9845, Attack Precision=0.4347

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
0.15       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344   <--
0.20       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.25       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.30       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.35       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.40       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.45       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.50       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.55       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.60       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.65       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.70       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.75       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
0.80       0.8834   0.7714   0.8582   0.9954   0.9841   0.6344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8834, F1=0.7714, Normal Recall=0.8582, Normal Precision=0.9954, Attack Recall=0.9841, Attack Precision=0.6344

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
0.15       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487   <--
0.20       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.25       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.30       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.35       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.40       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.45       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.50       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.55       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.60       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.65       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.70       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.75       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
0.80       0.8961   0.8504   0.8584   0.9921   0.9841   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8961, F1=0.8504, Normal Recall=0.8584, Normal Precision=0.9921, Attack Recall=0.9841, Attack Precision=0.7487

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
0.15       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215   <--
0.20       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.25       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.30       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.35       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.40       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.45       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.50       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.55       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.60       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.65       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.70       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.75       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
0.80       0.9081   0.8954   0.8574   0.9878   0.9841   0.8215  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.8954, Normal Recall=0.8574, Normal Precision=0.9878, Attack Recall=0.9841, Attack Precision=0.8215

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
0.15       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727   <--
0.20       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.25       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.30       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.35       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.40       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.45       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.50       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.55       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.60       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.65       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.70       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.75       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
0.80       0.9203   0.9250   0.8565   0.9817   0.9841   0.8727  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9203, F1=0.9250, Normal Recall=0.8565, Normal Precision=0.9817, Attack Recall=0.9841, Attack Precision=0.8727

```

