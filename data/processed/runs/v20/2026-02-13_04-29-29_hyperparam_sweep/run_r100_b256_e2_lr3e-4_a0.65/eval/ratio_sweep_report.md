# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-22 13:41:44 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8514 | 0.8232 | 0.7962 | 0.7702 | 0.7413 | 0.7157 | 0.6878 | 0.6613 | 0.6342 | 0.6072 | 0.5801 |
| QAT+Prune only | 0.7925 | 0.8027 | 0.8121 | 0.8234 | 0.8334 | 0.8414 | 0.8526 | 0.8628 | 0.8737 | 0.8836 | 0.8938 |
| QAT+PTQ | 0.7924 | 0.8028 | 0.8126 | 0.8242 | 0.8347 | 0.8429 | 0.8545 | 0.8651 | 0.8762 | 0.8865 | 0.8970 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7924 | 0.8028 | 0.8126 | 0.8242 | 0.8347 | 0.8429 | 0.8545 | 0.8651 | 0.8762 | 0.8865 | 0.8970 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3960 | 0.5324 | 0.6023 | 0.6420 | 0.6711 | 0.6904 | 0.7057 | 0.7173 | 0.7266 | 0.7343 |
| QAT+Prune only | 0.0000 | 0.4753 | 0.6555 | 0.7523 | 0.8110 | 0.8493 | 0.8792 | 0.9012 | 0.9189 | 0.9325 | 0.9439 |
| QAT+PTQ | 0.0000 | 0.4763 | 0.6569 | 0.7538 | 0.8128 | 0.8510 | 0.8810 | 0.9030 | 0.9206 | 0.9343 | 0.9457 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4763 | 0.6569 | 0.7538 | 0.8128 | 0.8510 | 0.8810 | 0.9030 | 0.9206 | 0.9343 | 0.9457 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8514 | 0.8502 | 0.8502 | 0.8516 | 0.8487 | 0.8514 | 0.8494 | 0.8509 | 0.8508 | 0.8511 | 0.0000 |
| QAT+Prune only | 0.7925 | 0.7925 | 0.7917 | 0.7932 | 0.7931 | 0.7891 | 0.7907 | 0.7905 | 0.7932 | 0.7910 | 0.0000 |
| QAT+PTQ | 0.7924 | 0.7924 | 0.7915 | 0.7929 | 0.7931 | 0.7889 | 0.7908 | 0.7905 | 0.7928 | 0.7921 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7924 | 0.7924 | 0.7915 | 0.7929 | 0.7931 | 0.7889 | 0.7908 | 0.7905 | 0.7928 | 0.7921 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8514 | 0.0000 | 0.0000 | 0.0000 | 0.8514 | 1.0000 |
| 90 | 10 | 299,940 | 0.8232 | 0.3007 | 0.5796 | 0.3960 | 0.8502 | 0.9479 |
| 80 | 20 | 291,350 | 0.7962 | 0.4920 | 0.5801 | 0.5324 | 0.8502 | 0.8901 |
| 70 | 30 | 194,230 | 0.7702 | 0.6262 | 0.5801 | 0.6023 | 0.8516 | 0.8256 |
| 60 | 40 | 145,675 | 0.7413 | 0.7188 | 0.5801 | 0.6420 | 0.8487 | 0.7520 |
| 50 | 50 | 116,540 | 0.7157 | 0.7961 | 0.5801 | 0.6711 | 0.8514 | 0.6697 |
| 40 | 60 | 97,115 | 0.6878 | 0.8524 | 0.5801 | 0.6904 | 0.8494 | 0.5742 |
| 30 | 70 | 83,240 | 0.6613 | 0.9008 | 0.5801 | 0.7057 | 0.8509 | 0.4648 |
| 20 | 80 | 72,835 | 0.6342 | 0.9396 | 0.5801 | 0.7173 | 0.8508 | 0.3362 |
| 10 | 90 | 64,740 | 0.6072 | 0.9723 | 0.5801 | 0.7266 | 0.8511 | 0.1838 |
| 0 | 100 | 58,270 | 0.5801 | 1.0000 | 0.5801 | 0.7343 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7925 | 0.0000 | 0.0000 | 0.0000 | 0.7925 | 1.0000 |
| 90 | 10 | 299,940 | 0.8027 | 0.3237 | 0.8938 | 0.4753 | 0.7925 | 0.9853 |
| 80 | 20 | 291,350 | 0.8121 | 0.5175 | 0.8938 | 0.6555 | 0.7917 | 0.9676 |
| 70 | 30 | 194,230 | 0.8234 | 0.6495 | 0.8938 | 0.7523 | 0.7932 | 0.9458 |
| 60 | 40 | 145,675 | 0.8334 | 0.7423 | 0.8938 | 0.8110 | 0.7931 | 0.9181 |
| 50 | 50 | 116,540 | 0.8414 | 0.8091 | 0.8938 | 0.8493 | 0.7891 | 0.8814 |
| 40 | 60 | 97,115 | 0.8526 | 0.8650 | 0.8938 | 0.8792 | 0.7907 | 0.8324 |
| 30 | 70 | 83,240 | 0.8628 | 0.9087 | 0.8938 | 0.9012 | 0.7905 | 0.7614 |
| 20 | 80 | 72,835 | 0.8737 | 0.9453 | 0.8938 | 0.9189 | 0.7932 | 0.6513 |
| 10 | 90 | 64,740 | 0.8836 | 0.9747 | 0.8938 | 0.9325 | 0.7910 | 0.4529 |
| 0 | 100 | 58,270 | 0.8938 | 1.0000 | 0.8938 | 0.9439 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7924 | 0.0000 | 0.0000 | 0.0000 | 0.7924 | 1.0000 |
| 90 | 10 | 299,940 | 0.8028 | 0.3243 | 0.8967 | 0.4763 | 0.7924 | 0.9857 |
| 80 | 20 | 291,350 | 0.8126 | 0.5182 | 0.8970 | 0.6569 | 0.7915 | 0.9685 |
| 70 | 30 | 194,230 | 0.8242 | 0.6500 | 0.8970 | 0.7538 | 0.7929 | 0.9473 |
| 60 | 40 | 145,675 | 0.8347 | 0.7430 | 0.8970 | 0.8128 | 0.7931 | 0.9203 |
| 50 | 50 | 116,540 | 0.8429 | 0.8095 | 0.8970 | 0.8510 | 0.7889 | 0.8845 |
| 40 | 60 | 97,115 | 0.8545 | 0.8654 | 0.8970 | 0.8810 | 0.7908 | 0.8366 |
| 30 | 70 | 83,240 | 0.8651 | 0.9090 | 0.8970 | 0.9030 | 0.7905 | 0.7669 |
| 20 | 80 | 72,835 | 0.8762 | 0.9454 | 0.8970 | 0.9206 | 0.7928 | 0.6581 |
| 10 | 90 | 64,740 | 0.8865 | 0.9749 | 0.8970 | 0.9343 | 0.7921 | 0.4609 |
| 0 | 100 | 58,270 | 0.8970 | 1.0000 | 0.8970 | 0.9457 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7924 | 0.0000 | 0.0000 | 0.0000 | 0.7924 | 1.0000 |
| 90 | 10 | 299,940 | 0.8028 | 0.3243 | 0.8967 | 0.4763 | 0.7924 | 0.9857 |
| 80 | 20 | 291,350 | 0.8126 | 0.5182 | 0.8970 | 0.6569 | 0.7915 | 0.9685 |
| 70 | 30 | 194,230 | 0.8242 | 0.6500 | 0.8970 | 0.7538 | 0.7929 | 0.9473 |
| 60 | 40 | 145,675 | 0.8347 | 0.7430 | 0.8970 | 0.8128 | 0.7931 | 0.9203 |
| 50 | 50 | 116,540 | 0.8429 | 0.8095 | 0.8970 | 0.8510 | 0.7889 | 0.8845 |
| 40 | 60 | 97,115 | 0.8545 | 0.8654 | 0.8970 | 0.8810 | 0.7908 | 0.8366 |
| 30 | 70 | 83,240 | 0.8651 | 0.9090 | 0.8970 | 0.9030 | 0.7905 | 0.7669 |
| 20 | 80 | 72,835 | 0.8762 | 0.9454 | 0.8970 | 0.9206 | 0.7928 | 0.6581 |
| 10 | 90 | 64,740 | 0.8865 | 0.9749 | 0.8970 | 0.9343 | 0.7921 | 0.4609 |
| 0 | 100 | 58,270 | 0.8970 | 1.0000 | 0.8970 | 0.9457 | 0.0000 | 0.0000 |


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
0.15       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022   <--
0.20       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.25       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.30       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.35       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.40       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.45       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.50       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.55       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.60       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.65       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.70       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.75       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
0.80       0.8236   0.3981   0.8502   0.9484   0.5836   0.3022  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8236, F1=0.3981, Normal Recall=0.8502, Normal Precision=0.9484, Attack Recall=0.5836, Attack Precision=0.3022

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
0.15       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923   <--
0.20       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.25       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.30       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.35       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.40       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.45       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.50       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.55       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.60       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.65       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.70       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.75       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
0.80       0.7964   0.5326   0.8505   0.8901   0.5801   0.4923  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7964, F1=0.5326, Normal Recall=0.8505, Normal Precision=0.8901, Attack Recall=0.5801, Attack Precision=0.4923

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
0.15       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258   <--
0.20       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.25       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.30       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.35       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.40       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.45       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.50       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.55       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.60       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.65       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.70       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.75       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
0.80       0.7700   0.6021   0.8514   0.8255   0.5801   0.6258  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7700, F1=0.6021, Normal Recall=0.8514, Normal Precision=0.8255, Attack Recall=0.5801, Attack Precision=0.6258

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
0.15       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230   <--
0.20       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.25       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.30       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.35       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.40       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.45       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.50       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.55       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.60       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.65       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.70       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.75       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
0.80       0.7431   0.6437   0.8518   0.7527   0.5801   0.7230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7431, F1=0.6437, Normal Recall=0.8518, Normal Precision=0.7527, Attack Recall=0.5801, Attack Precision=0.7230

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
0.15       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970   <--
0.20       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.25       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.30       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.35       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.40       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.45       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.50       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.55       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.60       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.65       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.70       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.75       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
0.80       0.7162   0.6715   0.8523   0.6699   0.5801   0.7970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7162, F1=0.6715, Normal Recall=0.8523, Normal Precision=0.6699, Attack Recall=0.5801, Attack Precision=0.7970

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
0.15       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235   <--
0.20       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.25       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.30       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.35       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.40       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.45       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.50       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.55       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.60       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.65       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.70       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.75       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
0.80       0.8026   0.4750   0.7925   0.9852   0.8930   0.3235  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8026, F1=0.4750, Normal Recall=0.7925, Normal Precision=0.9852, Attack Recall=0.8930, Attack Precision=0.3235

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
0.15       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189   <--
0.20       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.25       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.30       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.35       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.40       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.45       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.50       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.55       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.60       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.65       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.70       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.75       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
0.80       0.8130   0.6566   0.7928   0.9676   0.8938   0.5189  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8130, F1=0.6566, Normal Recall=0.7928, Normal Precision=0.9676, Attack Recall=0.8938, Attack Precision=0.5189

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
0.15       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480   <--
0.20       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.25       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.30       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.35       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.40       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.45       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.50       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.55       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.60       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.65       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.70       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.75       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
0.80       0.8225   0.7514   0.7920   0.9457   0.8938   0.6480  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8225, F1=0.7514, Normal Recall=0.7920, Normal Precision=0.9457, Attack Recall=0.8938, Attack Precision=0.6480

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
0.15       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414   <--
0.20       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.25       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.30       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.35       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.40       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.45       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.50       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.55       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.60       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.65       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.70       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.75       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
0.80       0.8328   0.8105   0.7921   0.9180   0.8938   0.7414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8328, F1=0.8105, Normal Recall=0.7921, Normal Precision=0.9180, Attack Recall=0.8938, Attack Precision=0.7414

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
0.15       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101   <--
0.20       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.25       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.30       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.35       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.40       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.45       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.50       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.55       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.60       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.65       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.70       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.75       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
0.80       0.8421   0.8499   0.7905   0.8816   0.8938   0.8101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8421, F1=0.8499, Normal Recall=0.7905, Normal Precision=0.8816, Attack Recall=0.8938, Attack Precision=0.8101

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
0.15       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241   <--
0.20       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.25       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.30       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.35       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.40       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.45       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.50       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.55       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.60       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.65       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.70       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.75       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.80       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8027, F1=0.4761, Normal Recall=0.7924, Normal Precision=0.9856, Attack Recall=0.8961, Attack Precision=0.3241

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
0.15       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196   <--
0.20       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.25       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.30       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.35       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.40       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.45       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.50       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.55       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.60       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.65       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.70       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.75       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.80       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8135, F1=0.6580, Normal Recall=0.7926, Normal Precision=0.9685, Attack Recall=0.8970, Attack Precision=0.5196

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
0.15       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487   <--
0.20       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.25       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.30       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.35       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.40       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.45       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.50       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.55       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.60       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.65       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.70       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.75       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.80       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8234, F1=0.7529, Normal Recall=0.7918, Normal Precision=0.9472, Attack Recall=0.8970, Attack Precision=0.6487

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
0.15       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421   <--
0.20       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.25       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.30       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.35       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.40       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.45       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.50       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.55       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.60       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.65       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.70       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.75       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.80       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8341, F1=0.8122, Normal Recall=0.7921, Normal Precision=0.9203, Attack Recall=0.8970, Attack Precision=0.7421

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
0.15       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106   <--
0.20       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.25       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.30       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.35       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.40       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.45       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.50       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.55       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.60       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.65       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.70       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.75       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.80       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8437, F1=0.8516, Normal Recall=0.7904, Normal Precision=0.8847, Attack Recall=0.8970, Attack Precision=0.8106

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
0.15       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241   <--
0.20       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.25       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.30       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.35       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.40       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.45       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.50       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.55       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.60       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.65       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.70       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.75       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
0.80       0.8027   0.4761   0.7924   0.9856   0.8961   0.3241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8027, F1=0.4761, Normal Recall=0.7924, Normal Precision=0.9856, Attack Recall=0.8961, Attack Precision=0.3241

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
0.15       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196   <--
0.20       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.25       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.30       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.35       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.40       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.45       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.50       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.55       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.60       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.65       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.70       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.75       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
0.80       0.8135   0.6580   0.7926   0.9685   0.8970   0.5196  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8135, F1=0.6580, Normal Recall=0.7926, Normal Precision=0.9685, Attack Recall=0.8970, Attack Precision=0.5196

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
0.15       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487   <--
0.20       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.25       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.30       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.35       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.40       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.45       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.50       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.55       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.60       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.65       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.70       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.75       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
0.80       0.8234   0.7529   0.7918   0.9472   0.8970   0.6487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8234, F1=0.7529, Normal Recall=0.7918, Normal Precision=0.9472, Attack Recall=0.8970, Attack Precision=0.6487

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
0.15       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421   <--
0.20       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.25       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.30       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.35       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.40       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.45       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.50       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.55       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.60       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.65       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.70       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.75       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
0.80       0.8341   0.8122   0.7921   0.9203   0.8970   0.7421  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8341, F1=0.8122, Normal Recall=0.7921, Normal Precision=0.9203, Attack Recall=0.8970, Attack Precision=0.7421

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
0.15       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106   <--
0.20       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.25       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.30       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.35       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.40       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.45       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.50       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.55       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.60       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.65       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.70       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.75       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
0.80       0.8437   0.8516   0.7904   0.8847   0.8970   0.8106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8437, F1=0.8516, Normal Recall=0.7904, Normal Precision=0.8847, Attack Recall=0.8970, Attack Precision=0.8106

```

