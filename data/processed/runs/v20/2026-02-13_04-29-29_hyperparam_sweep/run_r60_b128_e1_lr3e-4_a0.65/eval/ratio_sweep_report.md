# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-14 08:40:17 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8253 | 0.8243 | 0.8232 | 0.8227 | 0.8210 | 0.8190 | 0.8181 | 0.8163 | 0.8163 | 0.8149 | 0.8138 |
| QAT+Prune only | 0.8131 | 0.8249 | 0.8348 | 0.8447 | 0.8549 | 0.8634 | 0.8756 | 0.8847 | 0.8949 | 0.9036 | 0.9147 |
| QAT+PTQ | 0.8132 | 0.8246 | 0.8342 | 0.8438 | 0.8535 | 0.8619 | 0.8737 | 0.8824 | 0.8923 | 0.9007 | 0.9116 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8132 | 0.8246 | 0.8342 | 0.8438 | 0.8535 | 0.8619 | 0.8737 | 0.8824 | 0.8923 | 0.9007 | 0.9116 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4803 | 0.6480 | 0.7336 | 0.7844 | 0.8180 | 0.8430 | 0.8611 | 0.8763 | 0.8878 | 0.8973 |
| QAT+Prune only | 0.0000 | 0.5108 | 0.6889 | 0.7794 | 0.8345 | 0.8700 | 0.8982 | 0.9174 | 0.9330 | 0.9447 | 0.9555 |
| QAT+PTQ | 0.0000 | 0.5095 | 0.6875 | 0.7779 | 0.8327 | 0.8685 | 0.8965 | 0.9156 | 0.9313 | 0.9429 | 0.9537 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5095 | 0.6875 | 0.7779 | 0.8327 | 0.8685 | 0.8965 | 0.9156 | 0.9313 | 0.9429 | 0.9537 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8253 | 0.8256 | 0.8256 | 0.8265 | 0.8259 | 0.8242 | 0.8245 | 0.8221 | 0.8261 | 0.8248 | 0.0000 |
| QAT+Prune only | 0.8131 | 0.8149 | 0.8148 | 0.8146 | 0.8150 | 0.8120 | 0.8169 | 0.8148 | 0.8157 | 0.8035 | 0.0000 |
| QAT+PTQ | 0.8132 | 0.8150 | 0.8149 | 0.8148 | 0.8148 | 0.8123 | 0.8169 | 0.8143 | 0.8153 | 0.8031 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8132 | 0.8150 | 0.8149 | 0.8148 | 0.8148 | 0.8123 | 0.8169 | 0.8143 | 0.8153 | 0.8031 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8253 | 0.0000 | 0.0000 | 0.0000 | 0.8253 | 1.0000 |
| 90 | 10 | 299,940 | 0.8243 | 0.3410 | 0.8121 | 0.4803 | 0.8256 | 0.9753 |
| 80 | 20 | 291,350 | 0.8232 | 0.5384 | 0.8138 | 0.6480 | 0.8256 | 0.9466 |
| 70 | 30 | 194,230 | 0.8227 | 0.6678 | 0.8138 | 0.7336 | 0.8265 | 0.9119 |
| 60 | 40 | 145,675 | 0.8210 | 0.7570 | 0.8138 | 0.7844 | 0.8259 | 0.8693 |
| 50 | 50 | 116,540 | 0.8190 | 0.8224 | 0.8138 | 0.8180 | 0.8242 | 0.8157 |
| 40 | 60 | 97,115 | 0.8181 | 0.8743 | 0.8138 | 0.8430 | 0.8245 | 0.7470 |
| 30 | 70 | 83,240 | 0.8163 | 0.9143 | 0.8138 | 0.8611 | 0.8221 | 0.6542 |
| 20 | 80 | 72,835 | 0.8163 | 0.9493 | 0.8138 | 0.8763 | 0.8261 | 0.5259 |
| 10 | 90 | 64,740 | 0.8149 | 0.9766 | 0.8138 | 0.8878 | 0.8248 | 0.3298 |
| 0 | 100 | 58,270 | 0.8138 | 1.0000 | 0.8138 | 0.8973 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8131 | 0.0000 | 0.0000 | 0.0000 | 0.8131 | 1.0000 |
| 90 | 10 | 299,940 | 0.8249 | 0.3544 | 0.9143 | 0.5108 | 0.8149 | 0.9884 |
| 80 | 20 | 291,350 | 0.8348 | 0.5526 | 0.9147 | 0.6889 | 0.8148 | 0.9745 |
| 70 | 30 | 194,230 | 0.8447 | 0.6789 | 0.9147 | 0.7794 | 0.8146 | 0.9571 |
| 60 | 40 | 145,675 | 0.8549 | 0.7673 | 0.9147 | 0.8345 | 0.8150 | 0.9348 |
| 50 | 50 | 116,540 | 0.8634 | 0.8295 | 0.9147 | 0.8700 | 0.8120 | 0.9050 |
| 40 | 60 | 97,115 | 0.8756 | 0.8823 | 0.9147 | 0.8982 | 0.8169 | 0.8646 |
| 30 | 70 | 83,240 | 0.8847 | 0.9201 | 0.9147 | 0.9174 | 0.8148 | 0.8037 |
| 20 | 80 | 72,835 | 0.8949 | 0.9520 | 0.9147 | 0.9330 | 0.8157 | 0.7052 |
| 10 | 90 | 64,740 | 0.9036 | 0.9767 | 0.9147 | 0.9447 | 0.8035 | 0.5115 |
| 0 | 100 | 58,270 | 0.9147 | 1.0000 | 0.9147 | 0.9555 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8132 | 0.0000 | 0.0000 | 0.0000 | 0.8132 | 1.0000 |
| 90 | 10 | 299,940 | 0.8246 | 0.3536 | 0.9111 | 0.5095 | 0.8150 | 0.9880 |
| 80 | 20 | 291,350 | 0.8342 | 0.5518 | 0.9116 | 0.6875 | 0.8149 | 0.9736 |
| 70 | 30 | 194,230 | 0.8438 | 0.6784 | 0.9116 | 0.7779 | 0.8148 | 0.9556 |
| 60 | 40 | 145,675 | 0.8535 | 0.7665 | 0.9116 | 0.8327 | 0.8148 | 0.9325 |
| 50 | 50 | 116,540 | 0.8619 | 0.8292 | 0.9116 | 0.8685 | 0.8123 | 0.9018 |
| 40 | 60 | 97,115 | 0.8737 | 0.8819 | 0.9116 | 0.8965 | 0.8169 | 0.8603 |
| 30 | 70 | 83,240 | 0.8824 | 0.9197 | 0.9116 | 0.9156 | 0.8143 | 0.7978 |
| 20 | 80 | 72,835 | 0.8923 | 0.9518 | 0.9116 | 0.9313 | 0.8153 | 0.6975 |
| 10 | 90 | 64,740 | 0.9007 | 0.9766 | 0.9116 | 0.9429 | 0.8031 | 0.5023 |
| 0 | 100 | 58,270 | 0.9116 | 1.0000 | 0.9116 | 0.9537 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8132 | 0.0000 | 0.0000 | 0.0000 | 0.8132 | 1.0000 |
| 90 | 10 | 299,940 | 0.8246 | 0.3536 | 0.9111 | 0.5095 | 0.8150 | 0.9880 |
| 80 | 20 | 291,350 | 0.8342 | 0.5518 | 0.9116 | 0.6875 | 0.8149 | 0.9736 |
| 70 | 30 | 194,230 | 0.8438 | 0.6784 | 0.9116 | 0.7779 | 0.8148 | 0.9556 |
| 60 | 40 | 145,675 | 0.8535 | 0.7665 | 0.9116 | 0.8327 | 0.8148 | 0.9325 |
| 50 | 50 | 116,540 | 0.8619 | 0.8292 | 0.9116 | 0.8685 | 0.8123 | 0.9018 |
| 40 | 60 | 97,115 | 0.8737 | 0.8819 | 0.9116 | 0.8965 | 0.8169 | 0.8603 |
| 30 | 70 | 83,240 | 0.8824 | 0.9197 | 0.9116 | 0.9156 | 0.8143 | 0.7978 |
| 20 | 80 | 72,835 | 0.8923 | 0.9518 | 0.9116 | 0.9313 | 0.8153 | 0.6975 |
| 10 | 90 | 64,740 | 0.9007 | 0.9766 | 0.9116 | 0.9429 | 0.8031 | 0.5023 |
| 0 | 100 | 58,270 | 0.9116 | 1.0000 | 0.9116 | 0.9537 | 0.0000 | 0.0000 |


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
0.15       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409   <--
0.20       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.25       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.30       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.35       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.40       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.45       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.50       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.55       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.60       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.65       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.70       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.75       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
0.80       0.8242   0.4801   0.8256   0.9753   0.8116   0.3409  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8242, F1=0.4801, Normal Recall=0.8256, Normal Precision=0.9753, Attack Recall=0.8116, Attack Precision=0.3409

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
0.15       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389   <--
0.20       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.25       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.30       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.35       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.40       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.45       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.50       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.55       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.60       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.65       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.70       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.75       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
0.80       0.8235   0.6484   0.8259   0.9466   0.8138   0.5389  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8235, F1=0.6484, Normal Recall=0.8259, Normal Precision=0.9466, Attack Recall=0.8138, Attack Precision=0.5389

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
0.15       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672   <--
0.20       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.25       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.30       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.35       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.40       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.45       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.50       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.55       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.60       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.65       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.70       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.75       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
0.80       0.8224   0.7332   0.8260   0.9119   0.8138   0.6672  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8224, F1=0.7332, Normal Recall=0.8260, Normal Precision=0.9119, Attack Recall=0.8138, Attack Precision=0.6672

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
0.15       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564   <--
0.20       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.25       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.30       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.35       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.40       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.45       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.50       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.55       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.60       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.65       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.70       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.75       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
0.80       0.8207   0.7840   0.8253   0.8692   0.8138   0.7564  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8207, F1=0.7840, Normal Recall=0.8253, Normal Precision=0.8692, Attack Recall=0.8138, Attack Precision=0.7564

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
0.15       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226   <--
0.20       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.25       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.30       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.35       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.40       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.45       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.50       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.55       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.60       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.65       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.70       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.75       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
0.80       0.8191   0.8182   0.8245   0.8158   0.8138   0.8226  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8191, F1=0.8182, Normal Recall=0.8245, Normal Precision=0.8158, Attack Recall=0.8138, Attack Precision=0.8226

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
0.15       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546   <--
0.20       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.25       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.30       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.35       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.40       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.45       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.50       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.55       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.60       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.65       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.70       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.75       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
0.80       0.8249   0.5111   0.8149   0.9885   0.9149   0.3546  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8249, F1=0.5111, Normal Recall=0.8149, Normal Precision=0.9885, Attack Recall=0.9149, Attack Precision=0.3546

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
0.15       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534   <--
0.20       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.25       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.30       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.35       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.40       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.45       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.50       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.55       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.60       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.65       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.70       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.75       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
0.80       0.8353   0.6896   0.8155   0.9745   0.9147   0.5534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8353, F1=0.6896, Normal Recall=0.8155, Normal Precision=0.9745, Attack Recall=0.9147, Attack Precision=0.5534

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
0.15       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773   <--
0.20       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.25       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.30       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.35       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.40       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.45       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.50       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.55       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.60       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.65       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.70       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.75       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
0.80       0.8437   0.7783   0.8132   0.9570   0.9147   0.6773  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8437, F1=0.7783, Normal Recall=0.8132, Normal Precision=0.9570, Attack Recall=0.9147, Attack Precision=0.6773

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
0.15       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656   <--
0.20       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.25       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.30       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.35       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.40       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.45       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.50       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.55       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.60       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.65       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.70       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.75       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
0.80       0.8539   0.8336   0.8133   0.9347   0.9147   0.7656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8539, F1=0.8336, Normal Recall=0.8133, Normal Precision=0.9347, Attack Recall=0.9147, Attack Precision=0.7656

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
0.15       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298   <--
0.20       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.25       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.30       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.35       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.40       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.45       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.50       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.55       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.60       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.65       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.70       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.75       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
0.80       0.8636   0.8702   0.8124   0.9050   0.9147   0.8298  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8636, F1=0.8702, Normal Recall=0.8124, Normal Precision=0.9050, Attack Recall=0.9147, Attack Precision=0.8298

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
0.15       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538   <--
0.20       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.25       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.30       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.35       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.40       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.45       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.50       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.55       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.60       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.65       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.70       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.75       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.80       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8247, F1=0.5099, Normal Recall=0.8150, Normal Precision=0.9881, Attack Recall=0.9119, Attack Precision=0.3538

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
0.15       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527   <--
0.20       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.25       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.30       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.35       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.40       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.45       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.50       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.55       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.60       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.65       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.70       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.75       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.80       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8348, F1=0.6881, Normal Recall=0.8155, Normal Precision=0.9736, Attack Recall=0.9116, Attack Precision=0.5527

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
0.15       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767   <--
0.20       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.25       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.30       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.35       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.40       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.45       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.50       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.55       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.60       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.65       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.70       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.75       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.80       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8428, F1=0.7768, Normal Recall=0.8134, Normal Precision=0.9555, Attack Recall=0.9116, Attack Precision=0.6767

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
0.15       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653   <--
0.20       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.25       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.30       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.35       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.40       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.45       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.50       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.55       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.60       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.65       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.70       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.75       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.80       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8528, F1=0.8320, Normal Recall=0.8136, Normal Precision=0.9324, Attack Recall=0.9116, Attack Precision=0.7653

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
0.15       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295   <--
0.20       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.25       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.30       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.35       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.40       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.45       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.50       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.55       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.60       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.65       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.70       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.75       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.80       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8621, F1=0.8686, Normal Recall=0.8127, Normal Precision=0.9019, Attack Recall=0.9116, Attack Precision=0.8295

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
0.15       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538   <--
0.20       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.25       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.30       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.35       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.40       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.45       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.50       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.55       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.60       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.65       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.70       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.75       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
0.80       0.8247   0.5099   0.8150   0.9881   0.9119   0.3538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8247, F1=0.5099, Normal Recall=0.8150, Normal Precision=0.9881, Attack Recall=0.9119, Attack Precision=0.3538

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
0.15       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527   <--
0.20       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.25       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.30       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.35       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.40       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.45       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.50       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.55       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.60       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.65       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.70       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.75       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
0.80       0.8348   0.6881   0.8155   0.9736   0.9116   0.5527  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8348, F1=0.6881, Normal Recall=0.8155, Normal Precision=0.9736, Attack Recall=0.9116, Attack Precision=0.5527

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
0.15       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767   <--
0.20       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.25       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.30       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.35       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.40       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.45       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.50       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.55       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.60       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.65       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.70       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.75       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
0.80       0.8428   0.7768   0.8134   0.9555   0.9116   0.6767  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8428, F1=0.7768, Normal Recall=0.8134, Normal Precision=0.9555, Attack Recall=0.9116, Attack Precision=0.6767

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
0.15       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653   <--
0.20       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.25       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.30       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.35       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.40       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.45       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.50       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.55       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.60       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.65       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.70       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.75       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
0.80       0.8528   0.8320   0.8136   0.9324   0.9116   0.7653  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8528, F1=0.8320, Normal Recall=0.8136, Normal Precision=0.9324, Attack Recall=0.9116, Attack Precision=0.7653

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
0.15       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295   <--
0.20       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.25       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.30       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.35       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.40       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.45       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.50       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.55       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.60       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.65       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.70       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.75       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
0.80       0.8621   0.8686   0.8127   0.9019   0.9116   0.8295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8621, F1=0.8686, Normal Recall=0.8127, Normal Precision=0.9019, Attack Recall=0.9116, Attack Precision=0.8295

```

