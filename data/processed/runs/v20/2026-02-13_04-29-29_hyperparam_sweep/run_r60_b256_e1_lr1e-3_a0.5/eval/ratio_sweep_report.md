# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-15 10:08:51 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6079 | 0.6447 | 0.6833 | 0.7243 | 0.7637 | 0.7999 | 0.8416 | 0.8807 | 0.9187 | 0.9589 | 0.9981 |
| QAT+Prune only | 0.8045 | 0.8214 | 0.8373 | 0.8537 | 0.8701 | 0.8852 | 0.9024 | 0.9175 | 0.9353 | 0.9502 | 0.9671 |
| QAT+PTQ | 0.8036 | 0.8205 | 0.8365 | 0.8530 | 0.8694 | 0.8846 | 0.9020 | 0.9172 | 0.9351 | 0.9502 | 0.9672 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8036 | 0.8205 | 0.8365 | 0.8530 | 0.8694 | 0.8846 | 0.9020 | 0.9172 | 0.9351 | 0.9502 | 0.9672 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3597 | 0.5576 | 0.6848 | 0.7717 | 0.8330 | 0.8832 | 0.9214 | 0.9515 | 0.9777 | 0.9991 |
| QAT+Prune only | 0.0000 | 0.5200 | 0.7040 | 0.7986 | 0.8562 | 0.8939 | 0.9224 | 0.9426 | 0.9598 | 0.9722 | 0.9833 |
| QAT+PTQ | 0.0000 | 0.5187 | 0.7029 | 0.7978 | 0.8556 | 0.8934 | 0.9221 | 0.9424 | 0.9597 | 0.9722 | 0.9833 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5187 | 0.7029 | 0.7978 | 0.8556 | 0.8934 | 0.9221 | 0.9424 | 0.9597 | 0.9722 | 0.9833 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6079 | 0.6054 | 0.6045 | 0.6070 | 0.6075 | 0.6016 | 0.6068 | 0.6067 | 0.6007 | 0.6060 | 0.0000 |
| QAT+Prune only | 0.8045 | 0.8052 | 0.8049 | 0.8051 | 0.8054 | 0.8032 | 0.8054 | 0.8018 | 0.8079 | 0.7987 | 0.0000 |
| QAT+PTQ | 0.8036 | 0.8041 | 0.8038 | 0.8040 | 0.8043 | 0.8021 | 0.8042 | 0.8007 | 0.8067 | 0.7977 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8036 | 0.8041 | 0.8038 | 0.8040 | 0.8043 | 0.8021 | 0.8042 | 0.8007 | 0.8067 | 0.7977 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6079 | 0.0000 | 0.0000 | 0.0000 | 0.6079 | 1.0000 |
| 90 | 10 | 299,940 | 0.6447 | 0.2194 | 0.9982 | 0.3597 | 0.6054 | 0.9997 |
| 80 | 20 | 291,350 | 0.6833 | 0.3869 | 0.9981 | 0.5576 | 0.6045 | 0.9992 |
| 70 | 30 | 194,230 | 0.7243 | 0.5212 | 0.9981 | 0.6848 | 0.6070 | 0.9987 |
| 60 | 40 | 145,675 | 0.7637 | 0.6290 | 0.9981 | 0.7717 | 0.6075 | 0.9980 |
| 50 | 50 | 116,540 | 0.7999 | 0.7147 | 0.9981 | 0.8330 | 0.6016 | 0.9969 |
| 40 | 60 | 97,115 | 0.8416 | 0.7920 | 0.9981 | 0.8832 | 0.6068 | 0.9954 |
| 30 | 70 | 83,240 | 0.8807 | 0.8555 | 0.9981 | 0.9214 | 0.6067 | 0.9929 |
| 20 | 80 | 72,835 | 0.9187 | 0.9091 | 0.9981 | 0.9515 | 0.6007 | 0.9878 |
| 10 | 90 | 64,740 | 0.9589 | 0.9580 | 0.9981 | 0.9777 | 0.6060 | 0.9732 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8045 | 0.0000 | 0.0000 | 0.0000 | 0.8045 | 1.0000 |
| 90 | 10 | 299,940 | 0.8214 | 0.3556 | 0.9674 | 0.5200 | 0.8052 | 0.9955 |
| 80 | 20 | 291,350 | 0.8373 | 0.5534 | 0.9671 | 0.7040 | 0.8049 | 0.9899 |
| 70 | 30 | 194,230 | 0.8537 | 0.6802 | 0.9671 | 0.7986 | 0.8051 | 0.9828 |
| 60 | 40 | 145,675 | 0.8701 | 0.7681 | 0.9671 | 0.8562 | 0.8054 | 0.9735 |
| 50 | 50 | 116,540 | 0.8852 | 0.8309 | 0.9671 | 0.8939 | 0.8032 | 0.9606 |
| 40 | 60 | 97,115 | 0.9024 | 0.8817 | 0.9671 | 0.9224 | 0.8054 | 0.9422 |
| 30 | 70 | 83,240 | 0.9175 | 0.9192 | 0.9671 | 0.9426 | 0.8018 | 0.9125 |
| 20 | 80 | 72,835 | 0.9353 | 0.9527 | 0.9671 | 0.9598 | 0.8079 | 0.8599 |
| 10 | 90 | 64,740 | 0.9502 | 0.9774 | 0.9671 | 0.9722 | 0.7987 | 0.7293 |
| 0 | 100 | 58,270 | 0.9671 | 1.0000 | 0.9671 | 0.9833 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8036 | 0.0000 | 0.0000 | 0.0000 | 0.8036 | 1.0000 |
| 90 | 10 | 299,940 | 0.8205 | 0.3544 | 0.9675 | 0.5187 | 0.8041 | 0.9955 |
| 80 | 20 | 291,350 | 0.8365 | 0.5521 | 0.9672 | 0.7029 | 0.8038 | 0.9899 |
| 70 | 30 | 194,230 | 0.8530 | 0.6790 | 0.9672 | 0.7978 | 0.8040 | 0.9828 |
| 60 | 40 | 145,675 | 0.8694 | 0.7671 | 0.9672 | 0.8556 | 0.8043 | 0.9735 |
| 50 | 50 | 116,540 | 0.8846 | 0.8301 | 0.9672 | 0.8934 | 0.8021 | 0.9607 |
| 40 | 60 | 97,115 | 0.9020 | 0.8811 | 0.9672 | 0.9221 | 0.8042 | 0.9423 |
| 30 | 70 | 83,240 | 0.9172 | 0.9189 | 0.9672 | 0.9424 | 0.8007 | 0.9126 |
| 20 | 80 | 72,835 | 0.9351 | 0.9524 | 0.9672 | 0.9597 | 0.8067 | 0.8600 |
| 10 | 90 | 64,740 | 0.9502 | 0.9773 | 0.9672 | 0.9722 | 0.7977 | 0.7296 |
| 0 | 100 | 58,270 | 0.9672 | 1.0000 | 0.9672 | 0.9833 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8036 | 0.0000 | 0.0000 | 0.0000 | 0.8036 | 1.0000 |
| 90 | 10 | 299,940 | 0.8205 | 0.3544 | 0.9675 | 0.5187 | 0.8041 | 0.9955 |
| 80 | 20 | 291,350 | 0.8365 | 0.5521 | 0.9672 | 0.7029 | 0.8038 | 0.9899 |
| 70 | 30 | 194,230 | 0.8530 | 0.6790 | 0.9672 | 0.7978 | 0.8040 | 0.9828 |
| 60 | 40 | 145,675 | 0.8694 | 0.7671 | 0.9672 | 0.8556 | 0.8043 | 0.9735 |
| 50 | 50 | 116,540 | 0.8846 | 0.8301 | 0.9672 | 0.8934 | 0.8021 | 0.9607 |
| 40 | 60 | 97,115 | 0.9020 | 0.8811 | 0.9672 | 0.9221 | 0.8042 | 0.9423 |
| 30 | 70 | 83,240 | 0.9172 | 0.9189 | 0.9672 | 0.9424 | 0.8007 | 0.9126 |
| 20 | 80 | 72,835 | 0.9351 | 0.9524 | 0.9672 | 0.9597 | 0.8067 | 0.8600 |
| 10 | 90 | 64,740 | 0.9502 | 0.9773 | 0.9672 | 0.9722 | 0.7977 | 0.7296 |
| 0 | 100 | 58,270 | 0.9672 | 1.0000 | 0.9672 | 0.9833 | 0.0000 | 0.0000 |


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
0.15       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194   <--
0.20       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.25       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.30       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.35       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.40       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.45       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.50       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.55       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.60       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.65       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.70       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.75       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
0.80       0.6447   0.3598   0.6054   0.9997   0.9983   0.2194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6447, F1=0.3598, Normal Recall=0.6054, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2194

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
0.15       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877   <--
0.20       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.25       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.30       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.35       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.40       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.45       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.50       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.55       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.60       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.65       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.70       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.75       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
0.80       0.6844   0.5585   0.6059   0.9992   0.9981   0.3877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6844, F1=0.5585, Normal Recall=0.6059, Normal Precision=0.9992, Attack Recall=0.9981, Attack Precision=0.3877

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
0.15       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219   <--
0.20       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.25       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.30       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.35       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.40       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.45       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.50       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.55       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.60       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.65       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.70       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.75       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
0.80       0.7252   0.6854   0.6082   0.9987   0.9981   0.5219  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7252, F1=0.6854, Normal Recall=0.6082, Normal Precision=0.9987, Attack Recall=0.9981, Attack Precision=0.5219

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
0.15       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286   <--
0.20       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.25       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.30       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.35       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.40       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.45       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.50       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.55       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.60       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.65       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.70       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.75       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
0.80       0.7633   0.7714   0.6068   0.9980   0.9981   0.6286  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7633, F1=0.7714, Normal Recall=0.6068, Normal Precision=0.9980, Attack Recall=0.9981, Attack Precision=0.6286

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
0.15       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174   <--
0.20       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.25       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.30       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.35       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.40       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.45       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.50       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.55       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.60       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.65       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.70       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.75       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
0.80       0.8025   0.8348   0.6068   0.9970   0.9981   0.7174  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8025, F1=0.8348, Normal Recall=0.6068, Normal Precision=0.9970, Attack Recall=0.9981, Attack Precision=0.7174

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
0.15       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556   <--
0.20       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.25       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.30       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.35       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.40       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.45       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.50       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.55       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.60       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.65       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.70       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.75       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
0.80       0.8215   0.5201   0.8052   0.9955   0.9674   0.3556  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8215, F1=0.5201, Normal Recall=0.8052, Normal Precision=0.9955, Attack Recall=0.9674, Attack Precision=0.3556

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
0.15       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543   <--
0.20       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.25       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.30       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.35       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.40       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.45       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.50       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.55       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.60       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.65       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.70       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.75       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
0.80       0.8379   0.7047   0.8056   0.9899   0.9671   0.5543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8379, F1=0.7047, Normal Recall=0.8056, Normal Precision=0.9899, Attack Recall=0.9671, Attack Precision=0.5543

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
0.15       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794   <--
0.20       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.25       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.30       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.35       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.40       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.45       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.50       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.55       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.60       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.65       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.70       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.75       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
0.80       0.8532   0.7981   0.8045   0.9828   0.9671   0.6794  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8532, F1=0.7981, Normal Recall=0.8045, Normal Precision=0.9828, Attack Recall=0.9671, Attack Precision=0.6794

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
0.15       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670   <--
0.20       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.25       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.30       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.35       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.40       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.45       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.50       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.55       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.60       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.65       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.70       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.75       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
0.80       0.8693   0.8555   0.8042   0.9734   0.9671   0.7670  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8693, F1=0.8555, Normal Recall=0.8042, Normal Precision=0.9734, Attack Recall=0.9671, Attack Precision=0.7670

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
0.15       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307   <--
0.20       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.25       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.30       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.35       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.40       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.45       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.50       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.55       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.60       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.65       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.70       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.75       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
0.80       0.8850   0.8937   0.8029   0.9606   0.9671   0.8307  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8850, F1=0.8937, Normal Recall=0.8029, Normal Precision=0.9606, Attack Recall=0.9671, Attack Precision=0.8307

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
0.15       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543   <--
0.20       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.25       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.30       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.35       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.40       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.45       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.50       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.55       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.60       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.65       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.70       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.75       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.80       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8205, F1=0.5187, Normal Recall=0.8041, Normal Precision=0.9955, Attack Recall=0.9674, Attack Precision=0.3543

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
0.15       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530   <--
0.20       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.25       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.30       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.35       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.40       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.45       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.50       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.55       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.60       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.65       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.70       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.75       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.80       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8371, F1=0.7036, Normal Recall=0.8045, Normal Precision=0.9899, Attack Recall=0.9672, Attack Precision=0.5530

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
0.15       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783   <--
0.20       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.25       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.30       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.35       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.40       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.45       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.50       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.55       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.60       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.65       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.70       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.75       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.80       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8526, F1=0.7974, Normal Recall=0.8034, Normal Precision=0.9828, Attack Recall=0.9672, Attack Precision=0.6783

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
0.15       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661   <--
0.20       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.25       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.30       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.35       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.40       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.45       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.50       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.55       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.60       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.65       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.70       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.75       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.80       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8688, F1=0.8550, Normal Recall=0.8032, Normal Precision=0.9735, Attack Recall=0.9672, Attack Precision=0.7661

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
0.15       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299   <--
0.20       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.25       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.30       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.35       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.40       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.45       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.50       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.55       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.60       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.65       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.70       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.75       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.80       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8845, F1=0.8933, Normal Recall=0.8018, Normal Precision=0.9606, Attack Recall=0.9672, Attack Precision=0.8299

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
0.15       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543   <--
0.20       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.25       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.30       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.35       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.40       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.45       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.50       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.55       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.60       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.65       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.70       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.75       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
0.80       0.8205   0.5187   0.8041   0.9955   0.9674   0.3543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8205, F1=0.5187, Normal Recall=0.8041, Normal Precision=0.9955, Attack Recall=0.9674, Attack Precision=0.3543

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
0.15       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530   <--
0.20       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.25       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.30       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.35       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.40       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.45       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.50       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.55       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.60       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.65       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.70       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.75       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
0.80       0.8371   0.7036   0.8045   0.9899   0.9672   0.5530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8371, F1=0.7036, Normal Recall=0.8045, Normal Precision=0.9899, Attack Recall=0.9672, Attack Precision=0.5530

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
0.15       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783   <--
0.20       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.25       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.30       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.35       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.40       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.45       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.50       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.55       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.60       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.65       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.70       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.75       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
0.80       0.8526   0.7974   0.8034   0.9828   0.9672   0.6783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8526, F1=0.7974, Normal Recall=0.8034, Normal Precision=0.9828, Attack Recall=0.9672, Attack Precision=0.6783

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
0.15       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661   <--
0.20       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.25       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.30       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.35       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.40       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.45       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.50       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.55       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.60       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.65       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.70       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.75       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
0.80       0.8688   0.8550   0.8032   0.9735   0.9672   0.7661  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8688, F1=0.8550, Normal Recall=0.8032, Normal Precision=0.9735, Attack Recall=0.9672, Attack Precision=0.7661

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
0.15       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299   <--
0.20       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.25       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.30       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.35       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.40       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.45       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.50       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.55       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.60       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.65       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.70       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.75       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
0.80       0.8845   0.8933   0.8018   0.9606   0.9672   0.8299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8845, F1=0.8933, Normal Recall=0.8018, Normal Precision=0.9606, Attack Recall=0.9672, Attack Precision=0.8299

```

