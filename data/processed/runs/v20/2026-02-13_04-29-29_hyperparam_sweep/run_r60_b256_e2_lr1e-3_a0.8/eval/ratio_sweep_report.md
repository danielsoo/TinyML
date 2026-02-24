# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-15 23:04:16 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5084 | 0.5281 | 0.5470 | 0.5657 | 0.5844 | 0.6036 | 0.6217 | 0.6402 | 0.6598 | 0.6788 | 0.6972 |
| QAT+Prune only | 0.7904 | 0.8113 | 0.8319 | 0.8534 | 0.8740 | 0.8940 | 0.9155 | 0.9365 | 0.9567 | 0.9778 | 0.9990 |
| QAT+PTQ | 0.7878 | 0.8090 | 0.8298 | 0.8516 | 0.8725 | 0.8928 | 0.9145 | 0.9357 | 0.9563 | 0.9776 | 0.9990 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7878 | 0.8090 | 0.8298 | 0.8516 | 0.8725 | 0.8928 | 0.9145 | 0.9357 | 0.9563 | 0.9776 | 0.9990 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2281 | 0.3810 | 0.4906 | 0.5730 | 0.6375 | 0.6886 | 0.7307 | 0.7663 | 0.7962 | 0.8216 |
| QAT+Prune only | 0.0000 | 0.5143 | 0.7038 | 0.8035 | 0.8638 | 0.9041 | 0.9342 | 0.9566 | 0.9736 | 0.9878 | 0.9995 |
| QAT+PTQ | 0.0000 | 0.5112 | 0.7013 | 0.8016 | 0.8624 | 0.9031 | 0.9334 | 0.9561 | 0.9734 | 0.9877 | 0.9995 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5112 | 0.7013 | 0.8016 | 0.8624 | 0.9031 | 0.9334 | 0.9561 | 0.9734 | 0.9877 | 0.9995 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5084 | 0.5093 | 0.5094 | 0.5093 | 0.5093 | 0.5100 | 0.5085 | 0.5074 | 0.5101 | 0.5130 | 0.0000 |
| QAT+Prune only | 0.7904 | 0.7905 | 0.7901 | 0.7911 | 0.7908 | 0.7890 | 0.7904 | 0.7908 | 0.7879 | 0.7878 | 0.0000 |
| QAT+PTQ | 0.7878 | 0.7879 | 0.7875 | 0.7885 | 0.7883 | 0.7866 | 0.7879 | 0.7882 | 0.7855 | 0.7854 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7878 | 0.7879 | 0.7875 | 0.7885 | 0.7883 | 0.7866 | 0.7879 | 0.7882 | 0.7855 | 0.7854 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5084 | 0.0000 | 0.0000 | 0.0000 | 0.5084 | 1.0000 |
| 90 | 10 | 299,940 | 0.5281 | 0.1363 | 0.6971 | 0.2281 | 0.5093 | 0.9380 |
| 80 | 20 | 291,350 | 0.5470 | 0.2622 | 0.6972 | 0.3810 | 0.5094 | 0.8706 |
| 70 | 30 | 194,230 | 0.5657 | 0.3785 | 0.6972 | 0.4906 | 0.5093 | 0.7969 |
| 60 | 40 | 145,675 | 0.5844 | 0.4864 | 0.6972 | 0.5730 | 0.5093 | 0.7161 |
| 50 | 50 | 116,540 | 0.6036 | 0.5873 | 0.6972 | 0.6375 | 0.5100 | 0.6275 |
| 40 | 60 | 97,115 | 0.6217 | 0.6803 | 0.6972 | 0.6886 | 0.5085 | 0.5282 |
| 30 | 70 | 83,240 | 0.6402 | 0.7676 | 0.6972 | 0.7307 | 0.5074 | 0.4180 |
| 20 | 80 | 72,835 | 0.6598 | 0.8506 | 0.6972 | 0.7663 | 0.5101 | 0.2964 |
| 10 | 90 | 64,740 | 0.6788 | 0.9280 | 0.6972 | 0.7962 | 0.5130 | 0.1584 |
| 0 | 100 | 58,270 | 0.6972 | 1.0000 | 0.6972 | 0.8216 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7904 | 0.0000 | 0.0000 | 0.0000 | 0.7904 | 1.0000 |
| 90 | 10 | 299,940 | 0.8113 | 0.3463 | 0.9990 | 0.5143 | 0.7905 | 0.9999 |
| 80 | 20 | 291,350 | 0.8319 | 0.5433 | 0.9990 | 0.7038 | 0.7901 | 0.9997 |
| 70 | 30 | 194,230 | 0.8534 | 0.6721 | 0.9990 | 0.8035 | 0.7911 | 0.9994 |
| 60 | 40 | 145,675 | 0.8740 | 0.7609 | 0.9990 | 0.8638 | 0.7908 | 0.9991 |
| 50 | 50 | 116,540 | 0.8940 | 0.8256 | 0.9990 | 0.9041 | 0.7890 | 0.9987 |
| 40 | 60 | 97,115 | 0.9155 | 0.8773 | 0.9990 | 0.9342 | 0.7904 | 0.9980 |
| 30 | 70 | 83,240 | 0.9365 | 0.9177 | 0.9990 | 0.9566 | 0.7908 | 0.9969 |
| 20 | 80 | 72,835 | 0.9567 | 0.9496 | 0.9990 | 0.9736 | 0.7879 | 0.9947 |
| 10 | 90 | 64,740 | 0.9778 | 0.9769 | 0.9990 | 0.9878 | 0.7878 | 0.9882 |
| 0 | 100 | 58,270 | 0.9990 | 1.0000 | 0.9990 | 0.9995 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7878 | 0.0000 | 0.0000 | 0.0000 | 0.7878 | 1.0000 |
| 90 | 10 | 299,940 | 0.8090 | 0.3435 | 0.9990 | 0.5112 | 0.7879 | 0.9999 |
| 80 | 20 | 291,350 | 0.8298 | 0.5403 | 0.9990 | 0.7013 | 0.7875 | 0.9997 |
| 70 | 30 | 194,230 | 0.8516 | 0.6693 | 0.9990 | 0.8016 | 0.7885 | 0.9994 |
| 60 | 40 | 145,675 | 0.8725 | 0.7588 | 0.9990 | 0.8624 | 0.7883 | 0.9991 |
| 50 | 50 | 116,540 | 0.8928 | 0.8240 | 0.9990 | 0.9031 | 0.7866 | 0.9987 |
| 40 | 60 | 97,115 | 0.9145 | 0.8760 | 0.9990 | 0.9334 | 0.7879 | 0.9980 |
| 30 | 70 | 83,240 | 0.9357 | 0.9167 | 0.9990 | 0.9561 | 0.7882 | 0.9969 |
| 20 | 80 | 72,835 | 0.9563 | 0.9491 | 0.9990 | 0.9734 | 0.7855 | 0.9947 |
| 10 | 90 | 64,740 | 0.9776 | 0.9767 | 0.9990 | 0.9877 | 0.7854 | 0.9881 |
| 0 | 100 | 58,270 | 0.9990 | 1.0000 | 0.9990 | 0.9995 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7878 | 0.0000 | 0.0000 | 0.0000 | 0.7878 | 1.0000 |
| 90 | 10 | 299,940 | 0.8090 | 0.3435 | 0.9990 | 0.5112 | 0.7879 | 0.9999 |
| 80 | 20 | 291,350 | 0.8298 | 0.5403 | 0.9990 | 0.7013 | 0.7875 | 0.9997 |
| 70 | 30 | 194,230 | 0.8516 | 0.6693 | 0.9990 | 0.8016 | 0.7885 | 0.9994 |
| 60 | 40 | 145,675 | 0.8725 | 0.7588 | 0.9990 | 0.8624 | 0.7883 | 0.9991 |
| 50 | 50 | 116,540 | 0.8928 | 0.8240 | 0.9990 | 0.9031 | 0.7866 | 0.9987 |
| 40 | 60 | 97,115 | 0.9145 | 0.8760 | 0.9990 | 0.9334 | 0.7879 | 0.9980 |
| 30 | 70 | 83,240 | 0.9357 | 0.9167 | 0.9990 | 0.9561 | 0.7882 | 0.9969 |
| 20 | 80 | 72,835 | 0.9563 | 0.9491 | 0.9990 | 0.9734 | 0.7855 | 0.9947 |
| 10 | 90 | 64,740 | 0.9776 | 0.9767 | 0.9990 | 0.9877 | 0.7854 | 0.9881 |
| 0 | 100 | 58,270 | 0.9990 | 1.0000 | 0.9990 | 0.9995 | 0.0000 | 0.0000 |


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
0.15       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361   <--
0.20       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.25       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.30       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.35       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.40       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.45       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.50       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.55       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.60       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.65       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.70       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.75       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
0.80       0.5280   0.2277   0.5093   0.9378   0.6958   0.1361  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5280, F1=0.2277, Normal Recall=0.5093, Normal Precision=0.9378, Attack Recall=0.6958, Attack Precision=0.1361

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
0.15       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623   <--
0.20       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.25       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.30       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.35       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.40       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.45       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.50       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.55       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.60       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.65       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.70       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.75       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
0.80       0.5472   0.3812   0.5097   0.8707   0.6972   0.2623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5472, F1=0.3812, Normal Recall=0.5097, Normal Precision=0.8707, Attack Recall=0.6972, Attack Precision=0.2623

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
0.15       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789   <--
0.20       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.25       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.30       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.35       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.40       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.45       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.50       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.55       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.60       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.65       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.70       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.75       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
0.80       0.5663   0.4910   0.5102   0.7972   0.6972   0.3789  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5663, F1=0.4910, Normal Recall=0.5102, Normal Precision=0.7972, Attack Recall=0.6972, Attack Precision=0.3789

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
0.15       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861   <--
0.20       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.25       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.30       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.35       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.40       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.45       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.50       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.55       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.60       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.65       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.70       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.75       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
0.80       0.5841   0.5728   0.5086   0.7159   0.6972   0.4861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5841, F1=0.5728, Normal Recall=0.5086, Normal Precision=0.7159, Attack Recall=0.6972, Attack Precision=0.4861

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
0.15       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856   <--
0.20       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.25       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.30       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.35       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.40       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.45       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.50       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.55       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.60       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.65       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.70       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.75       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
0.80       0.6019   0.6366   0.5067   0.6259   0.6972   0.5856  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6019, F1=0.6366, Normal Recall=0.5067, Normal Precision=0.6259, Attack Recall=0.6972, Attack Precision=0.5856

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
0.15       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463   <--
0.20       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.25       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.30       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.35       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.40       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.45       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.50       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.55       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.60       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.65       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.70       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.75       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
0.80       0.8113   0.5143   0.7905   0.9999   0.9990   0.3463  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8113, F1=0.5143, Normal Recall=0.7905, Normal Precision=0.9999, Attack Recall=0.9990, Attack Precision=0.3463

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
0.15       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445   <--
0.20       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.25       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.30       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.35       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.40       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.45       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.50       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.55       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.60       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.65       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.70       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.75       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
0.80       0.8327   0.7049   0.7911   0.9997   0.9990   0.5445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8327, F1=0.7049, Normal Recall=0.7911, Normal Precision=0.9997, Attack Recall=0.9990, Attack Precision=0.5445

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
0.15       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720   <--
0.20       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.25       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.30       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.35       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.40       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.45       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.50       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.55       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.60       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.65       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.70       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.75       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
0.80       0.8534   0.8035   0.7910   0.9994   0.9990   0.6720  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8534, F1=0.8035, Normal Recall=0.7910, Normal Precision=0.9994, Attack Recall=0.9990, Attack Precision=0.6720

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
0.15       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607   <--
0.20       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.25       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.30       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.35       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.40       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.45       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.50       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.55       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.60       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.65       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.70       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.75       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
0.80       0.8739   0.8637   0.7905   0.9991   0.9990   0.7607  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8739, F1=0.8637, Normal Recall=0.7905, Normal Precision=0.9991, Attack Recall=0.9990, Attack Precision=0.7607

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
0.15       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262   <--
0.20       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.25       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.30       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.35       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.40       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.45       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.50       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.55       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.60       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.65       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.70       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.75       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
0.80       0.8944   0.9044   0.7899   0.9987   0.9990   0.8262  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8944, F1=0.9044, Normal Recall=0.7899, Normal Precision=0.9987, Attack Recall=0.9990, Attack Precision=0.8262

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
0.15       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435   <--
0.20       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.25       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.30       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.35       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.40       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.45       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.50       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.55       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.60       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.65       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.70       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.75       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.80       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8090, F1=0.5112, Normal Recall=0.7879, Normal Precision=0.9999, Attack Recall=0.9990, Attack Precision=0.3435

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
0.15       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415   <--
0.20       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.25       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.30       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.35       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.40       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.45       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.50       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.55       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.60       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.65       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.70       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.75       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.80       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8306, F1=0.7023, Normal Recall=0.7885, Normal Precision=0.9997, Attack Recall=0.9990, Attack Precision=0.5415

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
0.15       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693   <--
0.20       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.25       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.30       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.35       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.40       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.45       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.50       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.55       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.60       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.65       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.70       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.75       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.80       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8516, F1=0.8016, Normal Recall=0.7885, Normal Precision=0.9994, Attack Recall=0.9990, Attack Precision=0.6693

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
0.15       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586   <--
0.20       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.25       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.30       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.35       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.40       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.45       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.50       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.55       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.60       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.65       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.70       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.75       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.80       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8724, F1=0.8623, Normal Recall=0.7881, Normal Precision=0.9991, Attack Recall=0.9990, Attack Precision=0.7586

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
0.15       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246   <--
0.20       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.25       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.30       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.35       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.40       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.45       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.50       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.55       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.60       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.65       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.70       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.75       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.80       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8932, F1=0.9034, Normal Recall=0.7875, Normal Precision=0.9987, Attack Recall=0.9990, Attack Precision=0.8246

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
0.15       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435   <--
0.20       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.25       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.30       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.35       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.40       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.45       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.50       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.55       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.60       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.65       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.70       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.75       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
0.80       0.8090   0.5112   0.7879   0.9999   0.9990   0.3435  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8090, F1=0.5112, Normal Recall=0.7879, Normal Precision=0.9999, Attack Recall=0.9990, Attack Precision=0.3435

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
0.15       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415   <--
0.20       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.25       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.30       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.35       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.40       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.45       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.50       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.55       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.60       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.65       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.70       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.75       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
0.80       0.8306   0.7023   0.7885   0.9997   0.9990   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8306, F1=0.7023, Normal Recall=0.7885, Normal Precision=0.9997, Attack Recall=0.9990, Attack Precision=0.5415

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
0.15       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693   <--
0.20       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.25       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.30       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.35       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.40       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.45       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.50       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.55       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.60       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.65       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.70       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.75       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
0.80       0.8516   0.8016   0.7885   0.9994   0.9990   0.6693  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8516, F1=0.8016, Normal Recall=0.7885, Normal Precision=0.9994, Attack Recall=0.9990, Attack Precision=0.6693

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
0.15       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586   <--
0.20       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.25       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.30       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.35       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.40       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.45       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.50       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.55       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.60       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.65       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.70       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.75       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
0.80       0.8724   0.8623   0.7881   0.9991   0.9990   0.7586  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8724, F1=0.8623, Normal Recall=0.7881, Normal Precision=0.9991, Attack Recall=0.9990, Attack Precision=0.7586

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
0.15       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246   <--
0.20       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.25       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.30       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.35       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.40       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.45       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.50       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.55       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.60       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.65       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.70       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.75       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
0.80       0.8932   0.9034   0.7875   0.9987   0.9990   0.8246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8932, F1=0.9034, Normal Recall=0.7875, Normal Precision=0.9987, Attack Recall=0.9990, Attack Precision=0.8246

```

