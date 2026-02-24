# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-18 06:01:41 |

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
| Original (TFLite) | 0.1434 | 0.2278 | 0.3136 | 0.3995 | 0.4858 | 0.5707 | 0.6564 | 0.7430 | 0.8278 | 0.9143 | 0.9999 |
| QAT+Prune only | 0.7774 | 0.7979 | 0.8179 | 0.8399 | 0.8603 | 0.8796 | 0.9018 | 0.9228 | 0.9430 | 0.9637 | 0.9849 |
| QAT+PTQ | 0.7764 | 0.7970 | 0.8171 | 0.8392 | 0.8598 | 0.8791 | 0.9014 | 0.9226 | 0.9429 | 0.9636 | 0.9850 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7764 | 0.7970 | 0.8171 | 0.8392 | 0.8598 | 0.8791 | 0.9014 | 0.9226 | 0.9429 | 0.9636 | 0.9850 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2057 | 0.3682 | 0.4998 | 0.6087 | 0.6996 | 0.7774 | 0.8449 | 0.9028 | 0.9545 | 0.9999 |
| QAT+Prune only | 0.0000 | 0.4935 | 0.6839 | 0.7869 | 0.8494 | 0.8911 | 0.9233 | 0.9470 | 0.9651 | 0.9799 | 0.9924 |
| QAT+PTQ | 0.0000 | 0.4924 | 0.6829 | 0.7861 | 0.8490 | 0.8907 | 0.9230 | 0.9469 | 0.9650 | 0.9799 | 0.9924 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4924 | 0.6829 | 0.7861 | 0.8490 | 0.8907 | 0.9230 | 0.9469 | 0.9650 | 0.9799 | 0.9924 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1434 | 0.1420 | 0.1420 | 0.1422 | 0.1431 | 0.1416 | 0.1412 | 0.1435 | 0.1393 | 0.1440 | 0.0000 |
| QAT+Prune only | 0.7774 | 0.7772 | 0.7762 | 0.7778 | 0.7772 | 0.7743 | 0.7772 | 0.7779 | 0.7753 | 0.7723 | 0.0000 |
| QAT+PTQ | 0.7764 | 0.7762 | 0.7751 | 0.7766 | 0.7764 | 0.7733 | 0.7761 | 0.7770 | 0.7744 | 0.7705 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7764 | 0.7762 | 0.7751 | 0.7766 | 0.7764 | 0.7733 | 0.7761 | 0.7770 | 0.7744 | 0.7705 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1434 | 0.0000 | 0.0000 | 0.0000 | 0.1434 | 1.0000 |
| 90 | 10 | 299,940 | 0.2278 | 0.1146 | 0.9999 | 0.2057 | 0.1420 | 0.9999 |
| 80 | 20 | 291,350 | 0.3136 | 0.2256 | 0.9999 | 0.3682 | 0.1420 | 0.9998 |
| 70 | 30 | 194,230 | 0.3995 | 0.3331 | 0.9999 | 0.4998 | 0.1422 | 0.9996 |
| 60 | 40 | 145,675 | 0.4858 | 0.4375 | 0.9999 | 0.6087 | 0.1431 | 0.9994 |
| 50 | 50 | 116,540 | 0.5707 | 0.5381 | 0.9999 | 0.6996 | 0.1416 | 0.9992 |
| 40 | 60 | 97,115 | 0.6564 | 0.6359 | 0.9999 | 0.7774 | 0.1412 | 0.9987 |
| 30 | 70 | 83,240 | 0.7430 | 0.7315 | 0.9999 | 0.8449 | 0.1435 | 0.9981 |
| 20 | 80 | 72,835 | 0.8278 | 0.8229 | 0.9999 | 0.9028 | 0.1393 | 0.9966 |
| 10 | 90 | 64,740 | 0.9143 | 0.9131 | 0.9999 | 0.9545 | 0.1440 | 0.9925 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 0.9999 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7774 | 0.0000 | 0.0000 | 0.0000 | 0.7774 | 1.0000 |
| 90 | 10 | 299,940 | 0.7979 | 0.3293 | 0.9844 | 0.4935 | 0.7772 | 0.9978 |
| 80 | 20 | 291,350 | 0.8179 | 0.5238 | 0.9849 | 0.6839 | 0.7762 | 0.9952 |
| 70 | 30 | 194,230 | 0.8399 | 0.6551 | 0.9849 | 0.7869 | 0.7778 | 0.9918 |
| 60 | 40 | 145,675 | 0.8603 | 0.7467 | 0.9849 | 0.8494 | 0.7772 | 0.9872 |
| 50 | 50 | 116,540 | 0.8796 | 0.8135 | 0.9849 | 0.8911 | 0.7743 | 0.9809 |
| 40 | 60 | 97,115 | 0.9018 | 0.8689 | 0.9849 | 0.9233 | 0.7772 | 0.9717 |
| 30 | 70 | 83,240 | 0.9228 | 0.9119 | 0.9849 | 0.9470 | 0.7779 | 0.9568 |
| 20 | 80 | 72,835 | 0.9430 | 0.9460 | 0.9849 | 0.9651 | 0.7753 | 0.9279 |
| 10 | 90 | 64,740 | 0.9637 | 0.9750 | 0.9849 | 0.9799 | 0.7723 | 0.8508 |
| 0 | 100 | 58,270 | 0.9849 | 1.0000 | 0.9849 | 0.9924 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7764 | 0.0000 | 0.0000 | 0.0000 | 0.7764 | 1.0000 |
| 90 | 10 | 299,940 | 0.7970 | 0.3283 | 0.9845 | 0.4924 | 0.7762 | 0.9978 |
| 80 | 20 | 291,350 | 0.8171 | 0.5227 | 0.9850 | 0.6829 | 0.7751 | 0.9952 |
| 70 | 30 | 194,230 | 0.8392 | 0.6540 | 0.9850 | 0.7861 | 0.7766 | 0.9918 |
| 60 | 40 | 145,675 | 0.8598 | 0.7460 | 0.9850 | 0.8490 | 0.7764 | 0.9873 |
| 50 | 50 | 116,540 | 0.8791 | 0.8129 | 0.9850 | 0.8907 | 0.7733 | 0.9810 |
| 40 | 60 | 97,115 | 0.9014 | 0.8684 | 0.9850 | 0.9230 | 0.7761 | 0.9718 |
| 30 | 70 | 83,240 | 0.9226 | 0.9116 | 0.9850 | 0.9469 | 0.7770 | 0.9569 |
| 20 | 80 | 72,835 | 0.9429 | 0.9458 | 0.9850 | 0.9650 | 0.7744 | 0.9281 |
| 10 | 90 | 64,740 | 0.9636 | 0.9748 | 0.9850 | 0.9799 | 0.7705 | 0.8510 |
| 0 | 100 | 58,270 | 0.9850 | 1.0000 | 0.9850 | 0.9924 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7764 | 0.0000 | 0.0000 | 0.0000 | 0.7764 | 1.0000 |
| 90 | 10 | 299,940 | 0.7970 | 0.3283 | 0.9845 | 0.4924 | 0.7762 | 0.9978 |
| 80 | 20 | 291,350 | 0.8171 | 0.5227 | 0.9850 | 0.6829 | 0.7751 | 0.9952 |
| 70 | 30 | 194,230 | 0.8392 | 0.6540 | 0.9850 | 0.7861 | 0.7766 | 0.9918 |
| 60 | 40 | 145,675 | 0.8598 | 0.7460 | 0.9850 | 0.8490 | 0.7764 | 0.9873 |
| 50 | 50 | 116,540 | 0.8791 | 0.8129 | 0.9850 | 0.8907 | 0.7733 | 0.9810 |
| 40 | 60 | 97,115 | 0.9014 | 0.8684 | 0.9850 | 0.9230 | 0.7761 | 0.9718 |
| 30 | 70 | 83,240 | 0.9226 | 0.9116 | 0.9850 | 0.9469 | 0.7770 | 0.9569 |
| 20 | 80 | 72,835 | 0.9429 | 0.9458 | 0.9850 | 0.9650 | 0.7744 | 0.9281 |
| 10 | 90 | 64,740 | 0.9636 | 0.9748 | 0.9850 | 0.9799 | 0.7705 | 0.8510 |
| 0 | 100 | 58,270 | 0.9850 | 1.0000 | 0.9850 | 0.9924 | 0.0000 | 0.0000 |


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
0.15       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146   <--
0.20       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.25       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.30       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.35       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.40       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.45       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.50       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.55       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.60       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.65       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.70       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.75       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
0.80       0.2278   0.2057   0.1420   0.9999   0.9999   0.1146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2278, F1=0.2057, Normal Recall=0.1420, Normal Precision=0.9999, Attack Recall=0.9999, Attack Precision=0.1146

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
0.15       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256   <--
0.20       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.25       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.30       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.35       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.40       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.45       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.50       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.55       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.60       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.65       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.70       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.75       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
0.80       0.3136   0.3682   0.1421   0.9998   0.9999   0.2256  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3136, F1=0.3682, Normal Recall=0.1421, Normal Precision=0.9998, Attack Recall=0.9999, Attack Precision=0.2256

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
0.15       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334   <--
0.20       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.25       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.30       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.35       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.40       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.45       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.50       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.55       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.60       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.65       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.70       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.75       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
0.80       0.4002   0.5000   0.1432   0.9996   0.9999   0.3334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4002, F1=0.5000, Normal Recall=0.1432, Normal Precision=0.9996, Attack Recall=0.9999, Attack Precision=0.3334

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
0.15       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377   <--
0.20       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.25       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.30       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.35       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.40       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.45       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.50       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.55       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.60       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.65       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.70       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.75       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
0.80       0.4861   0.6088   0.1436   0.9994   0.9999   0.4377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4861, F1=0.6088, Normal Recall=0.1436, Normal Precision=0.9994, Attack Recall=0.9999, Attack Precision=0.4377

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
0.15       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385   <--
0.20       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.25       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.30       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.35       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.40       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.45       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.50       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.55       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.60       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.65       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.70       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.75       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
0.80       0.5715   0.7000   0.1431   0.9992   0.9999   0.5385  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5715, F1=0.7000, Normal Recall=0.1431, Normal Precision=0.9992, Attack Recall=0.9999, Attack Precision=0.5385

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
0.15       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294   <--
0.20       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.25       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.30       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.35       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.40       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.45       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.50       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.55       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.60       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.65       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.70       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.75       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
0.80       0.7980   0.4936   0.7772   0.9978   0.9848   0.3294  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7980, F1=0.4936, Normal Recall=0.7772, Normal Precision=0.9978, Attack Recall=0.9848, Attack Precision=0.3294

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
0.15       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257   <--
0.20       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.25       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.30       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.35       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.40       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.45       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.50       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.55       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.60       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.65       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.70       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.75       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
0.80       0.8193   0.6855   0.7779   0.9952   0.9849   0.5257  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8193, F1=0.6855, Normal Recall=0.7779, Normal Precision=0.9952, Attack Recall=0.9849, Attack Precision=0.5257

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
0.15       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549   <--
0.20       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.25       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.30       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.35       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.40       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.45       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.50       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.55       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.60       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.65       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.70       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.75       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
0.80       0.8397   0.7867   0.7775   0.9918   0.9849   0.6549  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8397, F1=0.7867, Normal Recall=0.7775, Normal Precision=0.9918, Attack Recall=0.9849, Attack Precision=0.6549

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
0.15       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469   <--
0.20       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.25       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.30       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.35       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.40       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.45       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.50       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.55       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.60       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.65       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.70       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.75       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
0.80       0.8604   0.8495   0.7774   0.9872   0.9849   0.7469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8604, F1=0.8495, Normal Recall=0.7774, Normal Precision=0.9872, Attack Recall=0.9849, Attack Precision=0.7469

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
0.15       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146   <--
0.20       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.25       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.30       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.35       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.40       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.45       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.50       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.55       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.60       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.65       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.70       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.75       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
0.80       0.8804   0.8917   0.7759   0.9809   0.9849   0.8146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8804, F1=0.8917, Normal Recall=0.7759, Normal Precision=0.9809, Attack Recall=0.9849, Attack Precision=0.8146

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
0.15       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284   <--
0.20       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.25       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.30       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.35       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.40       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.45       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.50       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.55       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.60       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.65       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.70       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.75       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.80       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7970, F1=0.4925, Normal Recall=0.7762, Normal Precision=0.9978, Attack Recall=0.9849, Attack Precision=0.3284

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
0.15       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245   <--
0.20       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.25       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.30       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.35       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.40       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.45       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.50       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.55       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.60       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.65       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.70       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.75       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.80       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8184, F1=0.6845, Normal Recall=0.7768, Normal Precision=0.9952, Attack Recall=0.9850, Attack Precision=0.5245

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
0.15       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538   <--
0.20       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.25       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.30       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.35       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.40       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.45       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.50       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.55       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.60       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.65       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.70       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.75       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.80       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8390, F1=0.7859, Normal Recall=0.7765, Normal Precision=0.9918, Attack Recall=0.9850, Attack Precision=0.6538

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
0.15       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461   <--
0.20       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.25       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.30       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.35       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.40       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.45       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.50       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.55       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.60       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.65       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.70       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.75       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.80       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8599, F1=0.8490, Normal Recall=0.7765, Normal Precision=0.9873, Attack Recall=0.9850, Attack Precision=0.7461

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
0.15       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140   <--
0.20       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.25       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.30       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.35       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.40       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.45       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.50       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.55       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.60       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.65       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.70       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.75       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.80       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8800, F1=0.8914, Normal Recall=0.7750, Normal Precision=0.9810, Attack Recall=0.9850, Attack Precision=0.8140

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
0.15       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284   <--
0.20       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.25       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.30       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.35       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.40       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.45       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.50       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.55       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.60       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.65       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.70       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.75       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
0.80       0.7970   0.4925   0.7762   0.9978   0.9849   0.3284  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7970, F1=0.4925, Normal Recall=0.7762, Normal Precision=0.9978, Attack Recall=0.9849, Attack Precision=0.3284

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
0.15       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245   <--
0.20       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.25       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.30       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.35       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.40       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.45       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.50       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.55       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.60       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.65       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.70       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.75       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
0.80       0.8184   0.6845   0.7768   0.9952   0.9850   0.5245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8184, F1=0.6845, Normal Recall=0.7768, Normal Precision=0.9952, Attack Recall=0.9850, Attack Precision=0.5245

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
0.15       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538   <--
0.20       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.25       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.30       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.35       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.40       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.45       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.50       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.55       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.60       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.65       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.70       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.75       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
0.80       0.8390   0.7859   0.7765   0.9918   0.9850   0.6538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8390, F1=0.7859, Normal Recall=0.7765, Normal Precision=0.9918, Attack Recall=0.9850, Attack Precision=0.6538

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
0.15       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461   <--
0.20       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.25       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.30       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.35       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.40       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.45       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.50       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.55       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.60       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.65       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.70       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.75       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
0.80       0.8599   0.8490   0.7765   0.9873   0.9850   0.7461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8599, F1=0.8490, Normal Recall=0.7765, Normal Precision=0.9873, Attack Recall=0.9850, Attack Precision=0.7461

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
0.15       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140   <--
0.20       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.25       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.30       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.35       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.40       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.45       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.50       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.55       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.60       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.65       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.70       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.75       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
0.80       0.8800   0.8914   0.7750   0.9810   0.9850   0.8140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8800, F1=0.8914, Normal Recall=0.7750, Normal Precision=0.9810, Attack Recall=0.9850, Attack Precision=0.8140

```

