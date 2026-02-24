# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-22 18:16:50 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3727 | 0.4321 | 0.4926 | 0.5536 | 0.6121 | 0.6731 | 0.7340 | 0.7936 | 0.8525 | 0.9140 | 0.9738 |
| QAT+Prune only | 0.1880 | 0.2694 | 0.3501 | 0.4309 | 0.5137 | 0.5924 | 0.6741 | 0.7557 | 0.8359 | 0.9176 | 0.9982 |
| QAT+PTQ | 0.1877 | 0.2689 | 0.3496 | 0.4307 | 0.5135 | 0.5922 | 0.6738 | 0.7557 | 0.8359 | 0.9175 | 0.9982 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.1877 | 0.2689 | 0.3496 | 0.4307 | 0.5135 | 0.5922 | 0.6738 | 0.7557 | 0.8359 | 0.9175 | 0.9982 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2554 | 0.4343 | 0.5669 | 0.6676 | 0.7487 | 0.8146 | 0.8685 | 0.9135 | 0.9532 | 0.9867 |
| QAT+Prune only | 0.0000 | 0.2146 | 0.3806 | 0.5128 | 0.6215 | 0.7101 | 0.7861 | 0.8512 | 0.9068 | 0.9562 | 0.9991 |
| QAT+PTQ | 0.0000 | 0.2145 | 0.3804 | 0.5127 | 0.6214 | 0.7100 | 0.7860 | 0.8512 | 0.9068 | 0.9561 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2145 | 0.3804 | 0.5127 | 0.6214 | 0.7100 | 0.7860 | 0.8512 | 0.9068 | 0.9561 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3727 | 0.3718 | 0.3723 | 0.3735 | 0.3710 | 0.3724 | 0.3743 | 0.3733 | 0.3675 | 0.3761 | 0.0000 |
| QAT+Prune only | 0.1880 | 0.1884 | 0.1880 | 0.1878 | 0.1907 | 0.1866 | 0.1879 | 0.1897 | 0.1867 | 0.1920 | 0.0000 |
| QAT+PTQ | 0.1877 | 0.1879 | 0.1875 | 0.1874 | 0.1903 | 0.1862 | 0.1873 | 0.1897 | 0.1867 | 0.1906 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.1877 | 0.1879 | 0.1875 | 0.1874 | 0.1903 | 0.1862 | 0.1873 | 0.1897 | 0.1867 | 0.1906 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3727 | 0.0000 | 0.0000 | 0.0000 | 0.3727 | 1.0000 |
| 90 | 10 | 299,940 | 0.4321 | 0.1470 | 0.9741 | 0.2554 | 0.3718 | 0.9923 |
| 80 | 20 | 291,350 | 0.4926 | 0.2795 | 0.9738 | 0.4343 | 0.3723 | 0.9827 |
| 70 | 30 | 194,230 | 0.5536 | 0.3998 | 0.9738 | 0.5669 | 0.3735 | 0.9708 |
| 60 | 40 | 145,675 | 0.6121 | 0.5079 | 0.9738 | 0.6676 | 0.3710 | 0.9550 |
| 50 | 50 | 116,540 | 0.6731 | 0.6081 | 0.9738 | 0.7487 | 0.3724 | 0.9342 |
| 40 | 60 | 97,115 | 0.7340 | 0.7001 | 0.9738 | 0.8146 | 0.3743 | 0.9049 |
| 30 | 70 | 83,240 | 0.7936 | 0.7838 | 0.9738 | 0.8685 | 0.3733 | 0.8592 |
| 20 | 80 | 72,835 | 0.8525 | 0.8603 | 0.9738 | 0.9135 | 0.3675 | 0.7780 |
| 10 | 90 | 64,740 | 0.9140 | 0.9335 | 0.9738 | 0.9532 | 0.3761 | 0.6144 |
| 0 | 100 | 58,270 | 0.9738 | 1.0000 | 0.9738 | 0.9867 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1880 | 0.0000 | 0.0000 | 0.0000 | 0.1880 | 1.0000 |
| 90 | 10 | 299,940 | 0.2694 | 0.1202 | 0.9982 | 0.2146 | 0.1884 | 0.9989 |
| 80 | 20 | 291,350 | 0.3501 | 0.2351 | 0.9982 | 0.3806 | 0.1880 | 0.9977 |
| 70 | 30 | 194,230 | 0.4309 | 0.3450 | 0.9982 | 0.5128 | 0.1878 | 0.9960 |
| 60 | 40 | 145,675 | 0.5137 | 0.4512 | 0.9982 | 0.6215 | 0.1907 | 0.9939 |
| 50 | 50 | 116,540 | 0.5924 | 0.5510 | 0.9982 | 0.7101 | 0.1866 | 0.9906 |
| 40 | 60 | 97,115 | 0.6741 | 0.6483 | 0.9982 | 0.7861 | 0.1879 | 0.9861 |
| 30 | 70 | 83,240 | 0.7557 | 0.7419 | 0.9982 | 0.8512 | 0.1897 | 0.9787 |
| 20 | 80 | 72,835 | 0.8359 | 0.8308 | 0.9982 | 0.9068 | 0.1867 | 0.9635 |
| 10 | 90 | 64,740 | 0.9176 | 0.9175 | 0.9982 | 0.9562 | 0.1920 | 0.9235 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1877 | 0.0000 | 0.0000 | 0.0000 | 0.1877 | 1.0000 |
| 90 | 10 | 299,940 | 0.2689 | 0.1202 | 0.9982 | 0.2145 | 0.1879 | 0.9989 |
| 80 | 20 | 291,350 | 0.3496 | 0.2350 | 0.9982 | 0.3804 | 0.1875 | 0.9976 |
| 70 | 30 | 194,230 | 0.4307 | 0.3449 | 0.9982 | 0.5127 | 0.1874 | 0.9960 |
| 60 | 40 | 145,675 | 0.5135 | 0.4511 | 0.9982 | 0.6214 | 0.1903 | 0.9938 |
| 50 | 50 | 116,540 | 0.5922 | 0.5509 | 0.9982 | 0.7100 | 0.1862 | 0.9906 |
| 40 | 60 | 97,115 | 0.6738 | 0.6482 | 0.9982 | 0.7860 | 0.1873 | 0.9860 |
| 30 | 70 | 83,240 | 0.7557 | 0.7419 | 0.9982 | 0.8512 | 0.1897 | 0.9787 |
| 20 | 80 | 72,835 | 0.8359 | 0.8308 | 0.9982 | 0.9068 | 0.1867 | 0.9635 |
| 10 | 90 | 64,740 | 0.9175 | 0.9174 | 0.9982 | 0.9561 | 0.1906 | 0.9230 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.1877 | 0.0000 | 0.0000 | 0.0000 | 0.1877 | 1.0000 |
| 90 | 10 | 299,940 | 0.2689 | 0.1202 | 0.9982 | 0.2145 | 0.1879 | 0.9989 |
| 80 | 20 | 291,350 | 0.3496 | 0.2350 | 0.9982 | 0.3804 | 0.1875 | 0.9976 |
| 70 | 30 | 194,230 | 0.4307 | 0.3449 | 0.9982 | 0.5127 | 0.1874 | 0.9960 |
| 60 | 40 | 145,675 | 0.5135 | 0.4511 | 0.9982 | 0.6214 | 0.1903 | 0.9938 |
| 50 | 50 | 116,540 | 0.5922 | 0.5509 | 0.9982 | 0.7100 | 0.1862 | 0.9906 |
| 40 | 60 | 97,115 | 0.6738 | 0.6482 | 0.9982 | 0.7860 | 0.1873 | 0.9860 |
| 30 | 70 | 83,240 | 0.7557 | 0.7419 | 0.9982 | 0.8512 | 0.1897 | 0.9787 |
| 20 | 80 | 72,835 | 0.8359 | 0.8308 | 0.9982 | 0.9068 | 0.1867 | 0.9635 |
| 10 | 90 | 64,740 | 0.9175 | 0.9174 | 0.9982 | 0.9561 | 0.1906 | 0.9230 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469   <--
0.20       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.25       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.30       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.35       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.40       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.45       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.50       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.55       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.60       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.65       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.70       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.75       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
0.80       0.4320   0.2553   0.3718   0.9921   0.9735   0.1469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4320, F1=0.2553, Normal Recall=0.3718, Normal Precision=0.9921, Attack Recall=0.9735, Attack Precision=0.1469

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
0.15       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792   <--
0.20       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.25       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.30       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.35       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.40       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.45       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.50       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.55       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.60       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.65       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.70       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.75       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
0.80       0.4920   0.4340   0.3715   0.9827   0.9738   0.2792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4920, F1=0.4340, Normal Recall=0.3715, Normal Precision=0.9827, Attack Recall=0.9738, Attack Precision=0.2792

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
0.15       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998   <--
0.20       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.25       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.30       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.35       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.40       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.45       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.50       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.55       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.60       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.65       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.70       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.75       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
0.80       0.5535   0.5668   0.3734   0.9708   0.9738   0.3998  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5535, F1=0.5668, Normal Recall=0.3734, Normal Precision=0.9708, Attack Recall=0.9738, Attack Precision=0.3998

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
0.15       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086   <--
0.20       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.25       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.30       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.35       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.40       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.45       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.50       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.55       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.60       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.65       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.70       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.75       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
0.80       0.6131   0.6682   0.3727   0.9552   0.9738   0.5086  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6131, F1=0.6682, Normal Recall=0.3727, Normal Precision=0.9552, Attack Recall=0.9738, Attack Precision=0.5086

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
0.15       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083   <--
0.20       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.25       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.30       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.35       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.40       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.45       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.50       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.55       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.60       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.65       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.70       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.75       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
0.80       0.6733   0.7488   0.3729   0.9343   0.9738   0.6083  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6733, F1=0.7488, Normal Recall=0.3729, Normal Precision=0.9343, Attack Recall=0.9738, Attack Precision=0.6083

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
0.15       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202   <--
0.20       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.25       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.30       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.35       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.40       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.45       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.50       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.55       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.60       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.65       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.70       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.75       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
0.80       0.2694   0.2146   0.1884   0.9990   0.9983   0.1202  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2694, F1=0.2146, Normal Recall=0.1884, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.1202

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
0.15       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351   <--
0.20       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.25       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.30       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.35       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.40       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.45       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.50       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.55       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.60       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.65       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.70       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.75       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
0.80       0.3502   0.3806   0.1882   0.9977   0.9982   0.2351  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3502, F1=0.3806, Normal Recall=0.1882, Normal Precision=0.9977, Attack Recall=0.9982, Attack Precision=0.2351

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
0.15       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451   <--
0.20       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.25       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.30       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.35       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.40       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.45       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.50       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.55       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.60       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.65       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.70       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.75       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
0.80       0.4312   0.5129   0.1882   0.9960   0.9982   0.3451  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4312, F1=0.5129, Normal Recall=0.1882, Normal Precision=0.9960, Attack Recall=0.9982, Attack Precision=0.3451

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
0.15       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503   <--
0.20       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.25       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.30       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.35       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.40       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.45       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.50       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.55       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.60       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.65       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.70       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.75       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
0.80       0.5118   0.6206   0.1874   0.9938   0.9982   0.4503  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5118, F1=0.6206, Normal Recall=0.1874, Normal Precision=0.9938, Attack Recall=0.9982, Attack Precision=0.4503

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
0.15       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510   <--
0.20       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.25       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.30       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.35       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.40       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.45       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.50       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.55       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.60       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.65       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.70       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.75       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
0.80       0.5924   0.7101   0.1866   0.9906   0.9982   0.5510  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5924, F1=0.7101, Normal Recall=0.1866, Normal Precision=0.9906, Attack Recall=0.9982, Attack Precision=0.5510

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
0.15       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202   <--
0.20       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.25       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.30       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.35       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.40       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.45       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.50       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.55       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.60       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.65       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.70       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.75       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.80       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2689, F1=0.2145, Normal Recall=0.1879, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.1202

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
0.15       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350   <--
0.20       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.25       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.30       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.35       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.40       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.45       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.50       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.55       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.60       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.65       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.70       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.75       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.80       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3498, F1=0.3805, Normal Recall=0.1877, Normal Precision=0.9977, Attack Recall=0.9982, Attack Precision=0.2350

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
0.15       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450   <--
0.20       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.25       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.30       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.35       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.40       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.45       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.50       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.55       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.60       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.65       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.70       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.75       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.80       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4309, F1=0.5128, Normal Recall=0.1878, Normal Precision=0.9960, Attack Recall=0.9982, Attack Precision=0.3450

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
0.15       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501   <--
0.20       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.25       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.30       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.35       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.40       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.45       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.50       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.55       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.60       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.65       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.70       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.75       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.80       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5115, F1=0.6205, Normal Recall=0.1871, Normal Precision=0.9937, Attack Recall=0.9982, Attack Precision=0.4501

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
0.15       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509   <--
0.20       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.25       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.30       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.35       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.40       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.45       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.50       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.55       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.60       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.65       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.70       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.75       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.80       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5922, F1=0.7100, Normal Recall=0.1862, Normal Precision=0.9906, Attack Recall=0.9982, Attack Precision=0.5509

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
0.15       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202   <--
0.20       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.25       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.30       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.35       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.40       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.45       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.50       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.55       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.60       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.65       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.70       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.75       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
0.80       0.2689   0.2145   0.1879   0.9990   0.9983   0.1202  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2689, F1=0.2145, Normal Recall=0.1879, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.1202

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
0.15       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350   <--
0.20       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.25       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.30       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.35       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.40       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.45       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.50       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.55       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.60       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.65       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.70       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.75       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
0.80       0.3498   0.3805   0.1877   0.9977   0.9982   0.2350  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3498, F1=0.3805, Normal Recall=0.1877, Normal Precision=0.9977, Attack Recall=0.9982, Attack Precision=0.2350

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
0.15       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450   <--
0.20       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.25       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.30       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.35       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.40       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.45       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.50       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.55       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.60       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.65       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.70       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.75       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
0.80       0.4309   0.5128   0.1878   0.9960   0.9982   0.3450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4309, F1=0.5128, Normal Recall=0.1878, Normal Precision=0.9960, Attack Recall=0.9982, Attack Precision=0.3450

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
0.15       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501   <--
0.20       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.25       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.30       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.35       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.40       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.45       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.50       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.55       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.60       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.65       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.70       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.75       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
0.80       0.5115   0.6205   0.1871   0.9937   0.9982   0.4501  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5115, F1=0.6205, Normal Recall=0.1871, Normal Precision=0.9937, Attack Recall=0.9982, Attack Precision=0.4501

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
0.15       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509   <--
0.20       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.25       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.30       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.35       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.40       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.45       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.50       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.55       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.60       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.65       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.70       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.75       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
0.80       0.5922   0.7100   0.1862   0.9906   0.9982   0.5509  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5922, F1=0.7100, Normal Recall=0.1862, Normal Precision=0.9906, Attack Recall=0.9982, Attack Precision=0.5509

```

