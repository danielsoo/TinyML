# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-21 10:29:38 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8950 | 0.8836 | 0.8722 | 0.8619 | 0.8502 | 0.8373 | 0.8278 | 0.8163 | 0.8051 | 0.7940 | 0.7826 |
| QAT+Prune only | 0.1905 | 0.2714 | 0.3519 | 0.4338 | 0.5152 | 0.5953 | 0.6752 | 0.7582 | 0.8382 | 0.9185 | 0.9997 |
| QAT+PTQ | 0.1801 | 0.2626 | 0.3442 | 0.4267 | 0.5095 | 0.5905 | 0.6715 | 0.7549 | 0.8361 | 0.9176 | 0.9997 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.1801 | 0.2626 | 0.3442 | 0.4267 | 0.5095 | 0.5905 | 0.6715 | 0.7549 | 0.8361 | 0.9176 | 0.9997 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5730 | 0.7101 | 0.7727 | 0.8070 | 0.8279 | 0.8450 | 0.8564 | 0.8653 | 0.8724 | 0.8780 |
| QAT+Prune only | 0.0000 | 0.2153 | 0.3816 | 0.5144 | 0.6226 | 0.7118 | 0.7870 | 0.8527 | 0.9081 | 0.9567 | 0.9998 |
| QAT+PTQ | 0.0000 | 0.2133 | 0.3788 | 0.5113 | 0.6198 | 0.7094 | 0.7850 | 0.8510 | 0.9070 | 0.9562 | 0.9998 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2133 | 0.3788 | 0.5113 | 0.6198 | 0.7094 | 0.7850 | 0.8510 | 0.9070 | 0.9562 | 0.9998 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8950 | 0.8949 | 0.8946 | 0.8959 | 0.8953 | 0.8921 | 0.8956 | 0.8949 | 0.8951 | 0.8962 | 0.0000 |
| QAT+Prune only | 0.1905 | 0.1905 | 0.1899 | 0.1912 | 0.1923 | 0.1909 | 0.1885 | 0.1947 | 0.1922 | 0.1881 | 0.0000 |
| QAT+PTQ | 0.1801 | 0.1807 | 0.1804 | 0.1812 | 0.1827 | 0.1812 | 0.1791 | 0.1838 | 0.1816 | 0.1790 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.1801 | 0.1807 | 0.1804 | 0.1812 | 0.1827 | 0.1812 | 0.1791 | 0.1838 | 0.1816 | 0.1790 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8950 | 0.0000 | 0.0000 | 0.0000 | 0.8950 | 1.0000 |
| 90 | 10 | 299,940 | 0.8836 | 0.4524 | 0.7812 | 0.5730 | 0.8949 | 0.9736 |
| 80 | 20 | 291,350 | 0.8722 | 0.6499 | 0.7826 | 0.7101 | 0.8946 | 0.9427 |
| 70 | 30 | 194,230 | 0.8619 | 0.7631 | 0.7826 | 0.7727 | 0.8959 | 0.9058 |
| 60 | 40 | 145,675 | 0.8502 | 0.8329 | 0.7826 | 0.8070 | 0.8953 | 0.8607 |
| 50 | 50 | 116,540 | 0.8373 | 0.8788 | 0.7826 | 0.8279 | 0.8921 | 0.8041 |
| 40 | 60 | 97,115 | 0.8278 | 0.9183 | 0.7826 | 0.8450 | 0.8956 | 0.7331 |
| 30 | 70 | 83,240 | 0.8163 | 0.9456 | 0.7826 | 0.8564 | 0.8949 | 0.6382 |
| 20 | 80 | 72,835 | 0.8051 | 0.9676 | 0.7826 | 0.8653 | 0.8951 | 0.5072 |
| 10 | 90 | 64,740 | 0.7940 | 0.9855 | 0.7826 | 0.8724 | 0.8962 | 0.3141 |
| 0 | 100 | 58,270 | 0.7826 | 1.0000 | 0.7826 | 0.8780 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1905 | 0.0000 | 0.0000 | 0.0000 | 0.1905 | 1.0000 |
| 90 | 10 | 299,940 | 0.2714 | 0.1207 | 0.9997 | 0.2153 | 0.1905 | 0.9998 |
| 80 | 20 | 291,350 | 0.3519 | 0.2358 | 0.9997 | 0.3816 | 0.1899 | 0.9996 |
| 70 | 30 | 194,230 | 0.4338 | 0.3463 | 0.9997 | 0.5144 | 0.1912 | 0.9993 |
| 60 | 40 | 145,675 | 0.5152 | 0.4521 | 0.9997 | 0.6226 | 0.1923 | 0.9989 |
| 50 | 50 | 116,540 | 0.5953 | 0.5527 | 0.9997 | 0.7118 | 0.1909 | 0.9984 |
| 40 | 60 | 97,115 | 0.6752 | 0.6489 | 0.9997 | 0.7870 | 0.1885 | 0.9975 |
| 30 | 70 | 83,240 | 0.7582 | 0.7434 | 0.9997 | 0.8527 | 0.1947 | 0.9963 |
| 20 | 80 | 72,835 | 0.8382 | 0.8319 | 0.9997 | 0.9081 | 0.1922 | 0.9936 |
| 10 | 90 | 64,740 | 0.9185 | 0.9172 | 0.9997 | 0.9567 | 0.1881 | 0.9854 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9998 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1801 | 0.0000 | 0.0000 | 0.0000 | 0.1801 | 1.0000 |
| 90 | 10 | 299,940 | 0.2626 | 0.1194 | 0.9997 | 0.2133 | 0.1807 | 0.9998 |
| 80 | 20 | 291,350 | 0.3442 | 0.2337 | 0.9997 | 0.3788 | 0.1804 | 0.9996 |
| 70 | 30 | 194,230 | 0.4267 | 0.3435 | 0.9997 | 0.5113 | 0.1812 | 0.9993 |
| 60 | 40 | 145,675 | 0.5095 | 0.4492 | 0.9997 | 0.6198 | 0.1827 | 0.9989 |
| 50 | 50 | 116,540 | 0.5905 | 0.5497 | 0.9997 | 0.7094 | 0.1812 | 0.9983 |
| 40 | 60 | 97,115 | 0.6715 | 0.6462 | 0.9997 | 0.7850 | 0.1791 | 0.9974 |
| 30 | 70 | 83,240 | 0.7549 | 0.7408 | 0.9997 | 0.8510 | 0.1838 | 0.9961 |
| 20 | 80 | 72,835 | 0.8361 | 0.8301 | 0.9997 | 0.9070 | 0.1816 | 0.9932 |
| 10 | 90 | 64,740 | 0.9176 | 0.9164 | 0.9997 | 0.9562 | 0.1790 | 0.9847 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9998 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.1801 | 0.0000 | 0.0000 | 0.0000 | 0.1801 | 1.0000 |
| 90 | 10 | 299,940 | 0.2626 | 0.1194 | 0.9997 | 0.2133 | 0.1807 | 0.9998 |
| 80 | 20 | 291,350 | 0.3442 | 0.2337 | 0.9997 | 0.3788 | 0.1804 | 0.9996 |
| 70 | 30 | 194,230 | 0.4267 | 0.3435 | 0.9997 | 0.5113 | 0.1812 | 0.9993 |
| 60 | 40 | 145,675 | 0.5095 | 0.4492 | 0.9997 | 0.6198 | 0.1827 | 0.9989 |
| 50 | 50 | 116,540 | 0.5905 | 0.5497 | 0.9997 | 0.7094 | 0.1812 | 0.9983 |
| 40 | 60 | 97,115 | 0.6715 | 0.6462 | 0.9997 | 0.7850 | 0.1791 | 0.9974 |
| 30 | 70 | 83,240 | 0.7549 | 0.7408 | 0.9997 | 0.8510 | 0.1838 | 0.9961 |
| 20 | 80 | 72,835 | 0.8361 | 0.8301 | 0.9997 | 0.9070 | 0.1816 | 0.9932 |
| 10 | 90 | 64,740 | 0.9176 | 0.9164 | 0.9997 | 0.9562 | 0.1790 | 0.9847 |
| 0 | 100 | 58,270 | 0.9997 | 1.0000 | 0.9997 | 0.9998 | 0.0000 | 0.0000 |


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
0.15       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523   <--
0.20       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.25       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.30       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.35       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.40       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.45       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.50       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.55       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.60       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.65       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.70       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.75       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
0.80       0.8835   0.5728   0.8949   0.9735   0.7808   0.4523  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8835, F1=0.5728, Normal Recall=0.8949, Normal Precision=0.9735, Attack Recall=0.7808, Attack Precision=0.4523

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
0.15       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509   <--
0.20       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.25       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.30       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.35       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.40       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.45       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.50       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.55       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.60       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.65       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.70       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.75       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
0.80       0.8726   0.7107   0.8951   0.9428   0.7826   0.6509  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8726, F1=0.7107, Normal Recall=0.8951, Normal Precision=0.9428, Attack Recall=0.7826, Attack Precision=0.6509

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
0.15       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623   <--
0.20       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.25       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.30       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.35       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.40       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.45       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.50       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.55       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.60       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.65       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.70       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.75       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
0.80       0.8616   0.7723   0.8954   0.9057   0.7826   0.7623  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8616, F1=0.7723, Normal Recall=0.8954, Normal Precision=0.9057, Attack Recall=0.7826, Attack Precision=0.7623

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
0.15       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326   <--
0.20       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.25       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.30       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.35       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.40       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.45       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.50       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.55       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.60       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.65       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.70       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.75       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
0.80       0.8501   0.8068   0.8951   0.8606   0.7826   0.8326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8501, F1=0.8068, Normal Recall=0.8951, Normal Precision=0.8606, Attack Recall=0.7826, Attack Precision=0.8326

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
0.15       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811   <--
0.20       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.25       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.30       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.35       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.40       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.45       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.50       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.55       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.60       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.65       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.70       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.75       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
0.80       0.8385   0.8289   0.8944   0.8045   0.7826   0.8811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8385, F1=0.8289, Normal Recall=0.8944, Normal Precision=0.8045, Attack Recall=0.7826, Attack Precision=0.8811

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
0.15       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207   <--
0.20       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.25       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.30       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.35       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.40       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.45       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.50       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.55       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.60       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.65       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.70       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.75       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
0.80       0.2714   0.2153   0.1905   0.9999   0.9998   0.1207  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2714, F1=0.2153, Normal Recall=0.1905, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1207

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
0.15       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358   <--
0.20       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.25       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.30       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.35       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.40       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.45       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.50       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.55       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.60       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.65       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.70       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.75       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
0.80       0.3520   0.3816   0.1901   0.9996   0.9997   0.2358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3520, F1=0.3816, Normal Recall=0.1901, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2358

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
0.15       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462   <--
0.20       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.25       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.30       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.35       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.40       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.45       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.50       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.55       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.60       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.65       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.70       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.75       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
0.80       0.4334   0.5142   0.1907   0.9993   0.9997   0.3462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4334, F1=0.5142, Normal Recall=0.1907, Normal Precision=0.9993, Attack Recall=0.9997, Attack Precision=0.3462

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
0.15       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517   <--
0.20       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.25       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.30       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.35       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.40       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.45       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.50       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.55       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.60       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.65       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.70       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.75       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
0.80       0.5145   0.6223   0.1911   0.9989   0.9997   0.4517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5145, F1=0.6223, Normal Recall=0.1911, Normal Precision=0.9989, Attack Recall=0.9997, Attack Precision=0.4517

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
0.15       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524   <--
0.20       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.25       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.30       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.35       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.40       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.45       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.50       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.55       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.60       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.65       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.70       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.75       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
0.80       0.5949   0.7116   0.1900   0.9984   0.9997   0.5524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5949, F1=0.7116, Normal Recall=0.1900, Normal Precision=0.9984, Attack Recall=0.9997, Attack Precision=0.5524

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
0.15       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194   <--
0.20       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.25       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.30       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.35       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.40       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.45       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.50       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.55       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.60       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.65       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.70       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.75       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.80       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2626, F1=0.2133, Normal Recall=0.1807, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1194

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
0.15       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337   <--
0.20       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.25       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.30       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.35       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.40       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.45       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.50       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.55       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.60       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.65       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.70       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.75       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.80       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3442, F1=0.3788, Normal Recall=0.1803, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2337

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
0.15       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433   <--
0.20       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.25       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.30       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.35       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.40       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.45       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.50       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.55       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.60       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.65       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.70       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.75       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.80       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4262, F1=0.5111, Normal Recall=0.1804, Normal Precision=0.9993, Attack Recall=0.9997, Attack Precision=0.3433

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
0.15       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485   <--
0.20       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.25       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.30       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.35       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.40       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.45       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.50       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.55       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.60       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.65       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.70       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.75       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.80       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5082, F1=0.6192, Normal Recall=0.1806, Normal Precision=0.9989, Attack Recall=0.9997, Attack Precision=0.4485

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
0.15       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491   <--
0.20       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.25       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.30       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.35       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.40       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.45       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.50       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.55       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.60       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.65       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.70       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.75       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.80       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5894, F1=0.7088, Normal Recall=0.1790, Normal Precision=0.9983, Attack Recall=0.9997, Attack Precision=0.5491

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
0.15       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194   <--
0.20       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.25       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.30       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.35       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.40       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.45       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.50       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.55       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.60       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.65       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.70       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.75       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
0.80       0.2626   0.2133   0.1807   0.9999   0.9998   0.1194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2626, F1=0.2133, Normal Recall=0.1807, Normal Precision=0.9999, Attack Recall=0.9998, Attack Precision=0.1194

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
0.15       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337   <--
0.20       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.25       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.30       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.35       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.40       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.45       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.50       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.55       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.60       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.65       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.70       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.75       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
0.80       0.3442   0.3788   0.1803   0.9996   0.9997   0.2337  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3442, F1=0.3788, Normal Recall=0.1803, Normal Precision=0.9996, Attack Recall=0.9997, Attack Precision=0.2337

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
0.15       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433   <--
0.20       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.25       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.30       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.35       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.40       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.45       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.50       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.55       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.60       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.65       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.70       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.75       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
0.80       0.4262   0.5111   0.1804   0.9993   0.9997   0.3433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4262, F1=0.5111, Normal Recall=0.1804, Normal Precision=0.9993, Attack Recall=0.9997, Attack Precision=0.3433

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
0.15       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485   <--
0.20       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.25       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.30       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.35       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.40       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.45       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.50       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.55       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.60       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.65       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.70       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.75       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
0.80       0.5082   0.6192   0.1806   0.9989   0.9997   0.4485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5082, F1=0.6192, Normal Recall=0.1806, Normal Precision=0.9989, Attack Recall=0.9997, Attack Precision=0.4485

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
0.15       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491   <--
0.20       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.25       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.30       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.35       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.40       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.45       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.50       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.55       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.60       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.65       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.70       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.75       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
0.80       0.5894   0.7088   0.1790   0.9983   0.9997   0.5491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5894, F1=0.7088, Normal Recall=0.1790, Normal Precision=0.9983, Attack Recall=0.9997, Attack Precision=0.5491

```

