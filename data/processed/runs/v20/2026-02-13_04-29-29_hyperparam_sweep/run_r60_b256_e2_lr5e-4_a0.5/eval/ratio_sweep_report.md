# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-15 17:18:21 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9915 | 0.9433 | 0.8961 | 0.8487 | 0.8009 | 0.7531 | 0.7058 | 0.6582 | 0.6108 | 0.5632 | 0.5158 |
| QAT+Prune only | 0.6377 | 0.6732 | 0.7088 | 0.7460 | 0.7813 | 0.8158 | 0.8527 | 0.8883 | 0.9256 | 0.9611 | 0.9974 |
| QAT+PTQ | 0.6380 | 0.6732 | 0.7087 | 0.7460 | 0.7811 | 0.8158 | 0.8527 | 0.8882 | 0.9254 | 0.9611 | 0.9974 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6380 | 0.6732 | 0.7087 | 0.7460 | 0.7811 | 0.8158 | 0.8527 | 0.8882 | 0.9254 | 0.9611 | 0.9974 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6445 | 0.6651 | 0.6716 | 0.6746 | 0.6763 | 0.6778 | 0.6787 | 0.6795 | 0.6801 | 0.6806 |
| QAT+Prune only | 0.0000 | 0.3791 | 0.5780 | 0.7021 | 0.7849 | 0.8441 | 0.8904 | 0.9259 | 0.9555 | 0.9788 | 0.9987 |
| QAT+PTQ | 0.0000 | 0.3791 | 0.5780 | 0.7020 | 0.7847 | 0.8441 | 0.8904 | 0.9258 | 0.9554 | 0.9788 | 0.9987 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3791 | 0.5780 | 0.7020 | 0.7847 | 0.8441 | 0.8904 | 0.9258 | 0.9554 | 0.9788 | 0.9987 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9915 | 0.9911 | 0.9912 | 0.9913 | 0.9910 | 0.9903 | 0.9906 | 0.9903 | 0.9906 | 0.9897 | 0.0000 |
| QAT+Prune only | 0.6377 | 0.6372 | 0.6366 | 0.6383 | 0.6372 | 0.6342 | 0.6356 | 0.6336 | 0.6384 | 0.6339 | 0.0000 |
| QAT+PTQ | 0.6380 | 0.6372 | 0.6365 | 0.6382 | 0.6369 | 0.6342 | 0.6357 | 0.6333 | 0.6377 | 0.6341 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6380 | 0.6372 | 0.6365 | 0.6382 | 0.6369 | 0.6342 | 0.6357 | 0.6333 | 0.6377 | 0.6341 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9915 | 0.0000 | 0.0000 | 0.0000 | 0.9915 | 1.0000 |
| 90 | 10 | 299,940 | 0.9433 | 0.8650 | 0.5136 | 0.6445 | 0.9911 | 0.9483 |
| 80 | 20 | 291,350 | 0.8961 | 0.9360 | 0.5158 | 0.6651 | 0.9912 | 0.8912 |
| 70 | 30 | 194,230 | 0.8487 | 0.9623 | 0.5158 | 0.6716 | 0.9913 | 0.8269 |
| 60 | 40 | 145,675 | 0.8009 | 0.9745 | 0.5158 | 0.6746 | 0.9910 | 0.7543 |
| 50 | 50 | 116,540 | 0.7531 | 0.9815 | 0.5158 | 0.6763 | 0.9903 | 0.6716 |
| 40 | 60 | 97,115 | 0.7058 | 0.9880 | 0.5158 | 0.6778 | 0.9906 | 0.5770 |
| 30 | 70 | 83,240 | 0.6582 | 0.9920 | 0.5158 | 0.6787 | 0.9903 | 0.4671 |
| 20 | 80 | 72,835 | 0.6108 | 0.9955 | 0.5158 | 0.6795 | 0.9906 | 0.3384 |
| 10 | 90 | 64,740 | 0.5632 | 0.9978 | 0.5158 | 0.6801 | 0.9897 | 0.1851 |
| 0 | 100 | 58,270 | 0.5158 | 1.0000 | 0.5158 | 0.6806 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6377 | 0.0000 | 0.0000 | 0.0000 | 0.6377 | 1.0000 |
| 90 | 10 | 299,940 | 0.6732 | 0.2340 | 0.9976 | 0.3791 | 0.6372 | 0.9996 |
| 80 | 20 | 291,350 | 0.7088 | 0.4069 | 0.9974 | 0.5780 | 0.6366 | 0.9990 |
| 70 | 30 | 194,230 | 0.7460 | 0.5417 | 0.9974 | 0.7021 | 0.6383 | 0.9983 |
| 60 | 40 | 145,675 | 0.7813 | 0.6470 | 0.9974 | 0.7849 | 0.6372 | 0.9973 |
| 50 | 50 | 116,540 | 0.8158 | 0.7317 | 0.9974 | 0.8441 | 0.6342 | 0.9960 |
| 40 | 60 | 97,115 | 0.8527 | 0.8042 | 0.9974 | 0.8904 | 0.6356 | 0.9940 |
| 30 | 70 | 83,240 | 0.8883 | 0.8640 | 0.9974 | 0.9259 | 0.6336 | 0.9906 |
| 20 | 80 | 72,835 | 0.9256 | 0.9169 | 0.9974 | 0.9555 | 0.6384 | 0.9841 |
| 10 | 90 | 64,740 | 0.9611 | 0.9608 | 0.9974 | 0.9788 | 0.6339 | 0.9647 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6380 | 0.0000 | 0.0000 | 0.0000 | 0.6380 | 1.0000 |
| 90 | 10 | 299,940 | 0.6732 | 0.2340 | 0.9976 | 0.3791 | 0.6372 | 0.9996 |
| 80 | 20 | 291,350 | 0.7087 | 0.4069 | 0.9974 | 0.5780 | 0.6365 | 0.9990 |
| 70 | 30 | 194,230 | 0.7460 | 0.5416 | 0.9974 | 0.7020 | 0.6382 | 0.9983 |
| 60 | 40 | 145,675 | 0.7811 | 0.6468 | 0.9974 | 0.7847 | 0.6369 | 0.9973 |
| 50 | 50 | 116,540 | 0.8158 | 0.7317 | 0.9974 | 0.8441 | 0.6342 | 0.9959 |
| 40 | 60 | 97,115 | 0.8527 | 0.8042 | 0.9974 | 0.8904 | 0.6357 | 0.9939 |
| 30 | 70 | 83,240 | 0.8882 | 0.8639 | 0.9974 | 0.9258 | 0.6333 | 0.9905 |
| 20 | 80 | 72,835 | 0.9254 | 0.9167 | 0.9974 | 0.9554 | 0.6377 | 0.9839 |
| 10 | 90 | 64,740 | 0.9611 | 0.9608 | 0.9974 | 0.9788 | 0.6341 | 0.9643 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6380 | 0.0000 | 0.0000 | 0.0000 | 0.6380 | 1.0000 |
| 90 | 10 | 299,940 | 0.6732 | 0.2340 | 0.9976 | 0.3791 | 0.6372 | 0.9996 |
| 80 | 20 | 291,350 | 0.7087 | 0.4069 | 0.9974 | 0.5780 | 0.6365 | 0.9990 |
| 70 | 30 | 194,230 | 0.7460 | 0.5416 | 0.9974 | 0.7020 | 0.6382 | 0.9983 |
| 60 | 40 | 145,675 | 0.7811 | 0.6468 | 0.9974 | 0.7847 | 0.6369 | 0.9973 |
| 50 | 50 | 116,540 | 0.8158 | 0.7317 | 0.9974 | 0.8441 | 0.6342 | 0.9959 |
| 40 | 60 | 97,115 | 0.8527 | 0.8042 | 0.9974 | 0.8904 | 0.6357 | 0.9939 |
| 30 | 70 | 83,240 | 0.8882 | 0.8639 | 0.9974 | 0.9258 | 0.6333 | 0.9905 |
| 20 | 80 | 72,835 | 0.9254 | 0.9167 | 0.9974 | 0.9554 | 0.6377 | 0.9839 |
| 10 | 90 | 64,740 | 0.9611 | 0.9608 | 0.9974 | 0.9788 | 0.6341 | 0.9643 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |


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
0.15       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656   <--
0.20       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.25       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.30       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.35       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.40       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.45       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.50       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.55       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.60       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.65       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.70       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.75       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
0.80       0.9436   0.6469   0.9911   0.9486   0.5164   0.8656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9436, F1=0.6469, Normal Recall=0.9911, Normal Precision=0.9486, Attack Recall=0.5164, Attack Precision=0.8656

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
0.15       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361   <--
0.20       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.25       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.30       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.35       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.40       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.45       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.50       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.55       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.60       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.65       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.70       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.75       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
0.80       0.8961   0.6651   0.9912   0.8912   0.5158   0.9361  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8961, F1=0.6651, Normal Recall=0.9912, Normal Precision=0.8912, Attack Recall=0.5158, Attack Precision=0.9361

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
0.15       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619   <--
0.20       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.25       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.30       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.35       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.40       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.45       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.50       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.55       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.60       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.65       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.70       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.75       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
0.80       0.8486   0.6716   0.9913   0.8269   0.5158   0.9619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8486, F1=0.6716, Normal Recall=0.9913, Normal Precision=0.8269, Attack Recall=0.5158, Attack Precision=0.9619

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
0.15       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761   <--
0.20       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.25       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.30       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.35       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.40       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.45       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.50       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.55       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.60       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.65       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.70       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.75       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
0.80       0.8013   0.6750   0.9916   0.7544   0.5158   0.9761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8013, F1=0.6750, Normal Recall=0.9916, Normal Precision=0.7544, Attack Recall=0.5158, Attack Precision=0.9761

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
0.15       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835   <--
0.20       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.25       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.30       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.35       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.40       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.45       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.50       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.55       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.60       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.65       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.70       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.75       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
0.80       0.7536   0.6767   0.9914   0.6719   0.5158   0.9835  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7536, F1=0.6767, Normal Recall=0.9914, Normal Precision=0.6719, Attack Recall=0.5158, Attack Precision=0.9835

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
0.15       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340   <--
0.20       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.25       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.30       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.35       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.40       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.45       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.50       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.55       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.60       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.65       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.70       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.75       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
0.80       0.6732   0.3790   0.6372   0.9995   0.9974   0.2340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6732, F1=0.3790, Normal Recall=0.6372, Normal Precision=0.9995, Attack Recall=0.9974, Attack Precision=0.2340

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
0.15       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079   <--
0.20       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.25       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.30       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.35       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.40       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.45       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.50       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.55       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.60       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.65       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.70       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.75       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
0.80       0.7099   0.5790   0.6380   0.9990   0.9974   0.4079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7099, F1=0.5790, Normal Recall=0.6380, Normal Precision=0.9990, Attack Recall=0.9974, Attack Precision=0.4079

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
0.15       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414   <--
0.20       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.25       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.30       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.35       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.40       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.45       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.50       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.55       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.60       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.65       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.70       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.75       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
0.80       0.7458   0.7019   0.6380   0.9983   0.9974   0.5414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7458, F1=0.7019, Normal Recall=0.6380, Normal Precision=0.9983, Attack Recall=0.9974, Attack Precision=0.5414

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
0.15       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472   <--
0.20       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.25       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.30       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.35       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.40       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.45       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.50       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.55       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.60       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.65       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.70       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.75       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
0.80       0.7815   0.7850   0.6375   0.9973   0.9974   0.6472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7815, F1=0.7850, Normal Recall=0.6375, Normal Precision=0.9973, Attack Recall=0.9974, Attack Precision=0.6472

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
0.15       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323   <--
0.20       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.25       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.30       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.35       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.40       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.45       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.50       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.55       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.60       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.65       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.70       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.75       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
0.80       0.8164   0.8445   0.6354   0.9960   0.9974   0.7323  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8164, F1=0.8445, Normal Recall=0.6354, Normal Precision=0.9960, Attack Recall=0.9974, Attack Precision=0.7323

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
0.15       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340   <--
0.20       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.25       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.30       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.35       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.40       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.45       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.50       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.55       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.60       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.65       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.70       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.75       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.80       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6732, F1=0.3790, Normal Recall=0.6372, Normal Precision=0.9995, Attack Recall=0.9973, Attack Precision=0.2340

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
0.15       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078   <--
0.20       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.25       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.30       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.35       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.40       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.45       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.50       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.55       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.60       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.65       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.70       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.75       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.80       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7098, F1=0.5789, Normal Recall=0.6379, Normal Precision=0.9990, Attack Recall=0.9974, Attack Precision=0.4078

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
0.15       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416   <--
0.20       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.25       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.30       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.35       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.40       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.45       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.50       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.55       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.60       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.65       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.70       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.75       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.80       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7460, F1=0.7020, Normal Recall=0.6382, Normal Precision=0.9983, Attack Recall=0.9974, Attack Precision=0.5416

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
0.15       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473   <--
0.20       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.25       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.30       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.35       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.40       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.45       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.50       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.55       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.60       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.65       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.70       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.75       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.80       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7851, Normal Recall=0.6377, Normal Precision=0.9973, Attack Recall=0.9974, Attack Precision=0.6473

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
0.15       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324   <--
0.20       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.25       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.30       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.35       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.40       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.45       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.50       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.55       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.60       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.65       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.70       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.75       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.80       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8165, F1=0.8446, Normal Recall=0.6356, Normal Precision=0.9959, Attack Recall=0.9974, Attack Precision=0.7324

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
0.15       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340   <--
0.20       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.25       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.30       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.35       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.40       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.45       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.50       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.55       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.60       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.65       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.70       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.75       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
0.80       0.6732   0.3790   0.6372   0.9995   0.9973   0.2340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6732, F1=0.3790, Normal Recall=0.6372, Normal Precision=0.9995, Attack Recall=0.9973, Attack Precision=0.2340

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
0.15       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078   <--
0.20       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.25       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.30       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.35       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.40       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.45       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.50       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.55       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.60       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.65       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.70       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.75       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
0.80       0.7098   0.5789   0.6379   0.9990   0.9974   0.4078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7098, F1=0.5789, Normal Recall=0.6379, Normal Precision=0.9990, Attack Recall=0.9974, Attack Precision=0.4078

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
0.15       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416   <--
0.20       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.25       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.30       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.35       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.40       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.45       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.50       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.55       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.60       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.65       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.70       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.75       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
0.80       0.7460   0.7020   0.6382   0.9983   0.9974   0.5416  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7460, F1=0.7020, Normal Recall=0.6382, Normal Precision=0.9983, Attack Recall=0.9974, Attack Precision=0.5416

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
0.15       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473   <--
0.20       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.25       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.30       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.35       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.40       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.45       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.50       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.55       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.60       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.65       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.70       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.75       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
0.80       0.7816   0.7851   0.6377   0.9973   0.9974   0.6473  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7816, F1=0.7851, Normal Recall=0.6377, Normal Precision=0.9973, Attack Recall=0.9974, Attack Precision=0.6473

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
0.15       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324   <--
0.20       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.25       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.30       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.35       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.40       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.45       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.50       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.55       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.60       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.65       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.70       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.75       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
0.80       0.8165   0.8446   0.6356   0.9959   0.9974   0.7324  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8165, F1=0.8446, Normal Recall=0.6356, Normal Precision=0.9959, Attack Recall=0.9974, Attack Precision=0.7324

```

