# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-16 21:09:48 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1931 | 0.2663 | 0.3395 | 0.4132 | 0.4866 | 0.5593 | 0.6324 | 0.7048 | 0.7785 | 0.8531 | 0.9252 |
| QAT+Prune only | 0.8384 | 0.8540 | 0.8695 | 0.8851 | 0.9003 | 0.9154 | 0.9322 | 0.9471 | 0.9637 | 0.9785 | 0.9942 |
| QAT+PTQ | 0.8391 | 0.8545 | 0.8699 | 0.8855 | 0.9007 | 0.9157 | 0.9323 | 0.9474 | 0.9639 | 0.9786 | 0.9942 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8391 | 0.8545 | 0.8699 | 0.8855 | 0.9007 | 0.9157 | 0.9323 | 0.9474 | 0.9639 | 0.9786 | 0.9942 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2012 | 0.3591 | 0.4861 | 0.5904 | 0.6774 | 0.7512 | 0.8144 | 0.8699 | 0.9189 | 0.9611 |
| QAT+Prune only | 0.0000 | 0.5766 | 0.7529 | 0.8385 | 0.8886 | 0.9216 | 0.9463 | 0.9634 | 0.9777 | 0.9882 | 0.9971 |
| QAT+PTQ | 0.0000 | 0.5774 | 0.7535 | 0.8390 | 0.8890 | 0.9219 | 0.9463 | 0.9636 | 0.9778 | 0.9882 | 0.9971 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5774 | 0.7535 | 0.8390 | 0.8890 | 0.9219 | 0.9463 | 0.9636 | 0.9778 | 0.9882 | 0.9971 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1931 | 0.1932 | 0.1931 | 0.1938 | 0.1942 | 0.1935 | 0.1932 | 0.1905 | 0.1919 | 0.2039 | 0.0000 |
| QAT+Prune only | 0.8384 | 0.8384 | 0.8383 | 0.8384 | 0.8377 | 0.8367 | 0.8393 | 0.8373 | 0.8417 | 0.8378 | 0.0000 |
| QAT+PTQ | 0.8391 | 0.8390 | 0.8389 | 0.8390 | 0.8383 | 0.8373 | 0.8394 | 0.8381 | 0.8427 | 0.8380 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8391 | 0.8390 | 0.8389 | 0.8390 | 0.8383 | 0.8373 | 0.8394 | 0.8381 | 0.8427 | 0.8380 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1931 | 0.0000 | 0.0000 | 0.0000 | 0.1931 | 1.0000 |
| 90 | 10 | 299,940 | 0.2663 | 0.1129 | 0.9240 | 0.2012 | 0.1932 | 0.9581 |
| 80 | 20 | 291,350 | 0.3395 | 0.2228 | 0.9252 | 0.3591 | 0.1931 | 0.9117 |
| 70 | 30 | 194,230 | 0.4132 | 0.3297 | 0.9252 | 0.4861 | 0.1938 | 0.8581 |
| 60 | 40 | 145,675 | 0.4866 | 0.4336 | 0.9252 | 0.5904 | 0.1942 | 0.7957 |
| 50 | 50 | 116,540 | 0.5593 | 0.5343 | 0.9252 | 0.6774 | 0.1935 | 0.7212 |
| 40 | 60 | 97,115 | 0.6324 | 0.6324 | 0.9252 | 0.7512 | 0.1932 | 0.6326 |
| 30 | 70 | 83,240 | 0.7048 | 0.7273 | 0.9252 | 0.8144 | 0.1905 | 0.5219 |
| 20 | 80 | 72,835 | 0.7785 | 0.8208 | 0.9252 | 0.8699 | 0.1919 | 0.3907 |
| 10 | 90 | 64,740 | 0.8531 | 0.9127 | 0.9252 | 0.9189 | 0.2039 | 0.2324 |
| 0 | 100 | 58,270 | 0.9252 | 1.0000 | 0.9252 | 0.9611 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8384 | 0.0000 | 0.0000 | 0.0000 | 0.8384 | 1.0000 |
| 90 | 10 | 299,940 | 0.8540 | 0.4061 | 0.9942 | 0.5766 | 0.8384 | 0.9992 |
| 80 | 20 | 291,350 | 0.8695 | 0.6059 | 0.9942 | 0.7529 | 0.8383 | 0.9983 |
| 70 | 30 | 194,230 | 0.8851 | 0.7250 | 0.9942 | 0.8385 | 0.8384 | 0.9970 |
| 60 | 40 | 145,675 | 0.9003 | 0.8033 | 0.9942 | 0.8886 | 0.8377 | 0.9954 |
| 50 | 50 | 116,540 | 0.9154 | 0.8589 | 0.9942 | 0.9216 | 0.8367 | 0.9931 |
| 40 | 60 | 97,115 | 0.9322 | 0.9027 | 0.9942 | 0.9463 | 0.8393 | 0.9897 |
| 30 | 70 | 83,240 | 0.9471 | 0.9345 | 0.9942 | 0.9634 | 0.8373 | 0.9840 |
| 20 | 80 | 72,835 | 0.9637 | 0.9617 | 0.9942 | 0.9777 | 0.8417 | 0.9731 |
| 10 | 90 | 64,740 | 0.9785 | 0.9822 | 0.9942 | 0.9882 | 0.8378 | 0.9412 |
| 0 | 100 | 58,270 | 0.9942 | 1.0000 | 0.9942 | 0.9971 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8391 | 0.0000 | 0.0000 | 0.0000 | 0.8391 | 1.0000 |
| 90 | 10 | 299,940 | 0.8545 | 0.4069 | 0.9942 | 0.5774 | 0.8390 | 0.9992 |
| 80 | 20 | 291,350 | 0.8699 | 0.6067 | 0.9942 | 0.7535 | 0.8389 | 0.9983 |
| 70 | 30 | 194,230 | 0.8855 | 0.7257 | 0.9942 | 0.8390 | 0.8390 | 0.9970 |
| 60 | 40 | 145,675 | 0.9007 | 0.8039 | 0.9942 | 0.8890 | 0.8383 | 0.9954 |
| 50 | 50 | 116,540 | 0.9157 | 0.8594 | 0.9942 | 0.9219 | 0.8373 | 0.9931 |
| 40 | 60 | 97,115 | 0.9323 | 0.9028 | 0.9942 | 0.9463 | 0.8394 | 0.9897 |
| 30 | 70 | 83,240 | 0.9474 | 0.9348 | 0.9942 | 0.9636 | 0.8381 | 0.9841 |
| 20 | 80 | 72,835 | 0.9639 | 0.9620 | 0.9942 | 0.9778 | 0.8427 | 0.9731 |
| 10 | 90 | 64,740 | 0.9786 | 0.9822 | 0.9942 | 0.9882 | 0.8380 | 0.9412 |
| 0 | 100 | 58,270 | 0.9942 | 1.0000 | 0.9942 | 0.9971 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8391 | 0.0000 | 0.0000 | 0.0000 | 0.8391 | 1.0000 |
| 90 | 10 | 299,940 | 0.8545 | 0.4069 | 0.9942 | 0.5774 | 0.8390 | 0.9992 |
| 80 | 20 | 291,350 | 0.8699 | 0.6067 | 0.9942 | 0.7535 | 0.8389 | 0.9983 |
| 70 | 30 | 194,230 | 0.8855 | 0.7257 | 0.9942 | 0.8390 | 0.8390 | 0.9970 |
| 60 | 40 | 145,675 | 0.9007 | 0.8039 | 0.9942 | 0.8890 | 0.8383 | 0.9954 |
| 50 | 50 | 116,540 | 0.9157 | 0.8594 | 0.9942 | 0.9219 | 0.8373 | 0.9931 |
| 40 | 60 | 97,115 | 0.9323 | 0.9028 | 0.9942 | 0.9463 | 0.8394 | 0.9897 |
| 30 | 70 | 83,240 | 0.9474 | 0.9348 | 0.9942 | 0.9636 | 0.8381 | 0.9841 |
| 20 | 80 | 72,835 | 0.9639 | 0.9620 | 0.9942 | 0.9778 | 0.8427 | 0.9731 |
| 10 | 90 | 64,740 | 0.9786 | 0.9822 | 0.9942 | 0.9882 | 0.8380 | 0.9412 |
| 0 | 100 | 58,270 | 0.9942 | 1.0000 | 0.9942 | 0.9971 | 0.0000 | 0.0000 |


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
0.15       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130   <--
0.20       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.25       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.30       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.35       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.40       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.45       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.50       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.55       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.60       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.65       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.70       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.75       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
0.80       0.2663   0.2013   0.1932   0.9584   0.9246   0.1130  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2663, F1=0.2013, Normal Recall=0.1932, Normal Precision=0.9584, Attack Recall=0.9246, Attack Precision=0.1130

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
0.15       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227   <--
0.20       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.25       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.30       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.35       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.40       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.45       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.50       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.55       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.60       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.65       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.70       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.75       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
0.80       0.3391   0.3590   0.1926   0.9115   0.9252   0.2227  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3391, F1=0.3590, Normal Recall=0.1926, Normal Precision=0.9115, Attack Recall=0.9252, Attack Precision=0.2227

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
0.15       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295   <--
0.20       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.25       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.30       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.35       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.40       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.45       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.50       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.55       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.60       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.65       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.70       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.75       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
0.80       0.4129   0.4860   0.1933   0.8577   0.9252   0.3295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4129, F1=0.4860, Normal Recall=0.1933, Normal Precision=0.8577, Attack Recall=0.9252, Attack Precision=0.3295

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
0.15       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333   <--
0.20       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.25       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.30       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.35       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.40       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.45       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.50       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.55       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.60       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.65       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.70       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.75       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
0.80       0.4861   0.5902   0.1933   0.7949   0.9252   0.4333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4861, F1=0.5902, Normal Recall=0.1933, Normal Precision=0.7949, Attack Recall=0.9252, Attack Precision=0.4333

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
0.15       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340   <--
0.20       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.25       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.30       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.35       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.40       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.45       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.50       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.55       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.60       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.65       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.70       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.75       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
0.80       0.5589   0.6771   0.1925   0.7202   0.9252   0.5340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5589, F1=0.6771, Normal Recall=0.1925, Normal Precision=0.7202, Attack Recall=0.9252, Attack Precision=0.5340

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
0.15       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062   <--
0.20       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.25       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.30       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.35       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.40       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.45       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.50       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.55       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.60       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.65       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.70       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.75       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
0.80       0.8541   0.5768   0.8384   0.9993   0.9947   0.4062  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8541, F1=0.5768, Normal Recall=0.8384, Normal Precision=0.9993, Attack Recall=0.9947, Attack Precision=0.4062

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
0.15       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067   <--
0.20       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.25       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.30       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.35       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.40       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.45       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.50       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.55       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.60       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.65       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.70       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.75       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
0.80       0.8699   0.7536   0.8389   0.9983   0.9942   0.6067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8699, F1=0.7536, Normal Recall=0.8389, Normal Precision=0.9983, Attack Recall=0.9942, Attack Precision=0.6067

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
0.15       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253   <--
0.20       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.25       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.30       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.35       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.40       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.45       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.50       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.55       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.60       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.65       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.70       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.75       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
0.80       0.8853   0.8387   0.8386   0.9970   0.9942   0.7253  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8853, F1=0.8387, Normal Recall=0.8386, Normal Precision=0.9970, Attack Recall=0.9942, Attack Precision=0.7253

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
0.15       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042   <--
0.20       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.25       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.30       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.35       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.40       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.45       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.50       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.55       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.60       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.65       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.70       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.75       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
0.80       0.9008   0.8891   0.8386   0.9954   0.9942   0.8042  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9008, F1=0.8891, Normal Recall=0.8386, Normal Precision=0.9954, Attack Recall=0.9942, Attack Precision=0.8042

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
0.15       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594   <--
0.20       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.25       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.30       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.35       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.40       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.45       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.50       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.55       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.60       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.65       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.70       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.75       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
0.80       0.9158   0.9219   0.8374   0.9931   0.9942   0.8594  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9158, F1=0.9219, Normal Recall=0.8374, Normal Precision=0.9931, Attack Recall=0.9942, Attack Precision=0.8594

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
0.15       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070   <--
0.20       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.25       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.30       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.35       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.40       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.45       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.50       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.55       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.60       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.65       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.70       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.75       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.80       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8546, F1=0.5777, Normal Recall=0.8390, Normal Precision=0.9993, Attack Recall=0.9947, Attack Precision=0.4070

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
0.15       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075   <--
0.20       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.25       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.30       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.35       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.40       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.45       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.50       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.55       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.60       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.65       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.70       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.75       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.80       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.7542, Normal Recall=0.8394, Normal Precision=0.9983, Attack Recall=0.9942, Attack Precision=0.6075

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
0.15       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261   <--
0.20       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.25       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.30       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.35       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.40       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.45       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.50       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.55       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.60       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.65       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.70       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.75       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.80       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8392, Normal Recall=0.8393, Normal Precision=0.9970, Attack Recall=0.9942, Attack Precision=0.7261

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
0.15       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050   <--
0.20       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.25       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.30       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.35       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.40       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.45       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.50       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.55       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.60       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.65       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.70       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.75       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.80       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9013, F1=0.8896, Normal Recall=0.8394, Normal Precision=0.9954, Attack Recall=0.9942, Attack Precision=0.8050

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
0.15       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600   <--
0.20       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.25       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.30       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.35       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.40       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.45       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.50       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.55       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.60       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.65       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.70       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.75       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.80       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9162, F1=0.9222, Normal Recall=0.8382, Normal Precision=0.9931, Attack Recall=0.9942, Attack Precision=0.8600

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
0.15       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070   <--
0.20       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.25       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.30       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.35       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.40       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.45       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.50       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.55       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.60       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.65       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.70       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.75       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
0.80       0.8546   0.5777   0.8390   0.9993   0.9947   0.4070  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8546, F1=0.5777, Normal Recall=0.8390, Normal Precision=0.9993, Attack Recall=0.9947, Attack Precision=0.4070

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
0.15       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075   <--
0.20       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.25       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.30       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.35       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.40       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.45       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.50       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.55       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.60       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.65       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.70       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.75       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
0.80       0.8704   0.7542   0.8394   0.9983   0.9942   0.6075  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8704, F1=0.7542, Normal Recall=0.8394, Normal Precision=0.9983, Attack Recall=0.9942, Attack Precision=0.6075

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
0.15       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261   <--
0.20       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.25       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.30       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.35       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.40       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.45       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.50       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.55       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.60       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.65       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.70       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.75       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
0.80       0.8857   0.8392   0.8393   0.9970   0.9942   0.7261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.8392, Normal Recall=0.8393, Normal Precision=0.9970, Attack Recall=0.9942, Attack Precision=0.7261

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
0.15       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050   <--
0.20       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.25       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.30       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.35       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.40       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.45       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.50       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.55       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.60       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.65       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.70       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.75       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
0.80       0.9013   0.8896   0.8394   0.9954   0.9942   0.8050  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9013, F1=0.8896, Normal Recall=0.8394, Normal Precision=0.9954, Attack Recall=0.9942, Attack Precision=0.8050

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
0.15       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600   <--
0.20       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.25       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.30       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.35       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.40       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.45       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.50       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.55       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.60       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.65       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.70       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.75       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
0.80       0.9162   0.9222   0.8382   0.9931   0.9942   0.8600  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9162, F1=0.9222, Normal Recall=0.8382, Normal Precision=0.9931, Attack Recall=0.9942, Attack Precision=0.8600

```

