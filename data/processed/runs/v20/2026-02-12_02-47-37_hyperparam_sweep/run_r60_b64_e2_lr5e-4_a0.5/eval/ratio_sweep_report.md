# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-12 21:10:06 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3010 | 0.3552 | 0.4090 | 0.4643 | 0.5167 | 0.5727 | 0.6257 | 0.6784 | 0.7331 | 0.7885 | 0.8414 |
| QAT+Prune only | 0.7490 | 0.7735 | 0.7977 | 0.8228 | 0.8481 | 0.8709 | 0.8969 | 0.9201 | 0.9441 | 0.9694 | 0.9941 |
| QAT+PTQ | 0.7509 | 0.7752 | 0.7992 | 0.8244 | 0.8495 | 0.8721 | 0.8981 | 0.9211 | 0.9448 | 0.9704 | 0.9947 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7509 | 0.7752 | 0.7992 | 0.8244 | 0.8495 | 0.8721 | 0.8981 | 0.9211 | 0.9448 | 0.9704 | 0.9947 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2072 | 0.3629 | 0.4852 | 0.5821 | 0.6632 | 0.7296 | 0.7856 | 0.8345 | 0.8775 | 0.9139 |
| QAT+Prune only | 0.0000 | 0.4676 | 0.6628 | 0.7709 | 0.8396 | 0.8850 | 0.9205 | 0.9457 | 0.9660 | 0.9832 | 0.9970 |
| QAT+PTQ | 0.0000 | 0.4697 | 0.6646 | 0.7727 | 0.8410 | 0.8860 | 0.9214 | 0.9464 | 0.9665 | 0.9837 | 0.9974 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4697 | 0.6646 | 0.7727 | 0.8410 | 0.8860 | 0.9214 | 0.9464 | 0.9665 | 0.9837 | 0.9974 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3010 | 0.3010 | 0.3009 | 0.3027 | 0.3001 | 0.3040 | 0.3021 | 0.2981 | 0.2995 | 0.3125 | 0.0000 |
| QAT+Prune only | 0.7490 | 0.7489 | 0.7486 | 0.7494 | 0.7507 | 0.7476 | 0.7512 | 0.7474 | 0.7441 | 0.7470 | 0.0000 |
| QAT+PTQ | 0.7509 | 0.7508 | 0.7504 | 0.7514 | 0.7527 | 0.7494 | 0.7532 | 0.7492 | 0.7452 | 0.7510 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7509 | 0.7508 | 0.7504 | 0.7514 | 0.7527 | 0.7494 | 0.7532 | 0.7492 | 0.7452 | 0.7510 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3010 | 0.0000 | 0.0000 | 0.0000 | 0.3010 | 1.0000 |
| 90 | 10 | 299,940 | 0.3552 | 0.1181 | 0.8427 | 0.2072 | 0.3010 | 0.9451 |
| 80 | 20 | 291,350 | 0.4090 | 0.2313 | 0.8414 | 0.3629 | 0.3009 | 0.8836 |
| 70 | 30 | 194,230 | 0.4643 | 0.3409 | 0.8414 | 0.4852 | 0.3027 | 0.8166 |
| 60 | 40 | 145,675 | 0.5167 | 0.4449 | 0.8414 | 0.5821 | 0.3001 | 0.7395 |
| 50 | 50 | 116,540 | 0.5727 | 0.5473 | 0.8414 | 0.6632 | 0.3040 | 0.6572 |
| 40 | 60 | 97,115 | 0.6257 | 0.6440 | 0.8414 | 0.7296 | 0.3021 | 0.5595 |
| 30 | 70 | 83,240 | 0.6784 | 0.7366 | 0.8414 | 0.7856 | 0.2981 | 0.4462 |
| 20 | 80 | 72,835 | 0.7331 | 0.8277 | 0.8415 | 0.8345 | 0.2995 | 0.3208 |
| 10 | 90 | 64,740 | 0.7885 | 0.9168 | 0.8414 | 0.8775 | 0.3125 | 0.1796 |
| 0 | 100 | 58,270 | 0.8414 | 1.0000 | 0.8414 | 0.9139 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7490 | 0.0000 | 0.0000 | 0.0000 | 0.7490 | 1.0000 |
| 90 | 10 | 299,940 | 0.7735 | 0.3056 | 0.9945 | 0.4676 | 0.7489 | 0.9992 |
| 80 | 20 | 291,350 | 0.7977 | 0.4971 | 0.9941 | 0.6628 | 0.7486 | 0.9980 |
| 70 | 30 | 194,230 | 0.8228 | 0.6296 | 0.9941 | 0.7709 | 0.7494 | 0.9966 |
| 60 | 40 | 145,675 | 0.8481 | 0.7267 | 0.9941 | 0.8396 | 0.7507 | 0.9948 |
| 50 | 50 | 116,540 | 0.8709 | 0.7975 | 0.9941 | 0.8850 | 0.7476 | 0.9921 |
| 40 | 60 | 97,115 | 0.8969 | 0.8570 | 0.9941 | 0.9205 | 0.7512 | 0.9883 |
| 30 | 70 | 83,240 | 0.9201 | 0.9018 | 0.9941 | 0.9457 | 0.7474 | 0.9819 |
| 20 | 80 | 72,835 | 0.9441 | 0.9395 | 0.9941 | 0.9660 | 0.7441 | 0.9692 |
| 10 | 90 | 64,740 | 0.9694 | 0.9725 | 0.9941 | 0.9832 | 0.7470 | 0.9334 |
| 0 | 100 | 58,270 | 0.9941 | 1.0000 | 0.9941 | 0.9970 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7509 | 0.0000 | 0.0000 | 0.0000 | 0.7509 | 1.0000 |
| 90 | 10 | 299,940 | 0.7752 | 0.3074 | 0.9953 | 0.4697 | 0.7508 | 0.9993 |
| 80 | 20 | 291,350 | 0.7992 | 0.4990 | 0.9947 | 0.6646 | 0.7504 | 0.9982 |
| 70 | 30 | 194,230 | 0.8244 | 0.6316 | 0.9947 | 0.7727 | 0.7514 | 0.9970 |
| 60 | 40 | 145,675 | 0.8495 | 0.7284 | 0.9947 | 0.8410 | 0.7527 | 0.9954 |
| 50 | 50 | 116,540 | 0.8721 | 0.7988 | 0.9947 | 0.8860 | 0.7494 | 0.9930 |
| 40 | 60 | 97,115 | 0.8981 | 0.8581 | 0.9947 | 0.9214 | 0.7532 | 0.9896 |
| 30 | 70 | 83,240 | 0.9211 | 0.9025 | 0.9947 | 0.9464 | 0.7492 | 0.9839 |
| 20 | 80 | 72,835 | 0.9448 | 0.9398 | 0.9947 | 0.9665 | 0.7452 | 0.9725 |
| 10 | 90 | 64,740 | 0.9704 | 0.9729 | 0.9947 | 0.9837 | 0.7510 | 0.9406 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9974 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7509 | 0.0000 | 0.0000 | 0.0000 | 0.7509 | 1.0000 |
| 90 | 10 | 299,940 | 0.7752 | 0.3074 | 0.9953 | 0.4697 | 0.7508 | 0.9993 |
| 80 | 20 | 291,350 | 0.7992 | 0.4990 | 0.9947 | 0.6646 | 0.7504 | 0.9982 |
| 70 | 30 | 194,230 | 0.8244 | 0.6316 | 0.9947 | 0.7727 | 0.7514 | 0.9970 |
| 60 | 40 | 145,675 | 0.8495 | 0.7284 | 0.9947 | 0.8410 | 0.7527 | 0.9954 |
| 50 | 50 | 116,540 | 0.8721 | 0.7988 | 0.9947 | 0.8860 | 0.7494 | 0.9930 |
| 40 | 60 | 97,115 | 0.8981 | 0.8581 | 0.9947 | 0.9214 | 0.7532 | 0.9896 |
| 30 | 70 | 83,240 | 0.9211 | 0.9025 | 0.9947 | 0.9464 | 0.7492 | 0.9839 |
| 20 | 80 | 72,835 | 0.9448 | 0.9398 | 0.9947 | 0.9665 | 0.7452 | 0.9725 |
| 10 | 90 | 64,740 | 0.9704 | 0.9729 | 0.9947 | 0.9837 | 0.7510 | 0.9406 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9974 | 0.0000 | 0.0000 |


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
0.15       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183   <--
0.20       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.25       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.30       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.35       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.40       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.45       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.50       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.55       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.60       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.65       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.70       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.75       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
0.80       0.3553   0.2075   0.3010   0.9456   0.8442   0.1183  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3553, F1=0.2075, Normal Recall=0.3010, Normal Precision=0.9456, Attack Recall=0.8442, Attack Precision=0.1183

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
0.15       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313   <--
0.20       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.25       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.30       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.35       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.40       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.45       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.50       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.55       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.60       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.65       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.70       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.75       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
0.80       0.4089   0.3628   0.3008   0.8836   0.8414   0.2313  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4089, F1=0.3628, Normal Recall=0.3008, Normal Precision=0.8836, Attack Recall=0.8414, Attack Precision=0.2313

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
0.15       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402   <--
0.20       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.25       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.30       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.35       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.40       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.45       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.50       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.55       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.60       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.65       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.70       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.75       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
0.80       0.4629   0.4845   0.3007   0.8157   0.8414   0.3402  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4629, F1=0.4845, Normal Recall=0.3007, Normal Precision=0.8157, Attack Recall=0.8414, Attack Precision=0.3402

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
0.15       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453   <--
0.20       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.25       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.30       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.35       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.40       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.45       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.50       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.55       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.60       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.65       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.70       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.75       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
0.80       0.5174   0.5824   0.3013   0.7403   0.8414   0.4453  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5174, F1=0.5824, Normal Recall=0.3013, Normal Precision=0.7403, Attack Recall=0.8414, Attack Precision=0.4453

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
0.15       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464   <--
0.20       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.25       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.30       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.35       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.40       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.45       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.50       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.55       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.60       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.65       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.70       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.75       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
0.80       0.5715   0.6626   0.3015   0.6554   0.8414   0.5464  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5715, F1=0.6626, Normal Recall=0.3015, Normal Precision=0.6554, Attack Recall=0.8414, Attack Precision=0.5464

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
0.15       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056   <--
0.20       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.25       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.30       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.35       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.40       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.45       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.50       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.55       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.60       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.65       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.70       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.75       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
0.80       0.7735   0.4675   0.7489   0.9992   0.9943   0.3056  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7735, F1=0.4675, Normal Recall=0.7489, Normal Precision=0.9992, Attack Recall=0.9943, Attack Precision=0.3056

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
0.15       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977   <--
0.20       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.25       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.30       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.35       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.40       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.45       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.50       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.55       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.60       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.65       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.70       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.75       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
0.80       0.7982   0.6633   0.7492   0.9980   0.9941   0.4977  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7982, F1=0.6633, Normal Recall=0.7492, Normal Precision=0.9980, Attack Recall=0.9941, Attack Precision=0.4977

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
0.15       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295   <--
0.20       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.25       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.30       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.35       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.40       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.45       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.50       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.55       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.60       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.65       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.70       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.75       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
0.80       0.8227   0.7709   0.7493   0.9966   0.9941   0.6295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8227, F1=0.7709, Normal Recall=0.7493, Normal Precision=0.9966, Attack Recall=0.9941, Attack Precision=0.6295

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
0.15       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251   <--
0.20       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.25       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.30       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.35       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.40       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.45       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.50       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.55       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.60       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.65       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.70       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.75       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
0.80       0.8469   0.8386   0.7488   0.9948   0.9941   0.7251  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8469, F1=0.8386, Normal Recall=0.7488, Normal Precision=0.9948, Attack Recall=0.9941, Attack Precision=0.7251

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
0.15       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971   <--
0.20       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.25       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.30       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.35       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.40       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.45       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.50       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.55       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.60       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.65       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.70       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.75       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
0.80       0.8705   0.8848   0.7470   0.9921   0.9941   0.7971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8705, F1=0.8848, Normal Recall=0.7470, Normal Precision=0.9921, Attack Recall=0.9941, Attack Precision=0.7971

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
0.15       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073   <--
0.20       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.25       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.30       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.35       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.40       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.45       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.50       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.55       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.60       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.65       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.70       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.75       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.80       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7752, F1=0.4695, Normal Recall=0.7508, Normal Precision=0.9992, Attack Recall=0.9948, Attack Precision=0.3073

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
0.15       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997   <--
0.20       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.25       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.30       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.35       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.40       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.45       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.50       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.55       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.60       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.65       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.70       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.75       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.80       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7997, F1=0.6652, Normal Recall=0.7510, Normal Precision=0.9982, Attack Recall=0.9947, Attack Precision=0.4997

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
0.15       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315   <--
0.20       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.25       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.30       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.35       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.40       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.45       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.50       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.55       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.60       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.65       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.70       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.75       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.80       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8243, F1=0.7726, Normal Recall=0.7512, Normal Precision=0.9970, Attack Recall=0.9947, Attack Precision=0.6315

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
0.15       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268   <--
0.20       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.25       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.30       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.35       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.40       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.45       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.50       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.55       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.60       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.65       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.70       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.75       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.80       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8483, F1=0.8399, Normal Recall=0.7507, Normal Precision=0.9953, Attack Recall=0.9947, Attack Precision=0.7268

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
0.15       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987   <--
0.20       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.25       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.30       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.35       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.40       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.45       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.50       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.55       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.60       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.65       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.70       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.75       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.80       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8720, F1=0.8860, Normal Recall=0.7492, Normal Precision=0.9930, Attack Recall=0.9947, Attack Precision=0.7987

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
0.15       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073   <--
0.20       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.25       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.30       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.35       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.40       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.45       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.50       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.55       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.60       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.65       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.70       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.75       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
0.80       0.7752   0.4695   0.7508   0.9992   0.9948   0.3073  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7752, F1=0.4695, Normal Recall=0.7508, Normal Precision=0.9992, Attack Recall=0.9948, Attack Precision=0.3073

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
0.15       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997   <--
0.20       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.25       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.30       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.35       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.40       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.45       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.50       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.55       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.60       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.65       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.70       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.75       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
0.80       0.7997   0.6652   0.7510   0.9982   0.9947   0.4997  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7997, F1=0.6652, Normal Recall=0.7510, Normal Precision=0.9982, Attack Recall=0.9947, Attack Precision=0.4997

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
0.15       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315   <--
0.20       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.25       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.30       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.35       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.40       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.45       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.50       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.55       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.60       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.65       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.70       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.75       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
0.80       0.8243   0.7726   0.7512   0.9970   0.9947   0.6315  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8243, F1=0.7726, Normal Recall=0.7512, Normal Precision=0.9970, Attack Recall=0.9947, Attack Precision=0.6315

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
0.15       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268   <--
0.20       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.25       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.30       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.35       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.40       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.45       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.50       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.55       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.60       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.65       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.70       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.75       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
0.80       0.8483   0.8399   0.7507   0.9953   0.9947   0.7268  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8483, F1=0.8399, Normal Recall=0.7507, Normal Precision=0.9953, Attack Recall=0.9947, Attack Precision=0.7268

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
0.15       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987   <--
0.20       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.25       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.30       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.35       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.40       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.45       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.50       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.55       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.60       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.65       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.70       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.75       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
0.80       0.8720   0.8860   0.7492   0.9930   0.9947   0.7987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8720, F1=0.8860, Normal Recall=0.7492, Normal Precision=0.9930, Attack Recall=0.9947, Attack Precision=0.7987

```

