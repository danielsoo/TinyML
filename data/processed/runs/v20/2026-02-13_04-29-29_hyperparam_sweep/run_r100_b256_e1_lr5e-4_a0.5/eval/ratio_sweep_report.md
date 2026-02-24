# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-22 05:13:58 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8495 | 0.8550 | 0.8602 | 0.8665 | 0.8720 | 0.8777 | 0.8837 | 0.8899 | 0.8953 | 0.9005 | 0.9065 |
| QAT+Prune only | 0.5568 | 0.5959 | 0.6343 | 0.6734 | 0.7128 | 0.7505 | 0.7905 | 0.8291 | 0.8696 | 0.9069 | 0.9472 |
| QAT+PTQ | 0.5568 | 0.5958 | 0.6341 | 0.6732 | 0.7124 | 0.7501 | 0.7900 | 0.8285 | 0.8689 | 0.9061 | 0.9464 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5568 | 0.5958 | 0.6341 | 0.6732 | 0.7124 | 0.7501 | 0.7900 | 0.8285 | 0.8689 | 0.9061 | 0.9464 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5558 | 0.7217 | 0.8029 | 0.8499 | 0.8811 | 0.9034 | 0.9202 | 0.9327 | 0.9425 | 0.9509 |
| QAT+Prune only | 0.0000 | 0.3194 | 0.5089 | 0.6350 | 0.7252 | 0.7915 | 0.8444 | 0.8858 | 0.9208 | 0.9482 | 0.9729 |
| QAT+PTQ | 0.0000 | 0.3192 | 0.5085 | 0.6347 | 0.7247 | 0.7911 | 0.8439 | 0.8854 | 0.9203 | 0.9478 | 0.9725 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3192 | 0.5085 | 0.6347 | 0.7247 | 0.7911 | 0.8439 | 0.8854 | 0.9203 | 0.9478 | 0.9725 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8495 | 0.8492 | 0.8486 | 0.8494 | 0.8490 | 0.8490 | 0.8497 | 0.8514 | 0.8506 | 0.8466 | 0.0000 |
| QAT+Prune only | 0.5568 | 0.5568 | 0.5561 | 0.5560 | 0.5566 | 0.5537 | 0.5554 | 0.5535 | 0.5593 | 0.5445 | 0.0000 |
| QAT+PTQ | 0.5568 | 0.5567 | 0.5560 | 0.5560 | 0.5564 | 0.5537 | 0.5553 | 0.5533 | 0.5589 | 0.5439 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5568 | 0.5567 | 0.5560 | 0.5560 | 0.5564 | 0.5537 | 0.5553 | 0.5533 | 0.5589 | 0.5439 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8495 | 0.0000 | 0.0000 | 0.0000 | 0.8495 | 1.0000 |
| 90 | 10 | 299,940 | 0.8550 | 0.4007 | 0.9073 | 0.5558 | 0.8492 | 0.9880 |
| 80 | 20 | 291,350 | 0.8602 | 0.5996 | 0.9065 | 0.7217 | 0.8486 | 0.9732 |
| 70 | 30 | 194,230 | 0.8665 | 0.7206 | 0.9065 | 0.8029 | 0.8494 | 0.9549 |
| 60 | 40 | 145,675 | 0.8720 | 0.8001 | 0.9065 | 0.8499 | 0.8490 | 0.9316 |
| 50 | 50 | 116,540 | 0.8777 | 0.8572 | 0.9065 | 0.8811 | 0.8490 | 0.9008 |
| 40 | 60 | 97,115 | 0.8837 | 0.9004 | 0.9065 | 0.9034 | 0.8497 | 0.8583 |
| 30 | 70 | 83,240 | 0.8899 | 0.9343 | 0.9065 | 0.9202 | 0.8514 | 0.7960 |
| 20 | 80 | 72,835 | 0.8953 | 0.9604 | 0.9065 | 0.9327 | 0.8506 | 0.6945 |
| 10 | 90 | 64,740 | 0.9005 | 0.9815 | 0.9065 | 0.9425 | 0.8466 | 0.5014 |
| 0 | 100 | 58,270 | 0.9065 | 1.0000 | 0.9065 | 0.9509 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5568 | 0.0000 | 0.0000 | 0.0000 | 0.5568 | 1.0000 |
| 90 | 10 | 299,940 | 0.5959 | 0.1921 | 0.9483 | 0.3194 | 0.5568 | 0.9898 |
| 80 | 20 | 291,350 | 0.6343 | 0.3479 | 0.9472 | 0.5089 | 0.5561 | 0.9768 |
| 70 | 30 | 194,230 | 0.6734 | 0.4776 | 0.9472 | 0.6350 | 0.5560 | 0.9609 |
| 60 | 40 | 145,675 | 0.7128 | 0.5875 | 0.9472 | 0.7252 | 0.5566 | 0.9405 |
| 50 | 50 | 116,540 | 0.7505 | 0.6798 | 0.9472 | 0.7915 | 0.5537 | 0.9129 |
| 40 | 60 | 97,115 | 0.7905 | 0.7617 | 0.9472 | 0.8444 | 0.5554 | 0.8752 |
| 30 | 70 | 83,240 | 0.8291 | 0.8319 | 0.9472 | 0.8858 | 0.5535 | 0.8179 |
| 20 | 80 | 72,835 | 0.8696 | 0.8958 | 0.9472 | 0.9208 | 0.5593 | 0.7259 |
| 10 | 90 | 64,740 | 0.9069 | 0.9493 | 0.9472 | 0.9482 | 0.5445 | 0.5339 |
| 0 | 100 | 58,270 | 0.9472 | 1.0000 | 0.9472 | 0.9729 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5568 | 0.0000 | 0.0000 | 0.0000 | 0.5568 | 1.0000 |
| 90 | 10 | 299,940 | 0.5958 | 0.1919 | 0.9476 | 0.3192 | 0.5567 | 0.9896 |
| 80 | 20 | 291,350 | 0.6341 | 0.3477 | 0.9464 | 0.5085 | 0.5560 | 0.9765 |
| 70 | 30 | 194,230 | 0.6732 | 0.4774 | 0.9464 | 0.6347 | 0.5560 | 0.9603 |
| 60 | 40 | 145,675 | 0.7124 | 0.5872 | 0.9464 | 0.7247 | 0.5564 | 0.9397 |
| 50 | 50 | 116,540 | 0.7501 | 0.6796 | 0.9464 | 0.7911 | 0.5537 | 0.9117 |
| 40 | 60 | 97,115 | 0.7900 | 0.7615 | 0.9464 | 0.8439 | 0.5553 | 0.8735 |
| 30 | 70 | 83,240 | 0.8285 | 0.8317 | 0.9464 | 0.8854 | 0.5533 | 0.8156 |
| 20 | 80 | 72,835 | 0.8689 | 0.8956 | 0.9464 | 0.9203 | 0.5589 | 0.7228 |
| 10 | 90 | 64,740 | 0.9061 | 0.9492 | 0.9464 | 0.9478 | 0.5439 | 0.5300 |
| 0 | 100 | 58,270 | 0.9464 | 1.0000 | 0.9464 | 0.9725 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5568 | 0.0000 | 0.0000 | 0.0000 | 0.5568 | 1.0000 |
| 90 | 10 | 299,940 | 0.5958 | 0.1919 | 0.9476 | 0.3192 | 0.5567 | 0.9896 |
| 80 | 20 | 291,350 | 0.6341 | 0.3477 | 0.9464 | 0.5085 | 0.5560 | 0.9765 |
| 70 | 30 | 194,230 | 0.6732 | 0.4774 | 0.9464 | 0.6347 | 0.5560 | 0.9603 |
| 60 | 40 | 145,675 | 0.7124 | 0.5872 | 0.9464 | 0.7247 | 0.5564 | 0.9397 |
| 50 | 50 | 116,540 | 0.7501 | 0.6796 | 0.9464 | 0.7911 | 0.5537 | 0.9117 |
| 40 | 60 | 97,115 | 0.7900 | 0.7615 | 0.9464 | 0.8439 | 0.5553 | 0.8735 |
| 30 | 70 | 83,240 | 0.8285 | 0.8317 | 0.9464 | 0.8854 | 0.5533 | 0.8156 |
| 20 | 80 | 72,835 | 0.8689 | 0.8956 | 0.9464 | 0.9203 | 0.5589 | 0.7228 |
| 10 | 90 | 64,740 | 0.9061 | 0.9492 | 0.9464 | 0.9478 | 0.5439 | 0.5300 |
| 0 | 100 | 58,270 | 0.9464 | 1.0000 | 0.9464 | 0.9725 | 0.0000 | 0.0000 |


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
0.15       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001   <--
0.20       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.25       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.30       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.35       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.40       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.45       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.50       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.55       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.60       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.65       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.70       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.75       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
0.80       0.8548   0.5550   0.8492   0.9878   0.9053   0.4001  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8548, F1=0.5550, Normal Recall=0.8492, Normal Precision=0.9878, Attack Recall=0.9053, Attack Precision=0.4001

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
0.15       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007   <--
0.20       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.25       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.30       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.35       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.40       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.45       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.50       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.55       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.60       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.65       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.70       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.75       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
0.80       0.8608   0.7226   0.8494   0.9732   0.9065   0.6007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8608, F1=0.7226, Normal Recall=0.8494, Normal Precision=0.9732, Attack Recall=0.9065, Attack Precision=0.6007

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
0.15       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215   <--
0.20       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.25       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.30       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.35       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.40       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.45       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.50       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.55       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.60       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.65       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.70       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.75       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
0.80       0.8670   0.8035   0.8500   0.9550   0.9065   0.7215  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8670, F1=0.8035, Normal Recall=0.8500, Normal Precision=0.9550, Attack Recall=0.9065, Attack Precision=0.7215

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
0.15       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018   <--
0.20       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.25       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.30       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.35       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.40       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.45       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.50       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.55       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.60       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.65       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.70       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.75       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
0.80       0.8730   0.8509   0.8506   0.9317   0.9065   0.8018  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8730, F1=0.8509, Normal Recall=0.8506, Normal Precision=0.9317, Attack Recall=0.9065, Attack Precision=0.8018

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
0.15       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588   <--
0.20       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.25       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.30       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.35       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.40       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.45       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.50       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.55       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.60       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.65       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.70       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.75       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
0.80       0.8787   0.8820   0.8510   0.9010   0.9065   0.8588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8787, F1=0.8820, Normal Recall=0.8510, Normal Precision=0.9010, Attack Recall=0.9065, Attack Precision=0.8588

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
0.15       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917   <--
0.20       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.25       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.30       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.35       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.40       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.45       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.50       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.55       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.60       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.65       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.70       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.75       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
0.80       0.5957   0.3188   0.5568   0.9893   0.9461   0.1917  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5957, F1=0.3188, Normal Recall=0.5568, Normal Precision=0.9893, Attack Recall=0.9461, Attack Precision=0.1917

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
0.15       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485   <--
0.20       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.25       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.30       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.35       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.40       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.45       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.50       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.55       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.60       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.65       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.70       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.75       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
0.80       0.6353   0.5095   0.5573   0.9769   0.9472   0.3485  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6353, F1=0.5095, Normal Recall=0.5573, Normal Precision=0.9769, Attack Recall=0.9472, Attack Precision=0.3485

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
0.15       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782   <--
0.20       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.25       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.30       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.35       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.40       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.45       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.50       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.55       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.60       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.65       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.70       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.75       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
0.80       0.6741   0.6356   0.5571   0.9610   0.9472   0.4782  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6741, F1=0.6356, Normal Recall=0.5571, Normal Precision=0.9610, Attack Recall=0.9472, Attack Precision=0.4782

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
0.15       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875   <--
0.20       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.25       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.30       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.35       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.40       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.45       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.50       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.55       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.60       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.65       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.70       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.75       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
0.80       0.7129   0.7252   0.5567   0.9405   0.9472   0.5875  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7129, F1=0.7252, Normal Recall=0.5567, Normal Precision=0.9405, Attack Recall=0.9472, Attack Precision=0.5875

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
0.15       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807   <--
0.20       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.25       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.30       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.35       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.40       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.45       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.50       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.55       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.60       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.65       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.70       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.75       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
0.80       0.7514   0.7921   0.5557   0.9132   0.9472   0.6807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7514, F1=0.7921, Normal Recall=0.5557, Normal Precision=0.9132, Attack Recall=0.9472, Attack Precision=0.6807

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
0.15       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915   <--
0.20       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.25       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.30       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.35       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.40       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.45       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.50       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.55       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.60       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.65       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.70       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.75       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.80       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5955, F1=0.3185, Normal Recall=0.5567, Normal Precision=0.9892, Attack Recall=0.9453, Attack Precision=0.1915

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
0.15       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482   <--
0.20       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.25       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.30       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.35       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.40       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.45       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.50       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.55       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.60       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.65       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.70       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.75       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.80       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6350, F1=0.5091, Normal Recall=0.5572, Normal Precision=0.9765, Attack Recall=0.9464, Attack Precision=0.3482

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
0.15       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780   <--
0.20       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.25       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.30       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.35       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.40       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.45       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.50       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.55       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.60       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.65       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.70       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.75       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.80       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6739, F1=0.6352, Normal Recall=0.5570, Normal Precision=0.9604, Attack Recall=0.9464, Attack Precision=0.4780

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
0.15       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873   <--
0.20       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.25       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.30       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.35       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.40       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.45       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.50       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.55       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.60       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.65       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.70       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.75       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.80       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7125, F1=0.7248, Normal Recall=0.5566, Normal Precision=0.9397, Attack Recall=0.9464, Attack Precision=0.5873

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
0.15       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804   <--
0.20       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.25       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.30       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.35       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.40       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.45       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.50       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.55       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.60       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.65       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.70       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.75       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.80       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7509, F1=0.7917, Normal Recall=0.5555, Normal Precision=0.9120, Attack Recall=0.9464, Attack Precision=0.6804

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
0.15       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915   <--
0.20       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.25       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.30       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.35       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.40       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.45       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.50       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.55       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.60       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.65       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.70       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.75       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
0.80       0.5955   0.3185   0.5567   0.9892   0.9453   0.1915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5955, F1=0.3185, Normal Recall=0.5567, Normal Precision=0.9892, Attack Recall=0.9453, Attack Precision=0.1915

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
0.15       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482   <--
0.20       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.25       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.30       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.35       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.40       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.45       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.50       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.55       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.60       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.65       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.70       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.75       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
0.80       0.6350   0.5091   0.5572   0.9765   0.9464   0.3482  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6350, F1=0.5091, Normal Recall=0.5572, Normal Precision=0.9765, Attack Recall=0.9464, Attack Precision=0.3482

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
0.15       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780   <--
0.20       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.25       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.30       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.35       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.40       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.45       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.50       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.55       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.60       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.65       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.70       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.75       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
0.80       0.6739   0.6352   0.5570   0.9604   0.9464   0.4780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6739, F1=0.6352, Normal Recall=0.5570, Normal Precision=0.9604, Attack Recall=0.9464, Attack Precision=0.4780

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
0.15       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873   <--
0.20       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.25       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.30       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.35       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.40       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.45       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.50       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.55       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.60       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.65       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.70       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.75       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
0.80       0.7125   0.7248   0.5566   0.9397   0.9464   0.5873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7125, F1=0.7248, Normal Recall=0.5566, Normal Precision=0.9397, Attack Recall=0.9464, Attack Precision=0.5873

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
0.15       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804   <--
0.20       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.25       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.30       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.35       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.40       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.45       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.50       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.55       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.60       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.65       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.70       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.75       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
0.80       0.7509   0.7917   0.5555   0.9120   0.9464   0.6804  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7509, F1=0.7917, Normal Recall=0.5555, Normal Precision=0.9120, Attack Recall=0.9464, Attack Precision=0.6804

```

