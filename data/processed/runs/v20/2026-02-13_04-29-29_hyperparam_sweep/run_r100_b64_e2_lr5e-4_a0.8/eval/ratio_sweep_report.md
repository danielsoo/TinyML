# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-20 12:29:12 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6119 | 0.6282 | 0.6443 | 0.6612 | 0.6767 | 0.6958 | 0.7105 | 0.7278 | 0.7438 | 0.7605 | 0.7773 |
| QAT+Prune only | 0.8485 | 0.8645 | 0.8785 | 0.8941 | 0.9081 | 0.9217 | 0.9377 | 0.9528 | 0.9664 | 0.9806 | 0.9957 |
| QAT+PTQ | 0.8467 | 0.8631 | 0.8773 | 0.8928 | 0.9074 | 0.9211 | 0.9370 | 0.9525 | 0.9660 | 0.9803 | 0.9957 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8467 | 0.8631 | 0.8773 | 0.8928 | 0.9074 | 0.9211 | 0.9370 | 0.9525 | 0.9660 | 0.9803 | 0.9957 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2953 | 0.4664 | 0.5792 | 0.6579 | 0.7187 | 0.7631 | 0.7999 | 0.8292 | 0.8538 | 0.8747 |
| QAT+Prune only | 0.0000 | 0.5952 | 0.7663 | 0.8495 | 0.8966 | 0.9271 | 0.9505 | 0.9672 | 0.9793 | 0.9893 | 0.9978 |
| QAT+PTQ | 0.0000 | 0.5927 | 0.7645 | 0.8479 | 0.8958 | 0.9266 | 0.9499 | 0.9671 | 0.9791 | 0.9891 | 0.9978 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5927 | 0.7645 | 0.8479 | 0.8958 | 0.9266 | 0.9499 | 0.9671 | 0.9791 | 0.9891 | 0.9978 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6119 | 0.6114 | 0.6110 | 0.6114 | 0.6096 | 0.6143 | 0.6102 | 0.6121 | 0.6095 | 0.6086 | 0.0000 |
| QAT+Prune only | 0.8485 | 0.8499 | 0.8493 | 0.8506 | 0.8498 | 0.8477 | 0.8508 | 0.8527 | 0.8492 | 0.8449 | 0.0000 |
| QAT+PTQ | 0.8467 | 0.8484 | 0.8477 | 0.8488 | 0.8485 | 0.8465 | 0.8490 | 0.8519 | 0.8472 | 0.8423 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8467 | 0.8484 | 0.8477 | 0.8488 | 0.8485 | 0.8465 | 0.8490 | 0.8519 | 0.8472 | 0.8423 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6119 | 0.0000 | 0.0000 | 0.0000 | 0.6119 | 1.0000 |
| 90 | 10 | 299,940 | 0.6282 | 0.1822 | 0.7790 | 0.2953 | 0.6114 | 0.9614 |
| 80 | 20 | 291,350 | 0.6443 | 0.3332 | 0.7773 | 0.4664 | 0.6110 | 0.9165 |
| 70 | 30 | 194,230 | 0.6612 | 0.4616 | 0.7773 | 0.5792 | 0.6114 | 0.8650 |
| 60 | 40 | 145,675 | 0.6767 | 0.5703 | 0.7773 | 0.6579 | 0.6096 | 0.8042 |
| 50 | 50 | 116,540 | 0.6958 | 0.6684 | 0.7773 | 0.7187 | 0.6143 | 0.7340 |
| 40 | 60 | 97,115 | 0.7105 | 0.7494 | 0.7773 | 0.7631 | 0.6102 | 0.6463 |
| 30 | 70 | 83,240 | 0.7278 | 0.8238 | 0.7773 | 0.7999 | 0.6121 | 0.5409 |
| 20 | 80 | 72,835 | 0.7438 | 0.8884 | 0.7774 | 0.8292 | 0.6095 | 0.4063 |
| 10 | 90 | 64,740 | 0.7605 | 0.9470 | 0.7773 | 0.8538 | 0.6086 | 0.2329 |
| 0 | 100 | 58,270 | 0.7773 | 1.0000 | 0.7773 | 0.8747 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8485 | 0.0000 | 0.0000 | 0.0000 | 0.8485 | 1.0000 |
| 90 | 10 | 299,940 | 0.8645 | 0.4244 | 0.9958 | 0.5952 | 0.8499 | 0.9995 |
| 80 | 20 | 291,350 | 0.8785 | 0.6228 | 0.9957 | 0.7663 | 0.8493 | 0.9987 |
| 70 | 30 | 194,230 | 0.8941 | 0.7407 | 0.9957 | 0.8495 | 0.8506 | 0.9978 |
| 60 | 40 | 145,675 | 0.9081 | 0.8154 | 0.9957 | 0.8966 | 0.8498 | 0.9966 |
| 50 | 50 | 116,540 | 0.9217 | 0.8674 | 0.9957 | 0.9271 | 0.8477 | 0.9949 |
| 40 | 60 | 97,115 | 0.9377 | 0.9092 | 0.9957 | 0.9505 | 0.8508 | 0.9924 |
| 30 | 70 | 83,240 | 0.9528 | 0.9404 | 0.9957 | 0.9672 | 0.8527 | 0.9883 |
| 20 | 80 | 72,835 | 0.9664 | 0.9635 | 0.9957 | 0.9793 | 0.8492 | 0.9800 |
| 10 | 90 | 64,740 | 0.9806 | 0.9830 | 0.9957 | 0.9893 | 0.8449 | 0.9560 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9978 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8467 | 0.0000 | 0.0000 | 0.0000 | 0.8467 | 1.0000 |
| 90 | 10 | 299,940 | 0.8631 | 0.4219 | 0.9958 | 0.5927 | 0.8484 | 0.9995 |
| 80 | 20 | 291,350 | 0.8773 | 0.6204 | 0.9957 | 0.7645 | 0.8477 | 0.9987 |
| 70 | 30 | 194,230 | 0.8928 | 0.7383 | 0.9957 | 0.8479 | 0.8488 | 0.9978 |
| 60 | 40 | 145,675 | 0.9074 | 0.8142 | 0.9957 | 0.8958 | 0.8485 | 0.9966 |
| 50 | 50 | 116,540 | 0.9211 | 0.8664 | 0.9957 | 0.9266 | 0.8465 | 0.9949 |
| 40 | 60 | 97,115 | 0.9370 | 0.9082 | 0.9957 | 0.9499 | 0.8490 | 0.9924 |
| 30 | 70 | 83,240 | 0.9525 | 0.9401 | 0.9957 | 0.9671 | 0.8519 | 0.9883 |
| 20 | 80 | 72,835 | 0.9660 | 0.9630 | 0.9957 | 0.9791 | 0.8472 | 0.9800 |
| 10 | 90 | 64,740 | 0.9803 | 0.9827 | 0.9957 | 0.9891 | 0.8423 | 0.9558 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9978 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8467 | 0.0000 | 0.0000 | 0.0000 | 0.8467 | 1.0000 |
| 90 | 10 | 299,940 | 0.8631 | 0.4219 | 0.9958 | 0.5927 | 0.8484 | 0.9995 |
| 80 | 20 | 291,350 | 0.8773 | 0.6204 | 0.9957 | 0.7645 | 0.8477 | 0.9987 |
| 70 | 30 | 194,230 | 0.8928 | 0.7383 | 0.9957 | 0.8479 | 0.8488 | 0.9978 |
| 60 | 40 | 145,675 | 0.9074 | 0.8142 | 0.9957 | 0.8958 | 0.8485 | 0.9966 |
| 50 | 50 | 116,540 | 0.9211 | 0.8664 | 0.9957 | 0.9266 | 0.8465 | 0.9949 |
| 40 | 60 | 97,115 | 0.9370 | 0.9082 | 0.9957 | 0.9499 | 0.8490 | 0.9924 |
| 30 | 70 | 83,240 | 0.9525 | 0.9401 | 0.9957 | 0.9671 | 0.8519 | 0.9883 |
| 20 | 80 | 72,835 | 0.9660 | 0.9630 | 0.9957 | 0.9791 | 0.8472 | 0.9800 |
| 10 | 90 | 64,740 | 0.9803 | 0.9827 | 0.9957 | 0.9891 | 0.8423 | 0.9558 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9978 | 0.0000 | 0.0000 |


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
0.15       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822   <--
0.20       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.25       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.30       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.35       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.40       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.45       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.50       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.55       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.60       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.65       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.70       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.75       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
0.80       0.6282   0.2953   0.6115   0.9614   0.7791   0.1822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6282, F1=0.2953, Normal Recall=0.6115, Normal Precision=0.9614, Attack Recall=0.7791, Attack Precision=0.1822

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
0.15       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335   <--
0.20       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.25       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.30       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.35       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.40       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.45       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.50       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.55       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.60       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.65       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.70       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.75       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
0.80       0.6448   0.4668   0.6117   0.9166   0.7773   0.3335  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6448, F1=0.4668, Normal Recall=0.6117, Normal Precision=0.9166, Attack Recall=0.7773, Attack Precision=0.3335

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
0.15       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616   <--
0.20       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.25       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.30       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.35       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.40       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.45       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.50       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.55       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.60       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.65       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.70       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.75       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
0.80       0.6612   0.5793   0.6115   0.8650   0.7773   0.4616  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6612, F1=0.5793, Normal Recall=0.6115, Normal Precision=0.8650, Attack Recall=0.7773, Attack Precision=0.4616

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
0.15       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716   <--
0.20       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.25       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.30       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.35       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.40       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.45       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.50       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.55       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.60       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.65       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.70       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.75       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
0.80       0.6779   0.6588   0.6116   0.8047   0.7773   0.5716  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6779, F1=0.6588, Normal Recall=0.6116, Normal Precision=0.8047, Attack Recall=0.7773, Attack Precision=0.5716

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
0.15       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671   <--
0.20       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.25       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.30       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.35       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.40       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.45       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.50       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.55       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.60       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.65       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.70       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.75       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
0.80       0.6947   0.7180   0.6121   0.7333   0.7773   0.6671  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6947, F1=0.7180, Normal Recall=0.6121, Normal Precision=0.7333, Attack Recall=0.7773, Attack Precision=0.6671

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
0.15       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245   <--
0.20       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.25       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.30       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.35       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.40       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.45       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.50       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.55       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.60       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.65       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.70       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.75       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
0.80       0.8646   0.5953   0.8499   0.9995   0.9962   0.4245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8646, F1=0.5953, Normal Recall=0.8499, Normal Precision=0.9995, Attack Recall=0.9962, Attack Precision=0.4245

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
0.15       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242   <--
0.20       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.25       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.30       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.35       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.40       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.45       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.50       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.55       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.60       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.65       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.70       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.75       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
0.80       0.8792   0.7673   0.8501   0.9987   0.9957   0.6242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8792, F1=0.7673, Normal Recall=0.8501, Normal Precision=0.9987, Attack Recall=0.9957, Attack Precision=0.6242

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
0.15       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386   <--
0.20       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.25       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.30       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.35       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.40       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.45       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.50       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.55       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.60       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.65       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.70       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.75       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
0.80       0.8930   0.8481   0.8490   0.9978   0.9957   0.7386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8930, F1=0.8481, Normal Recall=0.8490, Normal Precision=0.9978, Attack Recall=0.9957, Attack Precision=0.7386

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
0.15       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141   <--
0.20       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.25       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.30       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.35       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.40       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.45       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.50       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.55       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.60       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.65       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.70       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.75       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
0.80       0.9073   0.8958   0.8485   0.9966   0.9957   0.8141  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9073, F1=0.8958, Normal Recall=0.8485, Normal Precision=0.9966, Attack Recall=0.9957, Attack Precision=0.8141

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
0.15       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671   <--
0.20       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.25       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.30       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.35       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.40       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.45       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.50       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.55       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.60       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.65       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.70       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.75       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
0.80       0.9215   0.9269   0.8473   0.9949   0.9957   0.8671  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9215, F1=0.9269, Normal Recall=0.8473, Normal Precision=0.9949, Attack Recall=0.9957, Attack Precision=0.8671

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
0.15       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220   <--
0.20       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.25       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.30       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.35       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.40       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.45       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.50       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.55       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.60       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.65       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.70       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.75       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.80       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8632, F1=0.5928, Normal Recall=0.8484, Normal Precision=0.9995, Attack Recall=0.9962, Attack Precision=0.4220

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
0.15       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216   <--
0.20       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.25       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.30       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.35       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.40       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.45       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.50       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.55       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.60       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.65       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.70       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.75       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.80       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8779, F1=0.7654, Normal Recall=0.8485, Normal Precision=0.9987, Attack Recall=0.9957, Attack Precision=0.6216

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
0.15       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364   <--
0.20       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.25       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.30       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.35       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.40       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.45       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.50       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.55       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.60       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.65       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.70       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.75       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.80       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8918, F1=0.8466, Normal Recall=0.8472, Normal Precision=0.9978, Attack Recall=0.9957, Attack Precision=0.7364

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
0.15       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123   <--
0.20       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.25       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.30       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.35       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.40       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.45       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.50       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.55       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.60       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.65       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.70       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.75       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.80       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9062, F1=0.8947, Normal Recall=0.8466, Normal Precision=0.9966, Attack Recall=0.9957, Attack Precision=0.8123

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
0.15       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656   <--
0.20       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.25       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.30       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.35       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.40       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.45       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.50       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.55       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.60       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.65       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.70       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.75       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.80       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9206, F1=0.9261, Normal Recall=0.8454, Normal Precision=0.9949, Attack Recall=0.9957, Attack Precision=0.8656

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
0.15       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220   <--
0.20       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.25       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.30       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.35       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.40       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.45       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.50       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.55       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.60       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.65       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.70       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.75       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
0.80       0.8632   0.5928   0.8484   0.9995   0.9962   0.4220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8632, F1=0.5928, Normal Recall=0.8484, Normal Precision=0.9995, Attack Recall=0.9962, Attack Precision=0.4220

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
0.15       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216   <--
0.20       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.25       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.30       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.35       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.40       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.45       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.50       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.55       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.60       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.65       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.70       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.75       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
0.80       0.8779   0.7654   0.8485   0.9987   0.9957   0.6216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8779, F1=0.7654, Normal Recall=0.8485, Normal Precision=0.9987, Attack Recall=0.9957, Attack Precision=0.6216

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
0.15       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364   <--
0.20       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.25       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.30       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.35       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.40       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.45       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.50       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.55       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.60       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.65       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.70       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.75       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
0.80       0.8918   0.8466   0.8472   0.9978   0.9957   0.7364  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8918, F1=0.8466, Normal Recall=0.8472, Normal Precision=0.9978, Attack Recall=0.9957, Attack Precision=0.7364

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
0.15       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123   <--
0.20       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.25       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.30       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.35       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.40       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.45       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.50       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.55       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.60       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.65       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.70       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.75       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
0.80       0.9062   0.8947   0.8466   0.9966   0.9957   0.8123  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9062, F1=0.8947, Normal Recall=0.8466, Normal Precision=0.9966, Attack Recall=0.9957, Attack Precision=0.8123

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
0.15       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656   <--
0.20       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.25       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.30       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.35       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.40       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.45       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.50       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.55       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.60       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.65       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.70       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.75       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
0.80       0.9206   0.9261   0.8454   0.9949   0.9957   0.8656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9206, F1=0.9261, Normal Recall=0.8454, Normal Precision=0.9949, Attack Recall=0.9957, Attack Precision=0.8656

```

