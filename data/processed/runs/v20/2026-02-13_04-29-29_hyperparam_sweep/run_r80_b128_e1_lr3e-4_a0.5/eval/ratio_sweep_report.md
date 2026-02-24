# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-17 05:37:24 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6986 | 0.7281 | 0.7572 | 0.7871 | 0.8165 | 0.8454 | 0.8743 | 0.9055 | 0.9332 | 0.9628 | 0.9921 |
| QAT+Prune only | 0.6598 | 0.6877 | 0.7145 | 0.7419 | 0.7693 | 0.7960 | 0.8231 | 0.8507 | 0.8791 | 0.9055 | 0.9339 |
| QAT+PTQ | 0.6602 | 0.6879 | 0.7148 | 0.7421 | 0.7694 | 0.7961 | 0.8233 | 0.8507 | 0.8791 | 0.9054 | 0.9337 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6602 | 0.6879 | 0.7148 | 0.7421 | 0.7694 | 0.7961 | 0.8233 | 0.8507 | 0.8791 | 0.9054 | 0.9337 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4218 | 0.6204 | 0.7366 | 0.8122 | 0.8652 | 0.9045 | 0.9363 | 0.9596 | 0.9796 | 0.9960 |
| QAT+Prune only | 0.0000 | 0.3742 | 0.5668 | 0.6846 | 0.7640 | 0.8207 | 0.8637 | 0.8975 | 0.9252 | 0.9468 | 0.9658 |
| QAT+PTQ | 0.0000 | 0.3743 | 0.5670 | 0.6848 | 0.7641 | 0.8207 | 0.8638 | 0.8975 | 0.9252 | 0.9467 | 0.9657 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3743 | 0.5670 | 0.6848 | 0.7641 | 0.8207 | 0.8638 | 0.8975 | 0.9252 | 0.9467 | 0.9657 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6986 | 0.6988 | 0.6985 | 0.6992 | 0.6994 | 0.6987 | 0.6976 | 0.7035 | 0.6974 | 0.6985 | 0.0000 |
| QAT+Prune only | 0.6598 | 0.6603 | 0.6597 | 0.6596 | 0.6595 | 0.6581 | 0.6570 | 0.6565 | 0.6601 | 0.6501 | 0.0000 |
| QAT+PTQ | 0.6602 | 0.6607 | 0.6601 | 0.6600 | 0.6598 | 0.6584 | 0.6576 | 0.6570 | 0.6607 | 0.6504 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6602 | 0.6607 | 0.6601 | 0.6600 | 0.6598 | 0.6584 | 0.6576 | 0.6570 | 0.6607 | 0.6504 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6986 | 0.0000 | 0.0000 | 0.0000 | 0.6986 | 1.0000 |
| 90 | 10 | 299,940 | 0.7281 | 0.2678 | 0.9918 | 0.4218 | 0.6988 | 0.9987 |
| 80 | 20 | 291,350 | 0.7572 | 0.4513 | 0.9921 | 0.6204 | 0.6985 | 0.9972 |
| 70 | 30 | 194,230 | 0.7871 | 0.5857 | 0.9921 | 0.7366 | 0.6992 | 0.9952 |
| 60 | 40 | 145,675 | 0.8165 | 0.6875 | 0.9921 | 0.8122 | 0.6994 | 0.9925 |
| 50 | 50 | 116,540 | 0.8454 | 0.7670 | 0.9921 | 0.8652 | 0.6987 | 0.9889 |
| 40 | 60 | 97,115 | 0.8743 | 0.8311 | 0.9921 | 0.9045 | 0.6976 | 0.9833 |
| 30 | 70 | 83,240 | 0.9055 | 0.8865 | 0.9921 | 0.9363 | 0.7035 | 0.9745 |
| 20 | 80 | 72,835 | 0.9332 | 0.9292 | 0.9921 | 0.9596 | 0.6974 | 0.9568 |
| 10 | 90 | 64,740 | 0.9628 | 0.9673 | 0.9921 | 0.9796 | 0.6985 | 0.9078 |
| 0 | 100 | 58,270 | 0.9921 | 1.0000 | 0.9921 | 0.9960 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6598 | 0.0000 | 0.0000 | 0.0000 | 0.6598 | 1.0000 |
| 90 | 10 | 299,940 | 0.6877 | 0.2340 | 0.9337 | 0.3742 | 0.6603 | 0.9890 |
| 80 | 20 | 291,350 | 0.7145 | 0.4069 | 0.9339 | 0.5668 | 0.6597 | 0.9756 |
| 70 | 30 | 194,230 | 0.7419 | 0.5404 | 0.9339 | 0.6846 | 0.6596 | 0.9588 |
| 60 | 40 | 145,675 | 0.7693 | 0.6465 | 0.9339 | 0.7640 | 0.6595 | 0.9373 |
| 50 | 50 | 116,540 | 0.7960 | 0.7320 | 0.9339 | 0.8207 | 0.6581 | 0.9087 |
| 40 | 60 | 97,115 | 0.8231 | 0.8033 | 0.9339 | 0.8637 | 0.6570 | 0.8688 |
| 30 | 70 | 83,240 | 0.8507 | 0.8638 | 0.9339 | 0.8975 | 0.6565 | 0.8097 |
| 20 | 80 | 72,835 | 0.8791 | 0.9166 | 0.9339 | 0.9252 | 0.6601 | 0.7140 |
| 10 | 90 | 64,740 | 0.9055 | 0.9600 | 0.9339 | 0.9468 | 0.6501 | 0.5221 |
| 0 | 100 | 58,270 | 0.9339 | 1.0000 | 0.9339 | 0.9658 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6602 | 0.0000 | 0.0000 | 0.0000 | 0.6602 | 1.0000 |
| 90 | 10 | 299,940 | 0.6879 | 0.2341 | 0.9335 | 0.3743 | 0.6607 | 0.9889 |
| 80 | 20 | 291,350 | 0.7148 | 0.4071 | 0.9337 | 0.5670 | 0.6601 | 0.9755 |
| 70 | 30 | 194,230 | 0.7421 | 0.5406 | 0.9337 | 0.6848 | 0.6600 | 0.9587 |
| 60 | 40 | 145,675 | 0.7694 | 0.6466 | 0.9337 | 0.7641 | 0.6598 | 0.9373 |
| 50 | 50 | 116,540 | 0.7961 | 0.7322 | 0.9337 | 0.8207 | 0.6584 | 0.9086 |
| 40 | 60 | 97,115 | 0.8233 | 0.8036 | 0.9337 | 0.8638 | 0.6576 | 0.8687 |
| 30 | 70 | 83,240 | 0.8507 | 0.8640 | 0.9337 | 0.8975 | 0.6570 | 0.8095 |
| 20 | 80 | 72,835 | 0.8791 | 0.9167 | 0.9338 | 0.9252 | 0.6607 | 0.7137 |
| 10 | 90 | 64,740 | 0.9054 | 0.9601 | 0.9337 | 0.9467 | 0.6504 | 0.5217 |
| 0 | 100 | 58,270 | 0.9337 | 1.0000 | 0.9337 | 0.9657 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6602 | 0.0000 | 0.0000 | 0.0000 | 0.6602 | 1.0000 |
| 90 | 10 | 299,940 | 0.6879 | 0.2341 | 0.9335 | 0.3743 | 0.6607 | 0.9889 |
| 80 | 20 | 291,350 | 0.7148 | 0.4071 | 0.9337 | 0.5670 | 0.6601 | 0.9755 |
| 70 | 30 | 194,230 | 0.7421 | 0.5406 | 0.9337 | 0.6848 | 0.6600 | 0.9587 |
| 60 | 40 | 145,675 | 0.7694 | 0.6466 | 0.9337 | 0.7641 | 0.6598 | 0.9373 |
| 50 | 50 | 116,540 | 0.7961 | 0.7322 | 0.9337 | 0.8207 | 0.6584 | 0.9086 |
| 40 | 60 | 97,115 | 0.8233 | 0.8036 | 0.9337 | 0.8638 | 0.6576 | 0.8687 |
| 30 | 70 | 83,240 | 0.8507 | 0.8640 | 0.9337 | 0.8975 | 0.6570 | 0.8095 |
| 20 | 80 | 72,835 | 0.8791 | 0.9167 | 0.9338 | 0.9252 | 0.6607 | 0.7137 |
| 10 | 90 | 64,740 | 0.9054 | 0.9601 | 0.9337 | 0.9467 | 0.6504 | 0.5217 |
| 0 | 100 | 58,270 | 0.9337 | 1.0000 | 0.9337 | 0.9657 | 0.0000 | 0.0000 |


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
0.15       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678   <--
0.20       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.25       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.30       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.35       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.40       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.45       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.50       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.55       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.60       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.65       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.70       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.75       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
0.80       0.7281   0.4217   0.6988   0.9987   0.9916   0.2678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7281, F1=0.4217, Normal Recall=0.6988, Normal Precision=0.9987, Attack Recall=0.9916, Attack Precision=0.2678

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
0.15       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517   <--
0.20       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.25       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.30       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.35       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.40       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.45       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.50       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.55       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.60       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.65       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.70       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.75       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
0.80       0.7576   0.6208   0.6989   0.9972   0.9921   0.4517  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7576, F1=0.6208, Normal Recall=0.6989, Normal Precision=0.9972, Attack Recall=0.9921, Attack Precision=0.4517

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
0.15       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853   <--
0.20       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.25       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.30       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.35       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.40       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.45       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.50       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.55       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.60       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.65       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.70       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.75       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
0.80       0.7867   0.7362   0.6987   0.9952   0.9921   0.5853  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7867, F1=0.7362, Normal Recall=0.6987, Normal Precision=0.9952, Attack Recall=0.9921, Attack Precision=0.5853

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
0.15       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876   <--
0.20       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.25       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.30       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.35       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.40       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.45       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.50       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.55       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.60       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.65       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.70       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.75       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
0.80       0.8166   0.8123   0.6995   0.9925   0.9921   0.6876  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8166, F1=0.8123, Normal Recall=0.6995, Normal Precision=0.9925, Attack Recall=0.9921, Attack Precision=0.6876

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
0.15       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680   <--
0.20       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.25       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.30       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.35       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.40       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.45       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.50       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.55       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.60       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.65       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.70       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.75       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
0.80       0.8462   0.8658   0.7003   0.9889   0.9921   0.7680  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8462, F1=0.8658, Normal Recall=0.7003, Normal Precision=0.9889, Attack Recall=0.9921, Attack Precision=0.7680

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
0.15       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341   <--
0.20       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.25       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.30       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.35       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.40       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.45       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.50       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.55       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.60       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.65       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.70       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.75       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
0.80       0.6878   0.3745   0.6603   0.9891   0.9346   0.2341  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6878, F1=0.3745, Normal Recall=0.6603, Normal Precision=0.9891, Attack Recall=0.9346, Attack Precision=0.2341

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
0.15       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076   <--
0.20       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.25       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.30       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.35       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.40       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.45       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.50       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.55       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.60       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.65       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.70       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.75       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
0.80       0.7154   0.5676   0.6607   0.9756   0.9339   0.4076  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7154, F1=0.5676, Normal Recall=0.6607, Normal Precision=0.9756, Attack Recall=0.9339, Attack Precision=0.4076

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
0.15       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404   <--
0.20       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.25       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.30       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.35       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.40       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.45       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.50       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.55       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.60       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.65       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.70       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.75       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
0.80       0.7419   0.6847   0.6597   0.9588   0.9339   0.5404  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7419, F1=0.6847, Normal Recall=0.6597, Normal Precision=0.9588, Attack Recall=0.9339, Attack Precision=0.5404

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
0.15       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464   <--
0.20       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.25       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.30       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.35       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.40       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.45       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.50       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.55       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.60       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.65       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.70       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.75       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
0.80       0.7692   0.7640   0.6595   0.9373   0.9339   0.6464  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7692, F1=0.7640, Normal Recall=0.6595, Normal Precision=0.9373, Attack Recall=0.9339, Attack Precision=0.6464

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
0.15       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326   <--
0.20       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.25       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.30       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.35       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.40       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.45       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.50       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.55       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.60       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.65       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.70       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.75       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
0.80       0.7965   0.8211   0.6592   0.9088   0.9339   0.7326  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7965, F1=0.8211, Normal Recall=0.6592, Normal Precision=0.9088, Attack Recall=0.9339, Attack Precision=0.7326

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
0.15       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343   <--
0.20       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.25       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.30       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.35       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.40       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.45       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.50       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.55       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.60       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.65       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.70       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.75       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.80       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6881, F1=0.3747, Normal Recall=0.6607, Normal Precision=0.9891, Attack Recall=0.9345, Attack Precision=0.2343

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
0.15       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078   <--
0.20       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.25       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.30       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.35       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.40       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.45       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.50       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.55       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.60       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.65       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.70       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.75       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.80       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7156, F1=0.5677, Normal Recall=0.6611, Normal Precision=0.9756, Attack Recall=0.9337, Attack Precision=0.4078

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
0.15       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407   <--
0.20       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.25       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.30       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.35       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.40       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.45       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.50       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.55       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.60       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.65       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.70       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.75       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.80       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7422, F1=0.6849, Normal Recall=0.6601, Normal Precision=0.9588, Attack Recall=0.9337, Attack Precision=0.5407

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
0.15       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467   <--
0.20       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.25       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.30       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.35       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.40       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.45       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.50       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.55       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.60       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.65       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.70       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.75       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.80       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7694, F1=0.7641, Normal Recall=0.6599, Normal Precision=0.9373, Attack Recall=0.9337, Attack Precision=0.6467

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
0.15       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329   <--
0.20       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.25       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.30       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.35       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.40       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.45       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.50       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.55       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.60       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.65       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.70       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.75       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.80       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7968, F1=0.8212, Normal Recall=0.6598, Normal Precision=0.9087, Attack Recall=0.9337, Attack Precision=0.7329

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
0.15       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343   <--
0.20       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.25       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.30       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.35       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.40       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.45       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.50       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.55       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.60       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.65       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.70       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.75       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
0.80       0.6881   0.3747   0.6607   0.9891   0.9345   0.2343  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6881, F1=0.3747, Normal Recall=0.6607, Normal Precision=0.9891, Attack Recall=0.9345, Attack Precision=0.2343

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
0.15       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078   <--
0.20       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.25       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.30       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.35       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.40       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.45       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.50       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.55       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.60       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.65       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.70       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.75       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
0.80       0.7156   0.5677   0.6611   0.9756   0.9337   0.4078  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7156, F1=0.5677, Normal Recall=0.6611, Normal Precision=0.9756, Attack Recall=0.9337, Attack Precision=0.4078

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
0.15       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407   <--
0.20       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.25       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.30       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.35       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.40       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.45       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.50       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.55       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.60       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.65       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.70       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.75       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
0.80       0.7422   0.6849   0.6601   0.9588   0.9337   0.5407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7422, F1=0.6849, Normal Recall=0.6601, Normal Precision=0.9588, Attack Recall=0.9337, Attack Precision=0.5407

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
0.15       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467   <--
0.20       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.25       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.30       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.35       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.40       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.45       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.50       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.55       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.60       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.65       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.70       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.75       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
0.80       0.7694   0.7641   0.6599   0.9373   0.9337   0.6467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7694, F1=0.7641, Normal Recall=0.6599, Normal Precision=0.9373, Attack Recall=0.9337, Attack Precision=0.6467

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
0.15       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329   <--
0.20       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.25       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.30       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.35       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.40       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.45       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.50       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.55       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.60       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.65       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.70       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.75       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
0.80       0.7968   0.8212   0.6598   0.9087   0.9337   0.7329  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7968, F1=0.8212, Normal Recall=0.6598, Normal Precision=0.9087, Attack Recall=0.9337, Attack Precision=0.7329

```

