# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-19 07:47:13 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8629 | 0.8771 | 0.8901 | 0.9035 | 0.9167 | 0.9289 | 0.9437 | 0.9574 | 0.9702 | 0.9832 | 0.9969 |
| QAT+Prune only | 0.7427 | 0.7638 | 0.7829 | 0.8018 | 0.8208 | 0.8397 | 0.8599 | 0.8780 | 0.8987 | 0.9168 | 0.9370 |
| QAT+PTQ | 0.7425 | 0.7635 | 0.7828 | 0.8017 | 0.8208 | 0.8398 | 0.8599 | 0.8781 | 0.8989 | 0.9172 | 0.9373 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7425 | 0.7635 | 0.7828 | 0.8017 | 0.8208 | 0.8398 | 0.8599 | 0.8781 | 0.8989 | 0.9172 | 0.9373 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6187 | 0.7839 | 0.8611 | 0.9055 | 0.9334 | 0.9550 | 0.9704 | 0.9816 | 0.9907 | 0.9984 |
| QAT+Prune only | 0.0000 | 0.4423 | 0.6333 | 0.7394 | 0.8071 | 0.8539 | 0.8892 | 0.9149 | 0.9367 | 0.9530 | 0.9675 |
| QAT+PTQ | 0.0000 | 0.4421 | 0.6332 | 0.7393 | 0.8071 | 0.8541 | 0.8893 | 0.9150 | 0.9368 | 0.9532 | 0.9677 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4421 | 0.6332 | 0.7393 | 0.8071 | 0.8541 | 0.8893 | 0.9150 | 0.9368 | 0.9532 | 0.9677 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8629 | 0.8638 | 0.8633 | 0.8635 | 0.8633 | 0.8609 | 0.8639 | 0.8651 | 0.8633 | 0.8597 | 0.0000 |
| QAT+Prune only | 0.7427 | 0.7446 | 0.7444 | 0.7439 | 0.7434 | 0.7424 | 0.7441 | 0.7402 | 0.7455 | 0.7351 | 0.0000 |
| QAT+PTQ | 0.7425 | 0.7443 | 0.7441 | 0.7436 | 0.7431 | 0.7423 | 0.7438 | 0.7399 | 0.7448 | 0.7356 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7425 | 0.7443 | 0.7441 | 0.7436 | 0.7431 | 0.7423 | 0.7438 | 0.7399 | 0.7448 | 0.7356 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8629 | 0.0000 | 0.0000 | 0.0000 | 0.8629 | 1.0000 |
| 90 | 10 | 299,940 | 0.8771 | 0.4485 | 0.9971 | 0.6187 | 0.8638 | 0.9996 |
| 80 | 20 | 291,350 | 0.8901 | 0.6459 | 0.9969 | 0.7839 | 0.8633 | 0.9991 |
| 70 | 30 | 194,230 | 0.9035 | 0.7579 | 0.9969 | 0.8611 | 0.8635 | 0.9985 |
| 60 | 40 | 145,675 | 0.9167 | 0.8294 | 0.9969 | 0.9055 | 0.8633 | 0.9976 |
| 50 | 50 | 116,540 | 0.9289 | 0.8775 | 0.9969 | 0.9334 | 0.8609 | 0.9964 |
| 40 | 60 | 97,115 | 0.9437 | 0.9166 | 0.9969 | 0.9550 | 0.8639 | 0.9946 |
| 30 | 70 | 83,240 | 0.9574 | 0.9452 | 0.9969 | 0.9704 | 0.8651 | 0.9917 |
| 20 | 80 | 72,835 | 0.9702 | 0.9669 | 0.9969 | 0.9816 | 0.8633 | 0.9858 |
| 10 | 90 | 64,740 | 0.9832 | 0.9846 | 0.9969 | 0.9907 | 0.8597 | 0.9685 |
| 0 | 100 | 58,270 | 0.9969 | 1.0000 | 0.9969 | 0.9984 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7427 | 0.0000 | 0.0000 | 0.0000 | 0.7427 | 1.0000 |
| 90 | 10 | 299,940 | 0.7638 | 0.2895 | 0.9366 | 0.4423 | 0.7446 | 0.9906 |
| 80 | 20 | 291,350 | 0.7829 | 0.4782 | 0.9370 | 0.6333 | 0.7444 | 0.9793 |
| 70 | 30 | 194,230 | 0.8018 | 0.6106 | 0.9370 | 0.7394 | 0.7439 | 0.9650 |
| 60 | 40 | 145,675 | 0.8208 | 0.7088 | 0.9370 | 0.8071 | 0.7434 | 0.9465 |
| 50 | 50 | 116,540 | 0.8397 | 0.7844 | 0.9370 | 0.8539 | 0.7424 | 0.9218 |
| 40 | 60 | 97,115 | 0.8599 | 0.8460 | 0.9370 | 0.8892 | 0.7441 | 0.8873 |
| 30 | 70 | 83,240 | 0.8780 | 0.8938 | 0.9370 | 0.9149 | 0.7402 | 0.8343 |
| 20 | 80 | 72,835 | 0.8987 | 0.9364 | 0.9370 | 0.9367 | 0.7455 | 0.7475 |
| 10 | 90 | 64,740 | 0.9168 | 0.9695 | 0.9370 | 0.9530 | 0.7351 | 0.5646 |
| 0 | 100 | 58,270 | 0.9370 | 1.0000 | 0.9370 | 0.9675 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7425 | 0.0000 | 0.0000 | 0.0000 | 0.7425 | 1.0000 |
| 90 | 10 | 299,940 | 0.7635 | 0.2893 | 0.9369 | 0.4421 | 0.7443 | 0.9907 |
| 80 | 20 | 291,350 | 0.7828 | 0.4780 | 0.9373 | 0.6332 | 0.7441 | 0.9794 |
| 70 | 30 | 194,230 | 0.8017 | 0.6104 | 0.9373 | 0.7393 | 0.7436 | 0.9651 |
| 60 | 40 | 145,675 | 0.8208 | 0.7086 | 0.9373 | 0.8071 | 0.7431 | 0.9468 |
| 50 | 50 | 116,540 | 0.8398 | 0.7844 | 0.9373 | 0.8541 | 0.7423 | 0.9222 |
| 40 | 60 | 97,115 | 0.8599 | 0.8459 | 0.9373 | 0.8893 | 0.7438 | 0.8878 |
| 30 | 70 | 83,240 | 0.8781 | 0.8937 | 0.9373 | 0.9150 | 0.7399 | 0.8350 |
| 20 | 80 | 72,835 | 0.8989 | 0.9363 | 0.9374 | 0.9368 | 0.7448 | 0.7483 |
| 10 | 90 | 64,740 | 0.9172 | 0.9696 | 0.9373 | 0.9532 | 0.7356 | 0.5660 |
| 0 | 100 | 58,270 | 0.9373 | 1.0000 | 0.9373 | 0.9677 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7425 | 0.0000 | 0.0000 | 0.0000 | 0.7425 | 1.0000 |
| 90 | 10 | 299,940 | 0.7635 | 0.2893 | 0.9369 | 0.4421 | 0.7443 | 0.9907 |
| 80 | 20 | 291,350 | 0.7828 | 0.4780 | 0.9373 | 0.6332 | 0.7441 | 0.9794 |
| 70 | 30 | 194,230 | 0.8017 | 0.6104 | 0.9373 | 0.7393 | 0.7436 | 0.9651 |
| 60 | 40 | 145,675 | 0.8208 | 0.7086 | 0.9373 | 0.8071 | 0.7431 | 0.9468 |
| 50 | 50 | 116,540 | 0.8398 | 0.7844 | 0.9373 | 0.8541 | 0.7423 | 0.9222 |
| 40 | 60 | 97,115 | 0.8599 | 0.8459 | 0.9373 | 0.8893 | 0.7438 | 0.8878 |
| 30 | 70 | 83,240 | 0.8781 | 0.8937 | 0.9373 | 0.9150 | 0.7399 | 0.8350 |
| 20 | 80 | 72,835 | 0.8989 | 0.9363 | 0.9374 | 0.9368 | 0.7448 | 0.7483 |
| 10 | 90 | 64,740 | 0.9172 | 0.9696 | 0.9373 | 0.9532 | 0.7356 | 0.5660 |
| 0 | 100 | 58,270 | 0.9373 | 1.0000 | 0.9373 | 0.9677 | 0.0000 | 0.0000 |


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
0.15       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486   <--
0.20       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.25       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.30       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.35       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.40       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.45       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.50       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.55       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.60       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.65       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.70       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.75       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
0.80       0.8771   0.6188   0.8638   0.9996   0.9972   0.4486  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8771, F1=0.6188, Normal Recall=0.8638, Normal Precision=0.9996, Attack Recall=0.9972, Attack Precision=0.4486

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
0.15       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469   <--
0.20       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.25       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.30       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.35       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.40       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.45       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.50       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.55       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.60       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.65       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.70       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.75       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
0.80       0.8906   0.7846   0.8640   0.9991   0.9969   0.6469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8906, F1=0.7846, Normal Recall=0.8640, Normal Precision=0.9991, Attack Recall=0.9969, Attack Precision=0.6469

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
0.15       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577   <--
0.20       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.25       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.30       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.35       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.40       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.45       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.50       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.55       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.60       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.65       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.70       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.75       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
0.80       0.9034   0.8610   0.8634   0.9985   0.9969   0.7577  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9034, F1=0.8610, Normal Recall=0.8634, Normal Precision=0.9985, Attack Recall=0.9969, Attack Precision=0.7577

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
0.15       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292   <--
0.20       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.25       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.30       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.35       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.40       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.45       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.50       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.55       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.60       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.65       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.70       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.75       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
0.80       0.9166   0.9053   0.8631   0.9976   0.9969   0.8292  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9166, F1=0.9053, Normal Recall=0.8631, Normal Precision=0.9976, Attack Recall=0.9969, Attack Precision=0.8292

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
0.15       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793   <--
0.20       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.25       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.30       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.35       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.40       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.45       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.50       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.55       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.60       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.65       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.70       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.75       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
0.80       0.9300   0.9344   0.8632   0.9964   0.9969   0.8793  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9300, F1=0.9344, Normal Recall=0.8632, Normal Precision=0.9964, Attack Recall=0.9969, Attack Precision=0.8793

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
0.15       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897   <--
0.20       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.25       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.30       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.35       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.40       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.45       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.50       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.55       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.60       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.65       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.70       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.75       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
0.80       0.7639   0.4427   0.7446   0.9908   0.9377   0.2897  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7639, F1=0.4427, Normal Recall=0.7446, Normal Precision=0.9908, Attack Recall=0.9377, Attack Precision=0.2897

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
0.15       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785   <--
0.20       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.25       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.30       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.35       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.40       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.45       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.50       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.55       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.60       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.65       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.70       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.75       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
0.80       0.7831   0.6335   0.7447   0.9793   0.9370   0.4785  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7831, F1=0.6335, Normal Recall=0.7447, Normal Precision=0.9793, Attack Recall=0.9370, Attack Precision=0.4785

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
0.15       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101   <--
0.20       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.25       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.30       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.35       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.40       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.45       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.50       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.55       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.60       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.65       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.70       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.75       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
0.80       0.8014   0.7390   0.7433   0.9650   0.9370   0.6101  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8014, F1=0.7390, Normal Recall=0.7433, Normal Precision=0.9650, Attack Recall=0.9370, Attack Precision=0.6101

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
0.15       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080   <--
0.20       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.25       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.30       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.35       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.40       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.45       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.50       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.55       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.60       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.65       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.70       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.75       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
0.80       0.8202   0.8066   0.7424   0.9465   0.9370   0.7080  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8202, F1=0.8066, Normal Recall=0.7424, Normal Precision=0.9465, Attack Recall=0.9370, Attack Precision=0.7080

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
0.15       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842   <--
0.20       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.25       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.30       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.35       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.40       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.45       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.50       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.55       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.60       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.65       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.70       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.75       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
0.80       0.8396   0.8538   0.7421   0.9218   0.9370   0.7842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8396, F1=0.8538, Normal Recall=0.7421, Normal Precision=0.9218, Attack Recall=0.9370, Attack Precision=0.7842

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
0.15       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896   <--
0.20       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.25       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.30       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.35       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.40       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.45       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.50       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.55       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.60       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.65       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.70       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.75       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.80       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7637, F1=0.4426, Normal Recall=0.7443, Normal Precision=0.9908, Attack Recall=0.9381, Attack Precision=0.2896

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
0.15       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783   <--
0.20       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.25       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.30       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.35       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.40       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.45       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.50       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.55       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.60       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.65       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.70       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.75       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.80       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7830, F1=0.6334, Normal Recall=0.7444, Normal Precision=0.9794, Attack Recall=0.9373, Attack Precision=0.4783

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
0.15       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099   <--
0.20       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.25       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.30       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.35       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.40       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.45       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.50       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.55       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.60       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.65       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.70       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.75       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.80       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8013, F1=0.7390, Normal Recall=0.7431, Normal Precision=0.9651, Attack Recall=0.9373, Attack Precision=0.6099

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
0.15       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079   <--
0.20       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.25       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.30       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.35       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.40       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.45       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.50       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.55       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.60       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.65       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.70       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.75       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.80       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8203, F1=0.8066, Normal Recall=0.7422, Normal Precision=0.9467, Attack Recall=0.9373, Attack Precision=0.7079

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
0.15       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841   <--
0.20       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.25       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.30       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.35       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.40       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.45       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.50       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.55       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.60       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.65       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.70       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.75       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.80       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8396, F1=0.8539, Normal Recall=0.7419, Normal Precision=0.9221, Attack Recall=0.9373, Attack Precision=0.7841

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
0.15       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896   <--
0.20       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.25       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.30       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.35       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.40       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.45       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.50       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.55       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.60       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.65       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.70       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.75       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
0.80       0.7637   0.4426   0.7443   0.9908   0.9381   0.2896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7637, F1=0.4426, Normal Recall=0.7443, Normal Precision=0.9908, Attack Recall=0.9381, Attack Precision=0.2896

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
0.15       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783   <--
0.20       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.25       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.30       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.35       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.40       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.45       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.50       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.55       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.60       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.65       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.70       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.75       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
0.80       0.7830   0.6334   0.7444   0.9794   0.9373   0.4783  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7830, F1=0.6334, Normal Recall=0.7444, Normal Precision=0.9794, Attack Recall=0.9373, Attack Precision=0.4783

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
0.15       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099   <--
0.20       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.25       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.30       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.35       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.40       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.45       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.50       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.55       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.60       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.65       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.70       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.75       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
0.80       0.8013   0.7390   0.7431   0.9651   0.9373   0.6099  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8013, F1=0.7390, Normal Recall=0.7431, Normal Precision=0.9651, Attack Recall=0.9373, Attack Precision=0.6099

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
0.15       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079   <--
0.20       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.25       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.30       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.35       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.40       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.45       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.50       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.55       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.60       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.65       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.70       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.75       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
0.80       0.8203   0.8066   0.7422   0.9467   0.9373   0.7079  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8203, F1=0.8066, Normal Recall=0.7422, Normal Precision=0.9467, Attack Recall=0.9373, Attack Precision=0.7079

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
0.15       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841   <--
0.20       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.25       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.30       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.35       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.40       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.45       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.50       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.55       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.60       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.65       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.70       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.75       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
0.80       0.8396   0.8539   0.7419   0.9221   0.9373   0.7841  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8396, F1=0.8539, Normal Recall=0.7419, Normal Precision=0.9221, Attack Recall=0.9373, Attack Precision=0.7841

```

