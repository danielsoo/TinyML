# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-17 04:33:37 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6801 | 0.6771 | 0.6749 | 0.6719 | 0.6695 | 0.6669 | 0.6652 | 0.6610 | 0.6596 | 0.6573 | 0.6546 |
| QAT+Prune only | 0.8544 | 0.8689 | 0.8826 | 0.8971 | 0.9110 | 0.9240 | 0.9390 | 0.9537 | 0.9678 | 0.9814 | 0.9962 |
| QAT+PTQ | 0.8518 | 0.8666 | 0.8806 | 0.8953 | 0.9096 | 0.9227 | 0.9379 | 0.9529 | 0.9674 | 0.9812 | 0.9962 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8518 | 0.8666 | 0.8806 | 0.8953 | 0.9096 | 0.9227 | 0.9379 | 0.9529 | 0.9674 | 0.9812 | 0.9962 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2881 | 0.4461 | 0.5449 | 0.6131 | 0.6628 | 0.7012 | 0.7300 | 0.7547 | 0.7747 | 0.7912 |
| QAT+Prune only | 0.0000 | 0.6031 | 0.7724 | 0.8532 | 0.8995 | 0.9292 | 0.9514 | 0.9678 | 0.9802 | 0.9897 | 0.9981 |
| QAT+PTQ | 0.0000 | 0.5990 | 0.7694 | 0.8509 | 0.8982 | 0.9280 | 0.9506 | 0.9674 | 0.9799 | 0.9897 | 0.9981 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5990 | 0.7694 | 0.8509 | 0.8982 | 0.9280 | 0.9506 | 0.9674 | 0.9799 | 0.9897 | 0.9981 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6801 | 0.6797 | 0.6799 | 0.6794 | 0.6794 | 0.6793 | 0.6812 | 0.6760 | 0.6798 | 0.6818 | 0.0000 |
| QAT+Prune only | 0.8544 | 0.8547 | 0.8542 | 0.8547 | 0.8541 | 0.8519 | 0.8531 | 0.8544 | 0.8541 | 0.8480 | 0.0000 |
| QAT+PTQ | 0.8518 | 0.8522 | 0.8517 | 0.8520 | 0.8519 | 0.8493 | 0.8505 | 0.8520 | 0.8520 | 0.8468 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8518 | 0.8522 | 0.8517 | 0.8520 | 0.8519 | 0.8493 | 0.8505 | 0.8520 | 0.8520 | 0.8468 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6801 | 0.0000 | 0.0000 | 0.0000 | 0.6801 | 1.0000 |
| 90 | 10 | 299,940 | 0.6771 | 0.1848 | 0.6533 | 0.2881 | 0.6797 | 0.9464 |
| 80 | 20 | 291,350 | 0.6749 | 0.3383 | 0.6546 | 0.4461 | 0.6799 | 0.8873 |
| 70 | 30 | 194,230 | 0.6719 | 0.4666 | 0.6546 | 0.5449 | 0.6794 | 0.8211 |
| 60 | 40 | 145,675 | 0.6695 | 0.5765 | 0.6546 | 0.6131 | 0.6794 | 0.7469 |
| 50 | 50 | 116,540 | 0.6669 | 0.6711 | 0.6546 | 0.6628 | 0.6793 | 0.6629 |
| 40 | 60 | 97,115 | 0.6652 | 0.7549 | 0.6546 | 0.7012 | 0.6812 | 0.5680 |
| 30 | 70 | 83,240 | 0.6610 | 0.8250 | 0.6546 | 0.7300 | 0.6760 | 0.4561 |
| 20 | 80 | 72,835 | 0.6596 | 0.8910 | 0.6546 | 0.7547 | 0.6798 | 0.3297 |
| 10 | 90 | 64,740 | 0.6573 | 0.9488 | 0.6545 | 0.7747 | 0.6818 | 0.1799 |
| 0 | 100 | 58,270 | 0.6546 | 1.0000 | 0.6546 | 0.7912 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8544 | 0.0000 | 0.0000 | 0.0000 | 0.8544 | 1.0000 |
| 90 | 10 | 299,940 | 0.8689 | 0.4325 | 0.9963 | 0.6031 | 0.8547 | 0.9995 |
| 80 | 20 | 291,350 | 0.8826 | 0.6308 | 0.9962 | 0.7724 | 0.8542 | 0.9989 |
| 70 | 30 | 194,230 | 0.8971 | 0.7461 | 0.9962 | 0.8532 | 0.8547 | 0.9981 |
| 60 | 40 | 145,675 | 0.9110 | 0.8199 | 0.9962 | 0.8995 | 0.8541 | 0.9970 |
| 50 | 50 | 116,540 | 0.9240 | 0.8706 | 0.9962 | 0.9292 | 0.8519 | 0.9955 |
| 40 | 60 | 97,115 | 0.9390 | 0.9105 | 0.9962 | 0.9514 | 0.8531 | 0.9933 |
| 30 | 70 | 83,240 | 0.9537 | 0.9411 | 0.9962 | 0.9678 | 0.8544 | 0.9897 |
| 20 | 80 | 72,835 | 0.9678 | 0.9647 | 0.9962 | 0.9802 | 0.8541 | 0.9825 |
| 10 | 90 | 64,740 | 0.9814 | 0.9833 | 0.9962 | 0.9897 | 0.8480 | 0.9611 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8518 | 0.0000 | 0.0000 | 0.0000 | 0.8518 | 1.0000 |
| 90 | 10 | 299,940 | 0.8666 | 0.4283 | 0.9963 | 0.5990 | 0.8522 | 0.9995 |
| 80 | 20 | 291,350 | 0.8806 | 0.6268 | 0.9962 | 0.7694 | 0.8517 | 0.9989 |
| 70 | 30 | 194,230 | 0.8953 | 0.7426 | 0.9962 | 0.8509 | 0.8520 | 0.9981 |
| 60 | 40 | 145,675 | 0.9096 | 0.8177 | 0.9962 | 0.8982 | 0.8519 | 0.9970 |
| 50 | 50 | 116,540 | 0.9227 | 0.8686 | 0.9962 | 0.9280 | 0.8493 | 0.9955 |
| 40 | 60 | 97,115 | 0.9379 | 0.9090 | 0.9962 | 0.9506 | 0.8505 | 0.9933 |
| 30 | 70 | 83,240 | 0.9529 | 0.9402 | 0.9962 | 0.9674 | 0.8520 | 0.9897 |
| 20 | 80 | 72,835 | 0.9674 | 0.9642 | 0.9962 | 0.9799 | 0.8520 | 0.9824 |
| 10 | 90 | 64,740 | 0.9812 | 0.9832 | 0.9962 | 0.9897 | 0.8468 | 0.9611 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8518 | 0.0000 | 0.0000 | 0.0000 | 0.8518 | 1.0000 |
| 90 | 10 | 299,940 | 0.8666 | 0.4283 | 0.9963 | 0.5990 | 0.8522 | 0.9995 |
| 80 | 20 | 291,350 | 0.8806 | 0.6268 | 0.9962 | 0.7694 | 0.8517 | 0.9989 |
| 70 | 30 | 194,230 | 0.8953 | 0.7426 | 0.9962 | 0.8509 | 0.8520 | 0.9981 |
| 60 | 40 | 145,675 | 0.9096 | 0.8177 | 0.9962 | 0.8982 | 0.8519 | 0.9970 |
| 50 | 50 | 116,540 | 0.9227 | 0.8686 | 0.9962 | 0.9280 | 0.8493 | 0.9955 |
| 40 | 60 | 97,115 | 0.9379 | 0.9090 | 0.9962 | 0.9506 | 0.8505 | 0.9933 |
| 30 | 70 | 83,240 | 0.9529 | 0.9402 | 0.9962 | 0.9674 | 0.8520 | 0.9897 |
| 20 | 80 | 72,835 | 0.9674 | 0.9642 | 0.9962 | 0.9799 | 0.8520 | 0.9824 |
| 10 | 90 | 64,740 | 0.9812 | 0.9832 | 0.9962 | 0.9897 | 0.8468 | 0.9611 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |


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
0.15       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847   <--
0.20       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.25       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.30       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.35       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.40       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.45       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.50       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.55       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.60       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.65       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.70       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.75       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
0.80       0.6771   0.2880   0.6797   0.9464   0.6532   0.1847  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6771, F1=0.2880, Normal Recall=0.6797, Normal Precision=0.9464, Attack Recall=0.6532, Attack Precision=0.1847

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
0.15       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379   <--
0.20       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.25       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.30       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.35       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.40       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.45       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.50       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.55       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.60       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.65       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.70       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.75       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
0.80       0.6743   0.4457   0.6793   0.8872   0.6546   0.3379  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6743, F1=0.4457, Normal Recall=0.6793, Normal Precision=0.8872, Attack Recall=0.6546, Attack Precision=0.3379

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
0.15       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673   <--
0.20       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.25       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.30       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.35       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.40       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.45       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.50       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.55       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.60       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.65       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.70       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.75       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
0.80       0.6725   0.5453   0.6802   0.8213   0.6546   0.4673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6725, F1=0.5453, Normal Recall=0.6802, Normal Precision=0.8213, Attack Recall=0.6546, Attack Precision=0.4673

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
0.15       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770   <--
0.20       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.25       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.30       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.35       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.40       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.45       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.50       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.55       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.60       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.65       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.70       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.75       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
0.80       0.6699   0.6133   0.6801   0.7470   0.6546   0.5770  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6699, F1=0.6133, Normal Recall=0.6801, Normal Precision=0.7470, Attack Recall=0.6546, Attack Precision=0.5770

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
0.15       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719   <--
0.20       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.25       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.30       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.35       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.40       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.45       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.50       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.55       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.60       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.65       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.70       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.75       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
0.80       0.6675   0.6631   0.6804   0.6633   0.6546   0.6719  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6675, F1=0.6631, Normal Recall=0.6804, Normal Precision=0.6633, Attack Recall=0.6546, Attack Precision=0.6719

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
0.15       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325   <--
0.20       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.25       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.30       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.35       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.40       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.45       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.50       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.55       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.60       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.65       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.70       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.75       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
0.80       0.8689   0.6032   0.8547   0.9996   0.9966   0.4325  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8689, F1=0.6032, Normal Recall=0.8547, Normal Precision=0.9996, Attack Recall=0.9966, Attack Precision=0.4325

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
0.15       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323   <--
0.20       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.25       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.30       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.35       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.40       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.45       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.50       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.55       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.60       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.65       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.70       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.75       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
0.80       0.8834   0.7736   0.8551   0.9989   0.9962   0.6323  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8834, F1=0.7736, Normal Recall=0.8551, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6323

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
0.15       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467   <--
0.20       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.25       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.30       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.35       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.40       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.45       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.50       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.55       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.60       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.65       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.70       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.75       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
0.80       0.8975   0.8536   0.8552   0.9981   0.9962   0.7467  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8975, F1=0.8536, Normal Recall=0.8552, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7467

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
0.15       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203   <--
0.20       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.25       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.30       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.35       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.40       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.45       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.50       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.55       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.60       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.65       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.70       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.75       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
0.80       0.9112   0.8997   0.8545   0.9970   0.9962   0.8203  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9112, F1=0.8997, Normal Recall=0.8545, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8203

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
0.15       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721   <--
0.20       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.25       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.30       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.35       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.40       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.45       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.50       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.55       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.60       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.65       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.70       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.75       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
0.80       0.9251   0.9300   0.8539   0.9956   0.9962   0.8721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9251, F1=0.9300, Normal Recall=0.8539, Normal Precision=0.9956, Attack Recall=0.9962, Attack Precision=0.8721

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
0.15       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283   <--
0.20       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.25       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.30       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.35       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.40       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.45       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.50       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.55       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.60       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.65       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.70       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.75       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.80       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8667, F1=0.5992, Normal Recall=0.8522, Normal Precision=0.9996, Attack Recall=0.9966, Attack Precision=0.4283

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
0.15       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282   <--
0.20       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.25       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.30       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.35       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.40       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.45       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.50       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.55       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.60       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.65       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.70       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.75       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.80       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8813, F1=0.7705, Normal Recall=0.8526, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6282

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
0.15       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433   <--
0.20       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.25       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.30       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.35       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.40       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.45       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.50       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.55       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.60       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.65       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.70       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.75       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.80       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8957, F1=0.8514, Normal Recall=0.8526, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7433

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
0.15       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178   <--
0.20       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.25       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.30       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.35       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.40       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.45       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.50       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.55       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.60       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.65       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.70       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.75       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.80       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9097, F1=0.8982, Normal Recall=0.8520, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8178

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
0.15       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700   <--
0.20       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.25       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.30       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.35       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.40       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.45       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.50       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.55       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.60       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.65       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.70       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.75       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.80       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9237, F1=0.9288, Normal Recall=0.8512, Normal Precision=0.9955, Attack Recall=0.9962, Attack Precision=0.8700

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
0.15       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283   <--
0.20       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.25       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.30       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.35       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.40       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.45       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.50       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.55       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.60       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.65       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.70       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.75       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
0.80       0.8667   0.5992   0.8522   0.9996   0.9966   0.4283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8667, F1=0.5992, Normal Recall=0.8522, Normal Precision=0.9996, Attack Recall=0.9966, Attack Precision=0.4283

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
0.15       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282   <--
0.20       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.25       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.30       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.35       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.40       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.45       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.50       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.55       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.60       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.65       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.70       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.75       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
0.80       0.8813   0.7705   0.8526   0.9989   0.9962   0.6282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8813, F1=0.7705, Normal Recall=0.8526, Normal Precision=0.9989, Attack Recall=0.9962, Attack Precision=0.6282

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
0.15       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433   <--
0.20       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.25       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.30       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.35       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.40       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.45       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.50       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.55       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.60       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.65       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.70       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.75       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
0.80       0.8957   0.8514   0.8526   0.9981   0.9962   0.7433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8957, F1=0.8514, Normal Recall=0.8526, Normal Precision=0.9981, Attack Recall=0.9962, Attack Precision=0.7433

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
0.15       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178   <--
0.20       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.25       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.30       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.35       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.40       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.45       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.50       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.55       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.60       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.65       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.70       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.75       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
0.80       0.9097   0.8982   0.8520   0.9970   0.9962   0.8178  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9097, F1=0.8982, Normal Recall=0.8520, Normal Precision=0.9970, Attack Recall=0.9962, Attack Precision=0.8178

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
0.15       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700   <--
0.20       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.25       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.30       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.35       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.40       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.45       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.50       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.55       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.60       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.65       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.70       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.75       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
0.80       0.9237   0.9288   0.8512   0.9955   0.9962   0.8700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9237, F1=0.9288, Normal Recall=0.8512, Normal Precision=0.9955, Attack Recall=0.9962, Attack Precision=0.8700

```

