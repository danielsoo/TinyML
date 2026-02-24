# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-12 13:10:36 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3735 | 0.4237 | 0.4728 | 0.5226 | 0.5713 | 0.6219 | 0.6702 | 0.7204 | 0.7694 | 0.8186 | 0.8683 |
| QAT+Prune only | 0.7746 | 0.7974 | 0.8189 | 0.8416 | 0.8627 | 0.8854 | 0.9079 | 0.9296 | 0.9516 | 0.9738 | 0.9963 |
| QAT+PTQ | 0.7742 | 0.7969 | 0.8185 | 0.8412 | 0.8622 | 0.8851 | 0.9077 | 0.9297 | 0.9515 | 0.9736 | 0.9963 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7742 | 0.7969 | 0.8185 | 0.8412 | 0.8622 | 0.8851 | 0.9077 | 0.9297 | 0.9515 | 0.9736 | 0.9963 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2321 | 0.3972 | 0.5218 | 0.6184 | 0.6966 | 0.7596 | 0.8130 | 0.8577 | 0.8960 | 0.9295 |
| QAT+Prune only | 0.0000 | 0.4959 | 0.6876 | 0.7906 | 0.8530 | 0.8969 | 0.9285 | 0.9520 | 0.9705 | 0.9856 | 0.9981 |
| QAT+PTQ | 0.0000 | 0.4953 | 0.6871 | 0.7901 | 0.8526 | 0.8966 | 0.9283 | 0.9520 | 0.9705 | 0.9855 | 0.9982 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4953 | 0.6871 | 0.7901 | 0.8526 | 0.8966 | 0.9283 | 0.9520 | 0.9705 | 0.9855 | 0.9982 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3735 | 0.3740 | 0.3740 | 0.3744 | 0.3734 | 0.3755 | 0.3730 | 0.3752 | 0.3740 | 0.3710 | 0.0000 |
| QAT+Prune only | 0.7746 | 0.7752 | 0.7746 | 0.7753 | 0.7736 | 0.7745 | 0.7754 | 0.7741 | 0.7729 | 0.7711 | 0.0000 |
| QAT+PTQ | 0.7742 | 0.7747 | 0.7741 | 0.7747 | 0.7727 | 0.7738 | 0.7747 | 0.7743 | 0.7724 | 0.7691 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7742 | 0.7747 | 0.7741 | 0.7747 | 0.7727 | 0.7738 | 0.7747 | 0.7743 | 0.7724 | 0.7691 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3735 | 0.0000 | 0.0000 | 0.0000 | 0.3735 | 1.0000 |
| 90 | 10 | 299,940 | 0.4237 | 0.1339 | 0.8708 | 0.2321 | 0.3740 | 0.9630 |
| 80 | 20 | 291,350 | 0.4728 | 0.2575 | 0.8683 | 0.3972 | 0.3740 | 0.9191 |
| 70 | 30 | 194,230 | 0.5226 | 0.3730 | 0.8683 | 0.5218 | 0.3744 | 0.8690 |
| 60 | 40 | 145,675 | 0.5713 | 0.4802 | 0.8683 | 0.6184 | 0.3734 | 0.8096 |
| 50 | 50 | 116,540 | 0.6219 | 0.5816 | 0.8683 | 0.6966 | 0.3755 | 0.7403 |
| 40 | 60 | 97,115 | 0.6702 | 0.6750 | 0.8683 | 0.7596 | 0.3730 | 0.6537 |
| 30 | 70 | 83,240 | 0.7204 | 0.7643 | 0.8683 | 0.8130 | 0.3752 | 0.5497 |
| 20 | 80 | 72,835 | 0.7694 | 0.8473 | 0.8683 | 0.8577 | 0.3740 | 0.4151 |
| 10 | 90 | 64,740 | 0.8186 | 0.9255 | 0.8683 | 0.8960 | 0.3710 | 0.2384 |
| 0 | 100 | 58,270 | 0.8683 | 1.0000 | 0.8683 | 0.9295 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7746 | 0.0000 | 0.0000 | 0.0000 | 0.7746 | 1.0000 |
| 90 | 10 | 299,940 | 0.7974 | 0.3300 | 0.9965 | 0.4959 | 0.7752 | 0.9995 |
| 80 | 20 | 291,350 | 0.8189 | 0.5249 | 0.9963 | 0.6876 | 0.7746 | 0.9988 |
| 70 | 30 | 194,230 | 0.8416 | 0.6552 | 0.9963 | 0.7906 | 0.7753 | 0.9980 |
| 60 | 40 | 145,675 | 0.8627 | 0.7458 | 0.9963 | 0.8530 | 0.7736 | 0.9968 |
| 50 | 50 | 116,540 | 0.8854 | 0.8155 | 0.9963 | 0.8969 | 0.7745 | 0.9952 |
| 40 | 60 | 97,115 | 0.9079 | 0.8694 | 0.9963 | 0.9285 | 0.7754 | 0.9929 |
| 30 | 70 | 83,240 | 0.9296 | 0.9114 | 0.9963 | 0.9520 | 0.7741 | 0.9889 |
| 20 | 80 | 72,835 | 0.9516 | 0.9461 | 0.9963 | 0.9705 | 0.7729 | 0.9812 |
| 10 | 90 | 64,740 | 0.9738 | 0.9751 | 0.9963 | 0.9856 | 0.7711 | 0.9585 |
| 0 | 100 | 58,270 | 0.9963 | 1.0000 | 0.9963 | 0.9981 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7742 | 0.0000 | 0.0000 | 0.0000 | 0.7742 | 1.0000 |
| 90 | 10 | 299,940 | 0.7969 | 0.3295 | 0.9966 | 0.4953 | 0.7747 | 0.9995 |
| 80 | 20 | 291,350 | 0.8185 | 0.5244 | 0.9963 | 0.6871 | 0.7741 | 0.9988 |
| 70 | 30 | 194,230 | 0.8412 | 0.6546 | 0.9963 | 0.7901 | 0.7747 | 0.9980 |
| 60 | 40 | 145,675 | 0.8622 | 0.7450 | 0.9963 | 0.8526 | 0.7727 | 0.9969 |
| 50 | 50 | 116,540 | 0.8851 | 0.8150 | 0.9963 | 0.8966 | 0.7738 | 0.9953 |
| 40 | 60 | 97,115 | 0.9077 | 0.8690 | 0.9963 | 0.9283 | 0.7747 | 0.9930 |
| 30 | 70 | 83,240 | 0.9297 | 0.9115 | 0.9963 | 0.9520 | 0.7743 | 0.9891 |
| 20 | 80 | 72,835 | 0.9515 | 0.9460 | 0.9963 | 0.9705 | 0.7724 | 0.9814 |
| 10 | 90 | 64,740 | 0.9736 | 0.9749 | 0.9963 | 0.9855 | 0.7691 | 0.9590 |
| 0 | 100 | 58,270 | 0.9963 | 1.0000 | 0.9963 | 0.9982 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7742 | 0.0000 | 0.0000 | 0.0000 | 0.7742 | 1.0000 |
| 90 | 10 | 299,940 | 0.7969 | 0.3295 | 0.9966 | 0.4953 | 0.7747 | 0.9995 |
| 80 | 20 | 291,350 | 0.8185 | 0.5244 | 0.9963 | 0.6871 | 0.7741 | 0.9988 |
| 70 | 30 | 194,230 | 0.8412 | 0.6546 | 0.9963 | 0.7901 | 0.7747 | 0.9980 |
| 60 | 40 | 145,675 | 0.8622 | 0.7450 | 0.9963 | 0.8526 | 0.7727 | 0.9969 |
| 50 | 50 | 116,540 | 0.8851 | 0.8150 | 0.9963 | 0.8966 | 0.7738 | 0.9953 |
| 40 | 60 | 97,115 | 0.9077 | 0.8690 | 0.9963 | 0.9283 | 0.7747 | 0.9930 |
| 30 | 70 | 83,240 | 0.9297 | 0.9115 | 0.9963 | 0.9520 | 0.7743 | 0.9891 |
| 20 | 80 | 72,835 | 0.9515 | 0.9460 | 0.9963 | 0.9705 | 0.7724 | 0.9814 |
| 10 | 90 | 64,740 | 0.9736 | 0.9749 | 0.9963 | 0.9855 | 0.7691 | 0.9590 |
| 0 | 100 | 58,270 | 0.9963 | 1.0000 | 0.9963 | 0.9982 | 0.0000 | 0.0000 |


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
0.15       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336   <--
0.20       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.25       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.30       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.35       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.40       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.45       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.50       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.55       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.60       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.65       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.70       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.75       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
0.80       0.4235   0.2316   0.3740   0.9625   0.8688   0.1336  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4235, F1=0.2316, Normal Recall=0.3740, Normal Precision=0.9625, Attack Recall=0.8688, Attack Precision=0.1336

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
0.15       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575   <--
0.20       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.25       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.30       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.35       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.40       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.45       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.50       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.55       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.60       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.65       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.70       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.75       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
0.80       0.4729   0.3972   0.3741   0.9191   0.8683   0.2575  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4729, F1=0.3972, Normal Recall=0.3741, Normal Precision=0.9191, Attack Recall=0.8683, Attack Precision=0.2575

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
0.15       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734   <--
0.20       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.25       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.30       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.35       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.40       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.45       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.50       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.55       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.60       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.65       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.70       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.75       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
0.80       0.5233   0.5222   0.3755   0.8693   0.8683   0.3734  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5233, F1=0.5222, Normal Recall=0.3755, Normal Precision=0.8693, Attack Recall=0.8683, Attack Precision=0.3734

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
0.15       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803   <--
0.20       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.25       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.30       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.35       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.40       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.45       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.50       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.55       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.60       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.65       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.70       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.75       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
0.80       0.5715   0.6185   0.3737   0.8097   0.8683   0.4803  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5715, F1=0.6185, Normal Recall=0.3737, Normal Precision=0.8097, Attack Recall=0.8683, Attack Precision=0.4803

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
0.15       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811   <--
0.20       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.25       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.30       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.35       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.40       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.45       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.50       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.55       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.60       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.65       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.70       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.75       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
0.80       0.6212   0.6962   0.3741   0.7396   0.8683   0.5811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6212, F1=0.6962, Normal Recall=0.3741, Normal Precision=0.7396, Attack Recall=0.8683, Attack Precision=0.5811

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
0.15       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300   <--
0.20       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.25       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.30       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.35       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.40       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.45       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.50       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.55       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.60       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.65       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.70       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.75       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
0.80       0.7974   0.4958   0.7753   0.9995   0.9964   0.3300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7974, F1=0.4958, Normal Recall=0.7753, Normal Precision=0.9995, Attack Recall=0.9964, Attack Precision=0.3300

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
0.15       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261   <--
0.20       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.25       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.30       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.35       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.40       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.45       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.50       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.55       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.60       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.65       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.70       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.75       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
0.80       0.8198   0.6886   0.7756   0.9988   0.9963   0.5261  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8198, F1=0.6886, Normal Recall=0.7756, Normal Precision=0.9988, Attack Recall=0.9963, Attack Precision=0.5261

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
0.15       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552   <--
0.20       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.25       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.30       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.35       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.40       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.45       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.50       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.55       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.60       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.65       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.70       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.75       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
0.80       0.8416   0.7905   0.7753   0.9980   0.9963   0.6552  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8416, F1=0.7905, Normal Recall=0.7753, Normal Precision=0.9980, Attack Recall=0.9963, Attack Precision=0.6552

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
0.15       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466   <--
0.20       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.25       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.30       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.35       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.40       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.45       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.50       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.55       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.60       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.65       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.70       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.75       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
0.80       0.8633   0.8536   0.7746   0.9968   0.9963   0.7466  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8633, F1=0.8536, Normal Recall=0.7746, Normal Precision=0.9968, Attack Recall=0.9963, Attack Precision=0.7466

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
0.15       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140   <--
0.20       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.25       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.30       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.35       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.40       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.45       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.50       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.55       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.60       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.65       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.70       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.75       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
0.80       0.8843   0.8959   0.7723   0.9952   0.9963   0.8140  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8843, F1=0.8959, Normal Recall=0.7723, Normal Precision=0.9952, Attack Recall=0.9963, Attack Precision=0.8140

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
0.15       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295   <--
0.20       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.25       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.30       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.35       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.40       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.45       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.50       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.55       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.60       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.65       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.70       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.75       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.80       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7969, F1=0.4953, Normal Recall=0.7747, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3295

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
0.15       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257   <--
0.20       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.25       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.30       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.35       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.40       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.45       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.50       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.55       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.60       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.65       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.70       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.75       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.80       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8195, F1=0.6882, Normal Recall=0.7752, Normal Precision=0.9988, Attack Recall=0.9963, Attack Precision=0.5257

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
0.15       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548   <--
0.20       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.25       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.30       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.35       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.40       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.45       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.50       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.55       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.60       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.65       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.70       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.75       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.80       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8413, F1=0.7902, Normal Recall=0.7749, Normal Precision=0.9980, Attack Recall=0.9963, Attack Precision=0.6548

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
0.15       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462   <--
0.20       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.25       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.30       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.35       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.40       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.45       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.50       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.55       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.60       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.65       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.70       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.75       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.80       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8630, F1=0.8533, Normal Recall=0.7740, Normal Precision=0.9969, Attack Recall=0.9963, Attack Precision=0.7462

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
0.15       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134   <--
0.20       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.25       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.30       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.35       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.40       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.45       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.50       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.55       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.60       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.65       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.70       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.75       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.80       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8839, F1=0.8957, Normal Recall=0.7715, Normal Precision=0.9953, Attack Recall=0.9963, Attack Precision=0.8134

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
0.15       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295   <--
0.20       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.25       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.30       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.35       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.40       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.45       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.50       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.55       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.60       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.65       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.70       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.75       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
0.80       0.7969   0.4953   0.7747   0.9995   0.9965   0.3295  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7969, F1=0.4953, Normal Recall=0.7747, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3295

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
0.15       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257   <--
0.20       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.25       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.30       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.35       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.40       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.45       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.50       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.55       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.60       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.65       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.70       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.75       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
0.80       0.8195   0.6882   0.7752   0.9988   0.9963   0.5257  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8195, F1=0.6882, Normal Recall=0.7752, Normal Precision=0.9988, Attack Recall=0.9963, Attack Precision=0.5257

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
0.15       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548   <--
0.20       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.25       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.30       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.35       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.40       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.45       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.50       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.55       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.60       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.65       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.70       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.75       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
0.80       0.8413   0.7902   0.7749   0.9980   0.9963   0.6548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8413, F1=0.7902, Normal Recall=0.7749, Normal Precision=0.9980, Attack Recall=0.9963, Attack Precision=0.6548

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
0.15       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462   <--
0.20       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.25       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.30       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.35       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.40       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.45       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.50       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.55       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.60       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.65       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.70       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.75       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
0.80       0.8630   0.8533   0.7740   0.9969   0.9963   0.7462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8630, F1=0.8533, Normal Recall=0.7740, Normal Precision=0.9969, Attack Recall=0.9963, Attack Precision=0.7462

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
0.15       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134   <--
0.20       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.25       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.30       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.35       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.40       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.45       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.50       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.55       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.60       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.65       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.70       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.75       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
0.80       0.8839   0.8957   0.7715   0.9953   0.9963   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8839, F1=0.8957, Normal Recall=0.7715, Normal Precision=0.9953, Attack Recall=0.9963, Attack Precision=0.8134

```

