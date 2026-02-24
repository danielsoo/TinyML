# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-19 16:47:54 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2529 | 0.3215 | 0.3884 | 0.4551 | 0.5231 | 0.5890 | 0.6565 | 0.7230 | 0.7902 | 0.8582 | 0.9250 |
| QAT+Prune only | 0.8430 | 0.8592 | 0.8740 | 0.8895 | 0.9052 | 0.9184 | 0.9350 | 0.9491 | 0.9646 | 0.9793 | 0.9947 |
| QAT+PTQ | 0.8446 | 0.8605 | 0.8751 | 0.8905 | 0.9060 | 0.9191 | 0.9356 | 0.9495 | 0.9650 | 0.9793 | 0.9947 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8446 | 0.8605 | 0.8751 | 0.8905 | 0.9060 | 0.9191 | 0.9356 | 0.9495 | 0.9650 | 0.9793 | 0.9947 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2147 | 0.3769 | 0.5046 | 0.6081 | 0.6924 | 0.7637 | 0.8238 | 0.8759 | 0.9215 | 0.9610 |
| QAT+Prune only | 0.0000 | 0.5857 | 0.7595 | 0.8438 | 0.8936 | 0.9242 | 0.9484 | 0.9647 | 0.9783 | 0.9886 | 0.9973 |
| QAT+PTQ | 0.0000 | 0.5878 | 0.7611 | 0.8450 | 0.8944 | 0.9247 | 0.9488 | 0.9650 | 0.9785 | 0.9886 | 0.9973 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5878 | 0.7611 | 0.8450 | 0.8944 | 0.9247 | 0.9488 | 0.9650 | 0.9785 | 0.9886 | 0.9973 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2529 | 0.2541 | 0.2543 | 0.2537 | 0.2551 | 0.2531 | 0.2537 | 0.2518 | 0.2512 | 0.2572 | 0.0000 |
| QAT+Prune only | 0.8430 | 0.8442 | 0.8438 | 0.8444 | 0.8456 | 0.8421 | 0.8455 | 0.8425 | 0.8443 | 0.8406 | 0.0000 |
| QAT+PTQ | 0.8446 | 0.8455 | 0.8452 | 0.8459 | 0.8469 | 0.8434 | 0.8469 | 0.8442 | 0.8464 | 0.8412 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8446 | 0.8455 | 0.8452 | 0.8459 | 0.8469 | 0.8434 | 0.8469 | 0.8442 | 0.8464 | 0.8412 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2529 | 0.0000 | 0.0000 | 0.0000 | 0.2529 | 1.0000 |
| 90 | 10 | 299,940 | 0.3215 | 0.1214 | 0.9278 | 0.2147 | 0.2541 | 0.9694 |
| 80 | 20 | 291,350 | 0.3884 | 0.2367 | 0.9250 | 0.3769 | 0.2543 | 0.9313 |
| 70 | 30 | 194,230 | 0.4551 | 0.3469 | 0.9250 | 0.5046 | 0.2537 | 0.8876 |
| 60 | 40 | 145,675 | 0.5231 | 0.4529 | 0.9250 | 0.6081 | 0.2551 | 0.8361 |
| 50 | 50 | 116,540 | 0.5890 | 0.5533 | 0.9250 | 0.6924 | 0.2531 | 0.7714 |
| 40 | 60 | 97,115 | 0.6565 | 0.6502 | 0.9250 | 0.7637 | 0.2537 | 0.6928 |
| 30 | 70 | 83,240 | 0.7230 | 0.7426 | 0.9250 | 0.8238 | 0.2518 | 0.5900 |
| 20 | 80 | 72,835 | 0.7902 | 0.8317 | 0.9250 | 0.8759 | 0.2512 | 0.4557 |
| 10 | 90 | 64,740 | 0.8582 | 0.9181 | 0.9250 | 0.9215 | 0.2572 | 0.2759 |
| 0 | 100 | 58,270 | 0.9250 | 1.0000 | 0.9250 | 0.9610 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8430 | 0.0000 | 0.0000 | 0.0000 | 0.8430 | 1.0000 |
| 90 | 10 | 299,940 | 0.8592 | 0.4150 | 0.9948 | 0.5857 | 0.8442 | 0.9993 |
| 80 | 20 | 291,350 | 0.8740 | 0.6142 | 0.9947 | 0.7595 | 0.8438 | 0.9984 |
| 70 | 30 | 194,230 | 0.8895 | 0.7327 | 0.9947 | 0.8438 | 0.8444 | 0.9973 |
| 60 | 40 | 145,675 | 0.9052 | 0.8111 | 0.9947 | 0.8936 | 0.8456 | 0.9958 |
| 50 | 50 | 116,540 | 0.9184 | 0.8630 | 0.9947 | 0.9242 | 0.8421 | 0.9937 |
| 40 | 60 | 97,115 | 0.9350 | 0.9061 | 0.9947 | 0.9484 | 0.8455 | 0.9907 |
| 30 | 70 | 83,240 | 0.9491 | 0.9365 | 0.9947 | 0.9647 | 0.8425 | 0.9855 |
| 20 | 80 | 72,835 | 0.9646 | 0.9623 | 0.9947 | 0.9783 | 0.8443 | 0.9755 |
| 10 | 90 | 64,740 | 0.9793 | 0.9825 | 0.9947 | 0.9886 | 0.8406 | 0.9463 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9973 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8446 | 0.0000 | 0.0000 | 0.0000 | 0.8446 | 1.0000 |
| 90 | 10 | 299,940 | 0.8605 | 0.4171 | 0.9948 | 0.5878 | 0.8455 | 0.9993 |
| 80 | 20 | 291,350 | 0.8751 | 0.6163 | 0.9947 | 0.7611 | 0.8452 | 0.9984 |
| 70 | 30 | 194,230 | 0.8905 | 0.7345 | 0.9947 | 0.8450 | 0.8459 | 0.9973 |
| 60 | 40 | 145,675 | 0.9060 | 0.8125 | 0.9947 | 0.8944 | 0.8469 | 0.9958 |
| 50 | 50 | 116,540 | 0.9191 | 0.8640 | 0.9947 | 0.9247 | 0.8434 | 0.9938 |
| 40 | 60 | 97,115 | 0.9356 | 0.9070 | 0.9947 | 0.9488 | 0.8469 | 0.9907 |
| 30 | 70 | 83,240 | 0.9495 | 0.9371 | 0.9947 | 0.9650 | 0.8442 | 0.9856 |
| 20 | 80 | 72,835 | 0.9650 | 0.9628 | 0.9947 | 0.9785 | 0.8464 | 0.9755 |
| 10 | 90 | 64,740 | 0.9793 | 0.9826 | 0.9947 | 0.9886 | 0.8412 | 0.9463 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9973 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8446 | 0.0000 | 0.0000 | 0.0000 | 0.8446 | 1.0000 |
| 90 | 10 | 299,940 | 0.8605 | 0.4171 | 0.9948 | 0.5878 | 0.8455 | 0.9993 |
| 80 | 20 | 291,350 | 0.8751 | 0.6163 | 0.9947 | 0.7611 | 0.8452 | 0.9984 |
| 70 | 30 | 194,230 | 0.8905 | 0.7345 | 0.9947 | 0.8450 | 0.8459 | 0.9973 |
| 60 | 40 | 145,675 | 0.9060 | 0.8125 | 0.9947 | 0.8944 | 0.8469 | 0.9958 |
| 50 | 50 | 116,540 | 0.9191 | 0.8640 | 0.9947 | 0.9247 | 0.8434 | 0.9938 |
| 40 | 60 | 97,115 | 0.9356 | 0.9070 | 0.9947 | 0.9488 | 0.8469 | 0.9907 |
| 30 | 70 | 83,240 | 0.9495 | 0.9371 | 0.9947 | 0.9650 | 0.8442 | 0.9856 |
| 20 | 80 | 72,835 | 0.9650 | 0.9628 | 0.9947 | 0.9785 | 0.8464 | 0.9755 |
| 10 | 90 | 64,740 | 0.9793 | 0.9826 | 0.9947 | 0.9886 | 0.8412 | 0.9463 |
| 0 | 100 | 58,270 | 0.9947 | 1.0000 | 0.9947 | 0.9973 | 0.0000 | 0.0000 |


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
0.15       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213   <--
0.20       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.25       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.30       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.35       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.40       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.45       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.50       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.55       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.60       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.65       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.70       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.75       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
0.80       0.3213   0.2144   0.2541   0.9688   0.9264   0.1213  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3213, F1=0.2144, Normal Recall=0.2541, Normal Precision=0.9688, Attack Recall=0.9264, Attack Precision=0.1213

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
0.15       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365   <--
0.20       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.25       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.30       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.35       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.40       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.45       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.50       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.55       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.60       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.65       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.70       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.75       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
0.80       0.3879   0.3768   0.2536   0.9312   0.9250   0.2365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3879, F1=0.3768, Normal Recall=0.2536, Normal Precision=0.9312, Attack Recall=0.9250, Attack Precision=0.2365

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
0.15       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468   <--
0.20       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.25       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.30       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.35       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.40       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.45       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.50       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.55       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.60       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.65       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.70       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.75       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
0.80       0.4549   0.5045   0.2534   0.8874   0.9250   0.3468  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4549, F1=0.5045, Normal Recall=0.2534, Normal Precision=0.8874, Attack Recall=0.9250, Attack Precision=0.3468

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
0.15       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521   <--
0.20       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.25       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.30       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.35       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.40       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.45       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.50       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.55       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.60       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.65       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.70       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.75       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
0.80       0.5216   0.6074   0.2527   0.8348   0.9250   0.4521  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5216, F1=0.6074, Normal Recall=0.2527, Normal Precision=0.8348, Attack Recall=0.9250, Attack Precision=0.4521

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
0.15       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533   <--
0.20       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.25       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.30       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.35       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.40       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.45       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.50       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.55       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.60       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.65       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.70       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.75       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
0.80       0.5891   0.6924   0.2532   0.7715   0.9250   0.5533  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5891, F1=0.6924, Normal Recall=0.2532, Normal Precision=0.7715, Attack Recall=0.9250, Attack Precision=0.5533

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
0.15       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151   <--
0.20       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.25       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.30       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.35       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.40       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.45       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.50       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.55       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.60       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.65       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.70       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.75       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
0.80       0.8593   0.5859   0.8442   0.9994   0.9952   0.4151  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8593, F1=0.5859, Normal Recall=0.8442, Normal Precision=0.9994, Attack Recall=0.9952, Attack Precision=0.4151

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
0.15       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153   <--
0.20       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.25       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.30       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.35       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.40       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.45       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.50       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.55       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.60       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.65       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.70       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.75       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
0.80       0.8746   0.7603   0.8445   0.9984   0.9947   0.6153  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8746, F1=0.7603, Normal Recall=0.8445, Normal Precision=0.9984, Attack Recall=0.9947, Attack Precision=0.6153

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
0.15       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322   <--
0.20       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.25       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.30       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.35       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.40       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.45       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.50       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.55       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.60       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.65       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.70       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.75       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
0.80       0.8892   0.8435   0.8441   0.9973   0.9947   0.7322  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8892, F1=0.8435, Normal Recall=0.8441, Normal Precision=0.9973, Attack Recall=0.9947, Attack Precision=0.7322

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
0.15       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090   <--
0.20       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.25       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.30       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.35       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.40       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.45       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.50       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.55       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.60       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.65       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.70       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.75       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
0.80       0.9039   0.8923   0.8434   0.9958   0.9947   0.8090  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9039, F1=0.8923, Normal Recall=0.8434, Normal Precision=0.9958, Attack Recall=0.9947, Attack Precision=0.8090

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
0.15       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633   <--
0.20       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.25       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.30       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.35       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.40       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.45       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.50       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.55       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.60       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.65       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.70       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.75       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
0.80       0.9186   0.9244   0.8426   0.9937   0.9947   0.8633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9186, F1=0.9244, Normal Recall=0.8426, Normal Precision=0.9937, Attack Recall=0.9947, Attack Precision=0.8633

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
0.15       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172   <--
0.20       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.25       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.30       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.35       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.40       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.45       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.50       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.55       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.60       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.65       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.70       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.75       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.80       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8605, F1=0.5879, Normal Recall=0.8455, Normal Precision=0.9994, Attack Recall=0.9952, Attack Precision=0.4172

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
0.15       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173   <--
0.20       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.25       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.30       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.35       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.40       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.45       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.50       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.55       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.60       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.65       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.70       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.75       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.80       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8756, F1=0.7619, Normal Recall=0.8459, Normal Precision=0.9984, Attack Recall=0.9947, Attack Precision=0.6173

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
0.15       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340   <--
0.20       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.25       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.30       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.35       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.40       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.45       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.50       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.55       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.60       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.65       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.70       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.75       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.80       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8903, F1=0.8447, Normal Recall=0.8455, Normal Precision=0.9973, Attack Recall=0.9947, Attack Precision=0.7340

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
0.15       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105   <--
0.20       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.25       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.30       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.35       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.40       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.45       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.50       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.55       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.60       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.65       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.70       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.75       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.80       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9049, F1=0.8932, Normal Recall=0.8450, Normal Precision=0.9958, Attack Recall=0.9947, Attack Precision=0.8105

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
0.15       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646   <--
0.20       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.25       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.30       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.35       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.40       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.45       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.50       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.55       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.60       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.65       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.70       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.75       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.80       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9195, F1=0.9251, Normal Recall=0.8442, Normal Precision=0.9938, Attack Recall=0.9947, Attack Precision=0.8646

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
0.15       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172   <--
0.20       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.25       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.30       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.35       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.40       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.45       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.50       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.55       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.60       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.65       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.70       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.75       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
0.80       0.8605   0.5879   0.8455   0.9994   0.9952   0.4172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8605, F1=0.5879, Normal Recall=0.8455, Normal Precision=0.9994, Attack Recall=0.9952, Attack Precision=0.4172

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
0.15       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173   <--
0.20       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.25       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.30       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.35       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.40       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.45       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.50       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.55       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.60       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.65       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.70       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.75       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
0.80       0.8756   0.7619   0.8459   0.9984   0.9947   0.6173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8756, F1=0.7619, Normal Recall=0.8459, Normal Precision=0.9984, Attack Recall=0.9947, Attack Precision=0.6173

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
0.15       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340   <--
0.20       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.25       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.30       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.35       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.40       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.45       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.50       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.55       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.60       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.65       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.70       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.75       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
0.80       0.8903   0.8447   0.8455   0.9973   0.9947   0.7340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8903, F1=0.8447, Normal Recall=0.8455, Normal Precision=0.9973, Attack Recall=0.9947, Attack Precision=0.7340

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
0.15       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105   <--
0.20       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.25       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.30       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.35       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.40       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.45       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.50       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.55       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.60       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.65       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.70       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.75       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
0.80       0.9049   0.8932   0.8450   0.9958   0.9947   0.8105  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9049, F1=0.8932, Normal Recall=0.8450, Normal Precision=0.9958, Attack Recall=0.9947, Attack Precision=0.8105

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
0.15       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646   <--
0.20       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.25       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.30       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.35       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.40       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.45       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.50       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.55       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.60       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.65       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.70       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.75       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
0.80       0.9195   0.9251   0.8442   0.9938   0.9947   0.8646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9195, F1=0.9251, Normal Recall=0.8442, Normal Precision=0.9938, Attack Recall=0.9947, Attack Precision=0.8646

```

