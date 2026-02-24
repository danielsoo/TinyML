# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-15 05:12:37 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8567 | 0.8709 | 0.8840 | 0.8984 | 0.9120 | 0.9241 | 0.9386 | 0.9524 | 0.9653 | 0.9784 | 0.9921 |
| QAT+Prune only | 0.8361 | 0.8386 | 0.8406 | 0.8436 | 0.8446 | 0.8452 | 0.8489 | 0.8497 | 0.8527 | 0.8541 | 0.8570 |
| QAT+PTQ | 0.8353 | 0.8378 | 0.8401 | 0.8433 | 0.8447 | 0.8454 | 0.8495 | 0.8505 | 0.8537 | 0.8554 | 0.8586 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8353 | 0.8378 | 0.8401 | 0.8433 | 0.8447 | 0.8454 | 0.8495 | 0.8505 | 0.8537 | 0.8554 | 0.8586 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6058 | 0.7738 | 0.8542 | 0.9002 | 0.9289 | 0.9509 | 0.9669 | 0.9786 | 0.9881 | 0.9960 |
| QAT+Prune only | 0.0000 | 0.5147 | 0.6826 | 0.7668 | 0.8152 | 0.8470 | 0.8719 | 0.8887 | 0.9030 | 0.9136 | 0.9230 |
| QAT+PTQ | 0.0000 | 0.5139 | 0.6823 | 0.7668 | 0.8156 | 0.8474 | 0.8725 | 0.8894 | 0.9038 | 0.9144 | 0.9239 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5139 | 0.6823 | 0.7668 | 0.8156 | 0.8474 | 0.8725 | 0.8894 | 0.9038 | 0.9144 | 0.9239 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8567 | 0.8575 | 0.8570 | 0.8582 | 0.8586 | 0.8560 | 0.8582 | 0.8597 | 0.8578 | 0.8553 | 0.0000 |
| QAT+Prune only | 0.8361 | 0.8367 | 0.8365 | 0.8379 | 0.8362 | 0.8333 | 0.8367 | 0.8327 | 0.8352 | 0.8273 | 0.0000 |
| QAT+PTQ | 0.8353 | 0.8356 | 0.8354 | 0.8367 | 0.8354 | 0.8322 | 0.8357 | 0.8316 | 0.8339 | 0.8261 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8353 | 0.8356 | 0.8354 | 0.8367 | 0.8354 | 0.8322 | 0.8357 | 0.8316 | 0.8339 | 0.8261 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8567 | 0.0000 | 0.0000 | 0.0000 | 0.8567 | 1.0000 |
| 90 | 10 | 299,940 | 0.8709 | 0.4361 | 0.9920 | 0.6058 | 0.8575 | 0.9990 |
| 80 | 20 | 291,350 | 0.8840 | 0.6342 | 0.9921 | 0.7738 | 0.8570 | 0.9977 |
| 70 | 30 | 194,230 | 0.8984 | 0.7499 | 0.9921 | 0.8542 | 0.8582 | 0.9961 |
| 60 | 40 | 145,675 | 0.9120 | 0.8238 | 0.9921 | 0.9002 | 0.8586 | 0.9939 |
| 50 | 50 | 116,540 | 0.9241 | 0.8733 | 0.9921 | 0.9289 | 0.8560 | 0.9909 |
| 40 | 60 | 97,115 | 0.9386 | 0.9130 | 0.9921 | 0.9509 | 0.8582 | 0.9864 |
| 30 | 70 | 83,240 | 0.9524 | 0.9429 | 0.9921 | 0.9669 | 0.8597 | 0.9791 |
| 20 | 80 | 72,835 | 0.9653 | 0.9654 | 0.9921 | 0.9786 | 0.8578 | 0.9646 |
| 10 | 90 | 64,740 | 0.9784 | 0.9840 | 0.9921 | 0.9881 | 0.8553 | 0.9234 |
| 0 | 100 | 58,270 | 0.9921 | 1.0000 | 0.9921 | 0.9960 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8361 | 0.0000 | 0.0000 | 0.0000 | 0.8361 | 1.0000 |
| 90 | 10 | 299,940 | 0.8386 | 0.3680 | 0.8559 | 0.5147 | 0.8367 | 0.9812 |
| 80 | 20 | 291,350 | 0.8406 | 0.5671 | 0.8570 | 0.6826 | 0.8365 | 0.9590 |
| 70 | 30 | 194,230 | 0.8436 | 0.6938 | 0.8570 | 0.7668 | 0.8379 | 0.9319 |
| 60 | 40 | 145,675 | 0.8446 | 0.7772 | 0.8570 | 0.8152 | 0.8362 | 0.8977 |
| 50 | 50 | 116,540 | 0.8452 | 0.8372 | 0.8570 | 0.8470 | 0.8333 | 0.8536 |
| 40 | 60 | 97,115 | 0.8489 | 0.8873 | 0.8570 | 0.8719 | 0.8367 | 0.7960 |
| 30 | 70 | 83,240 | 0.8497 | 0.9228 | 0.8570 | 0.8887 | 0.8327 | 0.7140 |
| 20 | 80 | 72,835 | 0.8527 | 0.9541 | 0.8571 | 0.9030 | 0.8352 | 0.5936 |
| 10 | 90 | 64,740 | 0.8541 | 0.9781 | 0.8570 | 0.9136 | 0.8273 | 0.3913 |
| 0 | 100 | 58,270 | 0.8570 | 1.0000 | 0.8570 | 0.9230 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8353 | 0.0000 | 0.0000 | 0.0000 | 0.8353 | 1.0000 |
| 90 | 10 | 299,940 | 0.8378 | 0.3669 | 0.8573 | 0.5139 | 0.8356 | 0.9814 |
| 80 | 20 | 291,350 | 0.8401 | 0.5661 | 0.8586 | 0.6823 | 0.8354 | 0.9594 |
| 70 | 30 | 194,230 | 0.8433 | 0.6927 | 0.8586 | 0.7668 | 0.8367 | 0.9325 |
| 60 | 40 | 145,675 | 0.8447 | 0.7766 | 0.8586 | 0.8156 | 0.8354 | 0.8986 |
| 50 | 50 | 116,540 | 0.8454 | 0.8365 | 0.8586 | 0.8474 | 0.8322 | 0.8548 |
| 40 | 60 | 97,115 | 0.8495 | 0.8869 | 0.8586 | 0.8725 | 0.8357 | 0.7976 |
| 30 | 70 | 83,240 | 0.8505 | 0.9225 | 0.8586 | 0.8894 | 0.8316 | 0.7160 |
| 20 | 80 | 72,835 | 0.8537 | 0.9539 | 0.8587 | 0.9038 | 0.8339 | 0.5959 |
| 10 | 90 | 64,740 | 0.8554 | 0.9780 | 0.8586 | 0.9144 | 0.8261 | 0.3937 |
| 0 | 100 | 58,270 | 0.8586 | 1.0000 | 0.8586 | 0.9239 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8353 | 0.0000 | 0.0000 | 0.0000 | 0.8353 | 1.0000 |
| 90 | 10 | 299,940 | 0.8378 | 0.3669 | 0.8573 | 0.5139 | 0.8356 | 0.9814 |
| 80 | 20 | 291,350 | 0.8401 | 0.5661 | 0.8586 | 0.6823 | 0.8354 | 0.9594 |
| 70 | 30 | 194,230 | 0.8433 | 0.6927 | 0.8586 | 0.7668 | 0.8367 | 0.9325 |
| 60 | 40 | 145,675 | 0.8447 | 0.7766 | 0.8586 | 0.8156 | 0.8354 | 0.8986 |
| 50 | 50 | 116,540 | 0.8454 | 0.8365 | 0.8586 | 0.8474 | 0.8322 | 0.8548 |
| 40 | 60 | 97,115 | 0.8495 | 0.8869 | 0.8586 | 0.8725 | 0.8357 | 0.7976 |
| 30 | 70 | 83,240 | 0.8505 | 0.9225 | 0.8586 | 0.8894 | 0.8316 | 0.7160 |
| 20 | 80 | 72,835 | 0.8537 | 0.9539 | 0.8587 | 0.9038 | 0.8339 | 0.5959 |
| 10 | 90 | 64,740 | 0.8554 | 0.9780 | 0.8586 | 0.9144 | 0.8261 | 0.3937 |
| 0 | 100 | 58,270 | 0.8586 | 1.0000 | 0.8586 | 0.9239 | 0.0000 | 0.0000 |


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
0.15       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363   <--
0.20       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.25       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.30       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.35       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.40       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.45       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.50       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.55       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.60       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.65       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.70       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.75       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
0.80       0.8710   0.6063   0.8575   0.9991   0.9931   0.4363  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8710, F1=0.6063, Normal Recall=0.8575, Normal Precision=0.9991, Attack Recall=0.9931, Attack Precision=0.4363

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
0.15       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356   <--
0.20       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.25       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.30       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.35       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.40       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.45       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.50       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.55       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.60       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.65       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.70       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.75       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
0.80       0.8847   0.7748   0.8578   0.9977   0.9921   0.6356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8847, F1=0.7748, Normal Recall=0.8578, Normal Precision=0.9977, Attack Recall=0.9921, Attack Precision=0.6356

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
0.15       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487   <--
0.20       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.25       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.30       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.35       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.40       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.45       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.50       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.55       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.60       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.65       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.70       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.75       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
0.80       0.8977   0.8534   0.8573   0.9961   0.9921   0.7487  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8977, F1=0.8534, Normal Recall=0.8573, Normal Precision=0.9961, Attack Recall=0.9921, Attack Precision=0.7487

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
0.15       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222   <--
0.20       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.25       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.30       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.35       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.40       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.45       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.50       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.55       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.60       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.65       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.70       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.75       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
0.80       0.9110   0.8992   0.8569   0.9939   0.9921   0.8222  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9110, F1=0.8992, Normal Recall=0.8569, Normal Precision=0.9939, Attack Recall=0.9921, Attack Precision=0.8222

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
0.15       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738   <--
0.20       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.25       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.30       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.35       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.40       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.45       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.50       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.55       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.60       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.65       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.70       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.75       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
0.80       0.9244   0.9292   0.8567   0.9909   0.9921   0.8738  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9244, F1=0.9292, Normal Recall=0.8567, Normal Precision=0.9909, Attack Recall=0.9921, Attack Precision=0.8738

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
0.15       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688   <--
0.20       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.25       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.30       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.35       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.40       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.45       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.50       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.55       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.60       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.65       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.70       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.75       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
0.80       0.8389   0.5160   0.8367   0.9816   0.8589   0.3688  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.5160, Normal Recall=0.8367, Normal Precision=0.9816, Attack Recall=0.8589, Attack Precision=0.3688

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
0.15       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678   <--
0.20       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.25       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.30       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.35       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.40       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.45       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.50       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.55       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.60       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.65       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.70       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.75       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
0.80       0.8409   0.6830   0.8369   0.9590   0.8570   0.5678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8409, F1=0.6830, Normal Recall=0.8369, Normal Precision=0.9590, Attack Recall=0.8570, Attack Precision=0.5678

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
0.15       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904   <--
0.20       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.25       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.30       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.35       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.40       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.45       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.50       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.55       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.60       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.65       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.70       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.75       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
0.80       0.8418   0.7647   0.8353   0.9317   0.8570   0.6904  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8418, F1=0.7647, Normal Recall=0.8353, Normal Precision=0.9317, Attack Recall=0.8570, Attack Precision=0.6904

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
0.15       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770   <--
0.20       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.25       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.30       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.35       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.40       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.45       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.50       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.55       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.60       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.65       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.70       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.75       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
0.80       0.8444   0.8151   0.8360   0.8977   0.8570   0.7770  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8444, F1=0.8151, Normal Recall=0.8360, Normal Precision=0.8977, Attack Recall=0.8570, Attack Precision=0.7770

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
0.15       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383   <--
0.20       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.25       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.30       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.35       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.40       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.45       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.50       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.55       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.60       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.65       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.70       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.75       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
0.80       0.8458   0.8476   0.8346   0.8538   0.8570   0.8383  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8458, F1=0.8476, Normal Recall=0.8346, Normal Precision=0.8538, Attack Recall=0.8570, Attack Precision=0.8383

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
0.15       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678   <--
0.20       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.25       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.30       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.35       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.40       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.45       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.50       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.55       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.60       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.65       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.70       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.75       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.80       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8382, F1=0.5154, Normal Recall=0.8356, Normal Precision=0.9818, Attack Recall=0.8607, Attack Precision=0.3678

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
0.15       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667   <--
0.20       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.25       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.30       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.35       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.40       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.45       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.50       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.55       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.60       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.65       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.70       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.75       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.80       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8404, F1=0.6828, Normal Recall=0.8359, Normal Precision=0.9594, Attack Recall=0.8586, Attack Precision=0.5667

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
0.15       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894   <--
0.20       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.25       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.30       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.35       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.40       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.45       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.50       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.55       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.60       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.65       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.70       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.75       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.80       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8416, F1=0.7648, Normal Recall=0.8342, Normal Precision=0.9323, Attack Recall=0.8586, Attack Precision=0.6894

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
0.15       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764   <--
0.20       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.25       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.30       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.35       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.40       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.45       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.50       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.55       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.60       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.65       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.70       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.75       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.80       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8445, F1=0.8154, Normal Recall=0.8351, Normal Precision=0.8986, Attack Recall=0.8586, Attack Precision=0.7764

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
0.15       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378   <--
0.20       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.25       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.30       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.35       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.40       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.45       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.50       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.55       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.60       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.65       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.70       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.75       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.80       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8462, F1=0.8481, Normal Recall=0.8338, Normal Precision=0.8550, Attack Recall=0.8586, Attack Precision=0.8378

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
0.15       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678   <--
0.20       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.25       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.30       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.35       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.40       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.45       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.50       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.55       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.60       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.65       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.70       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.75       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
0.80       0.8382   0.5154   0.8356   0.9818   0.8607   0.3678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8382, F1=0.5154, Normal Recall=0.8356, Normal Precision=0.9818, Attack Recall=0.8607, Attack Precision=0.3678

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
0.15       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667   <--
0.20       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.25       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.30       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.35       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.40       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.45       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.50       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.55       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.60       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.65       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.70       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.75       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
0.80       0.8404   0.6828   0.8359   0.9594   0.8586   0.5667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8404, F1=0.6828, Normal Recall=0.8359, Normal Precision=0.9594, Attack Recall=0.8586, Attack Precision=0.5667

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
0.15       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894   <--
0.20       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.25       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.30       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.35       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.40       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.45       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.50       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.55       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.60       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.65       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.70       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.75       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
0.80       0.8416   0.7648   0.8342   0.9323   0.8586   0.6894  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8416, F1=0.7648, Normal Recall=0.8342, Normal Precision=0.9323, Attack Recall=0.8586, Attack Precision=0.6894

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
0.15       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764   <--
0.20       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.25       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.30       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.35       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.40       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.45       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.50       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.55       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.60       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.65       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.70       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.75       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
0.80       0.8445   0.8154   0.8351   0.8986   0.8586   0.7764  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8445, F1=0.8154, Normal Recall=0.8351, Normal Precision=0.8986, Attack Recall=0.8586, Attack Precision=0.7764

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
0.15       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378   <--
0.20       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.25       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.30       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.35       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.40       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.45       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.50       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.55       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.60       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.65       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.70       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.75       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
0.80       0.8462   0.8481   0.8338   0.8550   0.8586   0.8378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8462, F1=0.8481, Normal Recall=0.8338, Normal Precision=0.8550, Attack Recall=0.8586, Attack Precision=0.8378

```

