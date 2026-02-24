# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-19 04:37:08 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2579 | 0.3191 | 0.3806 | 0.4433 | 0.5030 | 0.5658 | 0.6267 | 0.6897 | 0.7499 | 0.8131 | 0.8741 |
| QAT+Prune only | 0.8746 | 0.8804 | 0.8863 | 0.8944 | 0.9001 | 0.9064 | 0.9131 | 0.9201 | 0.9259 | 0.9334 | 0.9400 |
| QAT+PTQ | 0.8770 | 0.8816 | 0.8861 | 0.8929 | 0.8973 | 0.9023 | 0.9076 | 0.9134 | 0.9179 | 0.9240 | 0.9293 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8770 | 0.8816 | 0.8861 | 0.8929 | 0.8973 | 0.9023 | 0.9076 | 0.9134 | 0.9179 | 0.9240 | 0.9293 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2044 | 0.3608 | 0.4851 | 0.5845 | 0.6681 | 0.7375 | 0.7977 | 0.8483 | 0.8938 | 0.9328 |
| QAT+Prune only | 0.0000 | 0.6116 | 0.7678 | 0.8423 | 0.8827 | 0.9094 | 0.9285 | 0.9428 | 0.9531 | 0.9622 | 0.9691 |
| QAT+PTQ | 0.0000 | 0.6111 | 0.7655 | 0.8388 | 0.8786 | 0.9049 | 0.9235 | 0.9376 | 0.9477 | 0.9566 | 0.9634 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6111 | 0.7655 | 0.8388 | 0.8786 | 0.9049 | 0.9235 | 0.9376 | 0.9477 | 0.9566 | 0.9634 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2579 | 0.2573 | 0.2572 | 0.2587 | 0.2557 | 0.2575 | 0.2557 | 0.2595 | 0.2535 | 0.2641 | 0.0000 |
| QAT+Prune only | 0.8746 | 0.8736 | 0.8729 | 0.8749 | 0.8734 | 0.8727 | 0.8728 | 0.8736 | 0.8696 | 0.8741 | 0.0000 |
| QAT+PTQ | 0.8770 | 0.8761 | 0.8753 | 0.8772 | 0.8759 | 0.8753 | 0.8751 | 0.8761 | 0.8720 | 0.8761 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8770 | 0.8761 | 0.8753 | 0.8772 | 0.8759 | 0.8753 | 0.8751 | 0.8761 | 0.8720 | 0.8761 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2579 | 0.0000 | 0.0000 | 0.0000 | 0.2579 | 1.0000 |
| 90 | 10 | 299,940 | 0.3191 | 0.1157 | 0.8748 | 0.2044 | 0.2573 | 0.9487 |
| 80 | 20 | 291,350 | 0.3806 | 0.2273 | 0.8741 | 0.3608 | 0.2572 | 0.8909 |
| 70 | 30 | 194,230 | 0.4433 | 0.3357 | 0.8740 | 0.4851 | 0.2587 | 0.8274 |
| 60 | 40 | 145,675 | 0.5030 | 0.4391 | 0.8741 | 0.5845 | 0.2557 | 0.7528 |
| 50 | 50 | 116,540 | 0.5658 | 0.5407 | 0.8741 | 0.6681 | 0.2575 | 0.6716 |
| 40 | 60 | 97,115 | 0.6267 | 0.6379 | 0.8740 | 0.7375 | 0.2557 | 0.5751 |
| 30 | 70 | 83,240 | 0.6897 | 0.7336 | 0.8740 | 0.7977 | 0.2595 | 0.4689 |
| 20 | 80 | 72,835 | 0.7499 | 0.8241 | 0.8740 | 0.8483 | 0.2535 | 0.3348 |
| 10 | 90 | 64,740 | 0.8131 | 0.9145 | 0.8740 | 0.8938 | 0.2641 | 0.1890 |
| 0 | 100 | 58,270 | 0.8741 | 1.0000 | 0.8741 | 0.9328 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8746 | 0.0000 | 0.0000 | 0.0000 | 0.8746 | 1.0000 |
| 90 | 10 | 299,940 | 0.8804 | 0.4529 | 0.9416 | 0.6116 | 0.8736 | 0.9926 |
| 80 | 20 | 291,350 | 0.8863 | 0.6489 | 0.9400 | 0.7678 | 0.8729 | 0.9831 |
| 70 | 30 | 194,230 | 0.8944 | 0.7630 | 0.9400 | 0.8423 | 0.8749 | 0.9715 |
| 60 | 40 | 145,675 | 0.9001 | 0.8320 | 0.9400 | 0.8827 | 0.8734 | 0.9562 |
| 50 | 50 | 116,540 | 0.9064 | 0.8807 | 0.9400 | 0.9094 | 0.8727 | 0.9357 |
| 40 | 60 | 97,115 | 0.9131 | 0.9173 | 0.9400 | 0.9285 | 0.8728 | 0.9066 |
| 30 | 70 | 83,240 | 0.9201 | 0.9455 | 0.9400 | 0.9428 | 0.8736 | 0.8619 |
| 20 | 80 | 72,835 | 0.9259 | 0.9665 | 0.9400 | 0.9531 | 0.8696 | 0.7838 |
| 10 | 90 | 64,740 | 0.9334 | 0.9853 | 0.9400 | 0.9622 | 0.8741 | 0.6183 |
| 0 | 100 | 58,270 | 0.9400 | 1.0000 | 0.9400 | 0.9691 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8770 | 0.0000 | 0.0000 | 0.0000 | 0.8770 | 1.0000 |
| 90 | 10 | 299,940 | 0.8816 | 0.4549 | 0.9306 | 0.6111 | 0.8761 | 0.9913 |
| 80 | 20 | 291,350 | 0.8861 | 0.6507 | 0.9293 | 0.7655 | 0.8753 | 0.9802 |
| 70 | 30 | 194,230 | 0.8929 | 0.7644 | 0.9293 | 0.8388 | 0.8772 | 0.9666 |
| 60 | 40 | 145,675 | 0.8973 | 0.8331 | 0.9293 | 0.8786 | 0.8759 | 0.9490 |
| 50 | 50 | 116,540 | 0.9023 | 0.8817 | 0.9293 | 0.9049 | 0.8753 | 0.9253 |
| 40 | 60 | 97,115 | 0.9076 | 0.9178 | 0.9293 | 0.9235 | 0.8751 | 0.8920 |
| 30 | 70 | 83,240 | 0.9134 | 0.9460 | 0.9293 | 0.9376 | 0.8761 | 0.8416 |
| 20 | 80 | 72,835 | 0.9179 | 0.9667 | 0.9293 | 0.9477 | 0.8720 | 0.7552 |
| 10 | 90 | 64,740 | 0.9240 | 0.9854 | 0.9294 | 0.9566 | 0.8761 | 0.5795 |
| 0 | 100 | 58,270 | 0.9293 | 1.0000 | 0.9293 | 0.9634 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8770 | 0.0000 | 0.0000 | 0.0000 | 0.8770 | 1.0000 |
| 90 | 10 | 299,940 | 0.8816 | 0.4549 | 0.9306 | 0.6111 | 0.8761 | 0.9913 |
| 80 | 20 | 291,350 | 0.8861 | 0.6507 | 0.9293 | 0.7655 | 0.8753 | 0.9802 |
| 70 | 30 | 194,230 | 0.8929 | 0.7644 | 0.9293 | 0.8388 | 0.8772 | 0.9666 |
| 60 | 40 | 145,675 | 0.8973 | 0.8331 | 0.9293 | 0.8786 | 0.8759 | 0.9490 |
| 50 | 50 | 116,540 | 0.9023 | 0.8817 | 0.9293 | 0.9049 | 0.8753 | 0.9253 |
| 40 | 60 | 97,115 | 0.9076 | 0.9178 | 0.9293 | 0.9235 | 0.8751 | 0.8920 |
| 30 | 70 | 83,240 | 0.9134 | 0.9460 | 0.9293 | 0.9376 | 0.8761 | 0.8416 |
| 20 | 80 | 72,835 | 0.9179 | 0.9667 | 0.9293 | 0.9477 | 0.8720 | 0.7552 |
| 10 | 90 | 64,740 | 0.9240 | 0.9854 | 0.9294 | 0.9566 | 0.8761 | 0.5795 |
| 0 | 100 | 58,270 | 0.9293 | 1.0000 | 0.9293 | 0.9634 | 0.0000 | 0.0000 |


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
0.15       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157   <--
0.20       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.25       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.30       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.35       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.40       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.45       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.50       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.55       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.60       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.65       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.70       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.75       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
0.80       0.3191   0.2044   0.2573   0.9487   0.8748   0.1157  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3191, F1=0.2044, Normal Recall=0.2573, Normal Precision=0.9487, Attack Recall=0.8748, Attack Precision=0.1157

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
0.15       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274   <--
0.20       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.25       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.30       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.35       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.40       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.45       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.50       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.55       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.60       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.65       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.70       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.75       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
0.80       0.3807   0.3609   0.2574   0.8910   0.8741   0.2274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3807, F1=0.3609, Normal Recall=0.2574, Normal Precision=0.8910, Attack Recall=0.8741, Attack Precision=0.2274

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
0.15       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354   <--
0.20       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.25       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.30       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.35       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.40       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.45       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.50       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.55       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.60       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.65       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.70       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.75       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
0.80       0.4426   0.4848   0.2577   0.8268   0.8740   0.3354  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4426, F1=0.4848, Normal Recall=0.2577, Normal Precision=0.8268, Attack Recall=0.8740, Attack Precision=0.3354

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
0.15       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397   <--
0.20       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.25       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.30       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.35       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.40       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.45       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.50       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.55       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.60       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.65       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.70       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.75       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
0.80       0.5042   0.5851   0.2576   0.7541   0.8741   0.4397  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5042, F1=0.5851, Normal Recall=0.2576, Normal Precision=0.7541, Attack Recall=0.8741, Attack Precision=0.4397

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
0.15       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403   <--
0.20       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.25       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.30       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.35       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.40       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.45       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.50       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.55       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.60       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.65       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.70       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.75       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
0.80       0.5651   0.6678   0.2562   0.6705   0.8741   0.5403  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5651, F1=0.6678, Normal Recall=0.2562, Normal Precision=0.6705, Attack Recall=0.8741, Attack Precision=0.5403

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
0.15       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525   <--
0.20       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.25       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.30       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.35       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.40       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.45       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.50       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.55       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.60       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.65       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.70       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.75       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
0.80       0.8803   0.6109   0.8736   0.9924   0.9400   0.4525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8803, F1=0.6109, Normal Recall=0.8736, Normal Precision=0.9924, Attack Recall=0.9400, Attack Precision=0.4525

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
0.15       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512   <--
0.20       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.25       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.30       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.35       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.40       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.45       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.50       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.55       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.60       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.65       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.70       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.75       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
0.80       0.8873   0.7694   0.8741   0.9831   0.9400   0.6512  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8873, F1=0.7694, Normal Recall=0.8741, Normal Precision=0.9831, Attack Recall=0.9400, Attack Precision=0.6512

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
0.15       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631   <--
0.20       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.25       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.30       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.35       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.40       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.45       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.50       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.55       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.60       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.65       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.70       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.75       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
0.80       0.8945   0.8424   0.8749   0.9715   0.9400   0.7631  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8945, F1=0.8424, Normal Recall=0.8749, Normal Precision=0.9715, Attack Recall=0.9400, Attack Precision=0.7631

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
0.15       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333   <--
0.20       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.25       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.30       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.35       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.40       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.45       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.50       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.55       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.60       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.65       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.70       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.75       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
0.80       0.9008   0.8834   0.8746   0.9563   0.9400   0.8333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9008, F1=0.8834, Normal Recall=0.8746, Normal Precision=0.9563, Attack Recall=0.9400, Attack Precision=0.8333

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
0.15       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815   <--
0.20       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.25       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.30       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.35       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.40       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.45       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.50       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.55       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.60       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.65       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.70       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.75       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
0.80       0.9068   0.9098   0.8736   0.9358   0.9400   0.8815  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9068, F1=0.9098, Normal Recall=0.8736, Normal Precision=0.9358, Attack Recall=0.9400, Attack Precision=0.8815

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
0.15       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544   <--
0.20       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.25       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.30       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.35       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.40       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.45       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.50       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.55       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.60       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.65       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.70       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.75       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.80       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8814, F1=0.6102, Normal Recall=0.8761, Normal Precision=0.9910, Attack Recall=0.9285, Attack Precision=0.4544

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
0.15       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530   <--
0.20       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.25       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.30       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.35       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.40       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.45       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.50       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.55       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.60       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.65       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.70       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.75       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.80       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8871, F1=0.7671, Normal Recall=0.8765, Normal Precision=0.9802, Attack Recall=0.9293, Attack Precision=0.6530

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
0.15       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646   <--
0.20       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.25       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.30       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.35       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.40       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.45       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.50       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.55       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.60       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.65       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.70       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.75       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.80       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8930, F1=0.8390, Normal Recall=0.8774, Normal Precision=0.9666, Attack Recall=0.9293, Attack Precision=0.7646

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
0.15       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344   <--
0.20       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.25       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.30       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.35       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.40       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.45       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.50       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.55       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.60       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.65       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.70       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.75       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.80       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8979, F1=0.8793, Normal Recall=0.8770, Normal Precision=0.9490, Attack Recall=0.9293, Attack Precision=0.8344

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
0.15       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822   <--
0.20       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.25       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.30       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.35       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.40       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.45       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.50       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.55       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.60       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.65       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.70       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.75       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.80       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9026, F1=0.9052, Normal Recall=0.8759, Normal Precision=0.9254, Attack Recall=0.9293, Attack Precision=0.8822

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
0.15       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544   <--
0.20       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.25       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.30       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.35       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.40       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.45       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.50       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.55       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.60       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.65       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.70       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.75       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
0.80       0.8814   0.6102   0.8761   0.9910   0.9285   0.4544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8814, F1=0.6102, Normal Recall=0.8761, Normal Precision=0.9910, Attack Recall=0.9285, Attack Precision=0.4544

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
0.15       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530   <--
0.20       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.25       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.30       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.35       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.40       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.45       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.50       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.55       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.60       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.65       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.70       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.75       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
0.80       0.8871   0.7671   0.8765   0.9802   0.9293   0.6530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8871, F1=0.7671, Normal Recall=0.8765, Normal Precision=0.9802, Attack Recall=0.9293, Attack Precision=0.6530

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
0.15       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646   <--
0.20       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.25       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.30       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.35       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.40       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.45       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.50       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.55       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.60       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.65       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.70       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.75       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
0.80       0.8930   0.8390   0.8774   0.9666   0.9293   0.7646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8930, F1=0.8390, Normal Recall=0.8774, Normal Precision=0.9666, Attack Recall=0.9293, Attack Precision=0.7646

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
0.15       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344   <--
0.20       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.25       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.30       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.35       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.40       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.45       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.50       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.55       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.60       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.65       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.70       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.75       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
0.80       0.8979   0.8793   0.8770   0.9490   0.9293   0.8344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8979, F1=0.8793, Normal Recall=0.8770, Normal Precision=0.9490, Attack Recall=0.9293, Attack Precision=0.8344

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
0.15       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822   <--
0.20       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.25       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.30       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.35       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.40       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.45       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.50       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.55       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.60       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.65       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.70       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.75       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
0.80       0.9026   0.9052   0.8759   0.9254   0.9293   0.8822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9026, F1=0.9052, Normal Recall=0.8759, Normal Precision=0.9254, Attack Recall=0.9293, Attack Precision=0.8822

```

