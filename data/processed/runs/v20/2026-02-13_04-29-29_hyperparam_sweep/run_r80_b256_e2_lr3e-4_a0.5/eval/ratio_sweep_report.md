# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-18 18:03:16 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8641 | 0.8739 | 0.8851 | 0.8974 | 0.9080 | 0.9184 | 0.9302 | 0.9416 | 0.9533 | 0.9645 | 0.9760 |
| QAT+Prune only | 0.8345 | 0.8495 | 0.8638 | 0.8791 | 0.8946 | 0.9082 | 0.9243 | 0.9391 | 0.9535 | 0.9694 | 0.9840 |
| QAT+PTQ | 0.8351 | 0.8499 | 0.8642 | 0.8794 | 0.8948 | 0.9083 | 0.9245 | 0.9392 | 0.9536 | 0.9694 | 0.9840 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8351 | 0.8499 | 0.8642 | 0.8794 | 0.8948 | 0.9083 | 0.9245 | 0.9392 | 0.9536 | 0.9694 | 0.9840 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6075 | 0.7726 | 0.8509 | 0.8946 | 0.9229 | 0.9438 | 0.9590 | 0.9709 | 0.9802 | 0.9878 |
| QAT+Prune only | 0.0000 | 0.5665 | 0.7430 | 0.8300 | 0.8819 | 0.9147 | 0.9397 | 0.9577 | 0.9713 | 0.9830 | 0.9919 |
| QAT+PTQ | 0.0000 | 0.5671 | 0.7434 | 0.8304 | 0.8821 | 0.9147 | 0.9399 | 0.9577 | 0.9714 | 0.9830 | 0.9919 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5671 | 0.7434 | 0.8304 | 0.8821 | 0.9147 | 0.9399 | 0.9577 | 0.9714 | 0.9830 | 0.9919 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8641 | 0.8627 | 0.8623 | 0.8637 | 0.8627 | 0.8609 | 0.8616 | 0.8615 | 0.8624 | 0.8611 | 0.0000 |
| QAT+Prune only | 0.8345 | 0.8346 | 0.8338 | 0.8341 | 0.8350 | 0.8324 | 0.8347 | 0.8344 | 0.8315 | 0.8378 | 0.0000 |
| QAT+PTQ | 0.8351 | 0.8350 | 0.8342 | 0.8346 | 0.8354 | 0.8325 | 0.8354 | 0.8346 | 0.8321 | 0.8386 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8351 | 0.8350 | 0.8342 | 0.8346 | 0.8354 | 0.8325 | 0.8354 | 0.8346 | 0.8321 | 0.8386 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8641 | 0.0000 | 0.0000 | 0.0000 | 0.8641 | 1.0000 |
| 90 | 10 | 299,940 | 0.8739 | 0.4411 | 0.9755 | 0.6075 | 0.8627 | 0.9969 |
| 80 | 20 | 291,350 | 0.8851 | 0.6393 | 0.9760 | 0.7726 | 0.8623 | 0.9931 |
| 70 | 30 | 194,230 | 0.8974 | 0.7543 | 0.9760 | 0.8509 | 0.8637 | 0.9882 |
| 60 | 40 | 145,675 | 0.9080 | 0.8257 | 0.9760 | 0.8946 | 0.8627 | 0.9818 |
| 50 | 50 | 116,540 | 0.9184 | 0.8753 | 0.9760 | 0.9229 | 0.8609 | 0.9728 |
| 40 | 60 | 97,115 | 0.9302 | 0.9136 | 0.9760 | 0.9438 | 0.8616 | 0.9598 |
| 30 | 70 | 83,240 | 0.9416 | 0.9427 | 0.9760 | 0.9590 | 0.8615 | 0.9389 |
| 20 | 80 | 72,835 | 0.9533 | 0.9660 | 0.9760 | 0.9709 | 0.8624 | 0.8997 |
| 10 | 90 | 64,740 | 0.9645 | 0.9844 | 0.9760 | 0.9802 | 0.8611 | 0.7993 |
| 0 | 100 | 58,270 | 0.9760 | 1.0000 | 0.9760 | 0.9878 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8345 | 0.0000 | 0.0000 | 0.0000 | 0.8345 | 1.0000 |
| 90 | 10 | 299,940 | 0.8495 | 0.3978 | 0.9834 | 0.5665 | 0.8346 | 0.9978 |
| 80 | 20 | 291,350 | 0.8638 | 0.5968 | 0.9840 | 0.7430 | 0.8338 | 0.9952 |
| 70 | 30 | 194,230 | 0.8791 | 0.7177 | 0.9840 | 0.8300 | 0.8341 | 0.9918 |
| 60 | 40 | 145,675 | 0.8946 | 0.7990 | 0.9840 | 0.8819 | 0.8350 | 0.9874 |
| 50 | 50 | 116,540 | 0.9082 | 0.8545 | 0.9840 | 0.9147 | 0.8324 | 0.9811 |
| 40 | 60 | 97,115 | 0.9243 | 0.8993 | 0.9840 | 0.9397 | 0.8347 | 0.9720 |
| 30 | 70 | 83,240 | 0.9391 | 0.9327 | 0.9840 | 0.9577 | 0.8344 | 0.9571 |
| 20 | 80 | 72,835 | 0.9535 | 0.9589 | 0.9840 | 0.9713 | 0.8315 | 0.9285 |
| 10 | 90 | 64,740 | 0.9694 | 0.9820 | 0.9840 | 0.9830 | 0.8378 | 0.8532 |
| 0 | 100 | 58,270 | 0.9840 | 1.0000 | 0.9840 | 0.9919 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8351 | 0.0000 | 0.0000 | 0.0000 | 0.8351 | 1.0000 |
| 90 | 10 | 299,940 | 0.8499 | 0.3984 | 0.9834 | 0.5671 | 0.8350 | 0.9978 |
| 80 | 20 | 291,350 | 0.8642 | 0.5974 | 0.9840 | 0.7434 | 0.8342 | 0.9952 |
| 70 | 30 | 194,230 | 0.8794 | 0.7182 | 0.9840 | 0.8304 | 0.8346 | 0.9918 |
| 60 | 40 | 145,675 | 0.8948 | 0.7994 | 0.9840 | 0.8821 | 0.8354 | 0.9874 |
| 50 | 50 | 116,540 | 0.9083 | 0.8546 | 0.9840 | 0.9147 | 0.8325 | 0.9811 |
| 40 | 60 | 97,115 | 0.9245 | 0.8997 | 0.9840 | 0.9399 | 0.8354 | 0.9721 |
| 30 | 70 | 83,240 | 0.9392 | 0.9328 | 0.9840 | 0.9577 | 0.8346 | 0.9572 |
| 20 | 80 | 72,835 | 0.9536 | 0.9591 | 0.9840 | 0.9714 | 0.8321 | 0.9285 |
| 10 | 90 | 64,740 | 0.9694 | 0.9821 | 0.9840 | 0.9830 | 0.8386 | 0.8533 |
| 0 | 100 | 58,270 | 0.9840 | 1.0000 | 0.9840 | 0.9919 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8351 | 0.0000 | 0.0000 | 0.0000 | 0.8351 | 1.0000 |
| 90 | 10 | 299,940 | 0.8499 | 0.3984 | 0.9834 | 0.5671 | 0.8350 | 0.9978 |
| 80 | 20 | 291,350 | 0.8642 | 0.5974 | 0.9840 | 0.7434 | 0.8342 | 0.9952 |
| 70 | 30 | 194,230 | 0.8794 | 0.7182 | 0.9840 | 0.8304 | 0.8346 | 0.9918 |
| 60 | 40 | 145,675 | 0.8948 | 0.7994 | 0.9840 | 0.8821 | 0.8354 | 0.9874 |
| 50 | 50 | 116,540 | 0.9083 | 0.8546 | 0.9840 | 0.9147 | 0.8325 | 0.9811 |
| 40 | 60 | 97,115 | 0.9245 | 0.8997 | 0.9840 | 0.9399 | 0.8354 | 0.9721 |
| 30 | 70 | 83,240 | 0.9392 | 0.9328 | 0.9840 | 0.9577 | 0.8346 | 0.9572 |
| 20 | 80 | 72,835 | 0.9536 | 0.9591 | 0.9840 | 0.9714 | 0.8321 | 0.9285 |
| 10 | 90 | 64,740 | 0.9694 | 0.9821 | 0.9840 | 0.9830 | 0.8386 | 0.8533 |
| 0 | 100 | 58,270 | 0.9840 | 1.0000 | 0.9840 | 0.9919 | 0.0000 | 0.0000 |


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
0.15       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411   <--
0.20       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.25       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.30       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.35       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.40       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.45       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.50       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.55       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.60       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.65       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.70       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.75       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
0.80       0.8740   0.6076   0.8627   0.9969   0.9757   0.4411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8740, F1=0.6076, Normal Recall=0.8627, Normal Precision=0.9969, Attack Recall=0.9757, Attack Precision=0.4411

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
0.15       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407   <--
0.20       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.25       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.30       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.35       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.40       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.45       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.50       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.55       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.60       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.65       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.70       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.75       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
0.80       0.8857   0.7735   0.8631   0.9931   0.9760   0.6407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8857, F1=0.7735, Normal Recall=0.8631, Normal Precision=0.9931, Attack Recall=0.9760, Attack Precision=0.6407

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
0.15       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550   <--
0.20       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.25       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.30       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.35       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.40       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.45       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.50       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.55       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.60       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.65       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.70       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.75       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
0.80       0.8978   0.8514   0.8642   0.9882   0.9760   0.7550  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8978, F1=0.8514, Normal Recall=0.8642, Normal Precision=0.9882, Attack Recall=0.9760, Attack Precision=0.7550

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
0.15       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274   <--
0.20       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.25       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.30       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.35       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.40       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.45       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.50       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.55       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.60       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.65       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.70       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.75       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
0.80       0.9090   0.8956   0.8643   0.9818   0.9760   0.8274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9090, F1=0.8956, Normal Recall=0.8643, Normal Precision=0.9818, Attack Recall=0.9760, Attack Precision=0.8274

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
0.15       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773   <--
0.20       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.25       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.30       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.35       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.40       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.45       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.50       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.55       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.60       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.65       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.70       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.75       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
0.80       0.9198   0.9240   0.8635   0.9729   0.9760   0.8773  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9198, F1=0.9240, Normal Recall=0.8635, Normal Precision=0.9729, Attack Recall=0.9760, Attack Precision=0.8773

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
0.15       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982   <--
0.20       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.25       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.30       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.35       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.40       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.45       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.50       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.55       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.60       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.65       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.70       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.75       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
0.80       0.8496   0.5671   0.8346   0.9980   0.9847   0.3982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8496, F1=0.5671, Normal Recall=0.8346, Normal Precision=0.9980, Attack Recall=0.9847, Attack Precision=0.3982

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
0.15       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988   <--
0.20       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.25       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.30       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.35       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.40       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.45       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.50       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.55       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.60       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.65       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.70       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.75       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
0.80       0.8649   0.7445   0.8352   0.9952   0.9840   0.5988  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8649, F1=0.7445, Normal Recall=0.8352, Normal Precision=0.9952, Attack Recall=0.9840, Attack Precision=0.5988

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
0.15       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188   <--
0.20       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.25       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.30       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.35       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.40       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.45       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.50       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.55       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.60       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.65       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.70       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.75       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
0.80       0.8797   0.8308   0.8351   0.9918   0.9840   0.7188  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8797, F1=0.8308, Normal Recall=0.8351, Normal Precision=0.9918, Attack Recall=0.9840, Attack Precision=0.7188

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
0.15       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984   <--
0.20       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.25       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.30       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.35       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.40       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.45       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.50       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.55       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.60       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.65       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.70       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.75       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
0.80       0.8942   0.8816   0.8344   0.9874   0.9840   0.7984  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8942, F1=0.8816, Normal Recall=0.8344, Normal Precision=0.9874, Attack Recall=0.9840, Attack Precision=0.7984

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
0.15       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543   <--
0.20       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.25       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.30       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.35       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.40       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.45       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.50       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.55       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.60       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.65       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.70       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.75       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
0.80       0.9081   0.9146   0.8321   0.9811   0.9840   0.8543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.9146, Normal Recall=0.8321, Normal Precision=0.9811, Attack Recall=0.9840, Attack Precision=0.8543

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
0.15       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988   <--
0.20       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.25       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.30       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.35       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.40       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.45       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.50       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.55       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.60       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.65       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.70       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.75       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.80       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8500, F1=0.5676, Normal Recall=0.8350, Normal Precision=0.9980, Attack Recall=0.9847, Attack Precision=0.3988

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
0.15       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994   <--
0.20       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.25       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.30       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.35       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.40       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.45       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.50       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.55       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.60       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.65       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.70       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.75       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.80       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8653, F1=0.7450, Normal Recall=0.8356, Normal Precision=0.9952, Attack Recall=0.9840, Attack Precision=0.5994

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
0.15       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194   <--
0.20       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.25       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.30       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.35       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.40       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.45       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.50       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.55       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.60       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.65       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.70       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.75       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.80       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8800, F1=0.8311, Normal Recall=0.8355, Normal Precision=0.9919, Attack Recall=0.9840, Attack Precision=0.7194

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
0.15       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990   <--
0.20       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.25       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.30       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.35       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.40       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.45       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.50       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.55       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.60       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.65       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.70       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.75       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.80       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8946, F1=0.8819, Normal Recall=0.8350, Normal Precision=0.9874, Attack Recall=0.9840, Attack Precision=0.7990

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
0.15       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547   <--
0.20       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.25       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.30       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.35       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.40       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.45       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.50       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.55       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.60       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.65       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.70       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.75       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.80       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9084, F1=0.9148, Normal Recall=0.8327, Normal Precision=0.9811, Attack Recall=0.9840, Attack Precision=0.8547

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
0.15       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988   <--
0.20       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.25       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.30       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.35       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.40       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.45       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.50       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.55       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.60       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.65       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.70       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.75       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
0.80       0.8500   0.5676   0.8350   0.9980   0.9847   0.3988  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8500, F1=0.5676, Normal Recall=0.8350, Normal Precision=0.9980, Attack Recall=0.9847, Attack Precision=0.3988

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
0.15       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994   <--
0.20       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.25       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.30       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.35       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.40       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.45       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.50       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.55       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.60       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.65       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.70       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.75       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
0.80       0.8653   0.7450   0.8356   0.9952   0.9840   0.5994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8653, F1=0.7450, Normal Recall=0.8356, Normal Precision=0.9952, Attack Recall=0.9840, Attack Precision=0.5994

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
0.15       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194   <--
0.20       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.25       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.30       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.35       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.40       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.45       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.50       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.55       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.60       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.65       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.70       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.75       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
0.80       0.8800   0.8311   0.8355   0.9919   0.9840   0.7194  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8800, F1=0.8311, Normal Recall=0.8355, Normal Precision=0.9919, Attack Recall=0.9840, Attack Precision=0.7194

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
0.15       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990   <--
0.20       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.25       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.30       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.35       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.40       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.45       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.50       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.55       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.60       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.65       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.70       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.75       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
0.80       0.8946   0.8819   0.8350   0.9874   0.9840   0.7990  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8946, F1=0.8819, Normal Recall=0.8350, Normal Precision=0.9874, Attack Recall=0.9840, Attack Precision=0.7990

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
0.15       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547   <--
0.20       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.25       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.30       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.35       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.40       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.45       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.50       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.55       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.60       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.65       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.70       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.75       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
0.80       0.9084   0.9148   0.8327   0.9811   0.9840   0.8547  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9084, F1=0.9148, Normal Recall=0.8327, Normal Precision=0.9811, Attack Recall=0.9840, Attack Precision=0.8547

```

