# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-18 19:33:13 |

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
| Original (TFLite) | 0.8645 | 0.8761 | 0.8886 | 0.9028 | 0.9139 | 0.9264 | 0.9396 | 0.9528 | 0.9635 | 0.9774 | 0.9901 |
| QAT+Prune only | 0.5169 | 0.5590 | 0.6013 | 0.6445 | 0.6874 | 0.7283 | 0.7721 | 0.8141 | 0.8548 | 0.8994 | 0.9405 |
| QAT+PTQ | 0.5122 | 0.5548 | 0.5975 | 0.6414 | 0.6847 | 0.7261 | 0.7700 | 0.8127 | 0.8537 | 0.8988 | 0.9405 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5122 | 0.5548 | 0.5975 | 0.6414 | 0.6847 | 0.7261 | 0.7700 | 0.8127 | 0.8537 | 0.8988 | 0.9405 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6150 | 0.7805 | 0.8594 | 0.9019 | 0.9308 | 0.9516 | 0.9671 | 0.9775 | 0.9875 | 0.9950 |
| QAT+Prune only | 0.0000 | 0.2990 | 0.4855 | 0.6135 | 0.7065 | 0.7758 | 0.8320 | 0.8763 | 0.9120 | 0.9439 | 0.9693 |
| QAT+PTQ | 0.0000 | 0.2970 | 0.4831 | 0.6115 | 0.7047 | 0.7744 | 0.8307 | 0.8755 | 0.9114 | 0.9436 | 0.9693 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2970 | 0.4831 | 0.6115 | 0.7047 | 0.7744 | 0.8307 | 0.8755 | 0.9114 | 0.9436 | 0.9693 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8645 | 0.8634 | 0.8632 | 0.8654 | 0.8631 | 0.8627 | 0.8639 | 0.8657 | 0.8574 | 0.8638 | 0.0000 |
| QAT+Prune only | 0.5169 | 0.5166 | 0.5165 | 0.5177 | 0.5186 | 0.5160 | 0.5196 | 0.5193 | 0.5118 | 0.5298 | 0.0000 |
| QAT+PTQ | 0.5122 | 0.5119 | 0.5118 | 0.5133 | 0.5141 | 0.5117 | 0.5143 | 0.5145 | 0.5066 | 0.5239 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5122 | 0.5119 | 0.5118 | 0.5133 | 0.5141 | 0.5117 | 0.5143 | 0.5145 | 0.5066 | 0.5239 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8645 | 0.0000 | 0.0000 | 0.0000 | 0.8645 | 1.0000 |
| 90 | 10 | 299,940 | 0.8761 | 0.4461 | 0.9900 | 0.6150 | 0.8634 | 0.9987 |
| 80 | 20 | 291,350 | 0.8886 | 0.6441 | 0.9901 | 0.7805 | 0.8632 | 0.9971 |
| 70 | 30 | 194,230 | 0.9028 | 0.7591 | 0.9901 | 0.8594 | 0.8654 | 0.9951 |
| 60 | 40 | 145,675 | 0.9139 | 0.8282 | 0.9901 | 0.9019 | 0.8631 | 0.9924 |
| 50 | 50 | 116,540 | 0.9264 | 0.8782 | 0.9901 | 0.9308 | 0.8627 | 0.9886 |
| 40 | 60 | 97,115 | 0.9396 | 0.9161 | 0.9901 | 0.9516 | 0.8639 | 0.9831 |
| 30 | 70 | 83,240 | 0.9528 | 0.9451 | 0.9901 | 0.9671 | 0.8657 | 0.9740 |
| 20 | 80 | 72,835 | 0.9635 | 0.9652 | 0.9901 | 0.9775 | 0.8574 | 0.9558 |
| 10 | 90 | 64,740 | 0.9774 | 0.9849 | 0.9901 | 0.9875 | 0.8638 | 0.9063 |
| 0 | 100 | 58,270 | 0.9901 | 1.0000 | 0.9901 | 0.9950 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5169 | 0.0000 | 0.0000 | 0.0000 | 0.5169 | 1.0000 |
| 90 | 10 | 299,940 | 0.5590 | 0.1777 | 0.9404 | 0.2990 | 0.5166 | 0.9873 |
| 80 | 20 | 291,350 | 0.6013 | 0.3272 | 0.9405 | 0.4855 | 0.5165 | 0.9720 |
| 70 | 30 | 194,230 | 0.6445 | 0.4552 | 0.9405 | 0.6135 | 0.5177 | 0.9531 |
| 60 | 40 | 145,675 | 0.6874 | 0.5657 | 0.9405 | 0.7065 | 0.5186 | 0.9290 |
| 50 | 50 | 116,540 | 0.7283 | 0.6602 | 0.9405 | 0.7758 | 0.5160 | 0.8966 |
| 40 | 60 | 97,115 | 0.7721 | 0.7460 | 0.9405 | 0.8320 | 0.5196 | 0.8534 |
| 30 | 70 | 83,240 | 0.8141 | 0.8203 | 0.9405 | 0.8763 | 0.5193 | 0.7891 |
| 20 | 80 | 72,835 | 0.8548 | 0.8851 | 0.9405 | 0.9120 | 0.5118 | 0.6826 |
| 10 | 90 | 64,740 | 0.8994 | 0.9474 | 0.9405 | 0.9439 | 0.5298 | 0.4974 |
| 0 | 100 | 58,270 | 0.9405 | 1.0000 | 0.9405 | 0.9693 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5122 | 0.0000 | 0.0000 | 0.0000 | 0.5122 | 1.0000 |
| 90 | 10 | 299,940 | 0.5548 | 0.1763 | 0.9404 | 0.2970 | 0.5119 | 0.9872 |
| 80 | 20 | 291,350 | 0.5975 | 0.3251 | 0.9405 | 0.4831 | 0.5118 | 0.9718 |
| 70 | 30 | 194,230 | 0.6414 | 0.4530 | 0.9405 | 0.6115 | 0.5133 | 0.9527 |
| 60 | 40 | 145,675 | 0.6847 | 0.5634 | 0.9405 | 0.7047 | 0.5141 | 0.9284 |
| 50 | 50 | 116,540 | 0.7261 | 0.6582 | 0.9405 | 0.7744 | 0.5117 | 0.8958 |
| 40 | 60 | 97,115 | 0.7700 | 0.7439 | 0.9405 | 0.8307 | 0.5143 | 0.8521 |
| 30 | 70 | 83,240 | 0.8127 | 0.8189 | 0.9405 | 0.8755 | 0.5145 | 0.7875 |
| 20 | 80 | 72,835 | 0.8537 | 0.8841 | 0.9405 | 0.9114 | 0.5066 | 0.6803 |
| 10 | 90 | 64,740 | 0.8988 | 0.9468 | 0.9405 | 0.9436 | 0.5239 | 0.4945 |
| 0 | 100 | 58,270 | 0.9405 | 1.0000 | 0.9405 | 0.9693 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5122 | 0.0000 | 0.0000 | 0.0000 | 0.5122 | 1.0000 |
| 90 | 10 | 299,940 | 0.5548 | 0.1763 | 0.9404 | 0.2970 | 0.5119 | 0.9872 |
| 80 | 20 | 291,350 | 0.5975 | 0.3251 | 0.9405 | 0.4831 | 0.5118 | 0.9718 |
| 70 | 30 | 194,230 | 0.6414 | 0.4530 | 0.9405 | 0.6115 | 0.5133 | 0.9527 |
| 60 | 40 | 145,675 | 0.6847 | 0.5634 | 0.9405 | 0.7047 | 0.5141 | 0.9284 |
| 50 | 50 | 116,540 | 0.7261 | 0.6582 | 0.9405 | 0.7744 | 0.5117 | 0.8958 |
| 40 | 60 | 97,115 | 0.7700 | 0.7439 | 0.9405 | 0.8307 | 0.5143 | 0.8521 |
| 30 | 70 | 83,240 | 0.8127 | 0.8189 | 0.9405 | 0.8755 | 0.5145 | 0.7875 |
| 20 | 80 | 72,835 | 0.8537 | 0.8841 | 0.9405 | 0.9114 | 0.5066 | 0.6803 |
| 10 | 90 | 64,740 | 0.8988 | 0.9468 | 0.9405 | 0.9436 | 0.5239 | 0.4945 |
| 0 | 100 | 58,270 | 0.9405 | 1.0000 | 0.9405 | 0.9693 | 0.0000 | 0.0000 |


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
0.15       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462   <--
0.20       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.25       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.30       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.35       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.40       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.45       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.50       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.55       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.60       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.65       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.70       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.75       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
0.80       0.8761   0.6153   0.8634   0.9988   0.9906   0.4462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8761, F1=0.6153, Normal Recall=0.8634, Normal Precision=0.9988, Attack Recall=0.9906, Attack Precision=0.4462

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
0.15       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449   <--
0.20       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.25       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.30       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.35       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.40       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.45       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.50       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.55       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.60       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.65       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.70       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.75       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
0.80       0.8890   0.7811   0.8637   0.9971   0.9901   0.6449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8890, F1=0.7811, Normal Recall=0.8637, Normal Precision=0.9971, Attack Recall=0.9901, Attack Precision=0.6449

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
0.15       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584   <--
0.20       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.25       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.30       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.35       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.40       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.45       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.50       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.55       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.60       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.65       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.70       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.75       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
0.80       0.9024   0.8589   0.8648   0.9951   0.9901   0.7584  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9024, F1=0.8589, Normal Recall=0.8648, Normal Precision=0.9951, Attack Recall=0.9901, Attack Precision=0.7584

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
0.15       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297   <--
0.20       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.25       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.30       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.35       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.40       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.45       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.50       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.55       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.60       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.65       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.70       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.75       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
0.80       0.9147   0.9028   0.8645   0.9924   0.9901   0.8297  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9147, F1=0.9028, Normal Recall=0.8645, Normal Precision=0.9924, Attack Recall=0.9901, Attack Precision=0.8297

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
0.15       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797   <--
0.20       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.25       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.30       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.35       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.40       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.45       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.50       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.55       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.60       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.65       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.70       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.75       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
0.80       0.9274   0.9316   0.8646   0.9887   0.9901   0.8797  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9274, F1=0.9316, Normal Recall=0.8646, Normal Precision=0.9887, Attack Recall=0.9901, Attack Precision=0.8797

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
0.15       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779   <--
0.20       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.25       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.30       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.35       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.40       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.45       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.50       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.55       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.60       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.65       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.70       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.75       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
0.80       0.5591   0.2993   0.5166   0.9876   0.9414   0.1779  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5591, F1=0.2993, Normal Recall=0.5166, Normal Precision=0.9876, Attack Recall=0.9414, Attack Precision=0.1779

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
0.15       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271   <--
0.20       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.25       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.30       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.35       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.40       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.45       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.50       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.55       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.60       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.65       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.70       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.75       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
0.80       0.6012   0.4854   0.5164   0.9720   0.9405   0.3271  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6012, F1=0.4854, Normal Recall=0.5164, Normal Precision=0.9720, Attack Recall=0.9405, Attack Precision=0.3271

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
0.15       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550   <--
0.20       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.25       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.30       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.35       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.40       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.45       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.50       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.55       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.60       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.65       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.70       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.75       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
0.80       0.6442   0.6133   0.5172   0.9530   0.9405   0.4550  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6442, F1=0.6133, Normal Recall=0.5172, Normal Precision=0.9530, Attack Recall=0.9405, Attack Precision=0.4550

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
0.15       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643   <--
0.20       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.25       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.30       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.35       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.40       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.45       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.50       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.55       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.60       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.65       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.70       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.75       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
0.80       0.6858   0.7054   0.5160   0.9286   0.9405   0.5643  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6858, F1=0.7054, Normal Recall=0.5160, Normal Precision=0.9286, Attack Recall=0.9405, Attack Precision=0.5643

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
0.15       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592   <--
0.20       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.25       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.30       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.35       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.40       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.45       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.50       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.55       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.60       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.65       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.70       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.75       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
0.80       0.7272   0.7751   0.5138   0.8962   0.9405   0.6592  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7272, F1=0.7751, Normal Recall=0.5138, Normal Precision=0.8962, Attack Recall=0.9405, Attack Precision=0.6592

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
0.15       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765   <--
0.20       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.25       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.30       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.35       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.40       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.45       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.50       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.55       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.60       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.65       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.70       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.75       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.80       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5549, F1=0.2973, Normal Recall=0.5119, Normal Precision=0.9874, Attack Recall=0.9414, Attack Precision=0.1765

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
0.15       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250   <--
0.20       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.25       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.30       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.35       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.40       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.45       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.50       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.55       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.60       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.65       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.70       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.75       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.80       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5975, F1=0.4831, Normal Recall=0.5117, Normal Precision=0.9717, Attack Recall=0.9405, Attack Precision=0.3250

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
0.15       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526   <--
0.20       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.25       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.30       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.35       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.40       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.45       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.50       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.55       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.60       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.65       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.70       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.75       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.80       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6410, F1=0.6112, Normal Recall=0.5126, Normal Precision=0.9526, Attack Recall=0.9405, Attack Precision=0.4526

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
0.15       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619   <--
0.20       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.25       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.30       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.35       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.40       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.45       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.50       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.55       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.60       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.65       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.70       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.75       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.80       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6829, F1=0.7035, Normal Recall=0.5112, Normal Precision=0.9280, Attack Recall=0.9405, Attack Precision=0.5619

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
0.15       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570   <--
0.20       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.25       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.30       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.35       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.40       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.45       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.50       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.55       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.60       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.65       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.70       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.75       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.80       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7248, F1=0.7736, Normal Recall=0.5090, Normal Precision=0.8953, Attack Recall=0.9405, Attack Precision=0.6570

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
0.15       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765   <--
0.20       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.25       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.30       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.35       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.40       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.45       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.50       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.55       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.60       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.65       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.70       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.75       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
0.80       0.5549   0.2973   0.5119   0.9874   0.9414   0.1765  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5549, F1=0.2973, Normal Recall=0.5119, Normal Precision=0.9874, Attack Recall=0.9414, Attack Precision=0.1765

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
0.15       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250   <--
0.20       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.25       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.30       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.35       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.40       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.45       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.50       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.55       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.60       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.65       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.70       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.75       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
0.80       0.5975   0.4831   0.5117   0.9717   0.9405   0.3250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5975, F1=0.4831, Normal Recall=0.5117, Normal Precision=0.9717, Attack Recall=0.9405, Attack Precision=0.3250

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
0.15       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526   <--
0.20       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.25       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.30       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.35       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.40       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.45       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.50       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.55       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.60       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.65       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.70       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.75       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
0.80       0.6410   0.6112   0.5126   0.9526   0.9405   0.4526  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6410, F1=0.6112, Normal Recall=0.5126, Normal Precision=0.9526, Attack Recall=0.9405, Attack Precision=0.4526

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
0.15       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619   <--
0.20       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.25       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.30       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.35       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.40       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.45       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.50       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.55       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.60       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.65       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.70       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.75       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
0.80       0.6829   0.7035   0.5112   0.9280   0.9405   0.5619  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6829, F1=0.7035, Normal Recall=0.5112, Normal Precision=0.9280, Attack Recall=0.9405, Attack Precision=0.5619

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
0.15       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570   <--
0.20       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.25       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.30       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.35       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.40       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.45       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.50       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.55       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.60       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.65       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.70       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.75       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
0.80       0.7248   0.7736   0.5090   0.8953   0.9405   0.6570  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7248, F1=0.7736, Normal Recall=0.5090, Normal Precision=0.8953, Attack Recall=0.9405, Attack Precision=0.6570

```

