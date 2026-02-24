# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-15 19:49:06 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3722 | 0.4310 | 0.4903 | 0.5500 | 0.6085 | 0.6687 | 0.7259 | 0.7870 | 0.8457 | 0.9058 | 0.9645 |
| QAT+Prune only | 0.5771 | 0.6183 | 0.6596 | 0.7020 | 0.7433 | 0.7833 | 0.8264 | 0.8678 | 0.9111 | 0.9505 | 0.9937 |
| QAT+PTQ | 0.5767 | 0.6180 | 0.6594 | 0.7018 | 0.7430 | 0.7831 | 0.8264 | 0.8678 | 0.9110 | 0.9506 | 0.9938 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5767 | 0.6180 | 0.6594 | 0.7018 | 0.7430 | 0.7831 | 0.8264 | 0.8678 | 0.9110 | 0.9506 | 0.9938 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2535 | 0.4308 | 0.5626 | 0.6634 | 0.7443 | 0.8085 | 0.8638 | 0.9091 | 0.9485 | 0.9819 |
| QAT+Prune only | 0.0000 | 0.3423 | 0.5387 | 0.6667 | 0.7559 | 0.8210 | 0.8729 | 0.9132 | 0.9471 | 0.9731 | 0.9968 |
| QAT+PTQ | 0.0000 | 0.3422 | 0.5385 | 0.6666 | 0.7557 | 0.8208 | 0.8729 | 0.9132 | 0.9470 | 0.9731 | 0.9969 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3422 | 0.5385 | 0.6666 | 0.7557 | 0.8208 | 0.8729 | 0.9132 | 0.9470 | 0.9731 | 0.9969 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3722 | 0.3715 | 0.3718 | 0.3723 | 0.3712 | 0.3729 | 0.3680 | 0.3728 | 0.3704 | 0.3769 | 0.0000 |
| QAT+Prune only | 0.5771 | 0.5766 | 0.5761 | 0.5770 | 0.5763 | 0.5729 | 0.5756 | 0.5740 | 0.5808 | 0.5615 | 0.0000 |
| QAT+PTQ | 0.5767 | 0.5763 | 0.5758 | 0.5767 | 0.5759 | 0.5724 | 0.5754 | 0.5739 | 0.5801 | 0.5615 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5767 | 0.5763 | 0.5758 | 0.5767 | 0.5759 | 0.5724 | 0.5754 | 0.5739 | 0.5801 | 0.5615 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3722 | 0.0000 | 0.0000 | 0.0000 | 0.3722 | 1.0000 |
| 90 | 10 | 299,940 | 0.4310 | 0.1459 | 0.9664 | 0.2535 | 0.3715 | 0.9900 |
| 80 | 20 | 291,350 | 0.4903 | 0.2774 | 0.9645 | 0.4308 | 0.3718 | 0.9767 |
| 70 | 30 | 194,230 | 0.5500 | 0.3971 | 0.9645 | 0.5626 | 0.3723 | 0.9608 |
| 60 | 40 | 145,675 | 0.6085 | 0.5056 | 0.9645 | 0.6634 | 0.3712 | 0.9401 |
| 50 | 50 | 116,540 | 0.6687 | 0.6060 | 0.9645 | 0.7443 | 0.3729 | 0.9131 |
| 40 | 60 | 97,115 | 0.7259 | 0.6960 | 0.9645 | 0.8085 | 0.3680 | 0.8737 |
| 30 | 70 | 83,240 | 0.7870 | 0.7821 | 0.9645 | 0.8638 | 0.3728 | 0.8183 |
| 20 | 80 | 72,835 | 0.8457 | 0.8597 | 0.9645 | 0.9091 | 0.3704 | 0.7230 |
| 10 | 90 | 64,740 | 0.9058 | 0.9330 | 0.9645 | 0.9485 | 0.3769 | 0.5414 |
| 0 | 100 | 58,270 | 0.9645 | 1.0000 | 0.9645 | 0.9819 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5771 | 0.0000 | 0.0000 | 0.0000 | 0.5771 | 1.0000 |
| 90 | 10 | 299,940 | 0.6183 | 0.2068 | 0.9934 | 0.3423 | 0.5766 | 0.9987 |
| 80 | 20 | 291,350 | 0.6596 | 0.3695 | 0.9937 | 0.5387 | 0.5761 | 0.9973 |
| 70 | 30 | 194,230 | 0.7020 | 0.5017 | 0.9937 | 0.6667 | 0.5770 | 0.9953 |
| 60 | 40 | 145,675 | 0.7433 | 0.6099 | 0.9937 | 0.7559 | 0.5763 | 0.9928 |
| 50 | 50 | 116,540 | 0.7833 | 0.6994 | 0.9937 | 0.8210 | 0.5729 | 0.9891 |
| 40 | 60 | 97,115 | 0.8264 | 0.7784 | 0.9937 | 0.8729 | 0.5756 | 0.9839 |
| 30 | 70 | 83,240 | 0.8678 | 0.8448 | 0.9937 | 0.9132 | 0.5740 | 0.9750 |
| 20 | 80 | 72,835 | 0.9111 | 0.9046 | 0.9937 | 0.9471 | 0.5808 | 0.9584 |
| 10 | 90 | 64,740 | 0.9505 | 0.9533 | 0.9937 | 0.9731 | 0.5615 | 0.9085 |
| 0 | 100 | 58,270 | 0.9937 | 1.0000 | 0.9937 | 0.9968 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5767 | 0.0000 | 0.0000 | 0.0000 | 0.5767 | 1.0000 |
| 90 | 10 | 299,940 | 0.6180 | 0.2067 | 0.9935 | 0.3422 | 0.5763 | 0.9987 |
| 80 | 20 | 291,350 | 0.6594 | 0.3694 | 0.9938 | 0.5385 | 0.5758 | 0.9973 |
| 70 | 30 | 194,230 | 0.7018 | 0.5015 | 0.9938 | 0.6666 | 0.5767 | 0.9954 |
| 60 | 40 | 145,675 | 0.7430 | 0.6097 | 0.9938 | 0.7557 | 0.5759 | 0.9928 |
| 50 | 50 | 116,540 | 0.7831 | 0.6992 | 0.9938 | 0.8208 | 0.5724 | 0.9892 |
| 40 | 60 | 97,115 | 0.8264 | 0.7783 | 0.9938 | 0.8729 | 0.5754 | 0.9840 |
| 30 | 70 | 83,240 | 0.8678 | 0.8448 | 0.9938 | 0.9132 | 0.5739 | 0.9753 |
| 20 | 80 | 72,835 | 0.9110 | 0.9045 | 0.9938 | 0.9470 | 0.5801 | 0.9588 |
| 10 | 90 | 64,740 | 0.9506 | 0.9533 | 0.9938 | 0.9731 | 0.5615 | 0.9094 |
| 0 | 100 | 58,270 | 0.9938 | 1.0000 | 0.9938 | 0.9969 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5767 | 0.0000 | 0.0000 | 0.0000 | 0.5767 | 1.0000 |
| 90 | 10 | 299,940 | 0.6180 | 0.2067 | 0.9935 | 0.3422 | 0.5763 | 0.9987 |
| 80 | 20 | 291,350 | 0.6594 | 0.3694 | 0.9938 | 0.5385 | 0.5758 | 0.9973 |
| 70 | 30 | 194,230 | 0.7018 | 0.5015 | 0.9938 | 0.6666 | 0.5767 | 0.9954 |
| 60 | 40 | 145,675 | 0.7430 | 0.6097 | 0.9938 | 0.7557 | 0.5759 | 0.9928 |
| 50 | 50 | 116,540 | 0.7831 | 0.6992 | 0.9938 | 0.8208 | 0.5724 | 0.9892 |
| 40 | 60 | 97,115 | 0.8264 | 0.7783 | 0.9938 | 0.8729 | 0.5754 | 0.9840 |
| 30 | 70 | 83,240 | 0.8678 | 0.8448 | 0.9938 | 0.9132 | 0.5739 | 0.9753 |
| 20 | 80 | 72,835 | 0.9110 | 0.9045 | 0.9938 | 0.9470 | 0.5801 | 0.9588 |
| 10 | 90 | 64,740 | 0.9506 | 0.9533 | 0.9938 | 0.9731 | 0.5615 | 0.9094 |
| 0 | 100 | 58,270 | 0.9938 | 1.0000 | 0.9938 | 0.9969 | 0.0000 | 0.0000 |


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
0.15       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458   <--
0.20       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.25       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.30       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.35       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.40       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.45       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.50       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.55       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.60       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.65       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.70       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.75       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
0.80       0.4309   0.2533   0.3715   0.9897   0.9652   0.1458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4309, F1=0.2533, Normal Recall=0.3715, Normal Precision=0.9897, Attack Recall=0.9652, Attack Precision=0.1458

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
0.15       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774   <--
0.20       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.25       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.30       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.35       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.40       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.45       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.50       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.55       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.60       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.65       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.70       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.75       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
0.80       0.4905   0.4309   0.3720   0.9767   0.9645   0.2774  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4905, F1=0.4309, Normal Recall=0.3720, Normal Precision=0.9767, Attack Recall=0.9645, Attack Precision=0.2774

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
0.15       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970   <--
0.20       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.25       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.30       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.35       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.40       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.45       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.50       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.55       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.60       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.65       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.70       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.75       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
0.80       0.5498   0.5625   0.3721   0.9607   0.9645   0.3970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5498, F1=0.5625, Normal Recall=0.3721, Normal Precision=0.9607, Attack Recall=0.9645, Attack Precision=0.3970

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
0.15       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059   <--
0.20       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.25       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.30       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.35       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.40       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.45       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.50       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.55       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.60       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.65       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.70       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.75       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
0.80       0.6090   0.6637   0.3720   0.9402   0.9645   0.5059  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6090, F1=0.6637, Normal Recall=0.3720, Normal Precision=0.9402, Attack Recall=0.9645, Attack Precision=0.5059

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
0.15       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055   <--
0.20       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.25       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.30       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.35       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.40       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.45       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.50       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.55       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.60       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.65       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.70       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.75       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
0.80       0.6680   0.7439   0.3715   0.9128   0.9645   0.6055  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6680, F1=0.7439, Normal Recall=0.3715, Normal Precision=0.9128, Attack Recall=0.9645, Attack Precision=0.6055

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
0.15       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068   <--
0.20       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.25       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.30       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.35       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.40       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.45       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.50       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.55       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.60       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.65       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.70       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.75       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
0.80       0.6183   0.3423   0.5766   0.9987   0.9932   0.2068  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6183, F1=0.3423, Normal Recall=0.5766, Normal Precision=0.9987, Attack Recall=0.9932, Attack Precision=0.2068

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
0.15       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703   <--
0.20       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.25       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.30       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.35       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.40       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.45       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.50       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.55       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.60       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.65       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.70       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.75       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
0.80       0.6607   0.5395   0.5775   0.9973   0.9937   0.3703  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6607, F1=0.5395, Normal Recall=0.5775, Normal Precision=0.9973, Attack Recall=0.9937, Attack Precision=0.3703

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
0.15       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018   <--
0.20       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.25       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.30       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.35       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.40       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.45       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.50       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.55       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.60       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.65       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.70       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.75       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
0.80       0.7022   0.6669   0.5773   0.9953   0.9937   0.5018  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7022, F1=0.6669, Normal Recall=0.5773, Normal Precision=0.9953, Attack Recall=0.9937, Attack Precision=0.5018

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
0.15       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106   <--
0.20       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.25       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.30       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.35       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.40       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.45       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.50       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.55       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.60       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.65       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.70       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.75       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
0.80       0.7440   0.7564   0.5775   0.9928   0.9937   0.6106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7440, F1=0.7564, Normal Recall=0.5775, Normal Precision=0.9928, Attack Recall=0.9937, Attack Precision=0.6106

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
0.15       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011   <--
0.20       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.25       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.30       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.35       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.40       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.45       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.50       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.55       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.60       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.65       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.70       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.75       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
0.80       0.7850   0.8221   0.5764   0.9892   0.9937   0.7011  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7850, F1=0.8221, Normal Recall=0.5764, Normal Precision=0.9892, Attack Recall=0.9937, Attack Precision=0.7011

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
0.15       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067   <--
0.20       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.25       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.30       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.35       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.40       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.45       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.50       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.55       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.60       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.65       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.70       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.75       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.80       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6180, F1=0.3421, Normal Recall=0.5763, Normal Precision=0.9987, Attack Recall=0.9933, Attack Precision=0.2067

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
0.15       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701   <--
0.20       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.25       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.30       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.35       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.40       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.45       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.50       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.55       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.60       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.65       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.70       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.75       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.80       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6605, F1=0.5394, Normal Recall=0.5772, Normal Precision=0.9973, Attack Recall=0.9938, Attack Precision=0.3701

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
0.15       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017   <--
0.20       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.25       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.30       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.35       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.40       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.45       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.50       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.55       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.60       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.65       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.70       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.75       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.80       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7020, F1=0.6668, Normal Recall=0.5770, Normal Precision=0.9954, Attack Recall=0.9938, Attack Precision=0.5017

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
0.15       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104   <--
0.20       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.25       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.30       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.35       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.40       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.45       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.50       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.55       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.60       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.65       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.70       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.75       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.80       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7438, F1=0.7563, Normal Recall=0.5772, Normal Precision=0.9929, Attack Recall=0.9938, Attack Precision=0.6104

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
0.15       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010   <--
0.20       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.25       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.30       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.35       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.40       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.45       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.50       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.55       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.60       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.65       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.70       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.75       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.80       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7849, F1=0.8221, Normal Recall=0.5761, Normal Precision=0.9893, Attack Recall=0.9938, Attack Precision=0.7010

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
0.15       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067   <--
0.20       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.25       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.30       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.35       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.40       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.45       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.50       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.55       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.60       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.65       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.70       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.75       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
0.80       0.6180   0.3421   0.5763   0.9987   0.9933   0.2067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6180, F1=0.3421, Normal Recall=0.5763, Normal Precision=0.9987, Attack Recall=0.9933, Attack Precision=0.2067

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
0.15       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701   <--
0.20       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.25       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.30       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.35       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.40       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.45       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.50       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.55       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.60       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.65       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.70       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.75       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
0.80       0.6605   0.5394   0.5772   0.9973   0.9938   0.3701  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6605, F1=0.5394, Normal Recall=0.5772, Normal Precision=0.9973, Attack Recall=0.9938, Attack Precision=0.3701

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
0.15       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017   <--
0.20       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.25       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.30       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.35       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.40       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.45       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.50       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.55       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.60       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.65       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.70       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.75       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
0.80       0.7020   0.6668   0.5770   0.9954   0.9938   0.5017  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7020, F1=0.6668, Normal Recall=0.5770, Normal Precision=0.9954, Attack Recall=0.9938, Attack Precision=0.5017

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
0.15       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104   <--
0.20       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.25       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.30       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.35       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.40       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.45       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.50       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.55       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.60       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.65       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.70       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.75       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
0.80       0.7438   0.7563   0.5772   0.9929   0.9938   0.6104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7438, F1=0.7563, Normal Recall=0.5772, Normal Precision=0.9929, Attack Recall=0.9938, Attack Precision=0.6104

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
0.15       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010   <--
0.20       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.25       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.30       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.35       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.40       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.45       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.50       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.55       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.60       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.65       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.70       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.75       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
0.80       0.7849   0.8221   0.5761   0.9893   0.9938   0.7010  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7849, F1=0.8221, Normal Recall=0.5761, Normal Precision=0.9893, Attack Recall=0.9938, Attack Precision=0.7010

```

