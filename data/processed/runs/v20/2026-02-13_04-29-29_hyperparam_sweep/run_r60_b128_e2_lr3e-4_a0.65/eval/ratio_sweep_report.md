# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-14 18:40:18 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2473 | 0.3112 | 0.3766 | 0.4425 | 0.5087 | 0.5734 | 0.6384 | 0.7046 | 0.7689 | 0.8363 | 0.9012 |
| QAT+Prune only | 0.6766 | 0.7081 | 0.7388 | 0.7706 | 0.8019 | 0.8330 | 0.8659 | 0.8975 | 0.9296 | 0.9591 | 0.9926 |
| QAT+PTQ | 0.6778 | 0.7089 | 0.7396 | 0.7713 | 0.8025 | 0.8334 | 0.8663 | 0.8979 | 0.9299 | 0.9593 | 0.9926 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6778 | 0.7089 | 0.7396 | 0.7713 | 0.8025 | 0.8334 | 0.8663 | 0.8979 | 0.9299 | 0.9593 | 0.9926 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2075 | 0.3664 | 0.4924 | 0.5947 | 0.6787 | 0.7494 | 0.8103 | 0.8619 | 0.9083 | 0.9480 |
| QAT+Prune only | 0.0000 | 0.4049 | 0.6031 | 0.7219 | 0.8003 | 0.8560 | 0.8988 | 0.9313 | 0.9575 | 0.9776 | 0.9963 |
| QAT+PTQ | 0.0000 | 0.4056 | 0.6039 | 0.7225 | 0.8008 | 0.8562 | 0.8991 | 0.9315 | 0.9577 | 0.9777 | 0.9963 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4056 | 0.6039 | 0.7225 | 0.8008 | 0.8562 | 0.8991 | 0.9315 | 0.9577 | 0.9777 | 0.9963 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2473 | 0.2457 | 0.2454 | 0.2460 | 0.2471 | 0.2456 | 0.2442 | 0.2460 | 0.2398 | 0.2527 | 0.0000 |
| QAT+Prune only | 0.6766 | 0.6764 | 0.6753 | 0.6755 | 0.6748 | 0.6735 | 0.6760 | 0.6756 | 0.6778 | 0.6580 | 0.0000 |
| QAT+PTQ | 0.6778 | 0.6774 | 0.6763 | 0.6764 | 0.6759 | 0.6742 | 0.6770 | 0.6769 | 0.6791 | 0.6603 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6778 | 0.6774 | 0.6763 | 0.6764 | 0.6759 | 0.6742 | 0.6770 | 0.6769 | 0.6791 | 0.6603 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2473 | 0.0000 | 0.0000 | 0.0000 | 0.2473 | 1.0000 |
| 90 | 10 | 299,940 | 0.3112 | 0.1172 | 0.9014 | 0.2075 | 0.2457 | 0.9573 |
| 80 | 20 | 291,350 | 0.3766 | 0.2299 | 0.9012 | 0.3664 | 0.2454 | 0.9085 |
| 70 | 30 | 194,230 | 0.4425 | 0.3387 | 0.9012 | 0.4924 | 0.2460 | 0.8531 |
| 60 | 40 | 145,675 | 0.5087 | 0.4438 | 0.9012 | 0.5947 | 0.2471 | 0.7895 |
| 50 | 50 | 116,540 | 0.5734 | 0.5443 | 0.9012 | 0.6787 | 0.2456 | 0.7131 |
| 40 | 60 | 97,115 | 0.6384 | 0.6414 | 0.9012 | 0.7494 | 0.2442 | 0.6223 |
| 30 | 70 | 83,240 | 0.7046 | 0.7360 | 0.9012 | 0.8103 | 0.2460 | 0.5161 |
| 20 | 80 | 72,835 | 0.7689 | 0.8258 | 0.9012 | 0.8619 | 0.2398 | 0.3775 |
| 10 | 90 | 64,740 | 0.8363 | 0.9156 | 0.9012 | 0.9083 | 0.2527 | 0.2212 |
| 0 | 100 | 58,270 | 0.9012 | 1.0000 | 0.9012 | 0.9480 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6766 | 0.0000 | 0.0000 | 0.0000 | 0.6766 | 1.0000 |
| 90 | 10 | 299,940 | 0.7081 | 0.2543 | 0.9931 | 0.4049 | 0.6764 | 0.9989 |
| 80 | 20 | 291,350 | 0.7388 | 0.4332 | 0.9926 | 0.6031 | 0.6753 | 0.9973 |
| 70 | 30 | 194,230 | 0.7706 | 0.5672 | 0.9926 | 0.7219 | 0.6755 | 0.9953 |
| 60 | 40 | 145,675 | 0.8019 | 0.6705 | 0.9926 | 0.8003 | 0.6748 | 0.9927 |
| 50 | 50 | 116,540 | 0.8330 | 0.7524 | 0.9926 | 0.8560 | 0.6735 | 0.9891 |
| 40 | 60 | 97,115 | 0.8659 | 0.8213 | 0.9926 | 0.8988 | 0.6760 | 0.9837 |
| 30 | 70 | 83,240 | 0.8975 | 0.8771 | 0.9926 | 0.9313 | 0.6756 | 0.9749 |
| 20 | 80 | 72,835 | 0.9296 | 0.9249 | 0.9926 | 0.9575 | 0.6778 | 0.9579 |
| 10 | 90 | 64,740 | 0.9591 | 0.9631 | 0.9926 | 0.9776 | 0.6580 | 0.9075 |
| 0 | 100 | 58,270 | 0.9926 | 1.0000 | 0.9926 | 0.9963 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6778 | 0.0000 | 0.0000 | 0.0000 | 0.6778 | 1.0000 |
| 90 | 10 | 299,940 | 0.7089 | 0.2548 | 0.9930 | 0.4056 | 0.6774 | 0.9989 |
| 80 | 20 | 291,350 | 0.7396 | 0.4339 | 0.9926 | 0.6039 | 0.6763 | 0.9973 |
| 70 | 30 | 194,230 | 0.7713 | 0.5680 | 0.9926 | 0.7225 | 0.6764 | 0.9953 |
| 60 | 40 | 145,675 | 0.8025 | 0.6712 | 0.9926 | 0.8008 | 0.6759 | 0.9927 |
| 50 | 50 | 116,540 | 0.8334 | 0.7529 | 0.9926 | 0.8562 | 0.6742 | 0.9891 |
| 40 | 60 | 97,115 | 0.8663 | 0.8217 | 0.9926 | 0.8991 | 0.6770 | 0.9838 |
| 30 | 70 | 83,240 | 0.8979 | 0.8776 | 0.9926 | 0.9315 | 0.6769 | 0.9750 |
| 20 | 80 | 72,835 | 0.9299 | 0.9252 | 0.9926 | 0.9577 | 0.6791 | 0.9580 |
| 10 | 90 | 64,740 | 0.9593 | 0.9634 | 0.9926 | 0.9777 | 0.6603 | 0.9078 |
| 0 | 100 | 58,270 | 0.9926 | 1.0000 | 0.9926 | 0.9963 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6778 | 0.0000 | 0.0000 | 0.0000 | 0.6778 | 1.0000 |
| 90 | 10 | 299,940 | 0.7089 | 0.2548 | 0.9930 | 0.4056 | 0.6774 | 0.9989 |
| 80 | 20 | 291,350 | 0.7396 | 0.4339 | 0.9926 | 0.6039 | 0.6763 | 0.9973 |
| 70 | 30 | 194,230 | 0.7713 | 0.5680 | 0.9926 | 0.7225 | 0.6764 | 0.9953 |
| 60 | 40 | 145,675 | 0.8025 | 0.6712 | 0.9926 | 0.8008 | 0.6759 | 0.9927 |
| 50 | 50 | 116,540 | 0.8334 | 0.7529 | 0.9926 | 0.8562 | 0.6742 | 0.9891 |
| 40 | 60 | 97,115 | 0.8663 | 0.8217 | 0.9926 | 0.8991 | 0.6770 | 0.9838 |
| 30 | 70 | 83,240 | 0.8979 | 0.8776 | 0.9926 | 0.9315 | 0.6769 | 0.9750 |
| 20 | 80 | 72,835 | 0.9299 | 0.9252 | 0.9926 | 0.9577 | 0.6791 | 0.9580 |
| 10 | 90 | 64,740 | 0.9593 | 0.9634 | 0.9926 | 0.9777 | 0.6603 | 0.9078 |
| 0 | 100 | 58,270 | 0.9926 | 1.0000 | 0.9926 | 0.9963 | 0.0000 | 0.0000 |


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
0.15       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172   <--
0.20       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.25       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.30       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.35       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.40       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.45       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.50       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.55       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.60       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.65       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.70       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.75       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
0.80       0.3112   0.2075   0.2457   0.9573   0.9014   0.1172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3112, F1=0.2075, Normal Recall=0.2457, Normal Precision=0.9573, Attack Recall=0.9014, Attack Precision=0.1172

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
0.15       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300   <--
0.20       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.25       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.30       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.35       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.40       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.45       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.50       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.55       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.60       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.65       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.70       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.75       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
0.80       0.3767   0.3664   0.2456   0.9086   0.9012   0.2300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3767, F1=0.3664, Normal Recall=0.2456, Normal Precision=0.9086, Attack Recall=0.9012, Attack Precision=0.2300

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
0.15       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391   <--
0.20       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.25       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.30       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.35       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.40       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.45       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.50       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.55       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.60       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.65       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.70       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.75       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
0.80       0.4433   0.4927   0.2471   0.8537   0.9012   0.3391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4433, F1=0.4927, Normal Recall=0.2471, Normal Precision=0.8537, Attack Recall=0.9012, Attack Precision=0.3391

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
0.15       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439   <--
0.20       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.25       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.30       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.35       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.40       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.45       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.50       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.55       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.60       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.65       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.70       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.75       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
0.80       0.5088   0.5948   0.2472   0.7896   0.9012   0.4439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5088, F1=0.5948, Normal Recall=0.2472, Normal Precision=0.7896, Attack Recall=0.9012, Attack Precision=0.4439

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
0.15       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448   <--
0.20       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.25       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.30       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.35       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.40       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.45       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.50       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.55       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.60       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.65       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.70       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.75       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
0.80       0.5741   0.6791   0.2471   0.7143   0.9012   0.5448  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5741, F1=0.6791, Normal Recall=0.2471, Normal Precision=0.7143, Attack Recall=0.9012, Attack Precision=0.5448

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
0.15       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543   <--
0.20       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.25       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.30       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.35       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.40       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.45       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.50       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.55       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.60       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.65       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.70       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.75       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
0.80       0.7081   0.4049   0.6765   0.9989   0.9930   0.2543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7081, F1=0.4049, Normal Recall=0.6765, Normal Precision=0.9989, Attack Recall=0.9930, Attack Precision=0.2543

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
0.15       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344   <--
0.20       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.25       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.30       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.35       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.40       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.45       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.50       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.55       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.60       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.65       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.70       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.75       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
0.80       0.7401   0.6043   0.6769   0.9973   0.9926   0.4344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7401, F1=0.6043, Normal Recall=0.6769, Normal Precision=0.9973, Attack Recall=0.9926, Attack Precision=0.4344

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
0.15       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682   <--
0.20       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.25       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.30       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.35       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.40       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.45       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.50       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.55       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.60       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.65       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.70       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.75       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
0.80       0.7715   0.7227   0.6768   0.9953   0.9926   0.5682  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7715, F1=0.7227, Normal Recall=0.6768, Normal Precision=0.9953, Attack Recall=0.9926, Attack Precision=0.5682

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
0.15       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714   <--
0.20       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.25       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.30       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.35       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.40       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.45       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.50       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.55       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.60       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.65       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.70       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.75       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
0.80       0.8027   0.8010   0.6761   0.9927   0.9926   0.6714  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8027, F1=0.8010, Normal Recall=0.6761, Normal Precision=0.9927, Attack Recall=0.9926, Attack Precision=0.6714

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
0.15       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531   <--
0.20       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.25       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.30       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.35       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.40       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.45       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.50       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.55       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.60       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.65       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.70       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.75       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
0.80       0.8336   0.8564   0.6746   0.9891   0.9926   0.7531  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8336, F1=0.8564, Normal Recall=0.6746, Normal Precision=0.9891, Attack Recall=0.9926, Attack Precision=0.7531

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
0.15       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548   <--
0.20       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.25       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.30       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.35       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.40       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.45       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.50       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.55       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.60       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.65       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.70       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.75       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.80       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7089, F1=0.4056, Normal Recall=0.6774, Normal Precision=0.9988, Attack Recall=0.9929, Attack Precision=0.2548

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
0.15       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352   <--
0.20       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.25       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.30       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.35       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.40       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.45       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.50       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.55       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.60       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.65       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.70       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.75       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.80       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7408, F1=0.6051, Normal Recall=0.6779, Normal Precision=0.9973, Attack Recall=0.9926, Attack Precision=0.4352

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
0.15       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691   <--
0.20       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.25       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.30       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.35       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.40       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.45       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.50       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.55       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.60       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.65       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.70       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.75       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.80       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7723, F1=0.7234, Normal Recall=0.6779, Normal Precision=0.9953, Attack Recall=0.9926, Attack Precision=0.5691

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
0.15       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723   <--
0.20       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.25       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.30       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.35       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.40       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.45       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.50       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.55       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.60       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.65       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.70       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.75       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.80       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8035, F1=0.8016, Normal Recall=0.6774, Normal Precision=0.9927, Attack Recall=0.9926, Attack Precision=0.6723

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
0.15       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538   <--
0.20       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.25       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.30       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.35       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.40       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.45       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.50       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.55       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.60       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.65       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.70       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.75       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.80       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.8568, Normal Recall=0.6758, Normal Precision=0.9891, Attack Recall=0.9926, Attack Precision=0.7538

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
0.15       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548   <--
0.20       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.25       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.30       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.35       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.40       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.45       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.50       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.55       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.60       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.65       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.70       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.75       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
0.80       0.7089   0.4056   0.6774   0.9988   0.9929   0.2548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7089, F1=0.4056, Normal Recall=0.6774, Normal Precision=0.9988, Attack Recall=0.9929, Attack Precision=0.2548

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
0.15       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352   <--
0.20       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.25       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.30       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.35       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.40       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.45       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.50       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.55       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.60       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.65       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.70       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.75       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
0.80       0.7408   0.6051   0.6779   0.9973   0.9926   0.4352  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7408, F1=0.6051, Normal Recall=0.6779, Normal Precision=0.9973, Attack Recall=0.9926, Attack Precision=0.4352

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
0.15       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691   <--
0.20       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.25       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.30       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.35       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.40       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.45       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.50       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.55       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.60       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.65       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.70       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.75       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
0.80       0.7723   0.7234   0.6779   0.9953   0.9926   0.5691  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7723, F1=0.7234, Normal Recall=0.6779, Normal Precision=0.9953, Attack Recall=0.9926, Attack Precision=0.5691

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
0.15       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723   <--
0.20       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.25       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.30       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.35       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.40       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.45       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.50       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.55       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.60       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.65       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.70       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.75       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
0.80       0.8035   0.8016   0.6774   0.9927   0.9926   0.6723  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8035, F1=0.8016, Normal Recall=0.6774, Normal Precision=0.9927, Attack Recall=0.9926, Attack Precision=0.6723

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
0.15       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538   <--
0.20       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.25       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.30       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.35       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.40       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.45       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.50       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.55       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.60       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.65       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.70       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.75       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
0.80       0.8342   0.8568   0.6758   0.9891   0.9926   0.7538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8342, F1=0.8568, Normal Recall=0.6758, Normal Precision=0.9891, Attack Recall=0.9926, Attack Precision=0.7538

```

