# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-15 03:10:11 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3925 | 0.4267 | 0.4621 | 0.4971 | 0.5328 | 0.5681 | 0.6041 | 0.6396 | 0.6736 | 0.7088 | 0.7446 |
| QAT+Prune only | 0.8547 | 0.8684 | 0.8807 | 0.8939 | 0.9065 | 0.9173 | 0.9313 | 0.9439 | 0.9563 | 0.9692 | 0.9822 |
| QAT+PTQ | 0.8516 | 0.8650 | 0.8775 | 0.8908 | 0.9036 | 0.9145 | 0.9284 | 0.9412 | 0.9536 | 0.9668 | 0.9799 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8516 | 0.8650 | 0.8775 | 0.8908 | 0.9036 | 0.9145 | 0.9284 | 0.9412 | 0.9536 | 0.9668 | 0.9799 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2056 | 0.3564 | 0.4704 | 0.5604 | 0.6329 | 0.6930 | 0.7431 | 0.7850 | 0.8215 | 0.8536 |
| QAT+Prune only | 0.0000 | 0.5989 | 0.7670 | 0.8475 | 0.8936 | 0.9223 | 0.9449 | 0.9608 | 0.9729 | 0.9829 | 0.9910 |
| QAT+PTQ | 0.0000 | 0.5923 | 0.7619 | 0.8433 | 0.8905 | 0.9197 | 0.9426 | 0.9589 | 0.9712 | 0.9815 | 0.9898 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5923 | 0.7619 | 0.8433 | 0.8905 | 0.9197 | 0.9426 | 0.9589 | 0.9712 | 0.9815 | 0.9898 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3925 | 0.3917 | 0.3915 | 0.3911 | 0.3915 | 0.3916 | 0.3934 | 0.3947 | 0.3899 | 0.3871 | 0.0000 |
| QAT+Prune only | 0.8547 | 0.8556 | 0.8553 | 0.8561 | 0.8560 | 0.8524 | 0.8550 | 0.8546 | 0.8526 | 0.8525 | 0.0000 |
| QAT+PTQ | 0.8516 | 0.8522 | 0.8519 | 0.8526 | 0.8527 | 0.8491 | 0.8511 | 0.8509 | 0.8481 | 0.8491 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8516 | 0.8522 | 0.8519 | 0.8526 | 0.8527 | 0.8491 | 0.8511 | 0.8509 | 0.8481 | 0.8491 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3925 | 0.0000 | 0.0000 | 0.0000 | 0.3925 | 1.0000 |
| 90 | 10 | 299,940 | 0.4267 | 0.1193 | 0.7418 | 0.2056 | 0.3917 | 0.9318 |
| 80 | 20 | 291,350 | 0.4621 | 0.2342 | 0.7446 | 0.3564 | 0.3915 | 0.8598 |
| 70 | 30 | 194,230 | 0.4971 | 0.3438 | 0.7446 | 0.4704 | 0.3911 | 0.7813 |
| 60 | 40 | 145,675 | 0.5328 | 0.4493 | 0.7446 | 0.5604 | 0.3915 | 0.6969 |
| 50 | 50 | 116,540 | 0.5681 | 0.5503 | 0.7446 | 0.6329 | 0.3916 | 0.6052 |
| 40 | 60 | 97,115 | 0.6041 | 0.6480 | 0.7446 | 0.6930 | 0.3934 | 0.5066 |
| 30 | 70 | 83,240 | 0.6396 | 0.7416 | 0.7446 | 0.7431 | 0.3947 | 0.3984 |
| 20 | 80 | 72,835 | 0.6736 | 0.8300 | 0.7446 | 0.7850 | 0.3899 | 0.2762 |
| 10 | 90 | 64,740 | 0.7088 | 0.9162 | 0.7446 | 0.8215 | 0.3871 | 0.1441 |
| 0 | 100 | 58,270 | 0.7446 | 1.0000 | 0.7446 | 0.8536 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8547 | 0.0000 | 0.0000 | 0.0000 | 0.8547 | 1.0000 |
| 90 | 10 | 299,940 | 0.8684 | 0.4307 | 0.9829 | 0.5989 | 0.8556 | 0.9978 |
| 80 | 20 | 291,350 | 0.8807 | 0.6292 | 0.9822 | 0.7670 | 0.8553 | 0.9948 |
| 70 | 30 | 194,230 | 0.8939 | 0.7453 | 0.9822 | 0.8475 | 0.8561 | 0.9912 |
| 60 | 40 | 145,675 | 0.9065 | 0.8197 | 0.9822 | 0.8936 | 0.8560 | 0.9863 |
| 50 | 50 | 116,540 | 0.9173 | 0.8693 | 0.9822 | 0.9223 | 0.8524 | 0.9795 |
| 40 | 60 | 97,115 | 0.9313 | 0.9104 | 0.9822 | 0.9449 | 0.8550 | 0.9697 |
| 30 | 70 | 83,240 | 0.9439 | 0.9403 | 0.9822 | 0.9608 | 0.8546 | 0.9536 |
| 20 | 80 | 72,835 | 0.9563 | 0.9638 | 0.9822 | 0.9729 | 0.8526 | 0.9229 |
| 10 | 90 | 64,740 | 0.9692 | 0.9836 | 0.9822 | 0.9829 | 0.8525 | 0.8416 |
| 0 | 100 | 58,270 | 0.9822 | 1.0000 | 0.9822 | 0.9910 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8516 | 0.0000 | 0.0000 | 0.0000 | 0.8516 | 1.0000 |
| 90 | 10 | 299,940 | 0.8650 | 0.4243 | 0.9804 | 0.5923 | 0.8522 | 0.9975 |
| 80 | 20 | 291,350 | 0.8775 | 0.6232 | 0.9799 | 0.7619 | 0.8519 | 0.9941 |
| 70 | 30 | 194,230 | 0.8908 | 0.7402 | 0.9799 | 0.8433 | 0.8526 | 0.9900 |
| 60 | 40 | 145,675 | 0.9036 | 0.8160 | 0.9799 | 0.8905 | 0.8527 | 0.9845 |
| 50 | 50 | 116,540 | 0.9145 | 0.8666 | 0.9799 | 0.9197 | 0.8491 | 0.9769 |
| 40 | 60 | 97,115 | 0.9284 | 0.9080 | 0.9799 | 0.9426 | 0.8511 | 0.9658 |
| 30 | 70 | 83,240 | 0.9412 | 0.9388 | 0.9799 | 0.9589 | 0.8509 | 0.9477 |
| 20 | 80 | 72,835 | 0.9536 | 0.9627 | 0.9799 | 0.9712 | 0.8481 | 0.9134 |
| 10 | 90 | 64,740 | 0.9668 | 0.9832 | 0.9799 | 0.9815 | 0.8491 | 0.8243 |
| 0 | 100 | 58,270 | 0.9799 | 1.0000 | 0.9799 | 0.9898 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8516 | 0.0000 | 0.0000 | 0.0000 | 0.8516 | 1.0000 |
| 90 | 10 | 299,940 | 0.8650 | 0.4243 | 0.9804 | 0.5923 | 0.8522 | 0.9975 |
| 80 | 20 | 291,350 | 0.8775 | 0.6232 | 0.9799 | 0.7619 | 0.8519 | 0.9941 |
| 70 | 30 | 194,230 | 0.8908 | 0.7402 | 0.9799 | 0.8433 | 0.8526 | 0.9900 |
| 60 | 40 | 145,675 | 0.9036 | 0.8160 | 0.9799 | 0.8905 | 0.8527 | 0.9845 |
| 50 | 50 | 116,540 | 0.9145 | 0.8666 | 0.9799 | 0.9197 | 0.8491 | 0.9769 |
| 40 | 60 | 97,115 | 0.9284 | 0.9080 | 0.9799 | 0.9426 | 0.8511 | 0.9658 |
| 30 | 70 | 83,240 | 0.9412 | 0.9388 | 0.9799 | 0.9589 | 0.8509 | 0.9477 |
| 20 | 80 | 72,835 | 0.9536 | 0.9627 | 0.9799 | 0.9712 | 0.8481 | 0.9134 |
| 10 | 90 | 64,740 | 0.9668 | 0.9832 | 0.9799 | 0.9815 | 0.8491 | 0.8243 |
| 0 | 100 | 58,270 | 0.9799 | 1.0000 | 0.9799 | 0.9898 | 0.0000 | 0.0000 |


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
0.15       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199   <--
0.20       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.25       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.30       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.35       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.40       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.45       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.50       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.55       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.60       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.65       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.70       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.75       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
0.80       0.4271   0.2065   0.3917   0.9327   0.7456   0.1199  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4271, F1=0.2065, Normal Recall=0.3917, Normal Precision=0.9327, Attack Recall=0.7456, Attack Precision=0.1199

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
0.15       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341   <--
0.20       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.25       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.30       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.35       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.40       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.45       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.50       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.55       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.60       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.65       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.70       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.75       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
0.80       0.4617   0.3562   0.3910   0.8596   0.7446   0.2341  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4617, F1=0.3562, Normal Recall=0.3910, Normal Precision=0.8596, Attack Recall=0.7446, Attack Precision=0.2341

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
0.15       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446   <--
0.20       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.25       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.30       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.35       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.40       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.45       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.50       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.55       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.60       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.65       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.70       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.75       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
0.80       0.4986   0.4712   0.3932   0.7822   0.7446   0.3446  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4986, F1=0.4712, Normal Recall=0.3932, Normal Precision=0.7822, Attack Recall=0.7446, Attack Precision=0.3446

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
0.15       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493   <--
0.20       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.25       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.30       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.35       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.40       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.45       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.50       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.55       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.60       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.65       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.70       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.75       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
0.80       0.5328   0.5604   0.3917   0.6970   0.7446   0.4493  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5328, F1=0.5604, Normal Recall=0.3917, Normal Precision=0.6970, Attack Recall=0.7446, Attack Precision=0.4493

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
0.15       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506   <--
0.20       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.25       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.30       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.35       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.40       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.45       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.50       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.55       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.60       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.65       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.70       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.75       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
0.80       0.5685   0.6331   0.3924   0.6057   0.7446   0.5506  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5685, F1=0.6331, Normal Recall=0.3924, Normal Precision=0.6057, Attack Recall=0.7446, Attack Precision=0.5506

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
0.15       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305   <--
0.20       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.25       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.30       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.35       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.40       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.45       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.50       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.55       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.60       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.65       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.70       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.75       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
0.80       0.8683   0.5986   0.8556   0.9977   0.9821   0.4305  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8683, F1=0.5986, Normal Recall=0.8556, Normal Precision=0.9977, Attack Recall=0.9821, Attack Precision=0.4305

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
0.15       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299   <--
0.20       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.25       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.30       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.35       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.40       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.45       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.50       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.55       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.60       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.65       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.70       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.75       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
0.80       0.8810   0.7676   0.8558   0.9948   0.9822   0.6299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8810, F1=0.7676, Normal Recall=0.8558, Normal Precision=0.9948, Attack Recall=0.9822, Attack Precision=0.6299

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
0.15       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433   <--
0.20       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.25       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.30       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.35       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.40       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.45       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.50       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.55       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.60       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.65       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.70       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.75       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
0.80       0.8929   0.8462   0.8546   0.9911   0.9822   0.7433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.8462, Normal Recall=0.8546, Normal Precision=0.9911, Attack Recall=0.9822, Attack Precision=0.7433

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
0.15       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182   <--
0.20       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.25       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.30       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.35       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.40       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.45       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.50       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.55       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.60       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.65       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.70       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.75       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
0.80       0.9056   0.8927   0.8545   0.9863   0.9822   0.8182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9056, F1=0.8927, Normal Recall=0.8545, Normal Precision=0.9863, Attack Recall=0.9822, Attack Precision=0.8182

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
0.15       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705   <--
0.20       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.25       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.30       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.35       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.40       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.45       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.50       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.55       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.60       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.65       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.70       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.75       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
0.80       0.9180   0.9230   0.8539   0.9795   0.9822   0.8705  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9180, F1=0.9230, Normal Recall=0.8539, Normal Precision=0.9795, Attack Recall=0.9822, Attack Precision=0.8705

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
0.15       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242   <--
0.20       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.25       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.30       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.35       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.40       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.45       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.50       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.55       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.60       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.65       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.70       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.75       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.80       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8650, F1=0.5921, Normal Recall=0.8522, Normal Precision=0.9974, Attack Recall=0.9800, Attack Precision=0.4242

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
0.15       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241   <--
0.20       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.25       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.30       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.35       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.40       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.45       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.50       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.55       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.60       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.65       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.70       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.75       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.80       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8780, F1=0.7626, Normal Recall=0.8525, Normal Precision=0.9941, Attack Recall=0.9799, Attack Precision=0.6241

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
0.15       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387   <--
0.20       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.25       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.30       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.35       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.40       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.45       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.50       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.55       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.60       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.65       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.70       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.75       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.80       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8900, F1=0.8423, Normal Recall=0.8514, Normal Precision=0.9900, Attack Recall=0.9799, Attack Precision=0.7387

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
0.15       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146   <--
0.20       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.25       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.30       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.35       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.40       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.45       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.50       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.55       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.60       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.65       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.70       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.75       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.80       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9027, F1=0.8896, Normal Recall=0.8513, Normal Precision=0.9845, Attack Recall=0.9799, Attack Precision=0.8146

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
0.15       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676   <--
0.20       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.25       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.30       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.35       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.40       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.45       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.50       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.55       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.60       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.65       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.70       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.75       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.80       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9152, F1=0.9203, Normal Recall=0.8504, Normal Precision=0.9769, Attack Recall=0.9799, Attack Precision=0.8676

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
0.15       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242   <--
0.20       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.25       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.30       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.35       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.40       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.45       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.50       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.55       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.60       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.65       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.70       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.75       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
0.80       0.8650   0.5921   0.8522   0.9974   0.9800   0.4242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8650, F1=0.5921, Normal Recall=0.8522, Normal Precision=0.9974, Attack Recall=0.9800, Attack Precision=0.4242

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
0.15       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241   <--
0.20       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.25       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.30       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.35       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.40       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.45       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.50       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.55       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.60       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.65       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.70       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.75       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
0.80       0.8780   0.7626   0.8525   0.9941   0.9799   0.6241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8780, F1=0.7626, Normal Recall=0.8525, Normal Precision=0.9941, Attack Recall=0.9799, Attack Precision=0.6241

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
0.15       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387   <--
0.20       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.25       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.30       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.35       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.40       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.45       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.50       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.55       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.60       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.65       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.70       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.75       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
0.80       0.8900   0.8423   0.8514   0.9900   0.9799   0.7387  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8900, F1=0.8423, Normal Recall=0.8514, Normal Precision=0.9900, Attack Recall=0.9799, Attack Precision=0.7387

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
0.15       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146   <--
0.20       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.25       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.30       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.35       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.40       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.45       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.50       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.55       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.60       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.65       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.70       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.75       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
0.80       0.9027   0.8896   0.8513   0.9845   0.9799   0.8146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9027, F1=0.8896, Normal Recall=0.8513, Normal Precision=0.9845, Attack Recall=0.9799, Attack Precision=0.8146

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
0.15       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676   <--
0.20       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.25       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.30       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.35       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.40       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.45       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.50       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.55       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.60       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.65       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.70       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.75       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
0.80       0.9152   0.9203   0.8504   0.9769   0.9799   0.8676  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9152, F1=0.9203, Normal Recall=0.8504, Normal Precision=0.9769, Attack Recall=0.9799, Attack Precision=0.8676

```

