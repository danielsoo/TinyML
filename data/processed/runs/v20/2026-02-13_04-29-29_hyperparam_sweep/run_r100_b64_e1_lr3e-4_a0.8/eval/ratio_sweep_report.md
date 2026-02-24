# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-19 10:51:38 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7825 | 0.8040 | 0.8245 | 0.8466 | 0.8679 | 0.8889 | 0.9104 | 0.9323 | 0.9526 | 0.9740 | 0.9957 |
| QAT+Prune only | 0.8462 | 0.8523 | 0.8583 | 0.8647 | 0.8712 | 0.8768 | 0.8837 | 0.8891 | 0.8963 | 0.9021 | 0.9090 |
| QAT+PTQ | 0.8461 | 0.8520 | 0.8579 | 0.8643 | 0.8706 | 0.8762 | 0.8828 | 0.8882 | 0.8953 | 0.9011 | 0.9078 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8461 | 0.8520 | 0.8579 | 0.8643 | 0.8706 | 0.8762 | 0.8828 | 0.8882 | 0.8953 | 0.9011 | 0.9078 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5040 | 0.6941 | 0.7957 | 0.8577 | 0.8997 | 0.9302 | 0.9537 | 0.9711 | 0.9857 | 0.9979 |
| QAT+Prune only | 0.0000 | 0.5515 | 0.7196 | 0.8012 | 0.8495 | 0.8806 | 0.9036 | 0.9198 | 0.9334 | 0.9436 | 0.9523 |
| QAT+PTQ | 0.0000 | 0.5508 | 0.7188 | 0.8005 | 0.8488 | 0.8800 | 0.9029 | 0.9191 | 0.9327 | 0.9429 | 0.9517 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5508 | 0.7188 | 0.8005 | 0.8488 | 0.8800 | 0.9029 | 0.9191 | 0.9327 | 0.9429 | 0.9517 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7825 | 0.7827 | 0.7816 | 0.7827 | 0.7826 | 0.7822 | 0.7824 | 0.7842 | 0.7802 | 0.7783 | 0.0000 |
| QAT+Prune only | 0.8462 | 0.8460 | 0.8456 | 0.8457 | 0.8459 | 0.8446 | 0.8457 | 0.8425 | 0.8454 | 0.8403 | 0.0000 |
| QAT+PTQ | 0.8461 | 0.8459 | 0.8455 | 0.8456 | 0.8458 | 0.8446 | 0.8453 | 0.8424 | 0.8451 | 0.8409 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8461 | 0.8459 | 0.8455 | 0.8456 | 0.8458 | 0.8446 | 0.8453 | 0.8424 | 0.8451 | 0.8409 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7825 | 0.0000 | 0.0000 | 0.0000 | 0.7825 | 1.0000 |
| 90 | 10 | 299,940 | 0.8040 | 0.3374 | 0.9958 | 0.5040 | 0.7827 | 0.9994 |
| 80 | 20 | 291,350 | 0.8245 | 0.5327 | 0.9957 | 0.6941 | 0.7816 | 0.9986 |
| 70 | 30 | 194,230 | 0.8466 | 0.6626 | 0.9957 | 0.7957 | 0.7827 | 0.9977 |
| 60 | 40 | 145,675 | 0.8679 | 0.7533 | 0.9957 | 0.8577 | 0.7826 | 0.9964 |
| 50 | 50 | 116,540 | 0.8889 | 0.8205 | 0.9957 | 0.8997 | 0.7822 | 0.9946 |
| 40 | 60 | 97,115 | 0.9104 | 0.8728 | 0.9957 | 0.9302 | 0.7824 | 0.9919 |
| 30 | 70 | 83,240 | 0.9323 | 0.9150 | 0.9957 | 0.9537 | 0.7842 | 0.9874 |
| 20 | 80 | 72,835 | 0.9526 | 0.9477 | 0.9957 | 0.9711 | 0.7802 | 0.9786 |
| 10 | 90 | 64,740 | 0.9740 | 0.9759 | 0.9957 | 0.9857 | 0.7783 | 0.9529 |
| 0 | 100 | 58,270 | 0.9957 | 1.0000 | 0.9957 | 0.9979 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8462 | 0.0000 | 0.0000 | 0.0000 | 0.8462 | 1.0000 |
| 90 | 10 | 299,940 | 0.8523 | 0.3960 | 0.9083 | 0.5515 | 0.8460 | 0.9881 |
| 80 | 20 | 291,350 | 0.8583 | 0.5955 | 0.9090 | 0.7196 | 0.8456 | 0.9738 |
| 70 | 30 | 194,230 | 0.8647 | 0.7163 | 0.9090 | 0.8012 | 0.8457 | 0.9559 |
| 60 | 40 | 145,675 | 0.8712 | 0.7973 | 0.9090 | 0.8495 | 0.8459 | 0.9331 |
| 50 | 50 | 116,540 | 0.8768 | 0.8540 | 0.9090 | 0.8806 | 0.8446 | 0.9027 |
| 40 | 60 | 97,115 | 0.8837 | 0.8983 | 0.9090 | 0.9036 | 0.8457 | 0.8610 |
| 30 | 70 | 83,240 | 0.8891 | 0.9309 | 0.9090 | 0.9198 | 0.8425 | 0.7987 |
| 20 | 80 | 72,835 | 0.8963 | 0.9592 | 0.9090 | 0.9334 | 0.8454 | 0.6990 |
| 10 | 90 | 64,740 | 0.9021 | 0.9809 | 0.9090 | 0.9436 | 0.8403 | 0.5064 |
| 0 | 100 | 58,270 | 0.9090 | 1.0000 | 0.9090 | 0.9523 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8461 | 0.0000 | 0.0000 | 0.0000 | 0.8461 | 1.0000 |
| 90 | 10 | 299,940 | 0.8520 | 0.3955 | 0.9070 | 0.5508 | 0.8459 | 0.9879 |
| 80 | 20 | 291,350 | 0.8579 | 0.5949 | 0.9078 | 0.7188 | 0.8455 | 0.9735 |
| 70 | 30 | 194,230 | 0.8643 | 0.7160 | 0.9078 | 0.8005 | 0.8456 | 0.9554 |
| 60 | 40 | 145,675 | 0.8706 | 0.7970 | 0.9078 | 0.8488 | 0.8458 | 0.9323 |
| 50 | 50 | 116,540 | 0.8762 | 0.8538 | 0.9078 | 0.8800 | 0.8446 | 0.9016 |
| 40 | 60 | 97,115 | 0.8828 | 0.8980 | 0.9078 | 0.9029 | 0.8453 | 0.8594 |
| 30 | 70 | 83,240 | 0.8882 | 0.9308 | 0.9078 | 0.9191 | 0.8424 | 0.7966 |
| 20 | 80 | 72,835 | 0.8953 | 0.9591 | 0.9078 | 0.9327 | 0.8451 | 0.6962 |
| 10 | 90 | 64,740 | 0.9011 | 0.9809 | 0.9078 | 0.9429 | 0.8409 | 0.5033 |
| 0 | 100 | 58,270 | 0.9078 | 1.0000 | 0.9078 | 0.9517 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8461 | 0.0000 | 0.0000 | 0.0000 | 0.8461 | 1.0000 |
| 90 | 10 | 299,940 | 0.8520 | 0.3955 | 0.9070 | 0.5508 | 0.8459 | 0.9879 |
| 80 | 20 | 291,350 | 0.8579 | 0.5949 | 0.9078 | 0.7188 | 0.8455 | 0.9735 |
| 70 | 30 | 194,230 | 0.8643 | 0.7160 | 0.9078 | 0.8005 | 0.8456 | 0.9554 |
| 60 | 40 | 145,675 | 0.8706 | 0.7970 | 0.9078 | 0.8488 | 0.8458 | 0.9323 |
| 50 | 50 | 116,540 | 0.8762 | 0.8538 | 0.9078 | 0.8800 | 0.8446 | 0.9016 |
| 40 | 60 | 97,115 | 0.8828 | 0.8980 | 0.9078 | 0.9029 | 0.8453 | 0.8594 |
| 30 | 70 | 83,240 | 0.8882 | 0.9308 | 0.9078 | 0.9191 | 0.8424 | 0.7966 |
| 20 | 80 | 72,835 | 0.8953 | 0.9591 | 0.9078 | 0.9327 | 0.8451 | 0.6962 |
| 10 | 90 | 64,740 | 0.9011 | 0.9809 | 0.9078 | 0.9429 | 0.8409 | 0.5033 |
| 0 | 100 | 58,270 | 0.9078 | 1.0000 | 0.9078 | 0.9517 | 0.0000 | 0.0000 |


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
0.15       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375   <--
0.20       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.25       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.30       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.35       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.40       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.45       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.50       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.55       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.60       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.65       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.70       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.75       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
0.80       0.8040   0.5042   0.7827   0.9995   0.9963   0.3375  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8040, F1=0.5042, Normal Recall=0.7827, Normal Precision=0.9995, Attack Recall=0.9963, Attack Precision=0.3375

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
0.15       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341   <--
0.20       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.25       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.30       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.35       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.40       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.45       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.50       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.55       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.60       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.65       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.70       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.75       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
0.80       0.8254   0.6952   0.7828   0.9986   0.9957   0.5341  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8254, F1=0.6952, Normal Recall=0.7828, Normal Precision=0.9986, Attack Recall=0.9957, Attack Precision=0.5341

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
0.15       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634   <--
0.20       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.25       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.30       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.35       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.40       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.45       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.50       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.55       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.60       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.65       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.70       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.75       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
0.80       0.8472   0.7963   0.7835   0.9977   0.9957   0.6634  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8472, F1=0.7963, Normal Recall=0.7835, Normal Precision=0.9977, Attack Recall=0.9957, Attack Precision=0.6634

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
0.15       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534   <--
0.20       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.25       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.30       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.35       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.40       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.45       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.50       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.55       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.60       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.65       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.70       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.75       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
0.80       0.8679   0.8578   0.7827   0.9964   0.9957   0.7534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8679, F1=0.8578, Normal Recall=0.7827, Normal Precision=0.9964, Attack Recall=0.9957, Attack Precision=0.7534

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
0.15       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200   <--
0.20       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.25       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.30       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.35       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.40       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.45       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.50       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.55       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.60       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.65       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.70       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.75       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
0.80       0.8886   0.8994   0.7815   0.9946   0.9957   0.8200  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8886, F1=0.8994, Normal Recall=0.7815, Normal Precision=0.9946, Attack Recall=0.9957, Attack Precision=0.8200

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
0.15       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962   <--
0.20       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.25       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.30       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.35       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.40       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.45       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.50       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.55       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.60       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.65       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.70       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.75       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
0.80       0.8524   0.5519   0.8460   0.9882   0.9092   0.3962  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8524, F1=0.5519, Normal Recall=0.8460, Normal Precision=0.9882, Attack Recall=0.9092, Attack Precision=0.3962

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
0.15       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968   <--
0.20       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.25       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.30       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.35       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.40       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.45       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.50       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.55       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.60       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.65       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.70       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.75       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
0.80       0.8590   0.7206   0.8465   0.9738   0.9090   0.5968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8590, F1=0.7206, Normal Recall=0.8465, Normal Precision=0.9738, Attack Recall=0.9090, Attack Precision=0.5968

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
0.15       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167   <--
0.20       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.25       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.30       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.35       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.40       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.45       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.50       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.55       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.60       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.65       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.70       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.75       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
0.80       0.8649   0.8015   0.8460   0.9559   0.9090   0.7167  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8649, F1=0.8015, Normal Recall=0.8460, Normal Precision=0.9559, Attack Recall=0.9090, Attack Precision=0.7167

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
0.15       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973   <--
0.20       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.25       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.30       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.35       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.40       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.45       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.50       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.55       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.60       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.65       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.70       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.75       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
0.80       0.8712   0.8495   0.8459   0.9331   0.9090   0.7973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8712, F1=0.8495, Normal Recall=0.8459, Normal Precision=0.9331, Attack Recall=0.9090, Attack Precision=0.7973

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
0.15       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540   <--
0.20       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.25       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.30       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.35       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.40       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.45       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.50       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.55       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.60       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.65       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.70       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.75       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
0.80       0.8768   0.8807   0.8447   0.9027   0.9090   0.8540  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8768, F1=0.8807, Normal Recall=0.8447, Normal Precision=0.9027, Attack Recall=0.9090, Attack Precision=0.8540

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
0.15       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957   <--
0.20       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.25       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.30       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.35       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.40       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.45       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.50       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.55       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.60       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.65       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.70       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.75       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.80       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8521, F1=0.5512, Normal Recall=0.8459, Normal Precision=0.9881, Attack Recall=0.9081, Attack Precision=0.3957

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
0.15       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963   <--
0.20       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.25       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.30       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.35       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.40       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.45       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.50       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.55       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.60       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.65       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.70       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.75       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.80       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8587, F1=0.7198, Normal Recall=0.8464, Normal Precision=0.9735, Attack Recall=0.9078, Attack Precision=0.5963

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
0.15       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163   <--
0.20       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.25       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.30       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.35       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.40       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.45       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.50       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.55       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.60       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.65       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.70       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.75       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.80       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8645, F1=0.8007, Normal Recall=0.8459, Normal Precision=0.9554, Attack Recall=0.9078, Attack Precision=0.7163

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
0.15       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970   <--
0.20       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.25       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.30       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.35       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.40       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.45       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.50       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.55       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.60       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.65       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.70       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.75       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.80       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8706, F1=0.8488, Normal Recall=0.8459, Normal Precision=0.9323, Attack Recall=0.9078, Attack Precision=0.7970

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
0.15       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538   <--
0.20       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.25       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.30       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.35       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.40       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.45       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.50       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.55       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.60       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.65       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.70       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.75       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.80       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8762, F1=0.8800, Normal Recall=0.8446, Normal Precision=0.9016, Attack Recall=0.9078, Attack Precision=0.8538

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
0.15       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957   <--
0.20       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.25       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.30       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.35       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.40       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.45       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.50       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.55       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.60       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.65       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.70       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.75       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
0.80       0.8521   0.5512   0.8459   0.9881   0.9081   0.3957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8521, F1=0.5512, Normal Recall=0.8459, Normal Precision=0.9881, Attack Recall=0.9081, Attack Precision=0.3957

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
0.15       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963   <--
0.20       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.25       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.30       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.35       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.40       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.45       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.50       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.55       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.60       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.65       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.70       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.75       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
0.80       0.8587   0.7198   0.8464   0.9735   0.9078   0.5963  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8587, F1=0.7198, Normal Recall=0.8464, Normal Precision=0.9735, Attack Recall=0.9078, Attack Precision=0.5963

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
0.15       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163   <--
0.20       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.25       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.30       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.35       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.40       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.45       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.50       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.55       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.60       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.65       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.70       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.75       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
0.80       0.8645   0.8007   0.8459   0.9554   0.9078   0.7163  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8645, F1=0.8007, Normal Recall=0.8459, Normal Precision=0.9554, Attack Recall=0.9078, Attack Precision=0.7163

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
0.15       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970   <--
0.20       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.25       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.30       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.35       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.40       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.45       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.50       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.55       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.60       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.65       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.70       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.75       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
0.80       0.8706   0.8488   0.8459   0.9323   0.9078   0.7970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8706, F1=0.8488, Normal Recall=0.8459, Normal Precision=0.9323, Attack Recall=0.9078, Attack Precision=0.7970

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
0.15       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538   <--
0.20       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.25       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.30       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.35       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.40       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.45       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.50       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.55       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.60       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.65       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.70       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.75       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
0.80       0.8762   0.8800   0.8446   0.9016   0.9078   0.8538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8762, F1=0.8800, Normal Recall=0.8446, Normal Precision=0.9016, Attack Recall=0.9078, Attack Precision=0.8538

```

