# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-15 22:02:30 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7735 | 0.7466 | 0.7204 | 0.6941 | 0.6666 | 0.6402 | 0.6132 | 0.5883 | 0.5593 | 0.5334 | 0.5073 |
| QAT+Prune only | 0.8928 | 0.8976 | 0.9011 | 0.9049 | 0.9084 | 0.9126 | 0.9160 | 0.9200 | 0.9235 | 0.9262 | 0.9310 |
| QAT+PTQ | 0.8929 | 0.8978 | 0.9013 | 0.9051 | 0.9084 | 0.9124 | 0.9158 | 0.9199 | 0.9233 | 0.9259 | 0.9306 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8929 | 0.8978 | 0.9013 | 0.9051 | 0.9084 | 0.9124 | 0.9158 | 0.9199 | 0.9233 | 0.9259 | 0.9306 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2846 | 0.4205 | 0.4987 | 0.5490 | 0.5850 | 0.6115 | 0.6331 | 0.6481 | 0.6618 | 0.6731 |
| QAT+Prune only | 0.0000 | 0.6454 | 0.7902 | 0.8545 | 0.8904 | 0.9142 | 0.9301 | 0.9422 | 0.9512 | 0.9578 | 0.9643 |
| QAT+PTQ | 0.0000 | 0.6459 | 0.7905 | 0.8547 | 0.8904 | 0.9140 | 0.9299 | 0.9421 | 0.9510 | 0.9576 | 0.9640 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6459 | 0.7905 | 0.8547 | 0.8904 | 0.9140 | 0.9299 | 0.9421 | 0.9510 | 0.9576 | 0.9640 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7735 | 0.7735 | 0.7736 | 0.7741 | 0.7728 | 0.7731 | 0.7721 | 0.7773 | 0.7673 | 0.7683 | 0.0000 |
| QAT+Prune only | 0.8928 | 0.8938 | 0.8937 | 0.8937 | 0.8933 | 0.8941 | 0.8934 | 0.8944 | 0.8935 | 0.8829 | 0.0000 |
| QAT+PTQ | 0.8929 | 0.8941 | 0.8940 | 0.8941 | 0.8935 | 0.8943 | 0.8937 | 0.8951 | 0.8941 | 0.8835 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8929 | 0.8941 | 0.8940 | 0.8941 | 0.8935 | 0.8943 | 0.8937 | 0.8951 | 0.8941 | 0.8835 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7735 | 0.0000 | 0.0000 | 0.0000 | 0.7735 | 1.0000 |
| 90 | 10 | 299,940 | 0.7466 | 0.1983 | 0.5041 | 0.2846 | 0.7735 | 0.9335 |
| 80 | 20 | 291,350 | 0.7204 | 0.3591 | 0.5073 | 0.4205 | 0.7736 | 0.8627 |
| 70 | 30 | 194,230 | 0.6941 | 0.4904 | 0.5073 | 0.4987 | 0.7741 | 0.7857 |
| 60 | 40 | 145,675 | 0.6666 | 0.5981 | 0.5073 | 0.5490 | 0.7728 | 0.7017 |
| 50 | 50 | 116,540 | 0.6402 | 0.6909 | 0.5073 | 0.5850 | 0.7731 | 0.6108 |
| 40 | 60 | 97,115 | 0.6132 | 0.7695 | 0.5073 | 0.6115 | 0.7721 | 0.5109 |
| 30 | 70 | 83,240 | 0.5883 | 0.8417 | 0.5073 | 0.6331 | 0.7773 | 0.4034 |
| 20 | 80 | 72,835 | 0.5593 | 0.8971 | 0.5073 | 0.6481 | 0.7673 | 0.2802 |
| 10 | 90 | 64,740 | 0.5334 | 0.9517 | 0.5073 | 0.6618 | 0.7683 | 0.1477 |
| 0 | 100 | 58,270 | 0.5073 | 1.0000 | 0.5073 | 0.6731 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8928 | 0.0000 | 0.0000 | 0.0000 | 0.8928 | 1.0000 |
| 90 | 10 | 299,940 | 0.8976 | 0.4936 | 0.9320 | 0.6454 | 0.8938 | 0.9916 |
| 80 | 20 | 291,350 | 0.9011 | 0.6864 | 0.9310 | 0.7902 | 0.8937 | 0.9811 |
| 70 | 30 | 194,230 | 0.9049 | 0.7897 | 0.9310 | 0.8545 | 0.8937 | 0.9680 |
| 60 | 40 | 145,675 | 0.9084 | 0.8533 | 0.9310 | 0.8904 | 0.8933 | 0.9510 |
| 50 | 50 | 116,540 | 0.9126 | 0.8979 | 0.9310 | 0.9142 | 0.8941 | 0.9284 |
| 40 | 60 | 97,115 | 0.9160 | 0.9291 | 0.9310 | 0.9301 | 0.8934 | 0.8962 |
| 30 | 70 | 83,240 | 0.9200 | 0.9536 | 0.9310 | 0.9422 | 0.8944 | 0.8475 |
| 20 | 80 | 72,835 | 0.9235 | 0.9722 | 0.9310 | 0.9512 | 0.8935 | 0.7641 |
| 10 | 90 | 64,740 | 0.9262 | 0.9862 | 0.9310 | 0.9578 | 0.8829 | 0.5872 |
| 0 | 100 | 58,270 | 0.9310 | 1.0000 | 0.9310 | 0.9643 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8929 | 0.0000 | 0.0000 | 0.0000 | 0.8929 | 1.0000 |
| 90 | 10 | 299,940 | 0.8978 | 0.4942 | 0.9317 | 0.6459 | 0.8941 | 0.9916 |
| 80 | 20 | 291,350 | 0.9013 | 0.6871 | 0.9306 | 0.7905 | 0.8940 | 0.9810 |
| 70 | 30 | 194,230 | 0.9051 | 0.7902 | 0.9306 | 0.8547 | 0.8941 | 0.9678 |
| 60 | 40 | 145,675 | 0.9084 | 0.8535 | 0.9306 | 0.8904 | 0.8935 | 0.9508 |
| 50 | 50 | 116,540 | 0.9124 | 0.8980 | 0.9306 | 0.9140 | 0.8943 | 0.9280 |
| 40 | 60 | 97,115 | 0.9158 | 0.9292 | 0.9306 | 0.9299 | 0.8937 | 0.8956 |
| 30 | 70 | 83,240 | 0.9199 | 0.9539 | 0.9306 | 0.9421 | 0.8951 | 0.8468 |
| 20 | 80 | 72,835 | 0.9233 | 0.9723 | 0.9306 | 0.9510 | 0.8941 | 0.7631 |
| 10 | 90 | 64,740 | 0.9259 | 0.9863 | 0.9306 | 0.9576 | 0.8835 | 0.5858 |
| 0 | 100 | 58,270 | 0.9306 | 1.0000 | 0.9306 | 0.9640 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8929 | 0.0000 | 0.0000 | 0.0000 | 0.8929 | 1.0000 |
| 90 | 10 | 299,940 | 0.8978 | 0.4942 | 0.9317 | 0.6459 | 0.8941 | 0.9916 |
| 80 | 20 | 291,350 | 0.9013 | 0.6871 | 0.9306 | 0.7905 | 0.8940 | 0.9810 |
| 70 | 30 | 194,230 | 0.9051 | 0.7902 | 0.9306 | 0.8547 | 0.8941 | 0.9678 |
| 60 | 40 | 145,675 | 0.9084 | 0.8535 | 0.9306 | 0.8904 | 0.8935 | 0.9508 |
| 50 | 50 | 116,540 | 0.9124 | 0.8980 | 0.9306 | 0.9140 | 0.8943 | 0.9280 |
| 40 | 60 | 97,115 | 0.9158 | 0.9292 | 0.9306 | 0.9299 | 0.8937 | 0.8956 |
| 30 | 70 | 83,240 | 0.9199 | 0.9539 | 0.9306 | 0.9421 | 0.8951 | 0.8468 |
| 20 | 80 | 72,835 | 0.9233 | 0.9723 | 0.9306 | 0.9510 | 0.8941 | 0.7631 |
| 10 | 90 | 64,740 | 0.9259 | 0.9863 | 0.9306 | 0.9576 | 0.8835 | 0.5858 |
| 0 | 100 | 58,270 | 0.9306 | 1.0000 | 0.9306 | 0.9640 | 0.0000 | 0.0000 |


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
0.15       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991   <--
0.20       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.25       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.30       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.35       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.40       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.45       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.50       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.55       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.60       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.65       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.70       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.75       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
0.80       0.7468   0.2860   0.7735   0.9339   0.5069   0.1991  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7468, F1=0.2860, Normal Recall=0.7735, Normal Precision=0.9339, Attack Recall=0.5069, Attack Precision=0.1991

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
0.15       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596   <--
0.20       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.25       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.30       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.35       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.40       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.45       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.50       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.55       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.60       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.65       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.70       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.75       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
0.80       0.7208   0.4209   0.7742   0.8627   0.5073   0.3596  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7208, F1=0.4209, Normal Recall=0.7742, Normal Precision=0.8627, Attack Recall=0.5073, Attack Precision=0.3596

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
0.15       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904   <--
0.20       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.25       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.30       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.35       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.40       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.45       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.50       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.55       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.60       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.65       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.70       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.75       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
0.80       0.6940   0.4987   0.7741   0.7857   0.5073   0.4904  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6940, F1=0.4987, Normal Recall=0.7741, Normal Precision=0.7857, Attack Recall=0.5073, Attack Precision=0.4904

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
0.15       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989   <--
0.20       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.25       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.30       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.35       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.40       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.45       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.50       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.55       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.60       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.65       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.70       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.75       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
0.80       0.6670   0.5493   0.7735   0.7019   0.5073   0.5989  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6670, F1=0.5493, Normal Recall=0.7735, Normal Precision=0.7019, Attack Recall=0.5073, Attack Precision=0.5989

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
0.15       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912   <--
0.20       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.25       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.30       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.35       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.40       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.45       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.50       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.55       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.60       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.65       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.70       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.75       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
0.80       0.6403   0.5851   0.7733   0.6108   0.5073   0.6912  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6403, F1=0.5851, Normal Recall=0.7733, Normal Precision=0.6108, Attack Recall=0.5073, Attack Precision=0.6912

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
0.15       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936   <--
0.20       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.25       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.30       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.35       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.40       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.45       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.50       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.55       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.60       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.65       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.70       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.75       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
0.80       0.8976   0.6455   0.8938   0.9916   0.9322   0.4936  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8976, F1=0.6455, Normal Recall=0.8938, Normal Precision=0.9916, Attack Recall=0.9322, Attack Precision=0.4936

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
0.15       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869   <--
0.20       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.25       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.30       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.35       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.40       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.45       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.50       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.55       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.60       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.65       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.70       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.75       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
0.80       0.9013   0.7906   0.8939   0.9811   0.9310   0.6869  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9013, F1=0.7906, Normal Recall=0.8939, Normal Precision=0.9811, Attack Recall=0.9310, Attack Precision=0.6869

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
0.15       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885   <--
0.20       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.25       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.30       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.35       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.40       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.45       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.50       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.55       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.60       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.65       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.70       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.75       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
0.80       0.9044   0.8538   0.8930   0.9680   0.9310   0.7885  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9044, F1=0.8538, Normal Recall=0.8930, Normal Precision=0.9680, Attack Recall=0.9310, Attack Precision=0.7885

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
0.15       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529   <--
0.20       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.25       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.30       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.35       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.40       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.45       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.50       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.55       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.60       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.65       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.70       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.75       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
0.80       0.9082   0.8903   0.8930   0.9510   0.9310   0.8529  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9082, F1=0.8903, Normal Recall=0.8930, Normal Precision=0.9510, Attack Recall=0.9310, Attack Precision=0.8529

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
0.15       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971   <--
0.20       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.25       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.30       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.35       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.40       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.45       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.50       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.55       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.60       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.65       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.70       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.75       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
0.80       0.9121   0.9137   0.8932   0.9283   0.9310   0.8971  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9121, F1=0.9137, Normal Recall=0.8932, Normal Precision=0.9283, Attack Recall=0.9310, Attack Precision=0.8971

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
0.15       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942   <--
0.20       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.25       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.30       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.35       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.40       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.45       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.50       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.55       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.60       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.65       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.70       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.75       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.80       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8978, F1=0.6458, Normal Recall=0.8941, Normal Precision=0.9916, Attack Recall=0.9315, Attack Precision=0.4942

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
0.15       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873   <--
0.20       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.25       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.30       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.35       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.40       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.45       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.50       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.55       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.60       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.65       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.70       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.75       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.80       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9014, F1=0.7906, Normal Recall=0.8942, Normal Precision=0.9810, Attack Recall=0.9306, Attack Precision=0.6873

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
0.15       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888   <--
0.20       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.25       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.30       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.35       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.40       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.45       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.50       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.55       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.60       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.65       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.70       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.75       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.80       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9044, F1=0.8538, Normal Recall=0.8932, Normal Precision=0.9678, Attack Recall=0.9306, Attack Precision=0.7888

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
0.15       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531   <--
0.20       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.25       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.30       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.35       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.40       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.45       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.50       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.55       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.60       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.65       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.70       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.75       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.80       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.8901, Normal Recall=0.8932, Normal Precision=0.9507, Attack Recall=0.9306, Attack Precision=0.8531

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
0.15       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970   <--
0.20       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.25       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.30       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.35       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.40       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.45       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.50       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.55       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.60       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.65       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.70       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.75       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.80       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9119, F1=0.9135, Normal Recall=0.8932, Normal Precision=0.9279, Attack Recall=0.9306, Attack Precision=0.8970

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
0.15       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942   <--
0.20       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.25       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.30       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.35       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.40       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.45       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.50       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.55       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.60       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.65       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.70       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.75       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
0.80       0.8978   0.6458   0.8941   0.9916   0.9315   0.4942  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8978, F1=0.6458, Normal Recall=0.8941, Normal Precision=0.9916, Attack Recall=0.9315, Attack Precision=0.4942

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
0.15       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873   <--
0.20       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.25       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.30       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.35       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.40       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.45       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.50       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.55       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.60       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.65       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.70       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.75       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
0.80       0.9014   0.7906   0.8942   0.9810   0.9306   0.6873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9014, F1=0.7906, Normal Recall=0.8942, Normal Precision=0.9810, Attack Recall=0.9306, Attack Precision=0.6873

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
0.15       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888   <--
0.20       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.25       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.30       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.35       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.40       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.45       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.50       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.55       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.60       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.65       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.70       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.75       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
0.80       0.9044   0.8538   0.8932   0.9678   0.9306   0.7888  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9044, F1=0.8538, Normal Recall=0.8932, Normal Precision=0.9678, Attack Recall=0.9306, Attack Precision=0.7888

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
0.15       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531   <--
0.20       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.25       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.30       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.35       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.40       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.45       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.50       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.55       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.60       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.65       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.70       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.75       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
0.80       0.9081   0.8901   0.8932   0.9507   0.9306   0.8531  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9081, F1=0.8901, Normal Recall=0.8932, Normal Precision=0.9507, Attack Recall=0.9306, Attack Precision=0.8531

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
0.15       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970   <--
0.20       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.25       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.30       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.35       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.40       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.45       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.50       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.55       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.60       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.65       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.70       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.75       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
0.80       0.9119   0.9135   0.8932   0.9279   0.9306   0.8970  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9119, F1=0.9135, Normal Recall=0.8932, Normal Precision=0.9279, Attack Recall=0.9306, Attack Precision=0.8970

```

