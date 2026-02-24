# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-22 00:51:12 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1853 | 0.2635 | 0.3433 | 0.4226 | 0.5026 | 0.5818 | 0.6611 | 0.7417 | 0.8201 | 0.9006 | 0.9790 |
| QAT+Prune only | 0.6180 | 0.6549 | 0.6926 | 0.7311 | 0.7706 | 0.8079 | 0.8439 | 0.8835 | 0.9214 | 0.9596 | 0.9981 |
| QAT+PTQ | 0.6146 | 0.6521 | 0.6900 | 0.7288 | 0.7689 | 0.8061 | 0.8428 | 0.8827 | 0.9205 | 0.9593 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6146 | 0.6521 | 0.6900 | 0.7288 | 0.7689 | 0.8061 | 0.8428 | 0.8827 | 0.9205 | 0.9593 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2098 | 0.3736 | 0.5043 | 0.6116 | 0.7007 | 0.7761 | 0.8414 | 0.8970 | 0.9466 | 0.9894 |
| QAT+Prune only | 0.0000 | 0.3665 | 0.5650 | 0.6901 | 0.7768 | 0.8386 | 0.8847 | 0.9230 | 0.9531 | 0.9780 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.3646 | 0.5629 | 0.6883 | 0.7755 | 0.8373 | 0.8839 | 0.9226 | 0.9526 | 0.9779 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3646 | 0.5629 | 0.6883 | 0.7755 | 0.8373 | 0.8839 | 0.9226 | 0.9526 | 0.9779 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1853 | 0.1841 | 0.1844 | 0.1841 | 0.1850 | 0.1846 | 0.1843 | 0.1879 | 0.1845 | 0.1954 | 0.0000 |
| QAT+Prune only | 0.6180 | 0.6168 | 0.6163 | 0.6167 | 0.6190 | 0.6177 | 0.6127 | 0.6161 | 0.6150 | 0.6137 | 0.0000 |
| QAT+PTQ | 0.6146 | 0.6136 | 0.6130 | 0.6134 | 0.6162 | 0.6142 | 0.6098 | 0.6137 | 0.6103 | 0.6109 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6146 | 0.6136 | 0.6130 | 0.6134 | 0.6162 | 0.6142 | 0.6098 | 0.6137 | 0.6103 | 0.6109 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1853 | 0.0000 | 0.0000 | 0.0000 | 0.1853 | 1.0000 |
| 90 | 10 | 299,940 | 0.2635 | 0.1175 | 0.9778 | 0.2098 | 0.1841 | 0.9868 |
| 80 | 20 | 291,350 | 0.3433 | 0.2308 | 0.9790 | 0.3736 | 0.1844 | 0.9723 |
| 70 | 30 | 194,230 | 0.4226 | 0.3396 | 0.9790 | 0.5043 | 0.1841 | 0.9534 |
| 60 | 40 | 145,675 | 0.5026 | 0.4447 | 0.9790 | 0.6116 | 0.1850 | 0.9296 |
| 50 | 50 | 116,540 | 0.5818 | 0.5456 | 0.9790 | 0.7007 | 0.1846 | 0.8978 |
| 40 | 60 | 97,115 | 0.6611 | 0.6429 | 0.9790 | 0.7761 | 0.1843 | 0.8540 |
| 30 | 70 | 83,240 | 0.7417 | 0.7377 | 0.9790 | 0.8414 | 0.1879 | 0.7931 |
| 20 | 80 | 72,835 | 0.8201 | 0.8276 | 0.9790 | 0.8970 | 0.1845 | 0.6870 |
| 10 | 90 | 64,740 | 0.9006 | 0.9163 | 0.9790 | 0.9466 | 0.1954 | 0.5084 |
| 0 | 100 | 58,270 | 0.9790 | 1.0000 | 0.9790 | 0.9894 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6180 | 0.0000 | 0.0000 | 0.0000 | 0.6180 | 1.0000 |
| 90 | 10 | 299,940 | 0.6549 | 0.2245 | 0.9982 | 0.3665 | 0.6168 | 0.9997 |
| 80 | 20 | 291,350 | 0.6926 | 0.3940 | 0.9981 | 0.5650 | 0.6163 | 0.9992 |
| 70 | 30 | 194,230 | 0.7311 | 0.5274 | 0.9981 | 0.6901 | 0.6167 | 0.9987 |
| 60 | 40 | 145,675 | 0.7706 | 0.6359 | 0.9981 | 0.7768 | 0.6190 | 0.9979 |
| 50 | 50 | 116,540 | 0.8079 | 0.7230 | 0.9981 | 0.8386 | 0.6177 | 0.9969 |
| 40 | 60 | 97,115 | 0.8439 | 0.7945 | 0.9981 | 0.8847 | 0.6127 | 0.9953 |
| 30 | 70 | 83,240 | 0.8835 | 0.8585 | 0.9981 | 0.9230 | 0.6161 | 0.9927 |
| 20 | 80 | 72,835 | 0.9214 | 0.9120 | 0.9981 | 0.9531 | 0.6150 | 0.9875 |
| 10 | 90 | 64,740 | 0.9596 | 0.9588 | 0.9981 | 0.9780 | 0.6137 | 0.9723 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6146 | 0.0000 | 0.0000 | 0.0000 | 0.6146 | 1.0000 |
| 90 | 10 | 299,940 | 0.6521 | 0.2230 | 0.9982 | 0.3646 | 0.6136 | 0.9997 |
| 80 | 20 | 291,350 | 0.6900 | 0.3920 | 0.9980 | 0.5629 | 0.6130 | 0.9992 |
| 70 | 30 | 194,230 | 0.7288 | 0.5252 | 0.9980 | 0.6883 | 0.6134 | 0.9986 |
| 60 | 40 | 145,675 | 0.7689 | 0.6342 | 0.9980 | 0.7755 | 0.6162 | 0.9979 |
| 50 | 50 | 116,540 | 0.8061 | 0.7212 | 0.9980 | 0.8373 | 0.6142 | 0.9968 |
| 40 | 60 | 97,115 | 0.8428 | 0.7933 | 0.9980 | 0.8839 | 0.6098 | 0.9952 |
| 30 | 70 | 83,240 | 0.8827 | 0.8577 | 0.9980 | 0.9226 | 0.6137 | 0.9926 |
| 20 | 80 | 72,835 | 0.9205 | 0.9111 | 0.9980 | 0.9526 | 0.6103 | 0.9873 |
| 10 | 90 | 64,740 | 0.9593 | 0.9585 | 0.9980 | 0.9779 | 0.6109 | 0.9720 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6146 | 0.0000 | 0.0000 | 0.0000 | 0.6146 | 1.0000 |
| 90 | 10 | 299,940 | 0.6521 | 0.2230 | 0.9982 | 0.3646 | 0.6136 | 0.9997 |
| 80 | 20 | 291,350 | 0.6900 | 0.3920 | 0.9980 | 0.5629 | 0.6130 | 0.9992 |
| 70 | 30 | 194,230 | 0.7288 | 0.5252 | 0.9980 | 0.6883 | 0.6134 | 0.9986 |
| 60 | 40 | 145,675 | 0.7689 | 0.6342 | 0.9980 | 0.7755 | 0.6162 | 0.9979 |
| 50 | 50 | 116,540 | 0.8061 | 0.7212 | 0.9980 | 0.8373 | 0.6142 | 0.9968 |
| 40 | 60 | 97,115 | 0.8428 | 0.7933 | 0.9980 | 0.8839 | 0.6098 | 0.9952 |
| 30 | 70 | 83,240 | 0.8827 | 0.8577 | 0.9980 | 0.9226 | 0.6137 | 0.9926 |
| 20 | 80 | 72,835 | 0.9205 | 0.9111 | 0.9980 | 0.9526 | 0.6103 | 0.9873 |
| 10 | 90 | 64,740 | 0.9593 | 0.9585 | 0.9980 | 0.9779 | 0.6109 | 0.9720 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177   <--
0.20       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.25       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.30       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.35       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.40       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.45       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.50       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.55       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.60       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.65       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.70       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.75       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
0.80       0.2636   0.2101   0.1841   0.9877   0.9793   0.1177  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2636, F1=0.2101, Normal Recall=0.1841, Normal Precision=0.9877, Attack Recall=0.9793, Attack Precision=0.1177

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
0.15       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307   <--
0.20       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.25       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.30       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.35       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.40       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.45       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.50       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.55       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.60       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.65       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.70       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.75       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
0.80       0.3429   0.3734   0.1839   0.9722   0.9790   0.2307  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3429, F1=0.3734, Normal Recall=0.1839, Normal Precision=0.9722, Attack Recall=0.9790, Attack Precision=0.2307

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
0.15       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399   <--
0.20       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.25       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.30       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.35       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.40       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.45       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.50       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.55       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.60       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.65       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.70       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.75       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
0.80       0.4232   0.5046   0.1851   0.9536   0.9790   0.3399  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4232, F1=0.5046, Normal Recall=0.1851, Normal Precision=0.9536, Attack Recall=0.9790, Attack Precision=0.3399

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
0.15       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448   <--
0.20       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.25       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.30       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.35       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.40       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.45       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.50       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.55       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.60       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.65       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.70       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.75       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
0.80       0.5028   0.6117   0.1854   0.9298   0.9790   0.4448  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5028, F1=0.6117, Normal Recall=0.1854, Normal Precision=0.9298, Attack Recall=0.9790, Attack Precision=0.4448

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
0.15       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457   <--
0.20       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.25       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.30       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.35       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.40       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.45       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.50       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.55       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.60       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.65       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.70       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.75       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
0.80       0.5819   0.7007   0.1848   0.8979   0.9790   0.5457  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5819, F1=0.7007, Normal Recall=0.1848, Normal Precision=0.8979, Attack Recall=0.9790, Attack Precision=0.5457

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
0.15       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245   <--
0.20       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.25       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.30       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.35       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.40       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.45       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.50       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.55       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.60       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.65       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.70       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.75       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
0.80       0.6549   0.3665   0.6168   0.9997   0.9982   0.2245  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6549, F1=0.3665, Normal Recall=0.6168, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2245

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
0.15       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945   <--
0.20       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.25       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.30       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.35       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.40       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.45       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.50       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.55       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.60       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.65       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.70       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.75       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
0.80       0.6933   0.5655   0.6171   0.9992   0.9981   0.3945  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6933, F1=0.5655, Normal Recall=0.6171, Normal Precision=0.9992, Attack Recall=0.9981, Attack Precision=0.3945

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
0.15       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278   <--
0.20       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.25       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.30       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.35       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.40       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.45       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.50       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.55       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.60       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.65       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.70       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.75       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
0.80       0.7315   0.6904   0.6173   0.9987   0.9981   0.5278  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7315, F1=0.6904, Normal Recall=0.6173, Normal Precision=0.9987, Attack Recall=0.9981, Attack Precision=0.5278

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
0.15       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353   <--
0.20       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.25       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.30       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.35       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.40       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.45       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.50       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.55       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.60       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.65       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.70       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.75       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
0.80       0.7700   0.7764   0.6180   0.9979   0.9981   0.6353  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7700, F1=0.7764, Normal Recall=0.6180, Normal Precision=0.9979, Attack Recall=0.9981, Attack Precision=0.6353

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
0.15       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220   <--
0.20       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.25       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.30       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.35       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.40       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.45       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.50       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.55       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.60       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.65       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.70       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.75       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
0.80       0.8069   0.8379   0.6157   0.9969   0.9981   0.7220  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8069, F1=0.8379, Normal Recall=0.6157, Normal Precision=0.9969, Attack Recall=0.9981, Attack Precision=0.7220

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
0.15       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230   <--
0.20       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.25       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.30       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.35       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.40       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.45       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.50       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.55       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.60       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.65       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.70       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.75       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.80       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6521, F1=0.3646, Normal Recall=0.6136, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2230

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
0.15       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926   <--
0.20       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.25       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.30       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.35       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.40       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.45       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.50       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.55       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.60       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.65       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.70       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.75       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.80       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6908, F1=0.5635, Normal Recall=0.6140, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.3926

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
0.15       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256   <--
0.20       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.25       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.30       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.35       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.40       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.45       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.50       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.55       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.60       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.65       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.70       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.75       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.80       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7292, F1=0.6886, Normal Recall=0.6140, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5256

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
0.15       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332   <--
0.20       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.25       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.30       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.35       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.40       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.45       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.50       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.55       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.60       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.65       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.70       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.75       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.80       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7680, F1=0.7748, Normal Recall=0.6146, Normal Precision=0.9979, Attack Recall=0.9980, Attack Precision=0.6332

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
0.15       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202   <--
0.20       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.25       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.30       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.35       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.40       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.45       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.50       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.55       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.60       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.65       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.70       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.75       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.80       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8051, F1=0.8367, Normal Recall=0.6123, Normal Precision=0.9968, Attack Recall=0.9980, Attack Precision=0.7202

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
0.15       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230   <--
0.20       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.25       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.30       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.35       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.40       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.45       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.50       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.55       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.60       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.65       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.70       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.75       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
0.80       0.6521   0.3646   0.6136   0.9997   0.9982   0.2230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6521, F1=0.3646, Normal Recall=0.6136, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2230

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
0.15       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926   <--
0.20       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.25       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.30       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.35       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.40       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.45       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.50       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.55       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.60       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.65       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.70       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.75       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
0.80       0.6908   0.5635   0.6140   0.9992   0.9980   0.3926  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6908, F1=0.5635, Normal Recall=0.6140, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.3926

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
0.15       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256   <--
0.20       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.25       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.30       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.35       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.40       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.45       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.50       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.55       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.60       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.65       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.70       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.75       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
0.80       0.7292   0.6886   0.6140   0.9986   0.9980   0.5256  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7292, F1=0.6886, Normal Recall=0.6140, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5256

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
0.15       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332   <--
0.20       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.25       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.30       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.35       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.40       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.45       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.50       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.55       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.60       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.65       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.70       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.75       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
0.80       0.7680   0.7748   0.6146   0.9979   0.9980   0.6332  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7680, F1=0.7748, Normal Recall=0.6146, Normal Precision=0.9979, Attack Recall=0.9980, Attack Precision=0.6332

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
0.15       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202   <--
0.20       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.25       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.30       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.35       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.40       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.45       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.50       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.55       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.60       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.65       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.70       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.75       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
0.80       0.8051   0.8367   0.6123   0.9968   0.9980   0.7202  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8051, F1=0.8367, Normal Recall=0.6123, Normal Precision=0.9968, Attack Recall=0.9980, Attack Precision=0.7202

```

