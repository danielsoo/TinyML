# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-21 23:06:34 |

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
| Original (TFLite) | 0.2967 | 0.3547 | 0.4137 | 0.4724 | 0.5312 | 0.5914 | 0.6501 | 0.7088 | 0.7679 | 0.8271 | 0.8862 |
| QAT+Prune only | 0.7738 | 0.7967 | 0.8190 | 0.8417 | 0.8641 | 0.8848 | 0.9089 | 0.9306 | 0.9527 | 0.9755 | 0.9985 |
| QAT+PTQ | 0.7728 | 0.7959 | 0.8183 | 0.8411 | 0.8636 | 0.8845 | 0.9086 | 0.9303 | 0.9524 | 0.9753 | 0.9984 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7728 | 0.7959 | 0.8183 | 0.8411 | 0.8636 | 0.8845 | 0.9086 | 0.9303 | 0.9524 | 0.9753 | 0.9984 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2162 | 0.3768 | 0.5020 | 0.6020 | 0.6845 | 0.7524 | 0.8099 | 0.8594 | 0.9022 | 0.9397 |
| QAT+Prune only | 0.0000 | 0.4956 | 0.6881 | 0.7910 | 0.8546 | 0.8966 | 0.9294 | 0.9527 | 0.9713 | 0.9866 | 0.9992 |
| QAT+PTQ | 0.0000 | 0.4946 | 0.6873 | 0.7904 | 0.8542 | 0.8963 | 0.9291 | 0.9525 | 0.9711 | 0.9865 | 0.9992 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4946 | 0.6873 | 0.7904 | 0.8542 | 0.8963 | 0.9291 | 0.9525 | 0.9711 | 0.9865 | 0.9992 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2967 | 0.2953 | 0.2955 | 0.2950 | 0.2945 | 0.2966 | 0.2959 | 0.2949 | 0.2947 | 0.2946 | 0.0000 |
| QAT+Prune only | 0.7738 | 0.7743 | 0.7741 | 0.7746 | 0.7746 | 0.7711 | 0.7746 | 0.7722 | 0.7698 | 0.7691 | 0.0000 |
| QAT+PTQ | 0.7728 | 0.7734 | 0.7733 | 0.7737 | 0.7738 | 0.7706 | 0.7739 | 0.7713 | 0.7685 | 0.7677 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7728 | 0.7734 | 0.7733 | 0.7737 | 0.7738 | 0.7706 | 0.7739 | 0.7713 | 0.7685 | 0.7677 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2967 | 0.0000 | 0.0000 | 0.0000 | 0.2967 | 1.0000 |
| 90 | 10 | 299,940 | 0.3547 | 0.1230 | 0.8897 | 0.2162 | 0.2953 | 0.9602 |
| 80 | 20 | 291,350 | 0.4137 | 0.2393 | 0.8862 | 0.3768 | 0.2955 | 0.9122 |
| 70 | 30 | 194,230 | 0.4724 | 0.3501 | 0.8862 | 0.5020 | 0.2950 | 0.8582 |
| 60 | 40 | 145,675 | 0.5312 | 0.4558 | 0.8862 | 0.6020 | 0.2945 | 0.7952 |
| 50 | 50 | 116,540 | 0.5914 | 0.5575 | 0.8862 | 0.6845 | 0.2966 | 0.7228 |
| 40 | 60 | 97,115 | 0.6501 | 0.6537 | 0.8862 | 0.7524 | 0.2959 | 0.6342 |
| 30 | 70 | 83,240 | 0.7088 | 0.7457 | 0.8862 | 0.8099 | 0.2949 | 0.5263 |
| 20 | 80 | 72,835 | 0.7679 | 0.8341 | 0.8862 | 0.8594 | 0.2947 | 0.3931 |
| 10 | 90 | 64,740 | 0.8271 | 0.9187 | 0.8862 | 0.9022 | 0.2946 | 0.2234 |
| 0 | 100 | 58,270 | 0.8862 | 1.0000 | 0.8862 | 0.9397 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7738 | 0.0000 | 0.0000 | 0.0000 | 0.7738 | 1.0000 |
| 90 | 10 | 299,940 | 0.7967 | 0.3296 | 0.9985 | 0.4956 | 0.7743 | 0.9998 |
| 80 | 20 | 291,350 | 0.8190 | 0.5250 | 0.9985 | 0.6881 | 0.7741 | 0.9995 |
| 70 | 30 | 194,230 | 0.8417 | 0.6550 | 0.9985 | 0.7910 | 0.7746 | 0.9992 |
| 60 | 40 | 145,675 | 0.8641 | 0.7470 | 0.9985 | 0.8546 | 0.7746 | 0.9987 |
| 50 | 50 | 116,540 | 0.8848 | 0.8135 | 0.9985 | 0.8966 | 0.7711 | 0.9980 |
| 40 | 60 | 97,115 | 0.9089 | 0.8692 | 0.9985 | 0.9294 | 0.7746 | 0.9971 |
| 30 | 70 | 83,240 | 0.9306 | 0.9109 | 0.9985 | 0.9527 | 0.7722 | 0.9955 |
| 20 | 80 | 72,835 | 0.9527 | 0.9455 | 0.9985 | 0.9713 | 0.7698 | 0.9922 |
| 10 | 90 | 64,740 | 0.9755 | 0.9749 | 0.9985 | 0.9866 | 0.7691 | 0.9826 |
| 0 | 100 | 58,270 | 0.9985 | 1.0000 | 0.9985 | 0.9992 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7728 | 0.0000 | 0.0000 | 0.0000 | 0.7728 | 1.0000 |
| 90 | 10 | 299,940 | 0.7959 | 0.3287 | 0.9984 | 0.4946 | 0.7734 | 0.9998 |
| 80 | 20 | 291,350 | 0.8183 | 0.5240 | 0.9984 | 0.6873 | 0.7733 | 0.9995 |
| 70 | 30 | 194,230 | 0.8411 | 0.6541 | 0.9984 | 0.7904 | 0.7737 | 0.9991 |
| 60 | 40 | 145,675 | 0.8636 | 0.7463 | 0.9984 | 0.8542 | 0.7738 | 0.9986 |
| 50 | 50 | 116,540 | 0.8845 | 0.8132 | 0.9984 | 0.8963 | 0.7706 | 0.9980 |
| 40 | 60 | 97,115 | 0.9086 | 0.8688 | 0.9984 | 0.9291 | 0.7739 | 0.9969 |
| 30 | 70 | 83,240 | 0.9303 | 0.9106 | 0.9984 | 0.9525 | 0.7713 | 0.9952 |
| 20 | 80 | 72,835 | 0.9524 | 0.9452 | 0.9984 | 0.9711 | 0.7685 | 0.9918 |
| 10 | 90 | 64,740 | 0.9753 | 0.9748 | 0.9984 | 0.9865 | 0.7677 | 0.9818 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7728 | 0.0000 | 0.0000 | 0.0000 | 0.7728 | 1.0000 |
| 90 | 10 | 299,940 | 0.7959 | 0.3287 | 0.9984 | 0.4946 | 0.7734 | 0.9998 |
| 80 | 20 | 291,350 | 0.8183 | 0.5240 | 0.9984 | 0.6873 | 0.7733 | 0.9995 |
| 70 | 30 | 194,230 | 0.8411 | 0.6541 | 0.9984 | 0.7904 | 0.7737 | 0.9991 |
| 60 | 40 | 145,675 | 0.8636 | 0.7463 | 0.9984 | 0.8542 | 0.7738 | 0.9986 |
| 50 | 50 | 116,540 | 0.8845 | 0.8132 | 0.9984 | 0.8963 | 0.7706 | 0.9980 |
| 40 | 60 | 97,115 | 0.9086 | 0.8688 | 0.9984 | 0.9291 | 0.7739 | 0.9969 |
| 30 | 70 | 83,240 | 0.9303 | 0.9106 | 0.9984 | 0.9525 | 0.7713 | 0.9952 |
| 20 | 80 | 72,835 | 0.9524 | 0.9452 | 0.9984 | 0.9711 | 0.7685 | 0.9918 |
| 10 | 90 | 64,740 | 0.9753 | 0.9748 | 0.9984 | 0.9865 | 0.7677 | 0.9818 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |


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
0.15       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225   <--
0.20       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.25       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.30       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.35       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.40       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.45       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.50       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.55       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.60       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.65       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.70       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.75       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
0.80       0.3543   0.2153   0.2953   0.9587   0.8855   0.1225  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3543, F1=0.2153, Normal Recall=0.2953, Normal Precision=0.9587, Attack Recall=0.8855, Attack Precision=0.1225

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
0.15       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390   <--
0.20       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.25       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.30       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.35       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.40       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.45       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.50       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.55       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.60       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.65       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.70       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.75       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
0.80       0.4129   0.3765   0.2946   0.9120   0.8862   0.2390  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4129, F1=0.3765, Normal Recall=0.2946, Normal Precision=0.9120, Attack Recall=0.8862, Attack Precision=0.2390

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
0.15       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507   <--
0.20       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.25       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.30       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.35       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.40       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.45       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.50       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.55       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.60       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.65       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.70       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.75       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
0.80       0.4736   0.5025   0.2967   0.8589   0.8862   0.3507  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4736, F1=0.5025, Normal Recall=0.2967, Normal Precision=0.8589, Attack Recall=0.8862, Attack Precision=0.3507

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
0.15       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568   <--
0.20       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.25       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.30       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.35       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.40       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.45       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.50       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.55       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.60       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.65       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.70       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.75       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
0.80       0.5329   0.6029   0.2974   0.7968   0.8862   0.4568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5329, F1=0.6029, Normal Recall=0.2974, Normal Precision=0.7968, Attack Recall=0.8862, Attack Precision=0.4568

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
0.15       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576   <--
0.20       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.25       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.30       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.35       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.40       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.45       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.50       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.55       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.60       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.65       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.70       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.75       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
0.80       0.5915   0.6845   0.2969   0.7229   0.8862   0.5576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5915, F1=0.6845, Normal Recall=0.2969, Normal Precision=0.7229, Attack Recall=0.8862, Attack Precision=0.5576

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
0.15       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296   <--
0.20       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.25       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.30       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.35       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.40       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.45       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.50       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.55       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.60       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.65       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.70       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.75       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
0.80       0.7968   0.4956   0.7743   0.9998   0.9986   0.3296  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7968, F1=0.4956, Normal Recall=0.7743, Normal Precision=0.9998, Attack Recall=0.9986, Attack Precision=0.3296

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
0.15       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257   <--
0.20       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.25       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.30       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.35       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.40       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.45       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.50       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.55       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.60       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.65       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.70       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.75       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
0.80       0.8195   0.6887   0.7748   0.9995   0.9985   0.5257  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8195, F1=0.6887, Normal Recall=0.7748, Normal Precision=0.9995, Attack Recall=0.9985, Attack Precision=0.5257

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
0.15       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548   <--
0.20       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.25       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.30       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.35       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.40       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.45       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.50       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.55       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.60       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.65       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.70       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.75       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
0.80       0.8416   0.7909   0.7744   0.9992   0.9985   0.6548  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8416, F1=0.7909, Normal Recall=0.7744, Normal Precision=0.9992, Attack Recall=0.9985, Attack Precision=0.6548

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
0.15       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466   <--
0.20       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.25       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.30       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.35       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.40       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.45       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.50       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.55       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.60       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.65       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.70       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.75       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
0.80       0.8639   0.8544   0.7741   0.9987   0.9985   0.7466  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8639, F1=0.8544, Normal Recall=0.7741, Normal Precision=0.9987, Attack Recall=0.9985, Attack Precision=0.7466

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
0.15       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152   <--
0.20       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.25       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.30       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.35       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.40       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.45       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.50       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.55       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.60       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.65       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.70       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.75       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
0.80       0.8861   0.8976   0.7737   0.9981   0.9985   0.8152  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8861, F1=0.8976, Normal Recall=0.7737, Normal Precision=0.9981, Attack Recall=0.9985, Attack Precision=0.8152

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
0.15       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287   <--
0.20       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.25       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.30       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.35       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.40       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.45       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.50       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.55       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.60       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.65       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.70       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.75       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.80       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7959, F1=0.4946, Normal Recall=0.7734, Normal Precision=0.9998, Attack Recall=0.9985, Attack Precision=0.3287

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
0.15       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247   <--
0.20       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.25       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.30       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.35       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.40       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.45       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.50       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.55       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.60       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.65       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.70       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.75       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.80       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8188, F1=0.6879, Normal Recall=0.7739, Normal Precision=0.9995, Attack Recall=0.9984, Attack Precision=0.5247

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
0.15       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537   <--
0.20       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.25       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.30       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.35       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.40       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.45       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.50       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.55       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.60       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.65       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.70       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.75       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.80       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8409, F1=0.7901, Normal Recall=0.7733, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.6537

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
0.15       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458   <--
0.20       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.25       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.30       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.35       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.40       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.45       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.50       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.55       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.60       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.65       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.70       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.75       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.80       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8632, F1=0.8538, Normal Recall=0.7731, Normal Precision=0.9986, Attack Recall=0.9984, Attack Precision=0.7458

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
0.15       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145   <--
0.20       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.25       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.30       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.35       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.40       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.45       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.50       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.55       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.60       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.65       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.70       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.75       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.80       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8856, F1=0.8972, Normal Recall=0.7727, Normal Precision=0.9980, Attack Recall=0.9984, Attack Precision=0.8145

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
0.15       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287   <--
0.20       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.25       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.30       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.35       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.40       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.45       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.50       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.55       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.60       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.65       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.70       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.75       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
0.80       0.7959   0.4946   0.7734   0.9998   0.9985   0.3287  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7959, F1=0.4946, Normal Recall=0.7734, Normal Precision=0.9998, Attack Recall=0.9985, Attack Precision=0.3287

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
0.15       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247   <--
0.20       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.25       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.30       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.35       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.40       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.45       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.50       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.55       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.60       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.65       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.70       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.75       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
0.80       0.8188   0.6879   0.7739   0.9995   0.9984   0.5247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8188, F1=0.6879, Normal Recall=0.7739, Normal Precision=0.9995, Attack Recall=0.9984, Attack Precision=0.5247

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
0.15       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537   <--
0.20       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.25       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.30       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.35       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.40       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.45       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.50       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.55       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.60       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.65       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.70       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.75       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
0.80       0.8409   0.7901   0.7733   0.9991   0.9984   0.6537  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8409, F1=0.7901, Normal Recall=0.7733, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.6537

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
0.15       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458   <--
0.20       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.25       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.30       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.35       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.40       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.45       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.50       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.55       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.60       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.65       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.70       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.75       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
0.80       0.8632   0.8538   0.7731   0.9986   0.9984   0.7458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8632, F1=0.8538, Normal Recall=0.7731, Normal Precision=0.9986, Attack Recall=0.9984, Attack Precision=0.7458

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
0.15       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145   <--
0.20       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.25       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.30       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.35       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.40       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.45       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.50       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.55       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.60       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.65       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.70       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.75       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
0.80       0.8856   0.8972   0.7727   0.9980   0.9984   0.8145  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8856, F1=0.8972, Normal Recall=0.7727, Normal Precision=0.9980, Attack Recall=0.9984, Attack Precision=0.8145

```

