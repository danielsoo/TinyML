# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-13 21:28:20 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7931 | 0.7882 | 0.7808 | 0.7735 | 0.7652 | 0.7593 | 0.7509 | 0.7425 | 0.7358 | 0.7279 | 0.7206 |
| QAT+Prune only | 0.7984 | 0.8190 | 0.8382 | 0.8577 | 0.8785 | 0.8966 | 0.9175 | 0.9378 | 0.9574 | 0.9766 | 0.9970 |
| QAT+PTQ | 0.7973 | 0.8182 | 0.8374 | 0.8573 | 0.8781 | 0.8962 | 0.9172 | 0.9377 | 0.9573 | 0.9767 | 0.9971 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7973 | 0.8182 | 0.8374 | 0.8573 | 0.8781 | 0.8962 | 0.9172 | 0.9377 | 0.9573 | 0.9767 | 0.9971 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4048 | 0.5680 | 0.6562 | 0.7106 | 0.7496 | 0.7764 | 0.7966 | 0.8136 | 0.8266 | 0.8376 |
| QAT+Prune only | 0.0000 | 0.5242 | 0.7114 | 0.8078 | 0.8678 | 0.9060 | 0.9355 | 0.9573 | 0.9740 | 0.9871 | 0.9985 |
| QAT+PTQ | 0.0000 | 0.5232 | 0.7104 | 0.8074 | 0.8674 | 0.9057 | 0.9353 | 0.9573 | 0.9739 | 0.9872 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5232 | 0.7104 | 0.8074 | 0.8674 | 0.9057 | 0.9353 | 0.9573 | 0.9739 | 0.9872 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7931 | 0.7958 | 0.7958 | 0.7961 | 0.7950 | 0.7980 | 0.7963 | 0.7935 | 0.7963 | 0.7933 | 0.0000 |
| QAT+Prune only | 0.7984 | 0.7992 | 0.7985 | 0.7980 | 0.7996 | 0.7962 | 0.7982 | 0.7998 | 0.7991 | 0.7932 | 0.0000 |
| QAT+PTQ | 0.7973 | 0.7983 | 0.7975 | 0.7974 | 0.7988 | 0.7952 | 0.7974 | 0.7991 | 0.7979 | 0.7929 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7973 | 0.7983 | 0.7975 | 0.7974 | 0.7988 | 0.7952 | 0.7974 | 0.7991 | 0.7979 | 0.7929 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7931 | 0.0000 | 0.0000 | 0.0000 | 0.7931 | 1.0000 |
| 90 | 10 | 299,940 | 0.7882 | 0.2815 | 0.7201 | 0.4048 | 0.7958 | 0.9624 |
| 80 | 20 | 291,350 | 0.7808 | 0.4688 | 0.7206 | 0.5680 | 0.7958 | 0.9193 |
| 70 | 30 | 194,230 | 0.7735 | 0.6023 | 0.7206 | 0.6562 | 0.7961 | 0.8693 |
| 60 | 40 | 145,675 | 0.7652 | 0.7009 | 0.7206 | 0.7106 | 0.7950 | 0.8102 |
| 50 | 50 | 116,540 | 0.7593 | 0.7810 | 0.7206 | 0.7496 | 0.7980 | 0.7407 |
| 40 | 60 | 97,115 | 0.7509 | 0.8414 | 0.7206 | 0.7764 | 0.7963 | 0.6552 |
| 30 | 70 | 83,240 | 0.7425 | 0.8906 | 0.7206 | 0.7966 | 0.7935 | 0.5490 |
| 20 | 80 | 72,835 | 0.7358 | 0.9340 | 0.7206 | 0.8136 | 0.7963 | 0.4161 |
| 10 | 90 | 64,740 | 0.7279 | 0.9691 | 0.7206 | 0.8266 | 0.7933 | 0.2398 |
| 0 | 100 | 58,270 | 0.7206 | 1.0000 | 0.7206 | 0.8376 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7984 | 0.0000 | 0.0000 | 0.0000 | 0.7984 | 1.0000 |
| 90 | 10 | 299,940 | 0.8190 | 0.3556 | 0.9971 | 0.5242 | 0.7992 | 0.9996 |
| 80 | 20 | 291,350 | 0.8382 | 0.5529 | 0.9970 | 0.7114 | 0.7985 | 0.9990 |
| 70 | 30 | 194,230 | 0.8577 | 0.6790 | 0.9970 | 0.8078 | 0.7980 | 0.9984 |
| 60 | 40 | 145,675 | 0.8785 | 0.7683 | 0.9970 | 0.8678 | 0.7996 | 0.9975 |
| 50 | 50 | 116,540 | 0.8966 | 0.8303 | 0.9970 | 0.9060 | 0.7962 | 0.9962 |
| 40 | 60 | 97,115 | 0.9175 | 0.8811 | 0.9970 | 0.9355 | 0.7982 | 0.9943 |
| 30 | 70 | 83,240 | 0.9378 | 0.9208 | 0.9970 | 0.9573 | 0.7998 | 0.9912 |
| 20 | 80 | 72,835 | 0.9574 | 0.9520 | 0.9970 | 0.9740 | 0.7991 | 0.9850 |
| 10 | 90 | 64,740 | 0.9766 | 0.9775 | 0.9970 | 0.9871 | 0.7932 | 0.9667 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7973 | 0.0000 | 0.0000 | 0.0000 | 0.7973 | 1.0000 |
| 90 | 10 | 299,940 | 0.8182 | 0.3546 | 0.9973 | 0.5232 | 0.7983 | 0.9996 |
| 80 | 20 | 291,350 | 0.8374 | 0.5517 | 0.9971 | 0.7104 | 0.7975 | 0.9991 |
| 70 | 30 | 194,230 | 0.8573 | 0.6784 | 0.9971 | 0.8074 | 0.7974 | 0.9985 |
| 60 | 40 | 145,675 | 0.8781 | 0.7676 | 0.9971 | 0.8674 | 0.7988 | 0.9976 |
| 50 | 50 | 116,540 | 0.8962 | 0.8296 | 0.9971 | 0.9057 | 0.7952 | 0.9964 |
| 40 | 60 | 97,115 | 0.9172 | 0.8807 | 0.9971 | 0.9353 | 0.7974 | 0.9946 |
| 30 | 70 | 83,240 | 0.9377 | 0.9205 | 0.9971 | 0.9573 | 0.7991 | 0.9917 |
| 20 | 80 | 72,835 | 0.9573 | 0.9518 | 0.9971 | 0.9739 | 0.7979 | 0.9858 |
| 10 | 90 | 64,740 | 0.9767 | 0.9774 | 0.9971 | 0.9872 | 0.7929 | 0.9683 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7973 | 0.0000 | 0.0000 | 0.0000 | 0.7973 | 1.0000 |
| 90 | 10 | 299,940 | 0.8182 | 0.3546 | 0.9973 | 0.5232 | 0.7983 | 0.9996 |
| 80 | 20 | 291,350 | 0.8374 | 0.5517 | 0.9971 | 0.7104 | 0.7975 | 0.9991 |
| 70 | 30 | 194,230 | 0.8573 | 0.6784 | 0.9971 | 0.8074 | 0.7974 | 0.9985 |
| 60 | 40 | 145,675 | 0.8781 | 0.7676 | 0.9971 | 0.8674 | 0.7988 | 0.9976 |
| 50 | 50 | 116,540 | 0.8962 | 0.8296 | 0.9971 | 0.9057 | 0.7952 | 0.9964 |
| 40 | 60 | 97,115 | 0.9172 | 0.8807 | 0.9971 | 0.9353 | 0.7974 | 0.9946 |
| 30 | 70 | 83,240 | 0.9377 | 0.9205 | 0.9971 | 0.9573 | 0.7991 | 0.9917 |
| 20 | 80 | 72,835 | 0.9573 | 0.9518 | 0.9971 | 0.9739 | 0.7979 | 0.9858 |
| 10 | 90 | 64,740 | 0.9767 | 0.9774 | 0.9971 | 0.9872 | 0.7929 | 0.9683 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830   <--
0.20       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.25       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.30       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.35       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.40       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.45       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.50       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.55       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.60       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.65       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.70       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.75       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
0.80       0.7888   0.4071   0.7958   0.9631   0.7252   0.2830  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7888, F1=0.4071, Normal Recall=0.7958, Normal Precision=0.9631, Attack Recall=0.7252, Attack Precision=0.2830

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
0.15       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687   <--
0.20       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.25       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.30       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.35       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.40       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.45       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.50       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.55       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.60       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.65       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.70       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.75       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
0.80       0.7807   0.5680   0.7958   0.9193   0.7206   0.4687  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7807, F1=0.5680, Normal Recall=0.7958, Normal Precision=0.9193, Attack Recall=0.7206, Attack Precision=0.4687

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
0.15       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007   <--
0.20       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.25       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.30       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.35       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.40       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.45       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.50       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.55       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.60       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.65       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.70       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.75       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
0.80       0.7725   0.6552   0.7947   0.8691   0.7206   0.6007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7725, F1=0.6552, Normal Recall=0.7947, Normal Precision=0.8691, Attack Recall=0.7206, Attack Precision=0.6007

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
0.15       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994   <--
0.20       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.25       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.30       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.35       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.40       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.45       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.50       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.55       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.60       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.65       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.70       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.75       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
0.80       0.7643   0.7098   0.7935   0.8099   0.7206   0.6994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7643, F1=0.7098, Normal Recall=0.7935, Normal Precision=0.8099, Attack Recall=0.7206, Attack Precision=0.6994

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
0.15       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775   <--
0.20       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.25       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.30       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.35       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.40       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.45       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.50       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.55       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.60       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.65       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.70       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.75       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
0.80       0.7572   0.7480   0.7938   0.7397   0.7206   0.7775  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7572, F1=0.7480, Normal Recall=0.7938, Normal Precision=0.7397, Attack Recall=0.7206, Attack Precision=0.7775

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
0.15       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556   <--
0.20       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.25       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.30       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.35       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.40       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.45       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.50       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.55       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.60       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.65       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.70       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.75       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
0.80       0.8190   0.5243   0.7992   0.9996   0.9972   0.3556  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8190, F1=0.5243, Normal Recall=0.7992, Normal Precision=0.9996, Attack Recall=0.9972, Attack Precision=0.3556

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
0.15       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540   <--
0.20       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.25       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.30       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.35       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.40       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.45       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.50       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.55       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.60       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.65       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.70       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.75       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
0.80       0.8389   0.7123   0.7994   0.9991   0.9970   0.5540  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8389, F1=0.7123, Normal Recall=0.7994, Normal Precision=0.9991, Attack Recall=0.9970, Attack Precision=0.5540

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
0.15       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802   <--
0.20       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.25       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.30       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.35       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.40       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.45       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.50       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.55       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.60       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.65       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.70       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.75       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
0.80       0.8585   0.8087   0.7991   0.9984   0.9970   0.6802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8585, F1=0.8087, Normal Recall=0.7991, Normal Precision=0.9984, Attack Recall=0.9970, Attack Precision=0.6802

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
0.15       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678   <--
0.20       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.25       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.30       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.35       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.40       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.45       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.50       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.55       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.60       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.65       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.70       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.75       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
0.80       0.8782   0.8675   0.7990   0.9975   0.9970   0.7678  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8782, F1=0.8675, Normal Recall=0.7990, Normal Precision=0.9975, Attack Recall=0.9970, Attack Precision=0.7678

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
0.15       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318   <--
0.20       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.25       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.30       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.35       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.40       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.45       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.50       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.55       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.60       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.65       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.70       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.75       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
0.80       0.8977   0.9069   0.7984   0.9962   0.9970   0.8318  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8977, F1=0.9069, Normal Recall=0.7984, Normal Precision=0.9962, Attack Recall=0.9970, Attack Precision=0.8318

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
0.15       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546   <--
0.20       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.25       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.30       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.35       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.40       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.45       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.50       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.55       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.60       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.65       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.70       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.75       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.80       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8182, F1=0.5232, Normal Recall=0.7983, Normal Precision=0.9996, Attack Recall=0.9973, Attack Precision=0.3546

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
0.15       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530   <--
0.20       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.25       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.30       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.35       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.40       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.45       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.50       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.55       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.60       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.65       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.70       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.75       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.80       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8382, F1=0.7114, Normal Recall=0.7985, Normal Precision=0.9991, Attack Recall=0.9971, Attack Precision=0.5530

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
0.15       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791   <--
0.20       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.25       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.30       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.35       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.40       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.45       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.50       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.55       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.60       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.65       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.70       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.75       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.80       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8578, F1=0.8079, Normal Recall=0.7981, Normal Precision=0.9985, Attack Recall=0.9971, Attack Precision=0.6791

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
0.15       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667   <--
0.20       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.25       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.30       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.35       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.40       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.45       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.50       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.55       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.60       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.65       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.70       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.75       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.80       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8775, F1=0.8668, Normal Recall=0.7977, Normal Precision=0.9976, Attack Recall=0.9971, Attack Precision=0.7667

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
0.15       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309   <--
0.20       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.25       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.30       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.35       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.40       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.45       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.50       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.55       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.60       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.65       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.70       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.75       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.80       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8971, F1=0.9065, Normal Recall=0.7971, Normal Precision=0.9964, Attack Recall=0.9971, Attack Precision=0.8309

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
0.15       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546   <--
0.20       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.25       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.30       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.35       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.40       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.45       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.50       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.55       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.60       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.65       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.70       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.75       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
0.80       0.8182   0.5232   0.7983   0.9996   0.9973   0.3546  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8182, F1=0.5232, Normal Recall=0.7983, Normal Precision=0.9996, Attack Recall=0.9973, Attack Precision=0.3546

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
0.15       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530   <--
0.20       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.25       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.30       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.35       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.40       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.45       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.50       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.55       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.60       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.65       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.70       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.75       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
0.80       0.8382   0.7114   0.7985   0.9991   0.9971   0.5530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8382, F1=0.7114, Normal Recall=0.7985, Normal Precision=0.9991, Attack Recall=0.9971, Attack Precision=0.5530

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
0.15       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791   <--
0.20       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.25       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.30       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.35       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.40       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.45       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.50       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.55       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.60       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.65       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.70       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.75       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
0.80       0.8578   0.8079   0.7981   0.9985   0.9971   0.6791  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8578, F1=0.8079, Normal Recall=0.7981, Normal Precision=0.9985, Attack Recall=0.9971, Attack Precision=0.6791

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
0.15       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667   <--
0.20       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.25       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.30       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.35       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.40       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.45       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.50       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.55       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.60       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.65       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.70       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.75       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
0.80       0.8775   0.8668   0.7977   0.9976   0.9971   0.7667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8775, F1=0.8668, Normal Recall=0.7977, Normal Precision=0.9976, Attack Recall=0.9971, Attack Precision=0.7667

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
0.15       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309   <--
0.20       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.25       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.30       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.35       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.40       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.45       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.50       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.55       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.60       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.65       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.70       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.75       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
0.80       0.8971   0.9065   0.7971   0.9964   0.9971   0.8309  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8971, F1=0.9065, Normal Recall=0.7971, Normal Precision=0.9964, Attack Recall=0.9971, Attack Precision=0.8309

```

