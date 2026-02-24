# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-16 11:51:20 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0119 | 0.1108 | 0.2094 | 0.3084 | 0.4070 | 0.5059 | 0.6046 | 0.7038 | 0.8022 | 0.9005 | 0.9995 |
| QAT+Prune only | 0.6840 | 0.7148 | 0.7457 | 0.7779 | 0.8109 | 0.8393 | 0.8724 | 0.9022 | 0.9345 | 0.9656 | 0.9972 |
| QAT+PTQ | 0.6903 | 0.7207 | 0.7509 | 0.7826 | 0.8148 | 0.8426 | 0.8751 | 0.9043 | 0.9356 | 0.9661 | 0.9971 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6903 | 0.7207 | 0.7509 | 0.7826 | 0.8148 | 0.8426 | 0.8751 | 0.9043 | 0.9356 | 0.9661 | 0.9971 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1835 | 0.3359 | 0.4644 | 0.5742 | 0.6692 | 0.7521 | 0.8253 | 0.8899 | 0.9476 | 0.9998 |
| QAT+Prune only | 0.0000 | 0.4115 | 0.6107 | 0.7293 | 0.8084 | 0.8612 | 0.9037 | 0.9345 | 0.9605 | 0.9812 | 0.9986 |
| QAT+PTQ | 0.0000 | 0.4165 | 0.6156 | 0.7335 | 0.8116 | 0.8637 | 0.9055 | 0.9358 | 0.9612 | 0.9815 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4165 | 0.6156 | 0.7335 | 0.8116 | 0.8637 | 0.9055 | 0.9358 | 0.9612 | 0.9815 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0119 | 0.0121 | 0.0119 | 0.0122 | 0.0120 | 0.0122 | 0.0123 | 0.0137 | 0.0130 | 0.0091 | 0.0000 |
| QAT+Prune only | 0.6840 | 0.6834 | 0.6828 | 0.6839 | 0.6867 | 0.6813 | 0.6852 | 0.6804 | 0.6833 | 0.6810 | 0.0000 |
| QAT+PTQ | 0.6903 | 0.6900 | 0.6893 | 0.6907 | 0.6932 | 0.6881 | 0.6922 | 0.6876 | 0.6895 | 0.6867 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6903 | 0.6900 | 0.6893 | 0.6907 | 0.6932 | 0.6881 | 0.6922 | 0.6876 | 0.6895 | 0.6867 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0119 | 0.0000 | 0.0000 | 0.0000 | 0.0119 | 1.0000 |
| 90 | 10 | 299,940 | 0.1108 | 0.1011 | 0.9995 | 0.1835 | 0.0121 | 0.9954 |
| 80 | 20 | 291,350 | 0.2094 | 0.2018 | 0.9995 | 0.3359 | 0.0119 | 0.9896 |
| 70 | 30 | 194,230 | 0.3084 | 0.3025 | 0.9995 | 0.4644 | 0.0122 | 0.9828 |
| 60 | 40 | 145,675 | 0.4070 | 0.4028 | 0.9995 | 0.5742 | 0.0120 | 0.9731 |
| 50 | 50 | 116,540 | 0.5059 | 0.5029 | 0.9995 | 0.6692 | 0.0122 | 0.9608 |
| 40 | 60 | 97,115 | 0.6046 | 0.6028 | 0.9995 | 0.7521 | 0.0123 | 0.9427 |
| 30 | 70 | 83,240 | 0.7038 | 0.7028 | 0.9995 | 0.8253 | 0.0137 | 0.9220 |
| 20 | 80 | 72,835 | 0.8022 | 0.8020 | 0.9995 | 0.8899 | 0.0130 | 0.8676 |
| 10 | 90 | 64,740 | 0.9005 | 0.9008 | 0.9995 | 0.9476 | 0.0091 | 0.6705 |
| 0 | 100 | 58,270 | 0.9995 | 1.0000 | 0.9995 | 0.9998 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6840 | 0.0000 | 0.0000 | 0.0000 | 0.6840 | 1.0000 |
| 90 | 10 | 299,940 | 0.7148 | 0.2593 | 0.9972 | 0.4115 | 0.6834 | 0.9995 |
| 80 | 20 | 291,350 | 0.7457 | 0.4401 | 0.9972 | 0.6107 | 0.6828 | 0.9990 |
| 70 | 30 | 194,230 | 0.7779 | 0.5749 | 0.9972 | 0.7293 | 0.6839 | 0.9983 |
| 60 | 40 | 145,675 | 0.8109 | 0.6797 | 0.9972 | 0.8084 | 0.6867 | 0.9973 |
| 50 | 50 | 116,540 | 0.8393 | 0.7578 | 0.9972 | 0.8612 | 0.6813 | 0.9960 |
| 40 | 60 | 97,115 | 0.8724 | 0.8261 | 0.9972 | 0.9037 | 0.6852 | 0.9940 |
| 30 | 70 | 83,240 | 0.9022 | 0.8793 | 0.9972 | 0.9345 | 0.6804 | 0.9906 |
| 20 | 80 | 72,835 | 0.9345 | 0.9265 | 0.9972 | 0.9605 | 0.6833 | 0.9841 |
| 10 | 90 | 64,740 | 0.9656 | 0.9657 | 0.9972 | 0.9812 | 0.6810 | 0.9648 |
| 0 | 100 | 58,270 | 0.9972 | 1.0000 | 0.9972 | 0.9986 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6903 | 0.0000 | 0.0000 | 0.0000 | 0.6903 | 1.0000 |
| 90 | 10 | 299,940 | 0.7207 | 0.2633 | 0.9971 | 0.4165 | 0.6900 | 0.9995 |
| 80 | 20 | 291,350 | 0.7509 | 0.4452 | 0.9971 | 0.6156 | 0.6893 | 0.9990 |
| 70 | 30 | 194,230 | 0.7826 | 0.5801 | 0.9971 | 0.7335 | 0.6907 | 0.9982 |
| 60 | 40 | 145,675 | 0.8148 | 0.6842 | 0.9971 | 0.8116 | 0.6932 | 0.9973 |
| 50 | 50 | 116,540 | 0.8426 | 0.7617 | 0.9971 | 0.8637 | 0.6881 | 0.9959 |
| 40 | 60 | 97,115 | 0.8751 | 0.8293 | 0.9971 | 0.9055 | 0.6922 | 0.9938 |
| 30 | 70 | 83,240 | 0.9043 | 0.8816 | 0.9971 | 0.9358 | 0.6876 | 0.9904 |
| 20 | 80 | 72,835 | 0.9356 | 0.9278 | 0.9971 | 0.9612 | 0.6895 | 0.9836 |
| 10 | 90 | 64,740 | 0.9661 | 0.9663 | 0.9971 | 0.9815 | 0.6867 | 0.9638 |
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
| 100 | 0 | 100,000 | 0.6903 | 0.0000 | 0.0000 | 0.0000 | 0.6903 | 1.0000 |
| 90 | 10 | 299,940 | 0.7207 | 0.2633 | 0.9971 | 0.4165 | 0.6900 | 0.9995 |
| 80 | 20 | 291,350 | 0.7509 | 0.4452 | 0.9971 | 0.6156 | 0.6893 | 0.9990 |
| 70 | 30 | 194,230 | 0.7826 | 0.5801 | 0.9971 | 0.7335 | 0.6907 | 0.9982 |
| 60 | 40 | 145,675 | 0.8148 | 0.6842 | 0.9971 | 0.8116 | 0.6932 | 0.9973 |
| 50 | 50 | 116,540 | 0.8426 | 0.7617 | 0.9971 | 0.8637 | 0.6881 | 0.9959 |
| 40 | 60 | 97,115 | 0.8751 | 0.8293 | 0.9971 | 0.9055 | 0.6922 | 0.9938 |
| 30 | 70 | 83,240 | 0.9043 | 0.8816 | 0.9971 | 0.9358 | 0.6876 | 0.9904 |
| 20 | 80 | 72,835 | 0.9356 | 0.9278 | 0.9971 | 0.9612 | 0.6895 | 0.9836 |
| 10 | 90 | 64,740 | 0.9661 | 0.9663 | 0.9971 | 0.9815 | 0.6867 | 0.9638 |
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
0.15       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011   <--
0.20       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.25       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.30       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.35       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.40       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.45       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.50       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.55       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.60       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.65       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.70       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.75       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
0.80       0.1108   0.1836   0.0121   0.9960   0.9996   0.1011  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1108, F1=0.1836, Normal Recall=0.0121, Normal Precision=0.9960, Attack Recall=0.9996, Attack Precision=0.1011

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
0.15       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019   <--
0.20       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.25       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.30       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.35       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.40       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.45       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.50       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.55       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.60       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.65       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.70       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.75       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
0.80       0.2095   0.3359   0.0120   0.9898   0.9995   0.2019  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2095, F1=0.3359, Normal Recall=0.0120, Normal Precision=0.9898, Attack Recall=0.9995, Attack Precision=0.2019

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
0.15       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024   <--
0.20       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.25       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.30       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.35       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.40       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.45       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.50       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.55       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.60       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.65       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.70       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.75       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
0.80       0.3083   0.4644   0.0120   0.9826   0.9995   0.3024  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3083, F1=0.4644, Normal Recall=0.0120, Normal Precision=0.9826, Attack Recall=0.9995, Attack Precision=0.3024

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
0.15       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028   <--
0.20       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.25       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.30       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.35       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.40       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.45       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.50       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.55       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.60       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.65       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.70       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.75       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
0.80       0.4070   0.5742   0.0120   0.9732   0.9995   0.4028  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4070, F1=0.5742, Normal Recall=0.0120, Normal Precision=0.9732, Attack Recall=0.9995, Attack Precision=0.4028

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
0.15       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030   <--
0.20       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.25       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.30       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.35       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.40       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.45       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.50       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.55       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.60       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.65       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.70       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.75       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
0.80       0.5059   0.6692   0.0123   0.9612   0.9995   0.5030  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5059, F1=0.6692, Normal Recall=0.0123, Normal Precision=0.9612, Attack Recall=0.9995, Attack Precision=0.5030

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
0.15       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.20       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.25       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.30       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.35       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.40       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.45       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.50       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.55       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594   <--
0.60       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.65       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.70       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.75       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
0.80       0.7149   0.4117   0.6835   0.9996   0.9976   0.2594  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7149, F1=0.4117, Normal Recall=0.6835, Normal Precision=0.9996, Attack Recall=0.9976, Attack Precision=0.2594

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
0.15       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.20       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.25       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.30       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.35       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.40       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.45       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.50       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.55       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410   <--
0.60       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.65       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.70       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.75       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
0.80       0.7467   0.6116   0.6840   0.9990   0.9972   0.4410  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7467, F1=0.6116, Normal Recall=0.6840, Normal Precision=0.9990, Attack Recall=0.9972, Attack Precision=0.4410

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
0.15       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.20       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.25       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.30       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.35       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.40       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.45       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.50       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.55       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751   <--
0.60       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.65       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.70       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.75       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
0.80       0.7781   0.7295   0.6842   0.9983   0.9972   0.5751  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.55
  At threshold 0.55: Accuracy=0.7781, F1=0.7295, Normal Recall=0.6842, Normal Precision=0.9983, Attack Recall=0.9972, Attack Precision=0.5751

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
0.15       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777   <--
0.20       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.25       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.30       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.35       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.40       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.45       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.50       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.55       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.60       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.65       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.70       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.75       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
0.80       0.8092   0.8070   0.6838   0.9973   0.9972   0.6777  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8092, F1=0.8070, Normal Recall=0.6838, Normal Precision=0.9973, Attack Recall=0.9972, Attack Precision=0.6777

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
0.15       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588   <--
0.20       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.25       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.30       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.35       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.40       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.45       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.50       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.55       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.60       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.65       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.70       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.75       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
0.80       0.8401   0.8618   0.6829   0.9960   0.9972   0.7588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8401, F1=0.8618, Normal Recall=0.6829, Normal Precision=0.9960, Attack Recall=0.9972, Attack Precision=0.7588

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
0.15       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633   <--
0.20       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.25       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.30       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.35       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.40       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.45       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.50       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.55       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.60       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.65       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.70       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.75       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.80       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7207, F1=0.4167, Normal Recall=0.6900, Normal Precision=0.9996, Attack Recall=0.9975, Attack Precision=0.2633

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
0.15       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461   <--
0.20       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.25       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.30       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.35       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.40       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.45       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.50       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.55       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.60       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.65       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.70       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.75       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.80       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.6165, Normal Recall=0.6905, Normal Precision=0.9990, Attack Recall=0.9971, Attack Precision=0.4461

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
0.15       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801   <--
0.20       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.25       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.30       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.35       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.40       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.45       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.50       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.55       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.60       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.65       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.70       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.75       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.80       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7826, F1=0.7335, Normal Recall=0.6907, Normal Precision=0.9982, Attack Recall=0.9971, Attack Precision=0.5801

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
0.15       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821   <--
0.20       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.25       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.30       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.35       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.40       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.45       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.50       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.55       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.60       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.65       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.70       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.75       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.80       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8130, F1=0.8101, Normal Recall=0.6902, Normal Precision=0.9972, Attack Recall=0.9971, Attack Precision=0.6821

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
0.15       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624   <--
0.20       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.25       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.30       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.35       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.40       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.45       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.50       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.55       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.60       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.65       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.70       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.75       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.80       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8432, F1=0.8641, Normal Recall=0.6893, Normal Precision=0.9959, Attack Recall=0.9971, Attack Precision=0.7624

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
0.15       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633   <--
0.20       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.25       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.30       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.35       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.40       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.45       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.50       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.55       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.60       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.65       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.70       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.75       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
0.80       0.7207   0.4167   0.6900   0.9996   0.9975   0.2633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7207, F1=0.4167, Normal Recall=0.6900, Normal Precision=0.9996, Attack Recall=0.9975, Attack Precision=0.2633

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
0.15       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461   <--
0.20       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.25       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.30       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.35       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.40       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.45       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.50       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.55       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.60       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.65       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.70       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.75       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
0.80       0.7518   0.6165   0.6905   0.9990   0.9971   0.4461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.6165, Normal Recall=0.6905, Normal Precision=0.9990, Attack Recall=0.9971, Attack Precision=0.4461

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
0.15       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801   <--
0.20       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.25       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.30       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.35       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.40       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.45       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.50       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.55       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.60       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.65       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.70       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.75       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
0.80       0.7826   0.7335   0.6907   0.9982   0.9971   0.5801  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7826, F1=0.7335, Normal Recall=0.6907, Normal Precision=0.9982, Attack Recall=0.9971, Attack Precision=0.5801

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
0.15       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821   <--
0.20       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.25       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.30       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.35       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.40       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.45       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.50       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.55       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.60       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.65       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.70       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.75       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
0.80       0.8130   0.8101   0.6902   0.9972   0.9971   0.6821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8130, F1=0.8101, Normal Recall=0.6902, Normal Precision=0.9972, Attack Recall=0.9971, Attack Precision=0.6821

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
0.15       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624   <--
0.20       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.25       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.30       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.35       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.40       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.45       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.50       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.55       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.60       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.65       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.70       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.75       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
0.80       0.8432   0.8641   0.6893   0.9959   0.9971   0.7624  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8432, F1=0.8641, Normal Recall=0.6893, Normal Precision=0.9959, Attack Recall=0.9971, Attack Precision=0.7624

```

