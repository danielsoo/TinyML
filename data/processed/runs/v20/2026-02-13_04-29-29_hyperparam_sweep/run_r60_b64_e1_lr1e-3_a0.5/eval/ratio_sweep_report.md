# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-13 12:19:46 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 1 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2879 | 0.3574 | 0.4247 | 0.4915 | 0.5608 | 0.6274 | 0.6963 | 0.7628 | 0.8305 | 0.8985 | 0.9656 |
| QAT+Prune only | 0.7976 | 0.8184 | 0.8377 | 0.8579 | 0.8776 | 0.8950 | 0.9171 | 0.9360 | 0.9554 | 0.9750 | 0.9949 |
| QAT+PTQ | 0.7966 | 0.8175 | 0.8368 | 0.8571 | 0.8769 | 0.8944 | 0.9164 | 0.9358 | 0.9551 | 0.9749 | 0.9949 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7966 | 0.8175 | 0.8368 | 0.8571 | 0.8769 | 0.8944 | 0.9164 | 0.9358 | 0.9551 | 0.9749 | 0.9949 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2310 | 0.4017 | 0.5325 | 0.6375 | 0.7216 | 0.7923 | 0.8507 | 0.9011 | 0.9448 | 0.9825 |
| QAT+Prune only | 0.0000 | 0.5230 | 0.7103 | 0.8077 | 0.8667 | 0.9046 | 0.9351 | 0.9561 | 0.9728 | 0.9862 | 0.9975 |
| QAT+PTQ | 0.0000 | 0.5217 | 0.7092 | 0.8069 | 0.8660 | 0.9040 | 0.9346 | 0.9559 | 0.9726 | 0.9862 | 0.9975 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5217 | 0.7092 | 0.8069 | 0.8660 | 0.9040 | 0.9346 | 0.9559 | 0.9726 | 0.9862 | 0.9975 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2879 | 0.2899 | 0.2895 | 0.2883 | 0.2910 | 0.2893 | 0.2923 | 0.2897 | 0.2902 | 0.2949 | 0.0000 |
| QAT+Prune only | 0.7976 | 0.7988 | 0.7983 | 0.7991 | 0.7994 | 0.7952 | 0.8003 | 0.7986 | 0.7975 | 0.7956 | 0.0000 |
| QAT+PTQ | 0.7966 | 0.7977 | 0.7973 | 0.7981 | 0.7982 | 0.7939 | 0.7986 | 0.7978 | 0.7958 | 0.7946 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7966 | 0.7977 | 0.7973 | 0.7981 | 0.7982 | 0.7939 | 0.7986 | 0.7978 | 0.7958 | 0.7946 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2879 | 0.0000 | 0.0000 | 0.0000 | 0.2879 | 1.0000 |
| 90 | 10 | 299,940 | 0.3574 | 0.1312 | 0.9650 | 0.2310 | 0.2899 | 0.9868 |
| 80 | 20 | 291,350 | 0.4247 | 0.2536 | 0.9656 | 0.4017 | 0.2895 | 0.9711 |
| 70 | 30 | 194,230 | 0.4915 | 0.3677 | 0.9656 | 0.5325 | 0.2883 | 0.9513 |
| 60 | 40 | 145,675 | 0.5608 | 0.4759 | 0.9656 | 0.6375 | 0.2910 | 0.9269 |
| 50 | 50 | 116,540 | 0.6274 | 0.5760 | 0.9656 | 0.7216 | 0.2893 | 0.8937 |
| 40 | 60 | 97,115 | 0.6963 | 0.6718 | 0.9656 | 0.7923 | 0.2923 | 0.8499 |
| 30 | 70 | 83,240 | 0.7628 | 0.7603 | 0.9656 | 0.8507 | 0.2897 | 0.7830 |
| 20 | 80 | 72,835 | 0.8305 | 0.8448 | 0.9656 | 0.9011 | 0.2902 | 0.6783 |
| 10 | 90 | 64,740 | 0.8985 | 0.9250 | 0.9656 | 0.9448 | 0.2949 | 0.4879 |
| 0 | 100 | 58,270 | 0.9656 | 1.0000 | 0.9656 | 0.9825 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7976 | 0.0000 | 0.0000 | 0.0000 | 0.7976 | 1.0000 |
| 90 | 10 | 299,940 | 0.8184 | 0.3547 | 0.9955 | 0.5230 | 0.7988 | 0.9994 |
| 80 | 20 | 291,350 | 0.8377 | 0.5522 | 0.9949 | 0.7103 | 0.7983 | 0.9984 |
| 70 | 30 | 194,230 | 0.8579 | 0.6798 | 0.9949 | 0.8077 | 0.7991 | 0.9973 |
| 60 | 40 | 145,675 | 0.8776 | 0.7678 | 0.9949 | 0.8667 | 0.7994 | 0.9958 |
| 50 | 50 | 116,540 | 0.8950 | 0.8293 | 0.9949 | 0.9046 | 0.7952 | 0.9937 |
| 40 | 60 | 97,115 | 0.9171 | 0.8820 | 0.9949 | 0.9351 | 0.8003 | 0.9906 |
| 30 | 70 | 83,240 | 0.9360 | 0.9202 | 0.9949 | 0.9561 | 0.7986 | 0.9854 |
| 20 | 80 | 72,835 | 0.9554 | 0.9516 | 0.9949 | 0.9728 | 0.7975 | 0.9752 |
| 10 | 90 | 64,740 | 0.9750 | 0.9777 | 0.9949 | 0.9862 | 0.7956 | 0.9458 |
| 0 | 100 | 58,270 | 0.9949 | 1.0000 | 0.9949 | 0.9975 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7966 | 0.0000 | 0.0000 | 0.0000 | 0.7966 | 1.0000 |
| 90 | 10 | 299,940 | 0.8175 | 0.3535 | 0.9955 | 0.5217 | 0.7977 | 0.9994 |
| 80 | 20 | 291,350 | 0.8368 | 0.5510 | 0.9949 | 0.7092 | 0.7973 | 0.9984 |
| 70 | 30 | 194,230 | 0.8571 | 0.6786 | 0.9949 | 0.8069 | 0.7981 | 0.9973 |
| 60 | 40 | 145,675 | 0.8769 | 0.7667 | 0.9949 | 0.8660 | 0.7982 | 0.9958 |
| 50 | 50 | 116,540 | 0.8944 | 0.8284 | 0.9949 | 0.9040 | 0.7939 | 0.9936 |
| 40 | 60 | 97,115 | 0.9164 | 0.8811 | 0.9949 | 0.9346 | 0.7986 | 0.9905 |
| 30 | 70 | 83,240 | 0.9358 | 0.9199 | 0.9949 | 0.9559 | 0.7978 | 0.9854 |
| 20 | 80 | 72,835 | 0.9551 | 0.9512 | 0.9949 | 0.9726 | 0.7958 | 0.9751 |
| 10 | 90 | 64,740 | 0.9749 | 0.9776 | 0.9949 | 0.9862 | 0.7946 | 0.9456 |
| 0 | 100 | 58,270 | 0.9949 | 1.0000 | 0.9949 | 0.9975 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7966 | 0.0000 | 0.0000 | 0.0000 | 0.7966 | 1.0000 |
| 90 | 10 | 299,940 | 0.8175 | 0.3535 | 0.9955 | 0.5217 | 0.7977 | 0.9994 |
| 80 | 20 | 291,350 | 0.8368 | 0.5510 | 0.9949 | 0.7092 | 0.7973 | 0.9984 |
| 70 | 30 | 194,230 | 0.8571 | 0.6786 | 0.9949 | 0.8069 | 0.7981 | 0.9973 |
| 60 | 40 | 145,675 | 0.8769 | 0.7667 | 0.9949 | 0.8660 | 0.7982 | 0.9958 |
| 50 | 50 | 116,540 | 0.8944 | 0.8284 | 0.9949 | 0.9040 | 0.7939 | 0.9936 |
| 40 | 60 | 97,115 | 0.9164 | 0.8811 | 0.9949 | 0.9346 | 0.7986 | 0.9905 |
| 30 | 70 | 83,240 | 0.9358 | 0.9199 | 0.9949 | 0.9559 | 0.7978 | 0.9854 |
| 20 | 80 | 72,835 | 0.9551 | 0.9512 | 0.9949 | 0.9726 | 0.7958 | 0.9751 |
| 10 | 90 | 64,740 | 0.9749 | 0.9776 | 0.9949 | 0.9862 | 0.7946 | 0.9456 |
| 0 | 100 | 58,270 | 0.9949 | 1.0000 | 0.9949 | 0.9975 | 0.0000 | 0.0000 |


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
0.15       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313   <--
0.20       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.25       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.30       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.35       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.40       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.45       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.50       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.55       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.60       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.65       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.70       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.75       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
0.80       0.3575   0.2312   0.2899   0.9872   0.9661   0.1313  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3575, F1=0.2312, Normal Recall=0.2899, Normal Precision=0.9872, Attack Recall=0.9661, Attack Precision=0.1313

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
0.15       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536   <--
0.20       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.25       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.30       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.35       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.40       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.45       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.50       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.55       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.60       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.65       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.70       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.75       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
0.80       0.4247   0.4017   0.2895   0.9711   0.9656   0.2536  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4247, F1=0.4017, Normal Recall=0.2895, Normal Precision=0.9711, Attack Recall=0.9656, Attack Precision=0.2536

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
0.15       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680   <--
0.20       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.25       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.30       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.35       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.40       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.45       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.50       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.55       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.60       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.65       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.70       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.75       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
0.80       0.4922   0.5329   0.2893   0.9515   0.9656   0.3680  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4922, F1=0.5329, Normal Recall=0.2893, Normal Precision=0.9515, Attack Recall=0.9656, Attack Precision=0.3680

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
0.15       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751   <--
0.20       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.25       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.30       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.35       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.40       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.45       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.50       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.55       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.60       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.65       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.70       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.75       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
0.80       0.5595   0.6368   0.2887   0.9264   0.9656   0.4751  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5595, F1=0.6368, Normal Recall=0.2887, Normal Precision=0.9264, Attack Recall=0.9656, Attack Precision=0.4751

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
0.15       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759   <--
0.20       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.25       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.30       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.35       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.40       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.45       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.50       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.55       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.60       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.65       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.70       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.75       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
0.80       0.6272   0.7215   0.2889   0.8936   0.9656   0.5759  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6272, F1=0.7215, Normal Recall=0.2889, Normal Precision=0.8936, Attack Recall=0.9656, Attack Precision=0.5759

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
0.15       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546   <--
0.20       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.25       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.30       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.35       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.40       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.45       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.50       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.55       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.60       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.65       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.70       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.75       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
0.80       0.8184   0.5229   0.7988   0.9993   0.9953   0.3546  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8184, F1=0.5229, Normal Recall=0.7988, Normal Precision=0.9993, Attack Recall=0.9953, Attack Precision=0.3546

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
0.15       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534   <--
0.20       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.25       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.30       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.35       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.40       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.45       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.50       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.55       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.60       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.65       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.70       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.75       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
0.80       0.8384   0.7112   0.7993   0.9984   0.9949   0.5534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8384, F1=0.7112, Normal Recall=0.7993, Normal Precision=0.9984, Attack Recall=0.9949, Attack Precision=0.5534

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
0.15       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792   <--
0.20       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.25       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.30       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.35       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.40       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.45       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.50       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.55       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.60       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.65       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.70       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.75       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
0.80       0.8575   0.8073   0.7986   0.9973   0.9949   0.6792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8575, F1=0.8073, Normal Recall=0.7986, Normal Precision=0.9973, Attack Recall=0.9949, Attack Precision=0.6792

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
0.15       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664   <--
0.20       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.25       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.30       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.35       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.40       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.45       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.50       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.55       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.60       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.65       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.70       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.75       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
0.80       0.8767   0.8658   0.7978   0.9958   0.9949   0.7664  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8767, F1=0.8658, Normal Recall=0.7978, Normal Precision=0.9958, Attack Recall=0.9949, Attack Precision=0.7664

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
0.15       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300   <--
0.20       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.25       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.30       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.35       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.40       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.45       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.50       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.55       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.60       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.65       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.70       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.75       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
0.80       0.8955   0.9050   0.7962   0.9937   0.9949   0.8300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8955, F1=0.9050, Normal Recall=0.7962, Normal Precision=0.9937, Attack Recall=0.9949, Attack Precision=0.8300

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
0.15       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534   <--
0.20       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.25       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.30       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.35       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.40       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.45       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.50       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.55       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.60       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.65       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.70       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.75       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.80       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8174, F1=0.5216, Normal Recall=0.7977, Normal Precision=0.9993, Attack Recall=0.9952, Attack Precision=0.3534

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
0.15       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521   <--
0.20       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.25       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.30       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.35       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.40       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.45       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.50       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.55       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.60       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.65       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.70       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.75       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.80       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.7101, Normal Recall=0.7982, Normal Precision=0.9984, Attack Recall=0.9949, Attack Precision=0.5521

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
0.15       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780   <--
0.20       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.25       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.30       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.35       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.40       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.45       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.50       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.55       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.60       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.65       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.70       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.75       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.80       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8567, F1=0.8064, Normal Recall=0.7975, Normal Precision=0.9973, Attack Recall=0.9949, Attack Precision=0.6780

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
0.15       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655   <--
0.20       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.25       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.30       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.35       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.40       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.45       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.50       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.55       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.60       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.65       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.70       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.75       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.80       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8760, F1=0.8652, Normal Recall=0.7968, Normal Precision=0.9958, Attack Recall=0.9949, Attack Precision=0.7655

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
0.15       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292   <--
0.20       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.25       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.30       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.35       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.40       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.45       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.50       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.55       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.60       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.65       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.70       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.75       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.80       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8950, F1=0.9045, Normal Recall=0.7951, Normal Precision=0.9937, Attack Recall=0.9949, Attack Precision=0.8292

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
0.15       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534   <--
0.20       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.25       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.30       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.35       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.40       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.45       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.50       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.55       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.60       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.65       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.70       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.75       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
0.80       0.8174   0.5216   0.7977   0.9993   0.9952   0.3534  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8174, F1=0.5216, Normal Recall=0.7977, Normal Precision=0.9993, Attack Recall=0.9952, Attack Precision=0.3534

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
0.15       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521   <--
0.20       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.25       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.30       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.35       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.40       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.45       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.50       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.55       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.60       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.65       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.70       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.75       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
0.80       0.8376   0.7101   0.7982   0.9984   0.9949   0.5521  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8376, F1=0.7101, Normal Recall=0.7982, Normal Precision=0.9984, Attack Recall=0.9949, Attack Precision=0.5521

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
0.15       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780   <--
0.20       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.25       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.30       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.35       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.40       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.45       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.50       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.55       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.60       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.65       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.70       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.75       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
0.80       0.8567   0.8064   0.7975   0.9973   0.9949   0.6780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8567, F1=0.8064, Normal Recall=0.7975, Normal Precision=0.9973, Attack Recall=0.9949, Attack Precision=0.6780

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
0.15       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655   <--
0.20       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.25       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.30       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.35       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.40       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.45       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.50       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.55       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.60       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.65       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.70       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.75       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
0.80       0.8760   0.8652   0.7968   0.9958   0.9949   0.7655  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8760, F1=0.8652, Normal Recall=0.7968, Normal Precision=0.9958, Attack Recall=0.9949, Attack Precision=0.7655

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
0.15       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292   <--
0.20       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.25       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.30       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.35       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.40       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.45       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.50       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.55       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.60       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.65       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.70       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.75       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
0.80       0.8950   0.9045   0.7951   0.9937   0.9949   0.8292  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8950, F1=0.9045, Normal Recall=0.7951, Normal Precision=0.9937, Attack Recall=0.9949, Attack Precision=0.8292

```

