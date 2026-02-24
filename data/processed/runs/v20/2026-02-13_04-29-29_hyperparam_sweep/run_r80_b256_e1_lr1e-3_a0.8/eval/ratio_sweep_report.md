# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-18 16:32:02 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6099 | 0.6472 | 0.6861 | 0.7257 | 0.7653 | 0.8023 | 0.8417 | 0.8821 | 0.9181 | 0.9588 | 0.9982 |
| QAT+Prune only | 0.4243 | 0.4808 | 0.5376 | 0.5956 | 0.6543 | 0.7104 | 0.7682 | 0.8257 | 0.8831 | 0.9411 | 0.9983 |
| QAT+PTQ | 0.4245 | 0.4810 | 0.5377 | 0.5957 | 0.6543 | 0.7105 | 0.7684 | 0.8258 | 0.8830 | 0.9411 | 0.9983 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4245 | 0.4810 | 0.5377 | 0.5957 | 0.6543 | 0.7105 | 0.7684 | 0.8258 | 0.8830 | 0.9411 | 0.9983 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3614 | 0.5598 | 0.6858 | 0.7728 | 0.8347 | 0.8832 | 0.9222 | 0.9512 | 0.9776 | 0.9991 |
| QAT+Prune only | 0.0000 | 0.2778 | 0.4634 | 0.5970 | 0.6979 | 0.7751 | 0.8379 | 0.8891 | 0.9318 | 0.9683 | 0.9992 |
| QAT+PTQ | 0.0000 | 0.2778 | 0.4634 | 0.5970 | 0.6979 | 0.7752 | 0.8380 | 0.8892 | 0.9318 | 0.9682 | 0.9992 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2778 | 0.4634 | 0.5970 | 0.6979 | 0.7752 | 0.8380 | 0.8892 | 0.9318 | 0.9682 | 0.9992 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6099 | 0.6082 | 0.6080 | 0.6089 | 0.6100 | 0.6065 | 0.6069 | 0.6112 | 0.5979 | 0.6046 | 0.0000 |
| QAT+Prune only | 0.4243 | 0.4233 | 0.4224 | 0.4230 | 0.4249 | 0.4225 | 0.4229 | 0.4228 | 0.4219 | 0.4260 | 0.0000 |
| QAT+PTQ | 0.4245 | 0.4235 | 0.4225 | 0.4232 | 0.4250 | 0.4226 | 0.4235 | 0.4232 | 0.4218 | 0.4255 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4245 | 0.4235 | 0.4225 | 0.4232 | 0.4250 | 0.4226 | 0.4235 | 0.4232 | 0.4218 | 0.4255 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6099 | 0.0000 | 0.0000 | 0.0000 | 0.6099 | 1.0000 |
| 90 | 10 | 299,940 | 0.6472 | 0.2206 | 0.9981 | 0.3614 | 0.6082 | 0.9997 |
| 80 | 20 | 291,350 | 0.6861 | 0.3890 | 0.9982 | 0.5598 | 0.6080 | 0.9993 |
| 70 | 30 | 194,230 | 0.7257 | 0.5224 | 0.9982 | 0.6858 | 0.6089 | 0.9987 |
| 60 | 40 | 145,675 | 0.7653 | 0.6305 | 0.9982 | 0.7728 | 0.6100 | 0.9980 |
| 50 | 50 | 116,540 | 0.8023 | 0.7172 | 0.9982 | 0.8347 | 0.6065 | 0.9970 |
| 40 | 60 | 97,115 | 0.8417 | 0.7920 | 0.9982 | 0.8832 | 0.6069 | 0.9955 |
| 30 | 70 | 83,240 | 0.8821 | 0.8570 | 0.9982 | 0.9222 | 0.6112 | 0.9931 |
| 20 | 80 | 72,835 | 0.9181 | 0.9085 | 0.9982 | 0.9512 | 0.5979 | 0.9880 |
| 10 | 90 | 64,740 | 0.9588 | 0.9578 | 0.9982 | 0.9776 | 0.6046 | 0.9736 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4243 | 0.0000 | 0.0000 | 0.0000 | 0.4243 | 1.0000 |
| 90 | 10 | 299,940 | 0.4808 | 0.1613 | 0.9983 | 0.2778 | 0.4233 | 0.9996 |
| 80 | 20 | 291,350 | 0.5376 | 0.3017 | 0.9983 | 0.4634 | 0.4224 | 0.9990 |
| 70 | 30 | 194,230 | 0.5956 | 0.4258 | 0.9983 | 0.5970 | 0.4230 | 0.9983 |
| 60 | 40 | 145,675 | 0.6543 | 0.5365 | 0.9983 | 0.6979 | 0.4249 | 0.9974 |
| 50 | 50 | 116,540 | 0.7104 | 0.6335 | 0.9983 | 0.7751 | 0.4225 | 0.9961 |
| 40 | 60 | 97,115 | 0.7682 | 0.7218 | 0.9983 | 0.8379 | 0.4229 | 0.9941 |
| 30 | 70 | 83,240 | 0.8257 | 0.8014 | 0.9983 | 0.8891 | 0.4228 | 0.9909 |
| 20 | 80 | 72,835 | 0.8831 | 0.8735 | 0.9983 | 0.9318 | 0.4219 | 0.9845 |
| 10 | 90 | 64,740 | 0.9411 | 0.9400 | 0.9983 | 0.9683 | 0.4260 | 0.9660 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9992 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4245 | 0.0000 | 0.0000 | 0.0000 | 0.4245 | 1.0000 |
| 90 | 10 | 299,940 | 0.4810 | 0.1614 | 0.9983 | 0.2778 | 0.4235 | 0.9996 |
| 80 | 20 | 291,350 | 0.5377 | 0.3018 | 0.9983 | 0.4634 | 0.4225 | 0.9990 |
| 70 | 30 | 194,230 | 0.5957 | 0.4259 | 0.9983 | 0.5970 | 0.4232 | 0.9983 |
| 60 | 40 | 145,675 | 0.6543 | 0.5365 | 0.9983 | 0.6979 | 0.4250 | 0.9974 |
| 50 | 50 | 116,540 | 0.7105 | 0.6336 | 0.9983 | 0.7752 | 0.4226 | 0.9961 |
| 40 | 60 | 97,115 | 0.7684 | 0.7220 | 0.9983 | 0.8380 | 0.4235 | 0.9941 |
| 30 | 70 | 83,240 | 0.8258 | 0.8015 | 0.9983 | 0.8892 | 0.4232 | 0.9909 |
| 20 | 80 | 72,835 | 0.8830 | 0.8735 | 0.9983 | 0.9318 | 0.4218 | 0.9845 |
| 10 | 90 | 64,740 | 0.9411 | 0.9399 | 0.9983 | 0.9682 | 0.4255 | 0.9660 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9992 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4245 | 0.0000 | 0.0000 | 0.0000 | 0.4245 | 1.0000 |
| 90 | 10 | 299,940 | 0.4810 | 0.1614 | 0.9983 | 0.2778 | 0.4235 | 0.9996 |
| 80 | 20 | 291,350 | 0.5377 | 0.3018 | 0.9983 | 0.4634 | 0.4225 | 0.9990 |
| 70 | 30 | 194,230 | 0.5957 | 0.4259 | 0.9983 | 0.5970 | 0.4232 | 0.9983 |
| 60 | 40 | 145,675 | 0.6543 | 0.5365 | 0.9983 | 0.6979 | 0.4250 | 0.9974 |
| 50 | 50 | 116,540 | 0.7105 | 0.6336 | 0.9983 | 0.7752 | 0.4226 | 0.9961 |
| 40 | 60 | 97,115 | 0.7684 | 0.7220 | 0.9983 | 0.8380 | 0.4235 | 0.9941 |
| 30 | 70 | 83,240 | 0.8258 | 0.8015 | 0.9983 | 0.8892 | 0.4232 | 0.9909 |
| 20 | 80 | 72,835 | 0.8830 | 0.8735 | 0.9983 | 0.9318 | 0.4218 | 0.9845 |
| 10 | 90 | 64,740 | 0.9411 | 0.9399 | 0.9983 | 0.9682 | 0.4255 | 0.9660 |
| 0 | 100 | 58,270 | 0.9983 | 1.0000 | 0.9983 | 0.9992 | 0.0000 | 0.0000 |


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
0.15       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207   <--
0.20       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.25       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.30       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.35       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.40       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.45       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.50       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.55       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.60       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.65       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.70       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.75       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
0.80       0.6472   0.3614   0.6082   0.9997   0.9983   0.2207  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6472, F1=0.3614, Normal Recall=0.6082, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2207

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
0.15       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894   <--
0.20       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.25       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.30       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.35       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.40       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.45       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.50       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.55       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.60       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.65       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.70       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.75       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
0.80       0.6866   0.5603   0.6087   0.9993   0.9982   0.3894  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6866, F1=0.5603, Normal Recall=0.6087, Normal Precision=0.9993, Attack Recall=0.9982, Attack Precision=0.3894

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
0.15       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233   <--
0.20       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.25       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.30       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.35       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.40       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.45       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.50       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.55       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.60       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.65       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.70       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.75       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
0.80       0.7267   0.6866   0.6103   0.9987   0.9982   0.5233  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7267, F1=0.6866, Normal Recall=0.6103, Normal Precision=0.9987, Attack Recall=0.9982, Attack Precision=0.5233

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
0.15       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308   <--
0.20       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.25       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.30       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.35       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.40       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.45       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.50       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.55       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.60       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.65       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.70       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.75       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
0.80       0.7656   0.7731   0.6106   0.9980   0.9982   0.6308  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7656, F1=0.7731, Normal Recall=0.6106, Normal Precision=0.9980, Attack Recall=0.9982, Attack Precision=0.6308

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
0.15       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196   <--
0.20       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.25       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.30       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.35       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.40       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.45       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.50       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.55       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.60       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.65       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.70       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.75       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
0.80       0.8046   0.8363   0.6110   0.9970   0.9982   0.7196  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8046, F1=0.8363, Normal Recall=0.6110, Normal Precision=0.9970, Attack Recall=0.9982, Attack Precision=0.7196

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
0.15       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613   <--
0.20       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.25       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.30       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.35       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.40       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.45       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.50       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.55       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.60       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.65       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.70       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.75       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
0.80       0.4808   0.2778   0.4233   0.9996   0.9984   0.1613  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4808, F1=0.2778, Normal Recall=0.4233, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1613

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
0.15       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023   <--
0.20       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.25       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.30       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.35       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.40       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.45       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.50       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.55       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.60       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.65       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.70       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.75       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
0.80       0.5389   0.4641   0.4241   0.9990   0.9983   0.3023  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5389, F1=0.4641, Normal Recall=0.4241, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.3023

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
0.15       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263   <--
0.20       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.25       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.30       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.35       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.40       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.45       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.50       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.55       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.60       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.65       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.70       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.75       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
0.80       0.5964   0.5975   0.4242   0.9983   0.9983   0.4263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5964, F1=0.5975, Normal Recall=0.4242, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.4263

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
0.15       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355   <--
0.20       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.25       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.30       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.35       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.40       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.45       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.50       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.55       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.60       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.65       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.70       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.75       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
0.80       0.6530   0.6971   0.4227   0.9974   0.9983   0.5355  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6530, F1=0.6971, Normal Recall=0.4227, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.5355

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
0.15       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334   <--
0.20       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.25       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.30       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.35       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.40       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.45       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.50       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.55       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.60       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.65       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.70       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.75       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
0.80       0.7102   0.7750   0.4221   0.9961   0.9983   0.6334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7102, F1=0.7750, Normal Recall=0.4221, Normal Precision=0.9961, Attack Recall=0.9983, Attack Precision=0.6334

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
0.15       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614   <--
0.20       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.25       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.30       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.35       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.40       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.45       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.50       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.55       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.60       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.65       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.70       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.75       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.80       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4810, F1=0.2779, Normal Recall=0.4235, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1614

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
0.15       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024   <--
0.20       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.25       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.30       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.35       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.40       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.45       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.50       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.55       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.60       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.65       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.70       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.75       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.80       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5391, F1=0.4642, Normal Recall=0.4243, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.3024

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
0.15       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264   <--
0.20       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.25       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.30       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.35       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.40       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.45       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.50       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.55       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.60       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.65       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.70       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.75       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.80       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5966, F1=0.5976, Normal Recall=0.4244, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.4264

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
0.15       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356   <--
0.20       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.25       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.30       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.35       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.40       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.45       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.50       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.55       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.60       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.65       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.70       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.75       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.80       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6531, F1=0.6972, Normal Recall=0.4230, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.5356

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
0.15       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334   <--
0.20       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.25       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.30       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.35       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.40       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.45       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.50       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.55       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.60       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.65       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.70       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.75       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.80       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7103, F1=0.7751, Normal Recall=0.4222, Normal Precision=0.9961, Attack Recall=0.9983, Attack Precision=0.6334

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
0.15       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614   <--
0.20       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.25       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.30       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.35       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.40       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.45       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.50       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.55       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.60       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.65       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.70       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.75       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
0.80       0.4810   0.2779   0.4235   0.9996   0.9984   0.1614  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4810, F1=0.2779, Normal Recall=0.4235, Normal Precision=0.9996, Attack Recall=0.9984, Attack Precision=0.1614

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
0.15       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024   <--
0.20       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.25       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.30       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.35       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.40       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.45       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.50       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.55       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.60       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.65       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.70       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.75       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
0.80       0.5391   0.4642   0.4243   0.9990   0.9983   0.3024  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5391, F1=0.4642, Normal Recall=0.4243, Normal Precision=0.9990, Attack Recall=0.9983, Attack Precision=0.3024

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
0.15       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264   <--
0.20       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.25       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.30       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.35       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.40       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.45       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.50       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.55       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.60       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.65       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.70       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.75       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
0.80       0.5966   0.5976   0.4244   0.9983   0.9983   0.4264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5966, F1=0.5976, Normal Recall=0.4244, Normal Precision=0.9983, Attack Recall=0.9983, Attack Precision=0.4264

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
0.15       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356   <--
0.20       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.25       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.30       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.35       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.40       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.45       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.50       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.55       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.60       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.65       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.70       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.75       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
0.80       0.6531   0.6972   0.4230   0.9974   0.9983   0.5356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6531, F1=0.6972, Normal Recall=0.4230, Normal Precision=0.9974, Attack Recall=0.9983, Attack Precision=0.5356

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
0.15       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334   <--
0.20       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.25       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.30       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.35       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.40       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.45       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.50       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.55       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.60       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.65       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.70       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.75       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
0.80       0.7103   0.7751   0.4222   0.9961   0.9983   0.6334  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7103, F1=0.7751, Normal Recall=0.4222, Normal Precision=0.9961, Attack Recall=0.9983, Attack Precision=0.6334

```

