# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-20 20:42:16 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5270 | 0.5672 | 0.6042 | 0.6428 | 0.6795 | 0.7178 | 0.7549 | 0.7927 | 0.8305 | 0.8677 | 0.9053 |
| QAT+Prune only | 0.7447 | 0.7702 | 0.7953 | 0.8210 | 0.8458 | 0.8714 | 0.8971 | 0.9220 | 0.9469 | 0.9726 | 0.9980 |
| QAT+PTQ | 0.7428 | 0.7683 | 0.7936 | 0.8195 | 0.8445 | 0.8705 | 0.8963 | 0.9215 | 0.9465 | 0.9726 | 0.9982 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7428 | 0.7683 | 0.7936 | 0.8195 | 0.8445 | 0.8705 | 0.8963 | 0.9215 | 0.9465 | 0.9726 | 0.9982 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2951 | 0.4778 | 0.6033 | 0.6932 | 0.7624 | 0.8159 | 0.8594 | 0.8952 | 0.9249 | 0.9503 |
| QAT+Prune only | 0.0000 | 0.4649 | 0.6610 | 0.7699 | 0.8382 | 0.8859 | 0.9209 | 0.9471 | 0.9678 | 0.9850 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4628 | 0.6592 | 0.7684 | 0.8371 | 0.8852 | 0.9203 | 0.9468 | 0.9676 | 0.9850 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4628 | 0.6592 | 0.7684 | 0.8371 | 0.8852 | 0.9203 | 0.9468 | 0.9676 | 0.9850 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5270 | 0.5295 | 0.5289 | 0.5303 | 0.5290 | 0.5304 | 0.5293 | 0.5299 | 0.5313 | 0.5292 | 0.0000 |
| QAT+Prune only | 0.7447 | 0.7449 | 0.7446 | 0.7452 | 0.7444 | 0.7448 | 0.7456 | 0.7447 | 0.7426 | 0.7436 | 0.0000 |
| QAT+PTQ | 0.7428 | 0.7427 | 0.7424 | 0.7429 | 0.7421 | 0.7429 | 0.7434 | 0.7426 | 0.7397 | 0.7425 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7428 | 0.7427 | 0.7424 | 0.7429 | 0.7421 | 0.7429 | 0.7434 | 0.7426 | 0.7397 | 0.7425 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5270 | 0.0000 | 0.0000 | 0.0000 | 0.5270 | 1.0000 |
| 90 | 10 | 299,940 | 0.5672 | 0.1763 | 0.9061 | 0.2951 | 0.5295 | 0.9807 |
| 80 | 20 | 291,350 | 0.6042 | 0.3245 | 0.9053 | 0.4778 | 0.5289 | 0.9572 |
| 70 | 30 | 194,230 | 0.6428 | 0.4524 | 0.9053 | 0.6033 | 0.5303 | 0.9289 |
| 60 | 40 | 145,675 | 0.6795 | 0.5617 | 0.9053 | 0.6932 | 0.5290 | 0.8934 |
| 50 | 50 | 116,540 | 0.7178 | 0.6584 | 0.9053 | 0.7624 | 0.5304 | 0.8485 |
| 40 | 60 | 97,115 | 0.7549 | 0.7426 | 0.9053 | 0.8159 | 0.5293 | 0.7884 |
| 30 | 70 | 83,240 | 0.7927 | 0.8180 | 0.9053 | 0.8594 | 0.5299 | 0.7057 |
| 20 | 80 | 72,835 | 0.8305 | 0.8854 | 0.9053 | 0.8952 | 0.5313 | 0.5838 |
| 10 | 90 | 64,740 | 0.8677 | 0.9454 | 0.9053 | 0.9249 | 0.5292 | 0.3830 |
| 0 | 100 | 58,270 | 0.9053 | 1.0000 | 0.9053 | 0.9503 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7447 | 0.0000 | 0.0000 | 0.0000 | 0.7447 | 1.0000 |
| 90 | 10 | 299,940 | 0.7702 | 0.3030 | 0.9982 | 0.4649 | 0.7449 | 0.9997 |
| 80 | 20 | 291,350 | 0.7953 | 0.4941 | 0.9980 | 0.6610 | 0.7446 | 0.9993 |
| 70 | 30 | 194,230 | 0.8210 | 0.6267 | 0.9980 | 0.7699 | 0.7452 | 0.9989 |
| 60 | 40 | 145,675 | 0.8458 | 0.7224 | 0.9980 | 0.8382 | 0.7444 | 0.9982 |
| 50 | 50 | 116,540 | 0.8714 | 0.7964 | 0.9980 | 0.8859 | 0.7448 | 0.9974 |
| 40 | 60 | 97,115 | 0.8971 | 0.8548 | 0.9980 | 0.9209 | 0.7456 | 0.9960 |
| 30 | 70 | 83,240 | 0.9220 | 0.9012 | 0.9980 | 0.9471 | 0.7447 | 0.9939 |
| 20 | 80 | 72,835 | 0.9469 | 0.9394 | 0.9980 | 0.9678 | 0.7426 | 0.9895 |
| 10 | 90 | 64,740 | 0.9726 | 0.9722 | 0.9980 | 0.9850 | 0.7436 | 0.9767 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7428 | 0.0000 | 0.0000 | 0.0000 | 0.7428 | 1.0000 |
| 90 | 10 | 299,940 | 0.7683 | 0.3013 | 0.9983 | 0.4628 | 0.7427 | 0.9997 |
| 80 | 20 | 291,350 | 0.7936 | 0.4921 | 0.9982 | 0.6592 | 0.7424 | 0.9994 |
| 70 | 30 | 194,230 | 0.8195 | 0.6246 | 0.9982 | 0.7684 | 0.7429 | 0.9990 |
| 60 | 40 | 145,675 | 0.8445 | 0.7207 | 0.9982 | 0.8371 | 0.7421 | 0.9984 |
| 50 | 50 | 116,540 | 0.8705 | 0.7952 | 0.9982 | 0.8852 | 0.7429 | 0.9976 |
| 40 | 60 | 97,115 | 0.8963 | 0.8537 | 0.9982 | 0.9203 | 0.7434 | 0.9964 |
| 30 | 70 | 83,240 | 0.9215 | 0.9005 | 0.9982 | 0.9468 | 0.7426 | 0.9944 |
| 20 | 80 | 72,835 | 0.9465 | 0.9388 | 0.9982 | 0.9676 | 0.7397 | 0.9903 |
| 10 | 90 | 64,740 | 0.9726 | 0.9721 | 0.9982 | 0.9850 | 0.7425 | 0.9786 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7428 | 0.0000 | 0.0000 | 0.0000 | 0.7428 | 1.0000 |
| 90 | 10 | 299,940 | 0.7683 | 0.3013 | 0.9983 | 0.4628 | 0.7427 | 0.9997 |
| 80 | 20 | 291,350 | 0.7936 | 0.4921 | 0.9982 | 0.6592 | 0.7424 | 0.9994 |
| 70 | 30 | 194,230 | 0.8195 | 0.6246 | 0.9982 | 0.7684 | 0.7429 | 0.9990 |
| 60 | 40 | 145,675 | 0.8445 | 0.7207 | 0.9982 | 0.8371 | 0.7421 | 0.9984 |
| 50 | 50 | 116,540 | 0.8705 | 0.7952 | 0.9982 | 0.8852 | 0.7429 | 0.9976 |
| 40 | 60 | 97,115 | 0.8963 | 0.8537 | 0.9982 | 0.9203 | 0.7434 | 0.9964 |
| 30 | 70 | 83,240 | 0.9215 | 0.9005 | 0.9982 | 0.9468 | 0.7426 | 0.9944 |
| 20 | 80 | 72,835 | 0.9465 | 0.9388 | 0.9982 | 0.9676 | 0.7397 | 0.9903 |
| 10 | 90 | 64,740 | 0.9726 | 0.9721 | 0.9982 | 0.9850 | 0.7425 | 0.9786 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762   <--
0.20       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.25       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.30       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.35       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.40       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.45       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.50       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.55       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.60       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.65       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.70       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.75       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
0.80       0.5671   0.2951   0.5295   0.9807   0.9060   0.1762  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5671, F1=0.2951, Normal Recall=0.5295, Normal Precision=0.9807, Attack Recall=0.9060, Attack Precision=0.1762

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
0.15       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248   <--
0.20       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.25       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.30       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.35       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.40       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.45       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.50       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.55       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.60       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.65       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.70       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.75       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
0.80       0.6046   0.4781   0.5295   0.9572   0.9053   0.3248  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6046, F1=0.4781, Normal Recall=0.5295, Normal Precision=0.9572, Attack Recall=0.9053, Attack Precision=0.3248

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
0.15       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514   <--
0.20       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.25       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.30       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.35       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.40       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.45       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.50       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.55       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.60       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.65       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.70       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.75       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
0.80       0.6415   0.6024   0.5284   0.9287   0.9053   0.4514  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6415, F1=0.6024, Normal Recall=0.5284, Normal Precision=0.9287, Attack Recall=0.9053, Attack Precision=0.4514

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
0.15       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609   <--
0.20       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.25       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.30       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.35       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.40       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.45       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.50       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.55       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.60       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.65       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.70       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.75       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
0.80       0.6786   0.6926   0.5275   0.8931   0.9053   0.5609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6786, F1=0.6926, Normal Recall=0.5275, Normal Precision=0.8931, Attack Recall=0.9053, Attack Precision=0.5609

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
0.15       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577   <--
0.20       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.25       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.30       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.35       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.40       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.45       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.50       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.55       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.60       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.65       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.70       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.75       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
0.80       0.7171   0.7619   0.5289   0.8481   0.9053   0.6577  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7171, F1=0.7619, Normal Recall=0.5289, Normal Precision=0.8481, Attack Recall=0.9053, Attack Precision=0.6577

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
0.15       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030   <--
0.20       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.25       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.30       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.35       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.40       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.45       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.50       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.55       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.60       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.65       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.70       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.75       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
0.80       0.7702   0.4648   0.7449   0.9997   0.9980   0.3030  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7702, F1=0.4648, Normal Recall=0.7449, Normal Precision=0.9997, Attack Recall=0.9980, Attack Precision=0.3030

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
0.15       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952   <--
0.20       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.25       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.30       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.35       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.40       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.45       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.50       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.55       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.60       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.65       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.70       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.75       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
0.80       0.7961   0.6619   0.7456   0.9993   0.9980   0.4952  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7961, F1=0.6619, Normal Recall=0.7456, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4952

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
0.15       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268   <--
0.20       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.25       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.30       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.35       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.40       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.45       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.50       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.55       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.60       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.65       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.70       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.75       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
0.80       0.8211   0.7700   0.7453   0.9989   0.9980   0.6268  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8211, F1=0.7700, Normal Recall=0.7453, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.6268

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
0.15       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230   <--
0.20       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.25       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.30       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.35       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.40       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.45       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.50       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.55       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.60       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.65       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.70       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.75       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
0.80       0.8463   0.8385   0.7451   0.9982   0.9980   0.7230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8463, F1=0.8385, Normal Recall=0.7451, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.7230

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
0.15       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956   <--
0.20       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.25       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.30       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.35       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.40       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.45       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.50       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.55       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.60       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.65       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.70       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.75       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
0.80       0.8708   0.8854   0.7436   0.9974   0.9980   0.7956  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8708, F1=0.8854, Normal Recall=0.7436, Normal Precision=0.9974, Attack Recall=0.9980, Attack Precision=0.7956

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
0.15       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012   <--
0.20       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.25       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.30       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.35       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.40       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.45       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.50       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.55       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.60       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.65       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.70       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.75       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.80       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7683, F1=0.4628, Normal Recall=0.7427, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.3012

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
0.15       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931   <--
0.20       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.25       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.30       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.35       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.40       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.45       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.50       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.55       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.60       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.65       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.70       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.75       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.80       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7944, F1=0.6601, Normal Recall=0.7435, Normal Precision=0.9994, Attack Recall=0.9982, Attack Precision=0.4931

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
0.15       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251   <--
0.20       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.25       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.30       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.35       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.40       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.45       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.50       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.55       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.60       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.65       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.70       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.75       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.80       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8198, F1=0.7688, Normal Recall=0.7434, Normal Precision=0.9990, Attack Recall=0.9982, Attack Precision=0.6251

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
0.15       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216   <--
0.20       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.25       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.30       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.35       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.40       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.45       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.50       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.55       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.60       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.65       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.70       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.75       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.80       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8452, F1=0.8376, Normal Recall=0.7432, Normal Precision=0.9984, Attack Recall=0.9982, Attack Precision=0.7216

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
0.15       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945   <--
0.20       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.25       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.30       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.35       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.40       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.45       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.50       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.55       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.60       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.65       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.70       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.75       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.80       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8700, F1=0.8848, Normal Recall=0.7419, Normal Precision=0.9976, Attack Recall=0.9982, Attack Precision=0.7945

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
0.15       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012   <--
0.20       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.25       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.30       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.35       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.40       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.45       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.50       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.55       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.60       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.65       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.70       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.75       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
0.80       0.7683   0.4628   0.7427   0.9997   0.9982   0.3012  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7683, F1=0.4628, Normal Recall=0.7427, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.3012

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
0.15       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931   <--
0.20       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.25       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.30       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.35       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.40       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.45       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.50       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.55       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.60       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.65       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.70       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.75       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
0.80       0.7944   0.6601   0.7435   0.9994   0.9982   0.4931  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7944, F1=0.6601, Normal Recall=0.7435, Normal Precision=0.9994, Attack Recall=0.9982, Attack Precision=0.4931

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
0.15       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251   <--
0.20       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.25       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.30       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.35       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.40       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.45       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.50       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.55       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.60       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.65       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.70       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.75       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
0.80       0.8198   0.7688   0.7434   0.9990   0.9982   0.6251  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8198, F1=0.7688, Normal Recall=0.7434, Normal Precision=0.9990, Attack Recall=0.9982, Attack Precision=0.6251

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
0.15       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216   <--
0.20       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.25       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.30       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.35       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.40       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.45       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.50       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.55       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.60       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.65       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.70       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.75       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
0.80       0.8452   0.8376   0.7432   0.9984   0.9982   0.7216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8452, F1=0.8376, Normal Recall=0.7432, Normal Precision=0.9984, Attack Recall=0.9982, Attack Precision=0.7216

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
0.15       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945   <--
0.20       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.25       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.30       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.35       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.40       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.45       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.50       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.55       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.60       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.65       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.70       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.75       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
0.80       0.8700   0.8848   0.7419   0.9976   0.9982   0.7945  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8700, F1=0.8848, Normal Recall=0.7419, Normal Precision=0.9976, Attack Recall=0.9982, Attack Precision=0.7945

```

