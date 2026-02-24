# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-15 11:09:11 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2226 | 0.2962 | 0.3710 | 0.4464 | 0.5214 | 0.5957 | 0.6699 | 0.7444 | 0.8188 | 0.8954 | 0.9697 |
| QAT+Prune only | 0.6027 | 0.6425 | 0.6814 | 0.7208 | 0.7600 | 0.7991 | 0.8394 | 0.8789 | 0.9196 | 0.9570 | 0.9979 |
| QAT+PTQ | 0.6023 | 0.6422 | 0.6812 | 0.7206 | 0.7599 | 0.7989 | 0.8393 | 0.8788 | 0.9194 | 0.9570 | 0.9979 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6023 | 0.6422 | 0.6812 | 0.7206 | 0.7599 | 0.7989 | 0.8393 | 0.8788 | 0.9194 | 0.9570 | 0.9979 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2163 | 0.3814 | 0.5124 | 0.6184 | 0.7058 | 0.7790 | 0.8416 | 0.8954 | 0.9435 | 0.9846 |
| QAT+Prune only | 0.0000 | 0.3583 | 0.5561 | 0.6820 | 0.7689 | 0.8324 | 0.8817 | 0.9203 | 0.9520 | 0.9766 | 0.9989 |
| QAT+PTQ | 0.0000 | 0.3581 | 0.5560 | 0.6818 | 0.7687 | 0.8323 | 0.8817 | 0.9202 | 0.9520 | 0.9766 | 0.9989 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3581 | 0.5560 | 0.6818 | 0.7687 | 0.8323 | 0.8817 | 0.9202 | 0.9520 | 0.9766 | 0.9989 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2226 | 0.2213 | 0.2213 | 0.2222 | 0.2224 | 0.2218 | 0.2202 | 0.2188 | 0.2153 | 0.2271 | 0.0000 |
| QAT+Prune only | 0.6027 | 0.6030 | 0.6023 | 0.6020 | 0.6015 | 0.6004 | 0.6017 | 0.6015 | 0.6064 | 0.5896 | 0.0000 |
| QAT+PTQ | 0.6023 | 0.6027 | 0.6021 | 0.6018 | 0.6012 | 0.6000 | 0.6014 | 0.6009 | 0.6058 | 0.5894 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6023 | 0.6027 | 0.6021 | 0.6018 | 0.6012 | 0.6000 | 0.6014 | 0.6009 | 0.6058 | 0.5894 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2226 | 0.0000 | 0.0000 | 0.0000 | 0.2226 | 1.0000 |
| 90 | 10 | 299,940 | 0.2962 | 0.1217 | 0.9709 | 0.2163 | 0.2213 | 0.9856 |
| 80 | 20 | 291,350 | 0.3710 | 0.2374 | 0.9697 | 0.3814 | 0.2213 | 0.9669 |
| 70 | 30 | 194,230 | 0.4464 | 0.3482 | 0.9697 | 0.5124 | 0.2222 | 0.9448 |
| 60 | 40 | 145,675 | 0.5214 | 0.4540 | 0.9697 | 0.6184 | 0.2224 | 0.9168 |
| 50 | 50 | 116,540 | 0.5957 | 0.5548 | 0.9697 | 0.7058 | 0.2218 | 0.8798 |
| 40 | 60 | 97,115 | 0.6699 | 0.6510 | 0.9697 | 0.7790 | 0.2202 | 0.8290 |
| 30 | 70 | 83,240 | 0.7444 | 0.7434 | 0.9697 | 0.8416 | 0.2188 | 0.7558 |
| 20 | 80 | 72,835 | 0.8188 | 0.8317 | 0.9697 | 0.8954 | 0.2153 | 0.6399 |
| 10 | 90 | 64,740 | 0.8954 | 0.9186 | 0.9697 | 0.9435 | 0.2271 | 0.4544 |
| 0 | 100 | 58,270 | 0.9697 | 1.0000 | 0.9697 | 0.9846 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6027 | 0.0000 | 0.0000 | 0.0000 | 0.6027 | 1.0000 |
| 90 | 10 | 299,940 | 0.6425 | 0.2183 | 0.9979 | 0.3583 | 0.6030 | 0.9996 |
| 80 | 20 | 291,350 | 0.6814 | 0.3855 | 0.9979 | 0.5561 | 0.6023 | 0.9991 |
| 70 | 30 | 194,230 | 0.7208 | 0.5180 | 0.9979 | 0.6820 | 0.6020 | 0.9985 |
| 60 | 40 | 145,675 | 0.7600 | 0.6254 | 0.9979 | 0.7689 | 0.6015 | 0.9976 |
| 50 | 50 | 116,540 | 0.7991 | 0.7140 | 0.9979 | 0.8324 | 0.6004 | 0.9964 |
| 40 | 60 | 97,115 | 0.8394 | 0.7898 | 0.9979 | 0.8817 | 0.6017 | 0.9947 |
| 30 | 70 | 83,240 | 0.8789 | 0.8539 | 0.9979 | 0.9203 | 0.6015 | 0.9917 |
| 20 | 80 | 72,835 | 0.9196 | 0.9102 | 0.9979 | 0.9520 | 0.6064 | 0.9860 |
| 10 | 90 | 64,740 | 0.9570 | 0.9563 | 0.9979 | 0.9766 | 0.5896 | 0.9683 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6023 | 0.0000 | 0.0000 | 0.0000 | 0.6023 | 1.0000 |
| 90 | 10 | 299,940 | 0.6422 | 0.2182 | 0.9979 | 0.3581 | 0.6027 | 0.9996 |
| 80 | 20 | 291,350 | 0.6812 | 0.3853 | 0.9979 | 0.5560 | 0.6021 | 0.9991 |
| 70 | 30 | 194,230 | 0.7206 | 0.5178 | 0.9979 | 0.6818 | 0.6018 | 0.9985 |
| 60 | 40 | 145,675 | 0.7599 | 0.6252 | 0.9979 | 0.7687 | 0.6012 | 0.9976 |
| 50 | 50 | 116,540 | 0.7989 | 0.7139 | 0.9979 | 0.8323 | 0.6000 | 0.9964 |
| 40 | 60 | 97,115 | 0.8393 | 0.7897 | 0.9979 | 0.8817 | 0.6014 | 0.9947 |
| 30 | 70 | 83,240 | 0.8788 | 0.8537 | 0.9979 | 0.9202 | 0.6009 | 0.9917 |
| 20 | 80 | 72,835 | 0.9194 | 0.9101 | 0.9979 | 0.9520 | 0.6058 | 0.9860 |
| 10 | 90 | 64,740 | 0.9570 | 0.9563 | 0.9979 | 0.9766 | 0.5894 | 0.9683 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6023 | 0.0000 | 0.0000 | 0.0000 | 0.6023 | 1.0000 |
| 90 | 10 | 299,940 | 0.6422 | 0.2182 | 0.9979 | 0.3581 | 0.6027 | 0.9996 |
| 80 | 20 | 291,350 | 0.6812 | 0.3853 | 0.9979 | 0.5560 | 0.6021 | 0.9991 |
| 70 | 30 | 194,230 | 0.7206 | 0.5178 | 0.9979 | 0.6818 | 0.6018 | 0.9985 |
| 60 | 40 | 145,675 | 0.7599 | 0.6252 | 0.9979 | 0.7687 | 0.6012 | 0.9976 |
| 50 | 50 | 116,540 | 0.7989 | 0.7139 | 0.9979 | 0.8323 | 0.6000 | 0.9964 |
| 40 | 60 | 97,115 | 0.8393 | 0.7897 | 0.9979 | 0.8817 | 0.6014 | 0.9947 |
| 30 | 70 | 83,240 | 0.8788 | 0.8537 | 0.9979 | 0.9202 | 0.6009 | 0.9917 |
| 20 | 80 | 72,835 | 0.9194 | 0.9101 | 0.9979 | 0.9520 | 0.6058 | 0.9860 |
| 10 | 90 | 64,740 | 0.9570 | 0.9563 | 0.9979 | 0.9766 | 0.5894 | 0.9683 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |


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
0.15       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216   <--
0.20       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.25       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.30       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.35       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.40       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.45       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.50       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.55       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.60       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.65       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.70       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.75       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
0.80       0.2962   0.2162   0.2213   0.9854   0.9705   0.1216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2962, F1=0.2162, Normal Recall=0.2213, Normal Precision=0.9854, Attack Recall=0.9705, Attack Precision=0.1216

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
0.15       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374   <--
0.20       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.25       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.30       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.35       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.40       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.45       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.50       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.55       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.60       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.65       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.70       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.75       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
0.80       0.3710   0.3815   0.2214   0.9669   0.9697   0.2374  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3710, F1=0.3815, Normal Recall=0.2214, Normal Precision=0.9669, Attack Recall=0.9697, Attack Precision=0.2374

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
0.15       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484   <--
0.20       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.25       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.30       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.35       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.40       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.45       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.50       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.55       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.60       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.65       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.70       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.75       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
0.80       0.4469   0.5126   0.2228   0.9449   0.9697   0.3484  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4469, F1=0.5126, Normal Recall=0.2228, Normal Precision=0.9449, Attack Recall=0.9697, Attack Precision=0.3484

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
0.15       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540   <--
0.20       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.25       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.30       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.35       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.40       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.45       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.50       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.55       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.60       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.65       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.70       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.75       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
0.80       0.5213   0.6184   0.2224   0.9167   0.9697   0.4540  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5213, F1=0.6184, Normal Recall=0.2224, Normal Precision=0.9167, Attack Recall=0.9697, Attack Precision=0.4540

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
0.15       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544   <--
0.20       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.25       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.30       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.35       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.40       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.45       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.50       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.55       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.60       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.65       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.70       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.75       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
0.80       0.5952   0.7055   0.2206   0.8793   0.9697   0.5544  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5952, F1=0.7055, Normal Recall=0.2206, Normal Precision=0.8793, Attack Recall=0.9697, Attack Precision=0.5544

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
0.15       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183   <--
0.20       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.25       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.30       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.35       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.40       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.45       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.50       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.55       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.60       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.65       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.70       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.75       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
0.80       0.6425   0.3583   0.6030   0.9996   0.9979   0.2183  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6425, F1=0.3583, Normal Recall=0.6030, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.2183

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
0.15       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863   <--
0.20       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.25       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.30       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.35       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.40       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.45       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.50       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.55       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.60       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.65       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.70       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.75       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
0.80       0.6825   0.5569   0.6036   0.9991   0.9979   0.3863  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6825, F1=0.5569, Normal Recall=0.6036, Normal Precision=0.9991, Attack Recall=0.9979, Attack Precision=0.3863

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
0.15       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187   <--
0.20       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.25       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.30       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.35       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.40       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.45       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.50       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.55       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.60       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.65       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.70       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.75       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
0.80       0.7216   0.6826   0.6033   0.9985   0.9979   0.5187  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7216, F1=0.6826, Normal Recall=0.6033, Normal Precision=0.9985, Attack Recall=0.9979, Attack Precision=0.5187

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
0.15       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257   <--
0.20       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.25       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.30       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.35       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.40       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.45       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.50       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.55       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.60       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.65       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.70       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.75       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
0.80       0.7603   0.7691   0.6020   0.9976   0.9979   0.6257  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7603, F1=0.7691, Normal Recall=0.6020, Normal Precision=0.9976, Attack Recall=0.9979, Attack Precision=0.6257

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
0.15       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141   <--
0.20       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.25       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.30       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.35       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.40       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.45       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.50       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.55       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.60       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.65       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.70       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.75       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
0.80       0.7992   0.8325   0.6005   0.9964   0.9979   0.7141  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7992, F1=0.8325, Normal Recall=0.6005, Normal Precision=0.9964, Attack Recall=0.9979, Attack Precision=0.7141

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
0.15       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182   <--
0.20       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.25       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.30       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.35       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.40       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.45       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.50       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.55       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.60       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.65       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.70       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.75       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.80       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6422, F1=0.3581, Normal Recall=0.6027, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.2182

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
0.15       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861   <--
0.20       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.25       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.30       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.35       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.40       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.45       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.50       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.55       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.60       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.65       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.70       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.75       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.80       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6822, F1=0.5567, Normal Recall=0.6033, Normal Precision=0.9991, Attack Recall=0.9979, Attack Precision=0.3861

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
0.15       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185   <--
0.20       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.25       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.30       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.35       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.40       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.45       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.50       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.55       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.60       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.65       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.70       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.75       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.80       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7214, F1=0.6824, Normal Recall=0.6029, Normal Precision=0.9985, Attack Recall=0.9979, Attack Precision=0.5185

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
0.15       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255   <--
0.20       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.25       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.30       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.35       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.40       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.45       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.50       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.55       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.60       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.65       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.70       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.75       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.80       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7601, F1=0.7689, Normal Recall=0.6016, Normal Precision=0.9976, Attack Recall=0.9979, Attack Precision=0.6255

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
0.15       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139   <--
0.20       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.25       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.30       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.35       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.40       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.45       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.50       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.55       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.60       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.65       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.70       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.75       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.80       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7990, F1=0.8323, Normal Recall=0.6001, Normal Precision=0.9964, Attack Recall=0.9979, Attack Precision=0.7139

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
0.15       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182   <--
0.20       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.25       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.30       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.35       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.40       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.45       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.50       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.55       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.60       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.65       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.70       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.75       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
0.80       0.6422   0.3581   0.6027   0.9996   0.9979   0.2182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6422, F1=0.3581, Normal Recall=0.6027, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.2182

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
0.15       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861   <--
0.20       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.25       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.30       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.35       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.40       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.45       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.50       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.55       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.60       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.65       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.70       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.75       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
0.80       0.6822   0.5567   0.6033   0.9991   0.9979   0.3861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6822, F1=0.5567, Normal Recall=0.6033, Normal Precision=0.9991, Attack Recall=0.9979, Attack Precision=0.3861

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
0.15       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185   <--
0.20       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.25       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.30       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.35       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.40       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.45       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.50       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.55       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.60       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.65       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.70       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.75       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
0.80       0.7214   0.6824   0.6029   0.9985   0.9979   0.5185  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7214, F1=0.6824, Normal Recall=0.6029, Normal Precision=0.9985, Attack Recall=0.9979, Attack Precision=0.5185

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
0.15       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255   <--
0.20       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.25       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.30       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.35       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.40       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.45       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.50       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.55       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.60       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.65       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.70       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.75       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
0.80       0.7601   0.7689   0.6016   0.9976   0.9979   0.6255  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7601, F1=0.7689, Normal Recall=0.6016, Normal Precision=0.9976, Attack Recall=0.9979, Attack Precision=0.6255

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
0.15       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139   <--
0.20       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.25       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.30       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.35       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.40       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.45       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.50       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.55       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.60       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.65       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.70       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.75       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
0.80       0.7990   0.8323   0.6001   0.9964   0.9979   0.7139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7990, F1=0.8323, Normal Recall=0.6001, Normal Precision=0.9964, Attack Recall=0.9979, Attack Precision=0.7139

```

