# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-23 01:23:37 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5202 | 0.5689 | 0.6168 | 0.6656 | 0.7125 | 0.7584 | 0.8089 | 0.8563 | 0.9034 | 0.9513 | 0.9986 |
| QAT+Prune only | 0.6339 | 0.6703 | 0.7064 | 0.7433 | 0.7796 | 0.8145 | 0.8526 | 0.8888 | 0.9261 | 0.9610 | 0.9980 |
| QAT+PTQ | 0.6362 | 0.6722 | 0.7082 | 0.7448 | 0.7809 | 0.8157 | 0.8536 | 0.8895 | 0.9265 | 0.9612 | 0.9980 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6362 | 0.6722 | 0.7082 | 0.7448 | 0.7809 | 0.8157 | 0.8536 | 0.8895 | 0.9265 | 0.9612 | 0.9980 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3166 | 0.5104 | 0.6418 | 0.7354 | 0.8052 | 0.8625 | 0.9068 | 0.9430 | 0.9736 | 0.9993 |
| QAT+Prune only | 0.0000 | 0.3771 | 0.5762 | 0.6999 | 0.7837 | 0.8433 | 0.8904 | 0.9263 | 0.9557 | 0.9787 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.3785 | 0.5777 | 0.7012 | 0.7847 | 0.8441 | 0.8911 | 0.9267 | 0.9560 | 0.9788 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3785 | 0.5777 | 0.7012 | 0.7847 | 0.8441 | 0.8911 | 0.9267 | 0.9560 | 0.9788 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5202 | 0.5212 | 0.5213 | 0.5228 | 0.5218 | 0.5181 | 0.5243 | 0.5243 | 0.5226 | 0.5249 | 0.0000 |
| QAT+Prune only | 0.6339 | 0.6338 | 0.6335 | 0.6341 | 0.6340 | 0.6311 | 0.6347 | 0.6339 | 0.6384 | 0.6281 | 0.0000 |
| QAT+PTQ | 0.6362 | 0.6360 | 0.6357 | 0.6363 | 0.6362 | 0.6334 | 0.6371 | 0.6364 | 0.6407 | 0.6299 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6362 | 0.6360 | 0.6357 | 0.6363 | 0.6362 | 0.6334 | 0.6371 | 0.6364 | 0.6407 | 0.6299 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5202 | 0.0000 | 0.0000 | 0.0000 | 0.5202 | 1.0000 |
| 90 | 10 | 299,940 | 0.5689 | 0.1881 | 0.9987 | 0.3166 | 0.5212 | 0.9997 |
| 80 | 20 | 291,350 | 0.6168 | 0.3428 | 0.9986 | 0.5104 | 0.5213 | 0.9994 |
| 70 | 30 | 194,230 | 0.6656 | 0.4728 | 0.9986 | 0.6418 | 0.5228 | 0.9989 |
| 60 | 40 | 145,675 | 0.7125 | 0.5820 | 0.9986 | 0.7354 | 0.5218 | 0.9983 |
| 50 | 50 | 116,540 | 0.7584 | 0.6745 | 0.9986 | 0.8052 | 0.5181 | 0.9974 |
| 40 | 60 | 97,115 | 0.8089 | 0.7590 | 0.9986 | 0.8625 | 0.5243 | 0.9961 |
| 30 | 70 | 83,240 | 0.8563 | 0.8305 | 0.9986 | 0.9068 | 0.5243 | 0.9940 |
| 20 | 80 | 72,835 | 0.9034 | 0.8932 | 0.9986 | 0.9430 | 0.5226 | 0.9897 |
| 10 | 90 | 64,740 | 0.9513 | 0.9498 | 0.9986 | 0.9736 | 0.5249 | 0.9773 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6339 | 0.0000 | 0.0000 | 0.0000 | 0.6339 | 1.0000 |
| 90 | 10 | 299,940 | 0.6703 | 0.2325 | 0.9982 | 0.3771 | 0.6338 | 0.9997 |
| 80 | 20 | 291,350 | 0.7064 | 0.4050 | 0.9980 | 0.5762 | 0.6335 | 0.9992 |
| 70 | 30 | 194,230 | 0.7433 | 0.5390 | 0.9980 | 0.6999 | 0.6341 | 0.9986 |
| 60 | 40 | 145,675 | 0.7796 | 0.6451 | 0.9980 | 0.7837 | 0.6340 | 0.9979 |
| 50 | 50 | 116,540 | 0.8145 | 0.7301 | 0.9980 | 0.8433 | 0.6311 | 0.9968 |
| 40 | 60 | 97,115 | 0.8526 | 0.8038 | 0.9980 | 0.8904 | 0.6347 | 0.9952 |
| 30 | 70 | 83,240 | 0.8888 | 0.8641 | 0.9980 | 0.9263 | 0.6339 | 0.9926 |
| 20 | 80 | 72,835 | 0.9261 | 0.9169 | 0.9980 | 0.9557 | 0.6384 | 0.9875 |
| 10 | 90 | 64,740 | 0.9610 | 0.9602 | 0.9980 | 0.9787 | 0.6281 | 0.9718 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6362 | 0.0000 | 0.0000 | 0.0000 | 0.6362 | 1.0000 |
| 90 | 10 | 299,940 | 0.6722 | 0.2335 | 0.9982 | 0.3785 | 0.6360 | 0.9997 |
| 80 | 20 | 291,350 | 0.7082 | 0.4065 | 0.9980 | 0.5777 | 0.6357 | 0.9992 |
| 70 | 30 | 194,230 | 0.7448 | 0.5404 | 0.9980 | 0.7012 | 0.6363 | 0.9986 |
| 60 | 40 | 145,675 | 0.7809 | 0.6465 | 0.9980 | 0.7847 | 0.6362 | 0.9979 |
| 50 | 50 | 116,540 | 0.8157 | 0.7314 | 0.9980 | 0.8441 | 0.6334 | 0.9968 |
| 40 | 60 | 97,115 | 0.8536 | 0.8049 | 0.9980 | 0.8911 | 0.6371 | 0.9953 |
| 30 | 70 | 83,240 | 0.8895 | 0.8649 | 0.9980 | 0.9267 | 0.6364 | 0.9927 |
| 20 | 80 | 72,835 | 0.9265 | 0.9174 | 0.9980 | 0.9560 | 0.6407 | 0.9876 |
| 10 | 90 | 64,740 | 0.9612 | 0.9604 | 0.9980 | 0.9788 | 0.6299 | 0.9721 |
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
| 100 | 0 | 100,000 | 0.6362 | 0.0000 | 0.0000 | 0.0000 | 0.6362 | 1.0000 |
| 90 | 10 | 299,940 | 0.6722 | 0.2335 | 0.9982 | 0.3785 | 0.6360 | 0.9997 |
| 80 | 20 | 291,350 | 0.7082 | 0.4065 | 0.9980 | 0.5777 | 0.6357 | 0.9992 |
| 70 | 30 | 194,230 | 0.7448 | 0.5404 | 0.9980 | 0.7012 | 0.6363 | 0.9986 |
| 60 | 40 | 145,675 | 0.7809 | 0.6465 | 0.9980 | 0.7847 | 0.6362 | 0.9979 |
| 50 | 50 | 116,540 | 0.8157 | 0.7314 | 0.9980 | 0.8441 | 0.6334 | 0.9968 |
| 40 | 60 | 97,115 | 0.8536 | 0.8049 | 0.9980 | 0.8911 | 0.6371 | 0.9953 |
| 30 | 70 | 83,240 | 0.8895 | 0.8649 | 0.9980 | 0.9267 | 0.6364 | 0.9927 |
| 20 | 80 | 72,835 | 0.9265 | 0.9174 | 0.9980 | 0.9560 | 0.6407 | 0.9876 |
| 10 | 90 | 64,740 | 0.9612 | 0.9604 | 0.9980 | 0.9788 | 0.6299 | 0.9721 |
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
0.15       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882   <--
0.20       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.25       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.30       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.35       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.40       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.45       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.50       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.55       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.60       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.65       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.70       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.75       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
0.80       0.5689   0.3167   0.5211   0.9998   0.9990   0.1882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5689, F1=0.3167, Normal Recall=0.5211, Normal Precision=0.9998, Attack Recall=0.9990, Attack Precision=0.1882

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
0.15       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426   <--
0.20       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.25       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.30       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.35       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.40       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.45       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.50       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.55       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.60       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.65       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.70       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.75       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
0.80       0.6164   0.5101   0.5209   0.9993   0.9986   0.3426  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6164, F1=0.5101, Normal Recall=0.5209, Normal Precision=0.9993, Attack Recall=0.9986, Attack Precision=0.3426

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
0.15       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721   <--
0.20       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.25       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.30       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.35       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.40       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.45       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.50       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.55       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.60       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.65       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.70       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.75       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
0.80       0.6646   0.6411   0.5215   0.9989   0.9986   0.4721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6646, F1=0.6411, Normal Recall=0.5215, Normal Precision=0.9989, Attack Recall=0.9986, Attack Precision=0.4721

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
0.15       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811   <--
0.20       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.25       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.30       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.35       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.40       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.45       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.50       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.55       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.60       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.65       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.70       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.75       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
0.80       0.7115   0.7347   0.5201   0.9983   0.9986   0.5811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7115, F1=0.7347, Normal Recall=0.5201, Normal Precision=0.9983, Attack Recall=0.9986, Attack Precision=0.5811

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
0.15       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757   <--
0.20       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.25       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.30       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.35       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.40       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.45       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.50       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.55       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.60       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.65       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.70       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.75       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
0.80       0.7597   0.8061   0.5208   0.9974   0.9986   0.6757  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7597, F1=0.8061, Normal Recall=0.5208, Normal Precision=0.9974, Attack Recall=0.9986, Attack Precision=0.6757

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
0.15       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325   <--
0.20       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.25       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.30       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.35       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.40       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.45       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.50       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.55       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.60       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.65       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.70       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.75       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
0.80       0.6703   0.3771   0.6338   0.9997   0.9982   0.2325  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6703, F1=0.3771, Normal Recall=0.6338, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2325

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
0.15       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058   <--
0.20       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.25       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.30       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.35       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.40       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.45       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.50       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.55       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.60       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.65       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.70       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.75       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
0.80       0.7074   0.5770   0.6347   0.9992   0.9980   0.4058  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7074, F1=0.5770, Normal Recall=0.6347, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4058

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
0.15       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388   <--
0.20       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.25       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.30       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.35       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.40       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.45       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.50       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.55       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.60       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.65       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.70       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.75       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
0.80       0.7431   0.6998   0.6339   0.9986   0.9980   0.5388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7431, F1=0.6998, Normal Recall=0.6339, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5388

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
0.15       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454   <--
0.20       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.25       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.30       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.35       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.40       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.45       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.50       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.55       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.60       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.65       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.70       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.75       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
0.80       0.7799   0.7839   0.6345   0.9979   0.9980   0.6454  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7799, F1=0.7839, Normal Recall=0.6345, Normal Precision=0.9979, Attack Recall=0.9980, Attack Precision=0.6454

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
0.15       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312   <--
0.20       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.25       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.30       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.35       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.40       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.45       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.50       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.55       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.60       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.65       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.70       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.75       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
0.80       0.8155   0.8440   0.6331   0.9968   0.9980   0.7312  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8155, F1=0.8440, Normal Recall=0.6331, Normal Precision=0.9968, Attack Recall=0.9980, Attack Precision=0.7312

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
0.15       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336   <--
0.20       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.25       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.30       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.35       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.40       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.45       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.50       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.55       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.60       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.65       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.70       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.75       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.80       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6722, F1=0.3785, Normal Recall=0.6360, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2336

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
0.15       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072   <--
0.20       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.25       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.30       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.35       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.40       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.45       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.50       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.55       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.60       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.65       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.70       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.75       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.80       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7091, F1=0.5784, Normal Recall=0.6368, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4072

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
0.15       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404   <--
0.20       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.25       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.30       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.35       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.40       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.45       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.50       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.55       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.60       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.65       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.70       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.75       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.80       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7448, F1=0.7011, Normal Recall=0.6362, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5404

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
0.15       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469   <--
0.20       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.25       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.30       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.35       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.40       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.45       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.50       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.55       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.60       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.65       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.70       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.75       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.80       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7813, F1=0.7850, Normal Recall=0.6369, Normal Precision=0.9979, Attack Recall=0.9980, Attack Precision=0.6469

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
0.15       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324   <--
0.20       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.25       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.30       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.35       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.40       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.45       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.50       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.55       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.60       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.65       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.70       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.75       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.80       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8167, F1=0.8448, Normal Recall=0.6354, Normal Precision=0.9968, Attack Recall=0.9980, Attack Precision=0.7324

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
0.15       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336   <--
0.20       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.25       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.30       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.35       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.40       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.45       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.50       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.55       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.60       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.65       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.70       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.75       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
0.80       0.6722   0.3785   0.6360   0.9997   0.9982   0.2336  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6722, F1=0.3785, Normal Recall=0.6360, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2336

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
0.15       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072   <--
0.20       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.25       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.30       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.35       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.40       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.45       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.50       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.55       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.60       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.65       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.70       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.75       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
0.80       0.7091   0.5784   0.6368   0.9992   0.9980   0.4072  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7091, F1=0.5784, Normal Recall=0.6368, Normal Precision=0.9992, Attack Recall=0.9980, Attack Precision=0.4072

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
0.15       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404   <--
0.20       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.25       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.30       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.35       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.40       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.45       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.50       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.55       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.60       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.65       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.70       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.75       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
0.80       0.7448   0.7011   0.6362   0.9986   0.9980   0.5404  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7448, F1=0.7011, Normal Recall=0.6362, Normal Precision=0.9986, Attack Recall=0.9980, Attack Precision=0.5404

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
0.15       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469   <--
0.20       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.25       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.30       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.35       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.40       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.45       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.50       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.55       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.60       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.65       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.70       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.75       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
0.80       0.7813   0.7850   0.6369   0.9979   0.9980   0.6469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7813, F1=0.7850, Normal Recall=0.6369, Normal Precision=0.9979, Attack Recall=0.9980, Attack Precision=0.6469

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
0.15       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324   <--
0.20       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.25       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.30       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.35       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.40       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.45       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.50       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.55       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.60       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.65       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.70       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.75       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
0.80       0.8167   0.8448   0.6354   0.9968   0.9980   0.7324  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8167, F1=0.8448, Normal Recall=0.6354, Normal Precision=0.9968, Attack Recall=0.9980, Attack Precision=0.7324

```

