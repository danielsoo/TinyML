# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-13 14:42:46 |

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
| Original (TFLite) | 0.2264 | 0.3025 | 0.3790 | 0.4542 | 0.5313 | 0.6061 | 0.6832 | 0.7599 | 0.8350 | 0.9120 | 0.9879 |
| QAT+Prune only | 0.5420 | 0.5897 | 0.6348 | 0.6809 | 0.7270 | 0.7714 | 0.8151 | 0.8615 | 0.9071 | 0.9510 | 0.9978 |
| QAT+PTQ | 0.5441 | 0.5916 | 0.6365 | 0.6820 | 0.7283 | 0.7724 | 0.8159 | 0.8622 | 0.9075 | 0.9512 | 0.9978 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5441 | 0.5916 | 0.6365 | 0.6820 | 0.7283 | 0.7724 | 0.8159 | 0.8622 | 0.9075 | 0.9512 | 0.9978 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2207 | 0.3889 | 0.5206 | 0.6277 | 0.7150 | 0.7891 | 0.8521 | 0.9055 | 0.9529 | 0.9939 |
| QAT+Prune only | 0.0000 | 0.3273 | 0.5222 | 0.6523 | 0.7452 | 0.8136 | 0.8662 | 0.9098 | 0.9450 | 0.9734 | 0.9989 |
| QAT+PTQ | 0.0000 | 0.3283 | 0.5234 | 0.6531 | 0.7461 | 0.8143 | 0.8668 | 0.9102 | 0.9452 | 0.9736 | 0.9989 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3283 | 0.5234 | 0.6531 | 0.7461 | 0.8143 | 0.8668 | 0.9102 | 0.9452 | 0.9736 | 0.9989 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2264 | 0.2263 | 0.2268 | 0.2255 | 0.2270 | 0.2244 | 0.2261 | 0.2279 | 0.2234 | 0.2294 | 0.0000 |
| QAT+Prune only | 0.5420 | 0.5444 | 0.5441 | 0.5451 | 0.5465 | 0.5449 | 0.5411 | 0.5436 | 0.5443 | 0.5295 | 0.0000 |
| QAT+PTQ | 0.5441 | 0.5465 | 0.5462 | 0.5467 | 0.5486 | 0.5470 | 0.5431 | 0.5457 | 0.5461 | 0.5315 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5441 | 0.5465 | 0.5462 | 0.5467 | 0.5486 | 0.5470 | 0.5431 | 0.5457 | 0.5461 | 0.5315 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2264 | 0.0000 | 0.0000 | 0.0000 | 0.2264 | 1.0000 |
| 90 | 10 | 299,940 | 0.3025 | 0.1242 | 0.9877 | 0.2207 | 0.2263 | 0.9940 |
| 80 | 20 | 291,350 | 0.3790 | 0.2421 | 0.9879 | 0.3889 | 0.2268 | 0.9868 |
| 70 | 30 | 194,230 | 0.4542 | 0.3535 | 0.9879 | 0.5206 | 0.2255 | 0.9775 |
| 60 | 40 | 145,675 | 0.5313 | 0.4600 | 0.9879 | 0.6277 | 0.2270 | 0.9656 |
| 50 | 50 | 116,540 | 0.6061 | 0.5602 | 0.9879 | 0.7150 | 0.2244 | 0.9487 |
| 40 | 60 | 97,115 | 0.6832 | 0.6569 | 0.9879 | 0.7891 | 0.2261 | 0.9255 |
| 30 | 70 | 83,240 | 0.7599 | 0.7491 | 0.9879 | 0.8521 | 0.2279 | 0.8895 |
| 20 | 80 | 72,835 | 0.8350 | 0.8357 | 0.9879 | 0.9055 | 0.2234 | 0.8215 |
| 10 | 90 | 64,740 | 0.9120 | 0.9202 | 0.9879 | 0.9529 | 0.2294 | 0.6775 |
| 0 | 100 | 58,270 | 0.9879 | 1.0000 | 0.9879 | 0.9939 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5420 | 0.0000 | 0.0000 | 0.0000 | 0.5420 | 1.0000 |
| 90 | 10 | 299,940 | 0.5897 | 0.1957 | 0.9979 | 0.3273 | 0.5444 | 0.9996 |
| 80 | 20 | 291,350 | 0.6348 | 0.3537 | 0.9978 | 0.5222 | 0.5441 | 0.9990 |
| 70 | 30 | 194,230 | 0.6809 | 0.4845 | 0.9978 | 0.6523 | 0.5451 | 0.9983 |
| 60 | 40 | 145,675 | 0.7270 | 0.5946 | 0.9978 | 0.7452 | 0.5465 | 0.9973 |
| 50 | 50 | 116,540 | 0.7714 | 0.6868 | 0.9978 | 0.8136 | 0.5449 | 0.9960 |
| 40 | 60 | 97,115 | 0.8151 | 0.7653 | 0.9978 | 0.8662 | 0.5411 | 0.9940 |
| 30 | 70 | 83,240 | 0.8615 | 0.8361 | 0.9978 | 0.9098 | 0.5436 | 0.9907 |
| 20 | 80 | 72,835 | 0.9071 | 0.8975 | 0.9978 | 0.9450 | 0.5443 | 0.9842 |
| 10 | 90 | 64,740 | 0.9510 | 0.9502 | 0.9978 | 0.9734 | 0.5295 | 0.9643 |
| 0 | 100 | 58,270 | 0.9978 | 1.0000 | 0.9978 | 0.9989 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5441 | 0.0000 | 0.0000 | 0.0000 | 0.5441 | 1.0000 |
| 90 | 10 | 299,940 | 0.5916 | 0.1964 | 0.9979 | 0.3283 | 0.5465 | 0.9996 |
| 80 | 20 | 291,350 | 0.6365 | 0.3547 | 0.9978 | 0.5234 | 0.5462 | 0.9990 |
| 70 | 30 | 194,230 | 0.6820 | 0.4854 | 0.9978 | 0.6531 | 0.5467 | 0.9983 |
| 60 | 40 | 145,675 | 0.7283 | 0.5957 | 0.9978 | 0.7461 | 0.5486 | 0.9974 |
| 50 | 50 | 116,540 | 0.7724 | 0.6878 | 0.9978 | 0.8143 | 0.5470 | 0.9961 |
| 40 | 60 | 97,115 | 0.8159 | 0.7661 | 0.9978 | 0.8668 | 0.5431 | 0.9941 |
| 30 | 70 | 83,240 | 0.8622 | 0.8367 | 0.9978 | 0.9102 | 0.5457 | 0.9908 |
| 20 | 80 | 72,835 | 0.9075 | 0.8979 | 0.9978 | 0.9452 | 0.5461 | 0.9844 |
| 10 | 90 | 64,740 | 0.9512 | 0.9504 | 0.9978 | 0.9736 | 0.5315 | 0.9647 |
| 0 | 100 | 58,270 | 0.9978 | 1.0000 | 0.9978 | 0.9989 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5441 | 0.0000 | 0.0000 | 0.0000 | 0.5441 | 1.0000 |
| 90 | 10 | 299,940 | 0.5916 | 0.1964 | 0.9979 | 0.3283 | 0.5465 | 0.9996 |
| 80 | 20 | 291,350 | 0.6365 | 0.3547 | 0.9978 | 0.5234 | 0.5462 | 0.9990 |
| 70 | 30 | 194,230 | 0.6820 | 0.4854 | 0.9978 | 0.6531 | 0.5467 | 0.9983 |
| 60 | 40 | 145,675 | 0.7283 | 0.5957 | 0.9978 | 0.7461 | 0.5486 | 0.9974 |
| 50 | 50 | 116,540 | 0.7724 | 0.6878 | 0.9978 | 0.8143 | 0.5470 | 0.9961 |
| 40 | 60 | 97,115 | 0.8159 | 0.7661 | 0.9978 | 0.8668 | 0.5431 | 0.9941 |
| 30 | 70 | 83,240 | 0.8622 | 0.8367 | 0.9978 | 0.9102 | 0.5457 | 0.9908 |
| 20 | 80 | 72,835 | 0.9075 | 0.8979 | 0.9978 | 0.9452 | 0.5461 | 0.9844 |
| 10 | 90 | 64,740 | 0.9512 | 0.9504 | 0.9978 | 0.9736 | 0.5315 | 0.9647 |
| 0 | 100 | 58,270 | 0.9978 | 1.0000 | 0.9978 | 0.9989 | 0.0000 | 0.0000 |


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
0.15       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242   <--
0.20       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.25       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.30       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.35       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.40       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.45       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.50       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.55       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.60       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.65       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.70       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.75       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
0.80       0.3024   0.2206   0.2263   0.9939   0.9874   0.1242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3024, F1=0.2206, Normal Recall=0.2263, Normal Precision=0.9939, Attack Recall=0.9874, Attack Precision=0.1242

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
0.15       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419   <--
0.20       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.25       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.30       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.35       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.40       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.45       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.50       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.55       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.60       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.65       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.70       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.75       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
0.80       0.3784   0.3887   0.2261   0.9868   0.9879   0.2419  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3784, F1=0.3887, Normal Recall=0.2261, Normal Precision=0.9868, Attack Recall=0.9879, Attack Precision=0.2419

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
0.15       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538   <--
0.20       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.25       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.30       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.35       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.40       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.45       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.50       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.55       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.60       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.65       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.70       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.75       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
0.80       0.4552   0.5210   0.2269   0.9776   0.9879   0.3538  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4552, F1=0.5210, Normal Recall=0.2269, Normal Precision=0.9776, Attack Recall=0.9879, Attack Precision=0.3538

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
0.15       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599   <--
0.20       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.25       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.30       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.35       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.40       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.45       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.50       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.55       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.60       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.65       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.70       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.75       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
0.80       0.5311   0.6276   0.2266   0.9655   0.9879   0.4599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5311, F1=0.6276, Normal Recall=0.2266, Normal Precision=0.9655, Attack Recall=0.9879, Attack Precision=0.4599

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
0.15       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603   <--
0.20       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.25       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.30       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.35       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.40       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.45       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.50       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.55       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.60       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.65       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.70       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.75       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
0.80       0.6063   0.7151   0.2248   0.9488   0.9879   0.5603  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6063, F1=0.7151, Normal Recall=0.2248, Normal Precision=0.9488, Attack Recall=0.9879, Attack Precision=0.5603

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
0.15       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957   <--
0.20       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.25       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.30       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.35       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.40       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.45       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.50       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.55       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.60       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.65       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.70       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.75       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
0.80       0.5897   0.3273   0.5444   0.9996   0.9979   0.1957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5897, F1=0.3273, Normal Recall=0.5444, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.1957

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
0.15       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539   <--
0.20       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.25       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.30       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.35       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.40       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.45       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.50       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.55       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.60       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.65       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.70       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.75       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
0.80       0.6352   0.5224   0.5445   0.9990   0.9978   0.3539  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6352, F1=0.5224, Normal Recall=0.5445, Normal Precision=0.9990, Attack Recall=0.9978, Attack Precision=0.3539

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
0.15       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834   <--
0.20       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.25       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.30       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.35       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.40       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.45       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.50       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.55       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.60       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.65       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.70       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.75       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
0.80       0.6795   0.6513   0.5431   0.9983   0.9978   0.4834  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6795, F1=0.6513, Normal Recall=0.5431, Normal Precision=0.9983, Attack Recall=0.9978, Attack Precision=0.4834

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
0.15       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922   <--
0.20       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.25       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.30       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.35       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.40       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.45       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.50       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.55       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.60       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.65       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.70       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.75       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
0.80       0.7242   0.7432   0.5419   0.9973   0.9978   0.5922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7242, F1=0.7432, Normal Recall=0.5419, Normal Precision=0.9973, Attack Recall=0.9978, Attack Precision=0.5922

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
0.15       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852   <--
0.20       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.25       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.30       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.35       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.40       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.45       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.50       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.55       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.60       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.65       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.70       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.75       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
0.80       0.7697   0.8125   0.5415   0.9960   0.9978   0.6852  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7697, F1=0.8125, Normal Recall=0.5415, Normal Precision=0.9960, Attack Recall=0.9978, Attack Precision=0.6852

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
0.15       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965   <--
0.20       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.25       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.30       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.35       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.40       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.45       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.50       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.55       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.60       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.65       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.70       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.75       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.80       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5916, F1=0.3283, Normal Recall=0.5465, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.1965

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
0.15       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549   <--
0.20       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.25       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.30       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.35       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.40       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.45       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.50       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.55       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.60       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.65       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.70       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.75       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.80       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6369, F1=0.5236, Normal Recall=0.5466, Normal Precision=0.9990, Attack Recall=0.9978, Attack Precision=0.3549

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
0.15       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845   <--
0.20       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.25       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.30       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.35       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.40       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.45       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.50       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.55       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.60       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.65       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.70       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.75       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.80       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6809, F1=0.6523, Normal Recall=0.5451, Normal Precision=0.9983, Attack Recall=0.9978, Attack Precision=0.4845

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
0.15       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933   <--
0.20       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.25       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.30       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.35       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.40       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.45       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.50       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.55       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.60       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.65       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.70       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.75       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.80       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7256, F1=0.7442, Normal Recall=0.5441, Normal Precision=0.9974, Attack Recall=0.9978, Attack Precision=0.5933

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
0.15       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862   <--
0.20       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.25       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.30       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.35       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.40       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.45       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.50       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.55       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.60       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.65       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.70       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.75       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.80       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7708, F1=0.8132, Normal Recall=0.5437, Normal Precision=0.9960, Attack Recall=0.9978, Attack Precision=0.6862

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
0.15       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965   <--
0.20       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.25       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.30       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.35       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.40       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.45       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.50       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.55       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.60       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.65       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.70       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.75       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
0.80       0.5916   0.3283   0.5465   0.9996   0.9979   0.1965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5916, F1=0.3283, Normal Recall=0.5465, Normal Precision=0.9996, Attack Recall=0.9979, Attack Precision=0.1965

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
0.15       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549   <--
0.20       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.25       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.30       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.35       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.40       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.45       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.50       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.55       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.60       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.65       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.70       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.75       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
0.80       0.6369   0.5236   0.5466   0.9990   0.9978   0.3549  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6369, F1=0.5236, Normal Recall=0.5466, Normal Precision=0.9990, Attack Recall=0.9978, Attack Precision=0.3549

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
0.15       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845   <--
0.20       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.25       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.30       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.35       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.40       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.45       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.50       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.55       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.60       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.65       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.70       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.75       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
0.80       0.6809   0.6523   0.5451   0.9983   0.9978   0.4845  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6809, F1=0.6523, Normal Recall=0.5451, Normal Precision=0.9983, Attack Recall=0.9978, Attack Precision=0.4845

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
0.15       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933   <--
0.20       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.25       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.30       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.35       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.40       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.45       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.50       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.55       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.60       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.65       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.70       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.75       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
0.80       0.7256   0.7442   0.5441   0.9974   0.9978   0.5933  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7256, F1=0.7442, Normal Recall=0.5441, Normal Precision=0.9974, Attack Recall=0.9978, Attack Precision=0.5933

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
0.15       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862   <--
0.20       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.25       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.30       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.35       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.40       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.45       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.50       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.55       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.60       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.65       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.70       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.75       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
0.80       0.7708   0.8132   0.5437   0.9960   0.9978   0.6862  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7708, F1=0.8132, Normal Recall=0.5437, Normal Precision=0.9960, Attack Recall=0.9978, Attack Precision=0.6862

```

