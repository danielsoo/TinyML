# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-17 11:14:41 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8867 | 0.8977 | 0.9086 | 0.9208 | 0.9311 | 0.9412 | 0.9535 | 0.9647 | 0.9752 | 0.9862 | 0.9978 |
| QAT+Prune only | 0.4344 | 0.4893 | 0.5447 | 0.6003 | 0.6556 | 0.7117 | 0.7682 | 0.8241 | 0.8793 | 0.9337 | 0.9906 |
| QAT+PTQ | 0.4353 | 0.4900 | 0.5454 | 0.6009 | 0.6562 | 0.7122 | 0.7685 | 0.8243 | 0.8795 | 0.9338 | 0.9906 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4353 | 0.4900 | 0.5454 | 0.6009 | 0.6562 | 0.7122 | 0.7685 | 0.8243 | 0.8795 | 0.9338 | 0.9906 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6612 | 0.8136 | 0.8832 | 0.9205 | 0.9444 | 0.9626 | 0.9754 | 0.9847 | 0.9924 | 0.9989 |
| QAT+Prune only | 0.0000 | 0.2795 | 0.4653 | 0.5979 | 0.6971 | 0.7746 | 0.8368 | 0.8874 | 0.9292 | 0.9641 | 0.9953 |
| QAT+PTQ | 0.0000 | 0.2798 | 0.4657 | 0.5983 | 0.6974 | 0.7748 | 0.8370 | 0.8875 | 0.9294 | 0.9642 | 0.9953 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2798 | 0.4657 | 0.5983 | 0.6974 | 0.7748 | 0.8370 | 0.8875 | 0.9294 | 0.9642 | 0.9953 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8867 | 0.8866 | 0.8863 | 0.8879 | 0.8866 | 0.8847 | 0.8872 | 0.8875 | 0.8848 | 0.8823 | 0.0000 |
| QAT+Prune only | 0.4344 | 0.4335 | 0.4332 | 0.4330 | 0.4323 | 0.4329 | 0.4346 | 0.4356 | 0.4342 | 0.4217 | 0.0000 |
| QAT+PTQ | 0.4353 | 0.4344 | 0.4341 | 0.4339 | 0.4333 | 0.4337 | 0.4354 | 0.4362 | 0.4354 | 0.4228 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4353 | 0.4344 | 0.4341 | 0.4339 | 0.4333 | 0.4337 | 0.4354 | 0.4362 | 0.4354 | 0.4228 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8867 | 0.0000 | 0.0000 | 0.0000 | 0.8867 | 1.0000 |
| 90 | 10 | 299,940 | 0.8977 | 0.4944 | 0.9979 | 0.6612 | 0.8866 | 0.9997 |
| 80 | 20 | 291,350 | 0.9086 | 0.6868 | 0.9978 | 0.8136 | 0.8863 | 0.9994 |
| 70 | 30 | 194,230 | 0.9208 | 0.7923 | 0.9978 | 0.8832 | 0.8879 | 0.9989 |
| 60 | 40 | 145,675 | 0.9311 | 0.8543 | 0.9978 | 0.9205 | 0.8866 | 0.9983 |
| 50 | 50 | 116,540 | 0.9412 | 0.8964 | 0.9978 | 0.9444 | 0.8847 | 0.9975 |
| 40 | 60 | 97,115 | 0.9535 | 0.9299 | 0.9978 | 0.9626 | 0.8872 | 0.9963 |
| 30 | 70 | 83,240 | 0.9647 | 0.9539 | 0.9978 | 0.9754 | 0.8875 | 0.9942 |
| 20 | 80 | 72,835 | 0.9752 | 0.9719 | 0.9978 | 0.9847 | 0.8848 | 0.9901 |
| 10 | 90 | 64,740 | 0.9862 | 0.9871 | 0.9978 | 0.9924 | 0.8823 | 0.9779 |
| 0 | 100 | 58,270 | 0.9978 | 1.0000 | 0.9978 | 0.9989 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4344 | 0.0000 | 0.0000 | 0.0000 | 0.4344 | 1.0000 |
| 90 | 10 | 299,940 | 0.4893 | 0.1627 | 0.9907 | 0.2795 | 0.4335 | 0.9976 |
| 80 | 20 | 291,350 | 0.5447 | 0.3041 | 0.9906 | 0.4653 | 0.4332 | 0.9946 |
| 70 | 30 | 194,230 | 0.6003 | 0.4282 | 0.9906 | 0.5979 | 0.4330 | 0.9908 |
| 60 | 40 | 145,675 | 0.6556 | 0.5377 | 0.9906 | 0.6971 | 0.4323 | 0.9857 |
| 50 | 50 | 116,540 | 0.7117 | 0.6359 | 0.9906 | 0.7746 | 0.4329 | 0.9787 |
| 40 | 60 | 97,115 | 0.7682 | 0.7244 | 0.9906 | 0.8368 | 0.4346 | 0.9685 |
| 30 | 70 | 83,240 | 0.8241 | 0.8037 | 0.9906 | 0.8874 | 0.4356 | 0.9520 |
| 20 | 80 | 72,835 | 0.8793 | 0.8750 | 0.9906 | 0.9292 | 0.4342 | 0.9201 |
| 10 | 90 | 64,740 | 0.9337 | 0.9391 | 0.9906 | 0.9641 | 0.4217 | 0.8326 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4353 | 0.0000 | 0.0000 | 0.0000 | 0.4353 | 1.0000 |
| 90 | 10 | 299,940 | 0.4900 | 0.1629 | 0.9907 | 0.2798 | 0.4344 | 0.9976 |
| 80 | 20 | 291,350 | 0.5454 | 0.3044 | 0.9906 | 0.4657 | 0.4341 | 0.9946 |
| 70 | 30 | 194,230 | 0.6009 | 0.4285 | 0.9906 | 0.5983 | 0.4339 | 0.9908 |
| 60 | 40 | 145,675 | 0.6562 | 0.5382 | 0.9906 | 0.6974 | 0.4333 | 0.9857 |
| 50 | 50 | 116,540 | 0.7122 | 0.6363 | 0.9906 | 0.7748 | 0.4337 | 0.9787 |
| 40 | 60 | 97,115 | 0.7685 | 0.7247 | 0.9906 | 0.8370 | 0.4354 | 0.9686 |
| 30 | 70 | 83,240 | 0.8243 | 0.8039 | 0.9906 | 0.8875 | 0.4362 | 0.9520 |
| 20 | 80 | 72,835 | 0.8795 | 0.8753 | 0.9906 | 0.9294 | 0.4354 | 0.9203 |
| 10 | 90 | 64,740 | 0.9338 | 0.9392 | 0.9906 | 0.9642 | 0.4228 | 0.8329 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4353 | 0.0000 | 0.0000 | 0.0000 | 0.4353 | 1.0000 |
| 90 | 10 | 299,940 | 0.4900 | 0.1629 | 0.9907 | 0.2798 | 0.4344 | 0.9976 |
| 80 | 20 | 291,350 | 0.5454 | 0.3044 | 0.9906 | 0.4657 | 0.4341 | 0.9946 |
| 70 | 30 | 194,230 | 0.6009 | 0.4285 | 0.9906 | 0.5983 | 0.4339 | 0.9908 |
| 60 | 40 | 145,675 | 0.6562 | 0.5382 | 0.9906 | 0.6974 | 0.4333 | 0.9857 |
| 50 | 50 | 116,540 | 0.7122 | 0.6363 | 0.9906 | 0.7748 | 0.4337 | 0.9787 |
| 40 | 60 | 97,115 | 0.7685 | 0.7247 | 0.9906 | 0.8370 | 0.4354 | 0.9686 |
| 30 | 70 | 83,240 | 0.8243 | 0.8039 | 0.9906 | 0.8875 | 0.4362 | 0.9520 |
| 20 | 80 | 72,835 | 0.8795 | 0.8753 | 0.9906 | 0.9294 | 0.4354 | 0.9203 |
| 10 | 90 | 64,740 | 0.9338 | 0.9392 | 0.9906 | 0.9642 | 0.4228 | 0.8329 |
| 0 | 100 | 58,270 | 0.9906 | 1.0000 | 0.9906 | 0.9953 | 0.0000 | 0.0000 |


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
0.15       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944   <--
0.20       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.25       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.30       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.35       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.40       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.45       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.50       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.55       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.60       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.65       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.70       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.75       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
0.80       0.8978   0.6613   0.8866   0.9997   0.9980   0.4944  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8978, F1=0.6613, Normal Recall=0.8866, Normal Precision=0.9997, Attack Recall=0.9980, Attack Precision=0.4944

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
0.15       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882   <--
0.20       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.25       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.30       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.35       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.40       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.45       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.50       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.55       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.60       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.65       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.70       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.75       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
0.80       0.9092   0.8146   0.8870   0.9994   0.9978   0.6882  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9092, F1=0.8146, Normal Recall=0.8870, Normal Precision=0.9994, Attack Recall=0.9978, Attack Precision=0.6882

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
0.15       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915   <--
0.20       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.25       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.30       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.35       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.40       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.45       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.50       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.55       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.60       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.65       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.70       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.75       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
0.80       0.9205   0.8828   0.8874   0.9989   0.9978   0.7915  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9205, F1=0.8828, Normal Recall=0.8874, Normal Precision=0.9989, Attack Recall=0.9978, Attack Precision=0.7915

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
0.15       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551   <--
0.20       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.25       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.30       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.35       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.40       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.45       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.50       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.55       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.60       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.65       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.70       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.75       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
0.80       0.9315   0.9210   0.8873   0.9983   0.9978   0.8551  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9315, F1=0.9210, Normal Recall=0.8873, Normal Precision=0.9983, Attack Recall=0.9978, Attack Precision=0.8551

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
0.15       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983   <--
0.20       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.25       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.30       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.35       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.40       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.45       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.50       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.55       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.60       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.65       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.70       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.75       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
0.80       0.9424   0.9455   0.8871   0.9975   0.9978   0.8983  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9424, F1=0.9455, Normal Recall=0.8871, Normal Precision=0.9975, Attack Recall=0.9978, Attack Precision=0.8983

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
0.15       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628   <--
0.20       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.25       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.30       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.35       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.40       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.45       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.50       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.55       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.60       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.65       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.70       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.75       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
0.80       0.4893   0.2797   0.4336   0.9978   0.9913   0.1628  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4893, F1=0.2797, Normal Recall=0.4336, Normal Precision=0.9978, Attack Recall=0.9913, Attack Precision=0.1628

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
0.15       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045   <--
0.20       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.25       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.30       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.35       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.40       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.45       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.50       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.55       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.60       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.65       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.70       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.75       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
0.80       0.5457   0.4659   0.4345   0.9946   0.9906   0.3045  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5457, F1=0.4659, Normal Recall=0.4345, Normal Precision=0.9946, Attack Recall=0.9906, Attack Precision=0.3045

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
0.15       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289   <--
0.20       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.25       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.30       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.35       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.40       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.45       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.50       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.55       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.60       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.65       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.70       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.75       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
0.80       0.6014   0.5986   0.4346   0.9908   0.9906   0.4289  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6014, F1=0.5986, Normal Recall=0.4346, Normal Precision=0.9908, Attack Recall=0.9906, Attack Precision=0.4289

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
0.15       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388   <--
0.20       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.25       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.30       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.35       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.40       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.45       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.50       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.55       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.60       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.65       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.70       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.75       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
0.80       0.6571   0.6980   0.4348   0.9858   0.9906   0.5388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6571, F1=0.6980, Normal Recall=0.4348, Normal Precision=0.9858, Attack Recall=0.9906, Attack Precision=0.5388

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
0.15       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372   <--
0.20       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.25       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.30       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.35       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.40       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.45       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.50       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.55       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.60       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.65       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.70       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.75       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
0.80       0.7132   0.7755   0.4359   0.9788   0.9906   0.6372  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7132, F1=0.7755, Normal Recall=0.4359, Normal Precision=0.9788, Attack Recall=0.9906, Attack Precision=0.6372

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
0.15       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630   <--
0.20       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.25       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.30       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.35       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.40       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.45       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.50       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.55       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.60       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.65       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.70       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.75       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.80       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4901, F1=0.2800, Normal Recall=0.4344, Normal Precision=0.9978, Attack Recall=0.9913, Attack Precision=0.1630

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
0.15       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049   <--
0.20       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.25       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.30       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.35       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.40       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.45       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.50       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.55       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.60       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.65       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.70       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.75       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.80       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5464, F1=0.4663, Normal Recall=0.4354, Normal Precision=0.9946, Attack Recall=0.9906, Attack Precision=0.3049

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
0.15       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293   <--
0.20       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.25       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.30       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.35       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.40       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.45       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.50       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.55       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.60       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.65       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.70       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.75       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.80       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6021, F1=0.5990, Normal Recall=0.4355, Normal Precision=0.9908, Attack Recall=0.9906, Attack Precision=0.4293

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
0.15       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392   <--
0.20       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.25       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.30       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.35       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.40       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.45       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.50       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.55       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.60       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.65       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.70       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.75       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.80       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6576, F1=0.6983, Normal Recall=0.4357, Normal Precision=0.9858, Attack Recall=0.9906, Attack Precision=0.5392

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
0.15       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376   <--
0.20       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.25       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.30       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.35       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.40       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.45       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.50       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.55       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.60       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.65       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.70       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.75       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.80       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7137, F1=0.7758, Normal Recall=0.4369, Normal Precision=0.9789, Attack Recall=0.9906, Attack Precision=0.6376

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
0.15       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630   <--
0.20       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.25       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.30       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.35       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.40       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.45       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.50       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.55       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.60       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.65       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.70       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.75       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
0.80       0.4901   0.2800   0.4344   0.9978   0.9913   0.1630  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4901, F1=0.2800, Normal Recall=0.4344, Normal Precision=0.9978, Attack Recall=0.9913, Attack Precision=0.1630

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
0.15       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049   <--
0.20       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.25       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.30       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.35       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.40       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.45       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.50       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.55       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.60       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.65       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.70       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.75       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
0.80       0.5464   0.4663   0.4354   0.9946   0.9906   0.3049  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5464, F1=0.4663, Normal Recall=0.4354, Normal Precision=0.9946, Attack Recall=0.9906, Attack Precision=0.3049

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
0.15       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293   <--
0.20       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.25       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.30       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.35       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.40       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.45       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.50       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.55       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.60       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.65       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.70       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.75       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
0.80       0.6021   0.5990   0.4355   0.9908   0.9906   0.4293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6021, F1=0.5990, Normal Recall=0.4355, Normal Precision=0.9908, Attack Recall=0.9906, Attack Precision=0.4293

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
0.15       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392   <--
0.20       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.25       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.30       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.35       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.40       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.45       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.50       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.55       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.60       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.65       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.70       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.75       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
0.80       0.6576   0.6983   0.4357   0.9858   0.9906   0.5392  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6576, F1=0.6983, Normal Recall=0.4357, Normal Precision=0.9858, Attack Recall=0.9906, Attack Precision=0.5392

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
0.15       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376   <--
0.20       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.25       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.30       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.35       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.40       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.45       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.50       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.55       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.60       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.65       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.70       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.75       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
0.80       0.7137   0.7758   0.4369   0.9789   0.9906   0.6376  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7137, F1=0.7758, Normal Recall=0.4369, Normal Precision=0.9789, Attack Recall=0.9906, Attack Precision=0.6376

```

