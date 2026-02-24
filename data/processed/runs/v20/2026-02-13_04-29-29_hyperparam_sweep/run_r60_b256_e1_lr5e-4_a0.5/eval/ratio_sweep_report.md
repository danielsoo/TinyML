# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-15 07:41:04 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9008 | 0.8989 | 0.8960 | 0.8943 | 0.8911 | 0.8891 | 0.8866 | 0.8843 | 0.8819 | 0.8796 | 0.8771 |
| QAT+Prune only | 0.6357 | 0.6723 | 0.7075 | 0.7444 | 0.7798 | 0.8140 | 0.8512 | 0.8873 | 0.9240 | 0.9584 | 0.9955 |
| QAT+PTQ | 0.6349 | 0.6713 | 0.7066 | 0.7435 | 0.7791 | 0.8136 | 0.8507 | 0.8871 | 0.9237 | 0.9581 | 0.9954 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6349 | 0.6713 | 0.7066 | 0.7435 | 0.7791 | 0.8136 | 0.8507 | 0.8871 | 0.9237 | 0.9581 | 0.9954 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6344 | 0.7713 | 0.8327 | 0.8656 | 0.8878 | 0.9027 | 0.9139 | 0.9224 | 0.9291 | 0.9345 |
| QAT+Prune only | 0.0000 | 0.3780 | 0.5765 | 0.7003 | 0.7834 | 0.8426 | 0.8892 | 0.9252 | 0.9545 | 0.9773 | 0.9978 |
| QAT+PTQ | 0.0000 | 0.3772 | 0.5757 | 0.6995 | 0.7828 | 0.8423 | 0.8889 | 0.9251 | 0.9543 | 0.9771 | 0.9977 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3772 | 0.5757 | 0.6995 | 0.7828 | 0.8423 | 0.8889 | 0.9251 | 0.9543 | 0.9771 | 0.9977 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9008 | 0.9013 | 0.9007 | 0.9016 | 0.9004 | 0.9011 | 0.9009 | 0.9010 | 0.9011 | 0.9021 | 0.0000 |
| QAT+Prune only | 0.6357 | 0.6364 | 0.6355 | 0.6368 | 0.6360 | 0.6325 | 0.6347 | 0.6350 | 0.6379 | 0.6243 | 0.0000 |
| QAT+PTQ | 0.6349 | 0.6352 | 0.6344 | 0.6355 | 0.6350 | 0.6318 | 0.6337 | 0.6345 | 0.6372 | 0.6226 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6349 | 0.6352 | 0.6344 | 0.6355 | 0.6350 | 0.6318 | 0.6337 | 0.6345 | 0.6372 | 0.6226 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9008 | 0.0000 | 0.0000 | 0.0000 | 0.9008 | 1.0000 |
| 90 | 10 | 299,940 | 0.8989 | 0.4969 | 0.8770 | 0.6344 | 0.9013 | 0.9851 |
| 80 | 20 | 291,350 | 0.8960 | 0.6883 | 0.8771 | 0.7713 | 0.9007 | 0.9670 |
| 70 | 30 | 194,230 | 0.8943 | 0.7926 | 0.8771 | 0.8327 | 0.9016 | 0.9448 |
| 60 | 40 | 145,675 | 0.8911 | 0.8544 | 0.8771 | 0.8656 | 0.9004 | 0.9166 |
| 50 | 50 | 116,540 | 0.8891 | 0.8987 | 0.8771 | 0.8878 | 0.9011 | 0.8800 |
| 40 | 60 | 97,115 | 0.8866 | 0.9299 | 0.8771 | 0.9027 | 0.9009 | 0.8301 |
| 30 | 70 | 83,240 | 0.8843 | 0.9538 | 0.8771 | 0.9139 | 0.9010 | 0.7585 |
| 20 | 80 | 72,835 | 0.8819 | 0.9726 | 0.8771 | 0.9224 | 0.9011 | 0.6470 |
| 10 | 90 | 64,740 | 0.8796 | 0.9877 | 0.8771 | 0.9291 | 0.9021 | 0.4492 |
| 0 | 100 | 58,270 | 0.8771 | 1.0000 | 0.8771 | 0.9345 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6357 | 0.0000 | 0.0000 | 0.0000 | 0.6357 | 1.0000 |
| 90 | 10 | 299,940 | 0.6723 | 0.2333 | 0.9956 | 0.3780 | 0.6364 | 0.9992 |
| 80 | 20 | 291,350 | 0.7075 | 0.4058 | 0.9955 | 0.5765 | 0.6355 | 0.9982 |
| 70 | 30 | 194,230 | 0.7444 | 0.5402 | 0.9955 | 0.7003 | 0.6368 | 0.9970 |
| 60 | 40 | 145,675 | 0.7798 | 0.6458 | 0.9955 | 0.7834 | 0.6360 | 0.9953 |
| 50 | 50 | 116,540 | 0.8140 | 0.7304 | 0.9955 | 0.8426 | 0.6325 | 0.9930 |
| 40 | 60 | 97,115 | 0.8512 | 0.8035 | 0.9955 | 0.8892 | 0.6347 | 0.9895 |
| 30 | 70 | 83,240 | 0.8873 | 0.8642 | 0.9955 | 0.9252 | 0.6350 | 0.9838 |
| 20 | 80 | 72,835 | 0.9240 | 0.9166 | 0.9955 | 0.9545 | 0.6379 | 0.9727 |
| 10 | 90 | 64,740 | 0.9584 | 0.9598 | 0.9955 | 0.9773 | 0.6243 | 0.9393 |
| 0 | 100 | 58,270 | 0.9955 | 1.0000 | 0.9955 | 0.9978 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6349 | 0.0000 | 0.0000 | 0.0000 | 0.6349 | 1.0000 |
| 90 | 10 | 299,940 | 0.6713 | 0.2327 | 0.9954 | 0.3772 | 0.6352 | 0.9992 |
| 80 | 20 | 291,350 | 0.7066 | 0.4050 | 0.9954 | 0.5757 | 0.6344 | 0.9982 |
| 70 | 30 | 194,230 | 0.7435 | 0.5393 | 0.9954 | 0.6995 | 0.6355 | 0.9969 |
| 60 | 40 | 145,675 | 0.7791 | 0.6451 | 0.9954 | 0.7828 | 0.6350 | 0.9952 |
| 50 | 50 | 116,540 | 0.8136 | 0.7300 | 0.9954 | 0.8423 | 0.6318 | 0.9927 |
| 40 | 60 | 97,115 | 0.8507 | 0.8030 | 0.9954 | 0.8889 | 0.6337 | 0.9892 |
| 30 | 70 | 83,240 | 0.8871 | 0.8640 | 0.9954 | 0.9251 | 0.6345 | 0.9832 |
| 20 | 80 | 72,835 | 0.9237 | 0.9165 | 0.9954 | 0.9543 | 0.6372 | 0.9717 |
| 10 | 90 | 64,740 | 0.9581 | 0.9596 | 0.9954 | 0.9771 | 0.6226 | 0.9372 |
| 0 | 100 | 58,270 | 0.9954 | 1.0000 | 0.9954 | 0.9977 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6349 | 0.0000 | 0.0000 | 0.0000 | 0.6349 | 1.0000 |
| 90 | 10 | 299,940 | 0.6713 | 0.2327 | 0.9954 | 0.3772 | 0.6352 | 0.9992 |
| 80 | 20 | 291,350 | 0.7066 | 0.4050 | 0.9954 | 0.5757 | 0.6344 | 0.9982 |
| 70 | 30 | 194,230 | 0.7435 | 0.5393 | 0.9954 | 0.6995 | 0.6355 | 0.9969 |
| 60 | 40 | 145,675 | 0.7791 | 0.6451 | 0.9954 | 0.7828 | 0.6350 | 0.9952 |
| 50 | 50 | 116,540 | 0.8136 | 0.7300 | 0.9954 | 0.8423 | 0.6318 | 0.9927 |
| 40 | 60 | 97,115 | 0.8507 | 0.8030 | 0.9954 | 0.8889 | 0.6337 | 0.9892 |
| 30 | 70 | 83,240 | 0.8871 | 0.8640 | 0.9954 | 0.9251 | 0.6345 | 0.9832 |
| 20 | 80 | 72,835 | 0.9237 | 0.9165 | 0.9954 | 0.9543 | 0.6372 | 0.9717 |
| 10 | 90 | 64,740 | 0.9581 | 0.9596 | 0.9954 | 0.9771 | 0.6226 | 0.9372 |
| 0 | 100 | 58,270 | 0.9954 | 1.0000 | 0.9954 | 0.9977 | 0.0000 | 0.0000 |


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
0.15       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972   <--
0.20       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.25       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.30       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.35       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.40       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.45       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.50       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.55       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.60       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.65       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.70       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.75       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
0.80       0.8990   0.6349   0.9013   0.9852   0.8782   0.4972  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8990, F1=0.6349, Normal Recall=0.9013, Normal Precision=0.9852, Attack Recall=0.8782, Attack Precision=0.4972

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
0.15       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895   <--
0.20       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.25       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.30       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.35       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.40       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.45       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.50       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.55       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.60       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.65       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.70       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.75       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
0.80       0.8964   0.7721   0.9013   0.9670   0.8771   0.6895  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8964, F1=0.7721, Normal Recall=0.9013, Normal Precision=0.9670, Attack Recall=0.8771, Attack Precision=0.6895

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
0.15       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922   <--
0.20       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.25       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.30       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.35       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.40       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.45       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.50       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.55       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.60       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.65       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.70       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.75       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
0.80       0.8941   0.8325   0.9014   0.9448   0.8771   0.7922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8941, F1=0.8325, Normal Recall=0.9014, Normal Precision=0.9448, Attack Recall=0.8771, Attack Precision=0.7922

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
0.15       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550   <--
0.20       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.25       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.30       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.35       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.40       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.45       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.50       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.55       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.60       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.65       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.70       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.75       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
0.80       0.8914   0.8659   0.9009   0.9166   0.8771   0.8550  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8914, F1=0.8659, Normal Recall=0.9009, Normal Precision=0.9166, Attack Recall=0.8771, Attack Precision=0.8550

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
0.15       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986   <--
0.20       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.25       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.30       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.35       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.40       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.45       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.50       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.55       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.60       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.65       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.70       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.75       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
0.80       0.8891   0.8877   0.9010   0.8800   0.8771   0.8986  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8891, F1=0.8877, Normal Recall=0.9010, Normal Precision=0.8800, Attack Recall=0.8771, Attack Precision=0.8986

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
0.15       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333   <--
0.20       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.25       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.30       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.35       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.40       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.45       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.50       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.55       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.60       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.65       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.70       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.75       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
0.80       0.6723   0.3781   0.6364   0.9993   0.9959   0.2333  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6723, F1=0.3781, Normal Recall=0.6364, Normal Precision=0.9993, Attack Recall=0.9959, Attack Precision=0.2333

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
0.15       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069   <--
0.20       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.25       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.30       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.35       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.40       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.45       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.50       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.55       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.60       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.65       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.70       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.75       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
0.80       0.7088   0.5776   0.6372   0.9982   0.9955   0.4069  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7088, F1=0.5776, Normal Recall=0.6372, Normal Precision=0.9982, Attack Recall=0.9955, Attack Precision=0.4069

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
0.15       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397   <--
0.20       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.25       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.30       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.35       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.40       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.45       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.50       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.55       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.60       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.65       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.70       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.75       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
0.80       0.7439   0.6999   0.6361   0.9970   0.9955   0.5397  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7439, F1=0.6999, Normal Recall=0.6361, Normal Precision=0.9970, Attack Recall=0.9955, Attack Precision=0.5397

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
0.15       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455   <--
0.20       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.25       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.30       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.35       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.40       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.45       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.50       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.55       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.60       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.65       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.70       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.75       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
0.80       0.7795   0.7832   0.6355   0.9953   0.9955   0.6455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7795, F1=0.7832, Normal Recall=0.6355, Normal Precision=0.9953, Attack Recall=0.9955, Attack Precision=0.6455

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
0.15       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309   <--
0.20       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.25       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.30       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.35       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.40       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.45       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.50       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.55       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.60       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.65       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.70       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.75       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
0.80       0.8145   0.8429   0.6334   0.9930   0.9955   0.7309  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8145, F1=0.8429, Normal Recall=0.6334, Normal Precision=0.9930, Attack Recall=0.9955, Attack Precision=0.7309

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
0.15       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327   <--
0.20       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.25       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.30       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.35       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.40       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.45       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.50       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.55       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.60       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.65       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.70       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.75       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.80       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6713, F1=0.3773, Normal Recall=0.6352, Normal Precision=0.9992, Attack Recall=0.9957, Attack Precision=0.2327

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
0.15       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061   <--
0.20       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.25       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.30       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.35       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.40       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.45       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.50       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.55       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.60       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.65       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.70       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.75       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.80       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7079, F1=0.5768, Normal Recall=0.6360, Normal Precision=0.9982, Attack Recall=0.9954, Attack Precision=0.4061

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
0.15       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390   <--
0.20       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.25       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.30       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.35       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.40       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.45       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.50       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.55       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.60       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.65       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.70       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.75       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.80       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7432, F1=0.6993, Normal Recall=0.6352, Normal Precision=0.9969, Attack Recall=0.9954, Attack Precision=0.5390

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
0.15       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450   <--
0.20       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.25       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.30       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.35       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.40       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.45       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.50       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.55       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.60       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.65       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.70       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.75       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.80       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7790, F1=0.7827, Normal Recall=0.6347, Normal Precision=0.9952, Attack Recall=0.9954, Attack Precision=0.6450

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
0.15       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304   <--
0.20       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.25       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.30       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.35       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.40       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.45       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.50       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.55       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.60       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.65       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.70       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.75       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.80       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8140, F1=0.8426, Normal Recall=0.6327, Normal Precision=0.9927, Attack Recall=0.9954, Attack Precision=0.7304

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
0.15       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327   <--
0.20       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.25       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.30       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.35       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.40       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.45       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.50       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.55       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.60       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.65       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.70       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.75       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
0.80       0.6713   0.3773   0.6352   0.9992   0.9957   0.2327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6713, F1=0.3773, Normal Recall=0.6352, Normal Precision=0.9992, Attack Recall=0.9957, Attack Precision=0.2327

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
0.15       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061   <--
0.20       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.25       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.30       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.35       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.40       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.45       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.50       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.55       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.60       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.65       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.70       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.75       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
0.80       0.7079   0.5768   0.6360   0.9982   0.9954   0.4061  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7079, F1=0.5768, Normal Recall=0.6360, Normal Precision=0.9982, Attack Recall=0.9954, Attack Precision=0.4061

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
0.15       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390   <--
0.20       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.25       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.30       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.35       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.40       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.45       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.50       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.55       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.60       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.65       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.70       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.75       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
0.80       0.7432   0.6993   0.6352   0.9969   0.9954   0.5390  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7432, F1=0.6993, Normal Recall=0.6352, Normal Precision=0.9969, Attack Recall=0.9954, Attack Precision=0.5390

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
0.15       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450   <--
0.20       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.25       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.30       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.35       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.40       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.45       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.50       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.55       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.60       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.65       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.70       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.75       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
0.80       0.7790   0.7827   0.6347   0.9952   0.9954   0.6450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7790, F1=0.7827, Normal Recall=0.6347, Normal Precision=0.9952, Attack Recall=0.9954, Attack Precision=0.6450

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
0.15       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304   <--
0.20       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.25       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.30       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.35       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.40       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.45       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.50       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.55       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.60       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.65       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.70       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.75       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
0.80       0.8140   0.8426   0.6327   0.9927   0.9954   0.7304  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8140, F1=0.8426, Normal Recall=0.6327, Normal Precision=0.9927, Attack Recall=0.9954, Attack Precision=0.7304

```

