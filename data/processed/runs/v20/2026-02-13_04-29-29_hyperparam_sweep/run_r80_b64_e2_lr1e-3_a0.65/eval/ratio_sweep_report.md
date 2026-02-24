# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-17 02:42:15 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4996 | 0.5428 | 0.5870 | 0.6327 | 0.6777 | 0.7218 | 0.7651 | 0.8113 | 0.8530 | 0.8996 | 0.9443 |
| QAT+Prune only | 0.6979 | 0.7277 | 0.7575 | 0.7872 | 0.8176 | 0.8461 | 0.8782 | 0.9075 | 0.9377 | 0.9673 | 0.9980 |
| QAT+PTQ | 0.6980 | 0.7276 | 0.7573 | 0.7871 | 0.8177 | 0.8459 | 0.8782 | 0.9076 | 0.9375 | 0.9671 | 0.9979 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6980 | 0.7276 | 0.7573 | 0.7871 | 0.8177 | 0.8459 | 0.8782 | 0.9076 | 0.9375 | 0.9671 | 0.9979 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2928 | 0.4777 | 0.6067 | 0.7009 | 0.7724 | 0.8283 | 0.8751 | 0.9113 | 0.9442 | 0.9714 |
| QAT+Prune only | 0.0000 | 0.4230 | 0.6221 | 0.7378 | 0.8141 | 0.8664 | 0.9077 | 0.9379 | 0.9624 | 0.9821 | 0.9990 |
| QAT+PTQ | 0.0000 | 0.4228 | 0.6219 | 0.7377 | 0.8141 | 0.8662 | 0.9077 | 0.9379 | 0.9623 | 0.9820 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4228 | 0.6219 | 0.7377 | 0.8141 | 0.8662 | 0.9077 | 0.9379 | 0.9623 | 0.9820 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4996 | 0.4980 | 0.4977 | 0.4992 | 0.4999 | 0.4992 | 0.4962 | 0.5010 | 0.4879 | 0.4974 | 0.0000 |
| QAT+Prune only | 0.6979 | 0.6977 | 0.6974 | 0.6969 | 0.6974 | 0.6943 | 0.6986 | 0.6964 | 0.6966 | 0.6911 | 0.0000 |
| QAT+PTQ | 0.6980 | 0.6975 | 0.6972 | 0.6968 | 0.6975 | 0.6939 | 0.6987 | 0.6967 | 0.6956 | 0.6898 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6980 | 0.6975 | 0.6972 | 0.6968 | 0.6975 | 0.6939 | 0.6987 | 0.6967 | 0.6956 | 0.6898 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4996 | 0.0000 | 0.0000 | 0.0000 | 0.4996 | 1.0000 |
| 90 | 10 | 299,940 | 0.5428 | 0.1732 | 0.9462 | 0.2928 | 0.4980 | 0.9881 |
| 80 | 20 | 291,350 | 0.5870 | 0.3197 | 0.9443 | 0.4777 | 0.4977 | 0.9728 |
| 70 | 30 | 194,230 | 0.6327 | 0.4469 | 0.9443 | 0.6067 | 0.4992 | 0.9544 |
| 60 | 40 | 145,675 | 0.6777 | 0.5573 | 0.9443 | 0.7009 | 0.4999 | 0.9309 |
| 50 | 50 | 116,540 | 0.7218 | 0.6535 | 0.9443 | 0.7724 | 0.4992 | 0.8996 |
| 40 | 60 | 97,115 | 0.7651 | 0.7376 | 0.9443 | 0.8283 | 0.4962 | 0.8559 |
| 30 | 70 | 83,240 | 0.8113 | 0.8153 | 0.9443 | 0.8751 | 0.5010 | 0.7940 |
| 20 | 80 | 72,835 | 0.8530 | 0.8806 | 0.9443 | 0.9113 | 0.4879 | 0.6865 |
| 10 | 90 | 64,740 | 0.8996 | 0.9442 | 0.9443 | 0.9442 | 0.4974 | 0.4981 |
| 0 | 100 | 58,270 | 0.9443 | 1.0000 | 0.9443 | 0.9714 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6979 | 0.0000 | 0.0000 | 0.0000 | 0.6979 | 1.0000 |
| 90 | 10 | 299,940 | 0.7277 | 0.2684 | 0.9980 | 0.4230 | 0.6977 | 0.9997 |
| 80 | 20 | 291,350 | 0.7575 | 0.4519 | 0.9980 | 0.6221 | 0.6974 | 0.9993 |
| 70 | 30 | 194,230 | 0.7872 | 0.5853 | 0.9980 | 0.7378 | 0.6969 | 0.9987 |
| 60 | 40 | 145,675 | 0.8176 | 0.6874 | 0.9980 | 0.8141 | 0.6974 | 0.9981 |
| 50 | 50 | 116,540 | 0.8461 | 0.7655 | 0.9980 | 0.8664 | 0.6943 | 0.9971 |
| 40 | 60 | 97,115 | 0.8782 | 0.8324 | 0.9980 | 0.9077 | 0.6986 | 0.9956 |
| 30 | 70 | 83,240 | 0.9075 | 0.8847 | 0.9980 | 0.9379 | 0.6964 | 0.9932 |
| 20 | 80 | 72,835 | 0.9377 | 0.9294 | 0.9980 | 0.9624 | 0.6966 | 0.9884 |
| 10 | 90 | 64,740 | 0.9673 | 0.9667 | 0.9980 | 0.9821 | 0.6911 | 0.9741 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6980 | 0.0000 | 0.0000 | 0.0000 | 0.6980 | 1.0000 |
| 90 | 10 | 299,940 | 0.7276 | 0.2683 | 0.9980 | 0.4228 | 0.6975 | 0.9997 |
| 80 | 20 | 291,350 | 0.7573 | 0.4517 | 0.9979 | 0.6219 | 0.6972 | 0.9993 |
| 70 | 30 | 194,230 | 0.7871 | 0.5851 | 0.9979 | 0.7377 | 0.6968 | 0.9987 |
| 60 | 40 | 145,675 | 0.8177 | 0.6874 | 0.9979 | 0.8141 | 0.6975 | 0.9980 |
| 50 | 50 | 116,540 | 0.8459 | 0.7653 | 0.9979 | 0.8662 | 0.6939 | 0.9970 |
| 40 | 60 | 97,115 | 0.8782 | 0.8324 | 0.9979 | 0.9077 | 0.6987 | 0.9956 |
| 30 | 70 | 83,240 | 0.9076 | 0.8847 | 0.9979 | 0.9379 | 0.6967 | 0.9931 |
| 20 | 80 | 72,835 | 0.9375 | 0.9291 | 0.9979 | 0.9623 | 0.6956 | 0.9883 |
| 10 | 90 | 64,740 | 0.9671 | 0.9666 | 0.9979 | 0.9820 | 0.6898 | 0.9738 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6980 | 0.0000 | 0.0000 | 0.0000 | 0.6980 | 1.0000 |
| 90 | 10 | 299,940 | 0.7276 | 0.2683 | 0.9980 | 0.4228 | 0.6975 | 0.9997 |
| 80 | 20 | 291,350 | 0.7573 | 0.4517 | 0.9979 | 0.6219 | 0.6972 | 0.9993 |
| 70 | 30 | 194,230 | 0.7871 | 0.5851 | 0.9979 | 0.7377 | 0.6968 | 0.9987 |
| 60 | 40 | 145,675 | 0.8177 | 0.6874 | 0.9979 | 0.8141 | 0.6975 | 0.9980 |
| 50 | 50 | 116,540 | 0.8459 | 0.7653 | 0.9979 | 0.8662 | 0.6939 | 0.9970 |
| 40 | 60 | 97,115 | 0.8782 | 0.8324 | 0.9979 | 0.9077 | 0.6987 | 0.9956 |
| 30 | 70 | 83,240 | 0.9076 | 0.8847 | 0.9979 | 0.9379 | 0.6967 | 0.9931 |
| 20 | 80 | 72,835 | 0.9375 | 0.9291 | 0.9979 | 0.9623 | 0.6956 | 0.9883 |
| 10 | 90 | 64,740 | 0.9671 | 0.9666 | 0.9979 | 0.9820 | 0.6898 | 0.9738 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726   <--
0.20       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.25       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.30       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.35       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.40       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.45       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.50       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.55       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.60       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.65       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.70       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.75       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
0.80       0.5425   0.2918   0.4980   0.9874   0.9427   0.1726  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5425, F1=0.2918, Normal Recall=0.4980, Normal Precision=0.9874, Attack Recall=0.9427, Attack Precision=0.1726

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
0.15       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200   <--
0.20       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.25       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.30       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.35       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.40       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.45       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.50       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.55       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.60       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.65       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.70       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.75       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
0.80       0.5875   0.4780   0.4982   0.9728   0.9443   0.3200  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5875, F1=0.4780, Normal Recall=0.4982, Normal Precision=0.9728, Attack Recall=0.9443, Attack Precision=0.3200

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
0.15       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470   <--
0.20       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.25       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.30       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.35       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.40       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.45       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.50       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.55       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.60       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.65       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.70       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.75       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
0.80       0.6328   0.6067   0.4992   0.9544   0.9443   0.4470  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6328, F1=0.6067, Normal Recall=0.4992, Normal Precision=0.9544, Attack Recall=0.9443, Attack Precision=0.4470

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
0.15       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568   <--
0.20       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.25       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.30       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.35       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.40       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.45       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.50       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.55       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.60       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.65       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.70       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.75       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
0.80       0.6770   0.7005   0.4988   0.9307   0.9443   0.5568  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6770, F1=0.7005, Normal Recall=0.4988, Normal Precision=0.9307, Attack Recall=0.9443, Attack Precision=0.5568

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
0.15       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543   <--
0.20       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.25       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.30       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.35       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.40       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.45       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.50       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.55       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.60       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.65       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.70       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.75       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
0.80       0.7227   0.7730   0.5011   0.9000   0.9443   0.6543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7227, F1=0.7730, Normal Recall=0.5011, Normal Precision=0.9000, Attack Recall=0.9443, Attack Precision=0.6543

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
0.15       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684   <--
0.20       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.25       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.30       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.35       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.40       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.45       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.50       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.55       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.60       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.65       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.70       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.75       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
0.80       0.7277   0.4230   0.6977   0.9997   0.9981   0.2684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7277, F1=0.4230, Normal Recall=0.6977, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2684

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
0.15       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525   <--
0.20       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.25       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.30       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.35       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.40       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.45       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.50       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.55       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.60       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.65       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.70       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.75       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
0.80       0.7581   0.6227   0.6982   0.9993   0.9980   0.4525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7581, F1=0.6227, Normal Recall=0.6982, Normal Precision=0.9993, Attack Recall=0.9980, Attack Precision=0.4525

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
0.15       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861   <--
0.20       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.25       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.30       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.35       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.40       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.45       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.50       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.55       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.60       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.65       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.70       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.75       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
0.80       0.7880   0.7385   0.6980   0.9987   0.9980   0.5861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7880, F1=0.7385, Normal Recall=0.6980, Normal Precision=0.9987, Attack Recall=0.9980, Attack Precision=0.5861

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
0.15       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877   <--
0.20       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.25       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.30       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.35       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.40       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.45       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.50       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.55       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.60       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.65       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.70       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.75       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
0.80       0.8179   0.8143   0.6979   0.9981   0.9980   0.6877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8179, F1=0.8143, Normal Recall=0.6979, Normal Precision=0.9981, Attack Recall=0.9980, Attack Precision=0.6877

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
0.15       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669   <--
0.20       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.25       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.30       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.35       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.40       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.45       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.50       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.55       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.60       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.65       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.70       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.75       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
0.80       0.8473   0.8673   0.6967   0.9971   0.9980   0.7669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8473, F1=0.8673, Normal Recall=0.6967, Normal Precision=0.9971, Attack Recall=0.9980, Attack Precision=0.7669

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
0.15       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683   <--
0.20       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.25       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.30       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.35       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.40       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.45       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.50       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.55       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.60       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.65       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.70       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.75       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.80       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7276, F1=0.4229, Normal Recall=0.6975, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2683

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
0.15       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524   <--
0.20       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.25       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.30       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.35       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.40       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.45       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.50       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.55       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.60       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.65       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.70       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.75       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.80       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7580, F1=0.6226, Normal Recall=0.6980, Normal Precision=0.9993, Attack Recall=0.9979, Attack Precision=0.4524

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
0.15       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861   <--
0.20       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.25       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.30       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.35       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.40       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.45       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.50       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.55       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.60       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.65       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.70       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.75       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.80       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7880, F1=0.7385, Normal Recall=0.6980, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.5861

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
0.15       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878   <--
0.20       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.25       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.30       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.35       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.40       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.45       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.50       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.55       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.60       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.65       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.70       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.75       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.80       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8180, F1=0.8143, Normal Recall=0.6980, Normal Precision=0.9980, Attack Recall=0.9979, Attack Precision=0.6878

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
0.15       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671   <--
0.20       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.25       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.30       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.35       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.40       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.45       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.50       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.55       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.60       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.65       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.70       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.75       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.80       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8475, F1=0.8674, Normal Recall=0.6970, Normal Precision=0.9971, Attack Recall=0.9979, Attack Precision=0.7671

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
0.15       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683   <--
0.20       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.25       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.30       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.35       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.40       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.45       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.50       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.55       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.60       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.65       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.70       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.75       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
0.80       0.7276   0.4229   0.6975   0.9997   0.9981   0.2683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7276, F1=0.4229, Normal Recall=0.6975, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2683

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
0.15       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524   <--
0.20       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.25       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.30       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.35       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.40       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.45       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.50       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.55       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.60       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.65       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.70       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.75       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
0.80       0.7580   0.6226   0.6980   0.9993   0.9979   0.4524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7580, F1=0.6226, Normal Recall=0.6980, Normal Precision=0.9993, Attack Recall=0.9979, Attack Precision=0.4524

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
0.15       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861   <--
0.20       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.25       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.30       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.35       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.40       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.45       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.50       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.55       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.60       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.65       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.70       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.75       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
0.80       0.7880   0.7385   0.6980   0.9987   0.9979   0.5861  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7880, F1=0.7385, Normal Recall=0.6980, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.5861

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
0.15       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878   <--
0.20       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.25       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.30       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.35       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.40       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.45       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.50       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.55       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.60       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.65       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.70       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.75       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
0.80       0.8180   0.8143   0.6980   0.9980   0.9979   0.6878  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8180, F1=0.8143, Normal Recall=0.6980, Normal Precision=0.9980, Attack Recall=0.9979, Attack Precision=0.6878

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
0.15       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671   <--
0.20       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.25       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.30       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.35       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.40       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.45       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.50       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.55       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.60       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.65       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.70       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.75       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
0.80       0.8475   0.8674   0.6970   0.9971   0.9979   0.7671  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8475, F1=0.8674, Normal Recall=0.6970, Normal Precision=0.9971, Attack Recall=0.9979, Attack Precision=0.7671

```

