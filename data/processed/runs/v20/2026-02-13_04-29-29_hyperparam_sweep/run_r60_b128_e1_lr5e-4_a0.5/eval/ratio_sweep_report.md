# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-14 10:53:30 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2762 | 0.3485 | 0.4205 | 0.4925 | 0.5651 | 0.6371 | 0.7091 | 0.7819 | 0.8526 | 0.9262 | 0.9981 |
| QAT+Prune only | 0.4555 | 0.5083 | 0.5625 | 0.6170 | 0.6715 | 0.7261 | 0.7813 | 0.8356 | 0.8903 | 0.9444 | 0.9994 |
| QAT+PTQ | 0.4530 | 0.5062 | 0.5606 | 0.6154 | 0.6701 | 0.7250 | 0.7805 | 0.8347 | 0.8899 | 0.9441 | 0.9994 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4530 | 0.5062 | 0.5606 | 0.6154 | 0.6701 | 0.7250 | 0.7805 | 0.8347 | 0.8899 | 0.9441 | 0.9994 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2345 | 0.4079 | 0.5413 | 0.6474 | 0.7334 | 0.8046 | 0.8650 | 0.9155 | 0.9605 | 0.9991 |
| QAT+Prune only | 0.0000 | 0.2890 | 0.4774 | 0.6103 | 0.7088 | 0.7849 | 0.8458 | 0.8949 | 0.9358 | 0.9700 | 0.9997 |
| QAT+PTQ | 0.0000 | 0.2881 | 0.4764 | 0.6093 | 0.7079 | 0.7842 | 0.8453 | 0.8944 | 0.9356 | 0.9699 | 0.9997 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2881 | 0.4764 | 0.6093 | 0.7079 | 0.7842 | 0.8453 | 0.8944 | 0.9356 | 0.9699 | 0.9997 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2762 | 0.2763 | 0.2761 | 0.2759 | 0.2764 | 0.2761 | 0.2756 | 0.2773 | 0.2706 | 0.2785 | 0.0000 |
| QAT+Prune only | 0.4555 | 0.4537 | 0.4532 | 0.4532 | 0.4529 | 0.4528 | 0.4540 | 0.4533 | 0.4538 | 0.4492 | 0.0000 |
| QAT+PTQ | 0.4530 | 0.4514 | 0.4509 | 0.4509 | 0.4505 | 0.4506 | 0.4521 | 0.4505 | 0.4518 | 0.4462 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4530 | 0.4514 | 0.4509 | 0.4509 | 0.4505 | 0.4506 | 0.4521 | 0.4505 | 0.4518 | 0.4462 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2762 | 0.0000 | 0.0000 | 0.0000 | 0.2762 | 1.0000 |
| 90 | 10 | 299,940 | 0.3485 | 0.1329 | 0.9980 | 0.2345 | 0.2763 | 0.9992 |
| 80 | 20 | 291,350 | 0.4205 | 0.2563 | 0.9981 | 0.4079 | 0.2761 | 0.9983 |
| 70 | 30 | 194,230 | 0.4925 | 0.3714 | 0.9981 | 0.5413 | 0.2759 | 0.9971 |
| 60 | 40 | 145,675 | 0.5651 | 0.4790 | 0.9981 | 0.6474 | 0.2764 | 0.9955 |
| 50 | 50 | 116,540 | 0.6371 | 0.5796 | 0.9981 | 0.7334 | 0.2761 | 0.9932 |
| 40 | 60 | 97,115 | 0.7091 | 0.6739 | 0.9981 | 0.8046 | 0.2756 | 0.9898 |
| 30 | 70 | 83,240 | 0.7819 | 0.7632 | 0.9981 | 0.8650 | 0.2773 | 0.9844 |
| 20 | 80 | 72,835 | 0.8526 | 0.8455 | 0.9981 | 0.9155 | 0.2706 | 0.9729 |
| 10 | 90 | 64,740 | 0.9262 | 0.9257 | 0.9981 | 0.9605 | 0.2785 | 0.9425 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9991 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4555 | 0.0000 | 0.0000 | 0.0000 | 0.4555 | 1.0000 |
| 90 | 10 | 299,940 | 0.5083 | 0.1689 | 0.9992 | 0.2890 | 0.4537 | 0.9998 |
| 80 | 20 | 291,350 | 0.5625 | 0.3136 | 0.9994 | 0.4774 | 0.4532 | 0.9997 |
| 70 | 30 | 194,230 | 0.6170 | 0.4392 | 0.9994 | 0.6103 | 0.4532 | 0.9994 |
| 60 | 40 | 145,675 | 0.6715 | 0.5491 | 0.9994 | 0.7088 | 0.4529 | 0.9991 |
| 50 | 50 | 116,540 | 0.7261 | 0.6462 | 0.9994 | 0.7849 | 0.4528 | 0.9987 |
| 40 | 60 | 97,115 | 0.7813 | 0.7330 | 0.9994 | 0.8458 | 0.4540 | 0.9981 |
| 30 | 70 | 83,240 | 0.8356 | 0.8101 | 0.9994 | 0.8949 | 0.4533 | 0.9970 |
| 20 | 80 | 72,835 | 0.8903 | 0.8798 | 0.9994 | 0.9358 | 0.4538 | 0.9949 |
| 10 | 90 | 64,740 | 0.9444 | 0.9423 | 0.9994 | 0.9700 | 0.4492 | 0.9884 |
| 0 | 100 | 58,270 | 0.9994 | 1.0000 | 0.9994 | 0.9997 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4530 | 0.0000 | 0.0000 | 0.0000 | 0.4530 | 1.0000 |
| 90 | 10 | 299,940 | 0.5062 | 0.1683 | 0.9992 | 0.2881 | 0.4514 | 0.9998 |
| 80 | 20 | 291,350 | 0.5606 | 0.3127 | 0.9994 | 0.4764 | 0.4509 | 0.9997 |
| 70 | 30 | 194,230 | 0.6154 | 0.4382 | 0.9994 | 0.6093 | 0.4509 | 0.9994 |
| 60 | 40 | 145,675 | 0.6701 | 0.5480 | 0.9994 | 0.7079 | 0.4505 | 0.9991 |
| 50 | 50 | 116,540 | 0.7250 | 0.6453 | 0.9994 | 0.7842 | 0.4506 | 0.9987 |
| 40 | 60 | 97,115 | 0.7805 | 0.7323 | 0.9994 | 0.8453 | 0.4521 | 0.9981 |
| 30 | 70 | 83,240 | 0.8347 | 0.8093 | 0.9994 | 0.8944 | 0.4505 | 0.9970 |
| 20 | 80 | 72,835 | 0.8899 | 0.8794 | 0.9994 | 0.9356 | 0.4518 | 0.9949 |
| 10 | 90 | 64,740 | 0.9441 | 0.9420 | 0.9994 | 0.9699 | 0.4462 | 0.9884 |
| 0 | 100 | 58,270 | 0.9994 | 1.0000 | 0.9994 | 0.9997 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4530 | 0.0000 | 0.0000 | 0.0000 | 0.4530 | 1.0000 |
| 90 | 10 | 299,940 | 0.5062 | 0.1683 | 0.9992 | 0.2881 | 0.4514 | 0.9998 |
| 80 | 20 | 291,350 | 0.5606 | 0.3127 | 0.9994 | 0.4764 | 0.4509 | 0.9997 |
| 70 | 30 | 194,230 | 0.6154 | 0.4382 | 0.9994 | 0.6093 | 0.4509 | 0.9994 |
| 60 | 40 | 145,675 | 0.6701 | 0.5480 | 0.9994 | 0.7079 | 0.4505 | 0.9991 |
| 50 | 50 | 116,540 | 0.7250 | 0.6453 | 0.9994 | 0.7842 | 0.4506 | 0.9987 |
| 40 | 60 | 97,115 | 0.7805 | 0.7323 | 0.9994 | 0.8453 | 0.4521 | 0.9981 |
| 30 | 70 | 83,240 | 0.8347 | 0.8093 | 0.9994 | 0.8944 | 0.4505 | 0.9970 |
| 20 | 80 | 72,835 | 0.8899 | 0.8794 | 0.9994 | 0.9356 | 0.4518 | 0.9949 |
| 10 | 90 | 64,740 | 0.9441 | 0.9420 | 0.9994 | 0.9699 | 0.4462 | 0.9884 |
| 0 | 100 | 58,270 | 0.9994 | 1.0000 | 0.9994 | 0.9997 | 0.0000 | 0.0000 |


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
0.15       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329   <--
0.20       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.25       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.30       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.35       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.40       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.45       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.50       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.55       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.60       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.65       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.70       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.75       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
0.80       0.3485   0.2346   0.2763   0.9993   0.9982   0.1329  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3485, F1=0.2346, Normal Recall=0.2763, Normal Precision=0.9993, Attack Recall=0.9982, Attack Precision=0.1329

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
0.15       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563   <--
0.20       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.25       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.30       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.35       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.40       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.45       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.50       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.55       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.60       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.65       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.70       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.75       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
0.80       0.4204   0.4079   0.2760   0.9983   0.9981   0.2563  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4204, F1=0.4079, Normal Recall=0.2760, Normal Precision=0.9983, Attack Recall=0.9981, Attack Precision=0.2563

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
0.15       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718   <--
0.20       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.25       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.30       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.35       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.40       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.45       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.50       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.55       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.60       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.65       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.70       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.75       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
0.80       0.4935   0.5418   0.2773   0.9971   0.9981   0.3718  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4935, F1=0.5418, Normal Recall=0.2773, Normal Precision=0.9971, Attack Recall=0.9981, Attack Precision=0.3718

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
0.15       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790   <--
0.20       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.25       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.30       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.35       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.40       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.45       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.50       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.55       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.60       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.65       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.70       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.75       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
0.80       0.5649   0.6473   0.2762   0.9955   0.9981   0.4790  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5649, F1=0.6473, Normal Recall=0.2762, Normal Precision=0.9955, Attack Recall=0.9981, Attack Precision=0.4790

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
0.15       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792   <--
0.20       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.25       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.30       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.35       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.40       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.45       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.50       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.55       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.60       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.65       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.70       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.75       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
0.80       0.6365   0.7330   0.2749   0.9932   0.9981   0.5792  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6365, F1=0.7330, Normal Recall=0.2749, Normal Precision=0.9932, Attack Recall=0.9981, Attack Precision=0.5792

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
0.15       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690   <--
0.20       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.25       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.30       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.35       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.40       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.45       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.50       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.55       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.60       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.65       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.70       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.75       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
0.80       0.5083   0.2891   0.4538   0.9999   0.9995   0.1690  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5083, F1=0.2891, Normal Recall=0.4538, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1690

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
0.15       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142   <--
0.20       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.25       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.30       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.35       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.40       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.45       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.50       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.55       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.60       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.65       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.70       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.75       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
0.80       0.5635   0.4780   0.4545   0.9997   0.9994   0.3142  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5635, F1=0.4780, Normal Recall=0.4545, Normal Precision=0.9997, Attack Recall=0.9994, Attack Precision=0.3142

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
0.15       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401   <--
0.20       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.25       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.30       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.35       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.40       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.45       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.50       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.55       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.60       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.65       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.70       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.75       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
0.80       0.6184   0.6111   0.4551   0.9995   0.9994   0.4401  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6184, F1=0.6111, Normal Recall=0.4551, Normal Precision=0.9995, Attack Recall=0.9994, Attack Precision=0.4401

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
0.15       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501   <--
0.20       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.25       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.30       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.35       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.40       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.45       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.50       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.55       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.60       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.65       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.70       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.75       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
0.80       0.6728   0.7096   0.4551   0.9991   0.9994   0.5501  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6728, F1=0.7096, Normal Recall=0.4551, Normal Precision=0.9991, Attack Recall=0.9994, Attack Precision=0.5501

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
0.15       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473   <--
0.20       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.25       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.30       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.35       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.40       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.45       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.50       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.55       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.60       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.65       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.70       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.75       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
0.80       0.7274   0.7857   0.4554   0.9987   0.9994   0.6473  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7274, F1=0.7857, Normal Recall=0.4554, Normal Precision=0.9987, Attack Recall=0.9994, Attack Precision=0.6473

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
0.15       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684   <--
0.20       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.25       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.30       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.35       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.40       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.45       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.50       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.55       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.60       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.65       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.70       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.75       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.80       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5062, F1=0.2882, Normal Recall=0.4514, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1684

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
0.15       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133   <--
0.20       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.25       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.30       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.35       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.40       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.45       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.50       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.55       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.60       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.65       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.70       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.75       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.80       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5617, F1=0.4770, Normal Recall=0.4523, Normal Precision=0.9997, Attack Recall=0.9994, Attack Precision=0.3133

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
0.15       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391   <--
0.20       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.25       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.30       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.35       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.40       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.45       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.50       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.55       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.60       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.65       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.70       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.75       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.80       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6168, F1=0.6101, Normal Recall=0.4529, Normal Precision=0.9994, Attack Recall=0.9994, Attack Precision=0.4391

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
0.15       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490   <--
0.20       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.25       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.30       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.35       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.40       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.45       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.50       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.55       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.60       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.65       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.70       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.75       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.80       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6713, F1=0.7087, Normal Recall=0.4526, Normal Precision=0.9991, Attack Recall=0.9994, Attack Precision=0.5490

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
0.15       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462   <--
0.20       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.25       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.30       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.35       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.40       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.45       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.50       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.55       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.60       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.65       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.70       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.75       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.80       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7262, F1=0.7849, Normal Recall=0.4529, Normal Precision=0.9987, Attack Recall=0.9994, Attack Precision=0.6462

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
0.15       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684   <--
0.20       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.25       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.30       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.35       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.40       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.45       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.50       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.55       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.60       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.65       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.70       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.75       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
0.80       0.5062   0.2882   0.4514   0.9999   0.9995   0.1684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5062, F1=0.2882, Normal Recall=0.4514, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.1684

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
0.15       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133   <--
0.20       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.25       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.30       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.35       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.40       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.45       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.50       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.55       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.60       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.65       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.70       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.75       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
0.80       0.5617   0.4770   0.4523   0.9997   0.9994   0.3133  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5617, F1=0.4770, Normal Recall=0.4523, Normal Precision=0.9997, Attack Recall=0.9994, Attack Precision=0.3133

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
0.15       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391   <--
0.20       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.25       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.30       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.35       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.40       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.45       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.50       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.55       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.60       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.65       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.70       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.75       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
0.80       0.6168   0.6101   0.4529   0.9994   0.9994   0.4391  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6168, F1=0.6101, Normal Recall=0.4529, Normal Precision=0.9994, Attack Recall=0.9994, Attack Precision=0.4391

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
0.15       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490   <--
0.20       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.25       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.30       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.35       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.40       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.45       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.50       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.55       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.60       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.65       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.70       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.75       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
0.80       0.6713   0.7087   0.4526   0.9991   0.9994   0.5490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6713, F1=0.7087, Normal Recall=0.4526, Normal Precision=0.9991, Attack Recall=0.9994, Attack Precision=0.5490

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
0.15       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462   <--
0.20       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.25       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.30       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.35       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.40       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.45       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.50       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.55       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.60       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.65       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.70       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.75       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
0.80       0.7262   0.7849   0.4529   0.9987   0.9994   0.6462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7262, F1=0.7849, Normal Recall=0.4529, Normal Precision=0.9987, Attack Recall=0.9994, Attack Precision=0.6462

```

