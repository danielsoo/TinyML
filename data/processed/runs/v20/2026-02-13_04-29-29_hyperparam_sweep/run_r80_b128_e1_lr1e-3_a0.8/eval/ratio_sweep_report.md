# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-17 15:05:10 |

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
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6024 | 0.6299 | 0.6601 | 0.6914 | 0.7208 | 0.7505 | 0.7819 | 0.8131 | 0.8402 | 0.8720 | 0.9021 |
| QAT+Prune only | 0.7598 | 0.7843 | 0.8071 | 0.8312 | 0.8551 | 0.8771 | 0.9017 | 0.9261 | 0.9489 | 0.9715 | 0.9962 |
| QAT+PTQ | 0.7579 | 0.7825 | 0.8055 | 0.8295 | 0.8539 | 0.8761 | 0.9007 | 0.9257 | 0.9485 | 0.9714 | 0.9962 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7579 | 0.7825 | 0.8055 | 0.8295 | 0.8539 | 0.8761 | 0.9007 | 0.9257 | 0.9485 | 0.9714 | 0.9962 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3274 | 0.5149 | 0.6369 | 0.7210 | 0.7834 | 0.8323 | 0.8711 | 0.9003 | 0.9269 | 0.9485 |
| QAT+Prune only | 0.0000 | 0.4803 | 0.6738 | 0.7798 | 0.8461 | 0.8902 | 0.9240 | 0.9497 | 0.9689 | 0.9844 | 0.9981 |
| QAT+PTQ | 0.0000 | 0.4782 | 0.6720 | 0.7781 | 0.8451 | 0.8894 | 0.9233 | 0.9494 | 0.9687 | 0.9843 | 0.9981 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4782 | 0.6720 | 0.7781 | 0.8451 | 0.8894 | 0.9233 | 0.9494 | 0.9687 | 0.9843 | 0.9981 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6024 | 0.5998 | 0.5996 | 0.6012 | 0.5999 | 0.5990 | 0.6017 | 0.6056 | 0.5924 | 0.6010 | 0.0000 |
| QAT+Prune only | 0.7598 | 0.7607 | 0.7598 | 0.7605 | 0.7610 | 0.7581 | 0.7599 | 0.7626 | 0.7595 | 0.7490 | 0.0000 |
| QAT+PTQ | 0.7579 | 0.7587 | 0.7578 | 0.7580 | 0.7590 | 0.7561 | 0.7574 | 0.7610 | 0.7576 | 0.7478 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7579 | 0.7587 | 0.7578 | 0.7580 | 0.7590 | 0.7561 | 0.7574 | 0.7610 | 0.7576 | 0.7478 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6024 | 0.0000 | 0.0000 | 0.0000 | 0.6024 | 1.0000 |
| 90 | 10 | 299,940 | 0.6299 | 0.2001 | 0.9009 | 0.3274 | 0.5998 | 0.9820 |
| 80 | 20 | 291,350 | 0.6601 | 0.3603 | 0.9021 | 0.5149 | 0.5996 | 0.9608 |
| 70 | 30 | 194,230 | 0.6914 | 0.4922 | 0.9021 | 0.6369 | 0.6012 | 0.9348 |
| 60 | 40 | 145,675 | 0.7208 | 0.6005 | 0.9021 | 0.7210 | 0.5999 | 0.9019 |
| 50 | 50 | 116,540 | 0.7505 | 0.6923 | 0.9021 | 0.7834 | 0.5990 | 0.8595 |
| 40 | 60 | 97,115 | 0.7819 | 0.7726 | 0.9021 | 0.8323 | 0.6017 | 0.8038 |
| 30 | 70 | 83,240 | 0.8131 | 0.8422 | 0.9021 | 0.8711 | 0.6056 | 0.7261 |
| 20 | 80 | 72,835 | 0.8402 | 0.8985 | 0.9021 | 0.9003 | 0.5924 | 0.6020 |
| 10 | 90 | 64,740 | 0.8720 | 0.9532 | 0.9021 | 0.9269 | 0.6010 | 0.4055 |
| 0 | 100 | 58,270 | 0.9021 | 1.0000 | 0.9021 | 0.9485 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7598 | 0.0000 | 0.0000 | 0.0000 | 0.7598 | 1.0000 |
| 90 | 10 | 299,940 | 0.7843 | 0.3164 | 0.9966 | 0.4803 | 0.7607 | 0.9995 |
| 80 | 20 | 291,350 | 0.8071 | 0.5091 | 0.9962 | 0.6738 | 0.7598 | 0.9988 |
| 70 | 30 | 194,230 | 0.8312 | 0.6406 | 0.9962 | 0.7798 | 0.7605 | 0.9979 |
| 60 | 40 | 145,675 | 0.8551 | 0.7354 | 0.9962 | 0.8461 | 0.7610 | 0.9967 |
| 50 | 50 | 116,540 | 0.8771 | 0.8046 | 0.9962 | 0.8902 | 0.7581 | 0.9950 |
| 40 | 60 | 97,115 | 0.9017 | 0.8616 | 0.9962 | 0.9240 | 0.7599 | 0.9926 |
| 30 | 70 | 83,240 | 0.9261 | 0.9073 | 0.9962 | 0.9497 | 0.7626 | 0.9886 |
| 20 | 80 | 72,835 | 0.9489 | 0.9431 | 0.9962 | 0.9689 | 0.7595 | 0.9805 |
| 10 | 90 | 64,740 | 0.9715 | 0.9728 | 0.9962 | 0.9844 | 0.7490 | 0.9566 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7579 | 0.0000 | 0.0000 | 0.0000 | 0.7579 | 1.0000 |
| 90 | 10 | 299,940 | 0.7825 | 0.3146 | 0.9966 | 0.4782 | 0.7587 | 0.9995 |
| 80 | 20 | 291,350 | 0.8055 | 0.5070 | 0.9962 | 0.6720 | 0.7578 | 0.9988 |
| 70 | 30 | 194,230 | 0.8295 | 0.6383 | 0.9962 | 0.7781 | 0.7580 | 0.9979 |
| 60 | 40 | 145,675 | 0.8539 | 0.7338 | 0.9962 | 0.8451 | 0.7590 | 0.9967 |
| 50 | 50 | 116,540 | 0.8761 | 0.8033 | 0.9962 | 0.8894 | 0.7561 | 0.9950 |
| 40 | 60 | 97,115 | 0.9007 | 0.8603 | 0.9962 | 0.9233 | 0.7574 | 0.9926 |
| 30 | 70 | 83,240 | 0.9257 | 0.9068 | 0.9962 | 0.9494 | 0.7610 | 0.9886 |
| 20 | 80 | 72,835 | 0.9485 | 0.9427 | 0.9962 | 0.9687 | 0.7576 | 0.9805 |
| 10 | 90 | 64,740 | 0.9714 | 0.9726 | 0.9962 | 0.9843 | 0.7478 | 0.9565 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7579 | 0.0000 | 0.0000 | 0.0000 | 0.7579 | 1.0000 |
| 90 | 10 | 299,940 | 0.7825 | 0.3146 | 0.9966 | 0.4782 | 0.7587 | 0.9995 |
| 80 | 20 | 291,350 | 0.8055 | 0.5070 | 0.9962 | 0.6720 | 0.7578 | 0.9988 |
| 70 | 30 | 194,230 | 0.8295 | 0.6383 | 0.9962 | 0.7781 | 0.7580 | 0.9979 |
| 60 | 40 | 145,675 | 0.8539 | 0.7338 | 0.9962 | 0.8451 | 0.7590 | 0.9967 |
| 50 | 50 | 116,540 | 0.8761 | 0.8033 | 0.9962 | 0.8894 | 0.7561 | 0.9950 |
| 40 | 60 | 97,115 | 0.9007 | 0.8603 | 0.9962 | 0.9233 | 0.7574 | 0.9926 |
| 30 | 70 | 83,240 | 0.9257 | 0.9068 | 0.9962 | 0.9494 | 0.7610 | 0.9886 |
| 20 | 80 | 72,835 | 0.9485 | 0.9427 | 0.9962 | 0.9687 | 0.7576 | 0.9805 |
| 10 | 90 | 64,740 | 0.9714 | 0.9726 | 0.9962 | 0.9843 | 0.7478 | 0.9565 |
| 0 | 100 | 58,270 | 0.9962 | 1.0000 | 0.9962 | 0.9981 | 0.0000 | 0.0000 |


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
0.15       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007   <--
0.20       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.25       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.30       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.35       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.40       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.45       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.50       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.55       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.60       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.65       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.70       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.75       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
0.80       0.6303   0.3285   0.5998   0.9826   0.9042   0.2007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6303, F1=0.3285, Normal Recall=0.5998, Normal Precision=0.9826, Attack Recall=0.9042, Attack Precision=0.2007

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
0.15       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605   <--
0.20       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.25       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.30       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.35       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.40       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.45       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.50       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.55       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.60       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.65       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.70       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.75       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
0.80       0.6604   0.5151   0.5999   0.9608   0.9021   0.3605  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6604, F1=0.5151, Normal Recall=0.5999, Normal Precision=0.9608, Attack Recall=0.9021, Attack Precision=0.3605

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
0.15       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935   <--
0.20       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.25       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.30       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.35       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.40       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.45       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.50       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.55       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.60       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.65       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.70       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.75       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
0.80       0.6928   0.6380   0.6031   0.9350   0.9021   0.4935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6928, F1=0.6380, Normal Recall=0.6031, Normal Precision=0.9350, Attack Recall=0.9021, Attack Precision=0.4935

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
0.15       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013   <--
0.20       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.25       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.30       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.35       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.40       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.45       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.50       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.55       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.60       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.65       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.70       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.75       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
0.80       0.7216   0.7216   0.6013   0.9021   0.9021   0.6013  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7216, F1=0.7216, Normal Recall=0.6013, Normal Precision=0.9021, Attack Recall=0.9021, Attack Precision=0.6013

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
0.15       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936   <--
0.20       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.25       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.30       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.35       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.40       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.45       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.50       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.55       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.60       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.65       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.70       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.75       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
0.80       0.7518   0.7842   0.6015   0.8600   0.9021   0.6936  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.7842, Normal Recall=0.6015, Normal Precision=0.8600, Attack Recall=0.9021, Attack Precision=0.6936

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
0.15       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164   <--
0.20       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.25       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.30       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.35       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.40       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.45       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.50       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.55       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.60       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.65       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.70       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.75       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
0.80       0.7843   0.4803   0.7607   0.9995   0.9965   0.3164  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7843, F1=0.4803, Normal Recall=0.7607, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3164

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
0.15       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104   <--
0.20       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.25       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.30       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.35       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.40       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.45       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.50       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.55       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.60       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.65       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.70       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.75       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
0.80       0.8081   0.6750   0.7611   0.9988   0.9962   0.5104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8081, F1=0.6750, Normal Recall=0.7611, Normal Precision=0.9988, Attack Recall=0.9962, Attack Precision=0.5104

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
0.15       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405   <--
0.20       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.25       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.30       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.35       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.40       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.45       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.50       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.55       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.60       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.65       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.70       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.75       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
0.80       0.8311   0.7797   0.7604   0.9979   0.9962   0.6405  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8311, F1=0.7797, Normal Recall=0.7604, Normal Precision=0.9979, Attack Recall=0.9962, Attack Precision=0.6405

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
0.15       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347   <--
0.20       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.25       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.30       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.35       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.40       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.45       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.50       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.55       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.60       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.65       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.70       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.75       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
0.80       0.8546   0.8457   0.7602   0.9967   0.9962   0.7347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8546, F1=0.8457, Normal Recall=0.7602, Normal Precision=0.9967, Attack Recall=0.9962, Attack Precision=0.7347

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
0.15       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052   <--
0.20       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.25       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.30       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.35       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.40       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.45       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.50       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.55       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.60       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.65       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.70       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.75       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
0.80       0.8776   0.8906   0.7590   0.9951   0.9962   0.8052  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8776, F1=0.8906, Normal Recall=0.7590, Normal Precision=0.9951, Attack Recall=0.9962, Attack Precision=0.8052

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
0.15       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146   <--
0.20       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.25       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.30       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.35       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.40       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.45       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.50       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.55       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.60       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.65       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.70       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.75       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.80       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7825, F1=0.4782, Normal Recall=0.7587, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3146

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
0.15       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084   <--
0.20       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.25       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.30       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.35       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.40       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.45       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.50       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.55       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.60       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.65       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.70       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.75       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.80       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8066, F1=0.6733, Normal Recall=0.7592, Normal Precision=0.9988, Attack Recall=0.9962, Attack Precision=0.5084

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
0.15       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388   <--
0.20       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.25       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.30       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.35       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.40       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.45       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.50       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.55       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.60       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.65       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.70       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.75       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.80       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8299, F1=0.7784, Normal Recall=0.7586, Normal Precision=0.9979, Attack Recall=0.9962, Attack Precision=0.6388

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
0.15       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331   <--
0.20       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.25       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.30       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.35       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.40       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.45       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.50       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.55       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.60       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.65       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.70       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.75       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.80       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8534, F1=0.8447, Normal Recall=0.7583, Normal Precision=0.9967, Attack Recall=0.9962, Attack Precision=0.7331

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
0.15       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039   <--
0.20       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.25       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.30       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.35       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.40       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.45       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.50       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.55       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.60       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.65       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.70       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.75       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.80       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8766, F1=0.8898, Normal Recall=0.7570, Normal Precision=0.9950, Attack Recall=0.9962, Attack Precision=0.8039

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
0.15       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146   <--
0.20       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.25       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.30       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.35       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.40       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.45       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.50       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.55       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.60       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.65       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.70       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.75       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
0.80       0.7825   0.4782   0.7587   0.9995   0.9965   0.3146  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7825, F1=0.4782, Normal Recall=0.7587, Normal Precision=0.9995, Attack Recall=0.9965, Attack Precision=0.3146

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
0.15       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084   <--
0.20       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.25       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.30       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.35       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.40       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.45       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.50       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.55       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.60       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.65       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.70       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.75       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
0.80       0.8066   0.6733   0.7592   0.9988   0.9962   0.5084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8066, F1=0.6733, Normal Recall=0.7592, Normal Precision=0.9988, Attack Recall=0.9962, Attack Precision=0.5084

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
0.15       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388   <--
0.20       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.25       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.30       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.35       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.40       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.45       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.50       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.55       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.60       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.65       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.70       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.75       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
0.80       0.8299   0.7784   0.7586   0.9979   0.9962   0.6388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8299, F1=0.7784, Normal Recall=0.7586, Normal Precision=0.9979, Attack Recall=0.9962, Attack Precision=0.6388

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
0.15       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331   <--
0.20       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.25       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.30       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.35       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.40       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.45       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.50       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.55       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.60       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.65       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.70       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.75       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
0.80       0.8534   0.8447   0.7583   0.9967   0.9962   0.7331  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8534, F1=0.8447, Normal Recall=0.7583, Normal Precision=0.9967, Attack Recall=0.9962, Attack Precision=0.7331

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
0.15       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039   <--
0.20       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.25       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.30       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.35       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.40       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.45       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.50       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.55       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.60       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.65       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.70       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.75       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
0.80       0.8766   0.8898   0.7570   0.9950   0.9962   0.8039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8766, F1=0.8898, Normal Recall=0.7570, Normal Precision=0.9950, Attack Recall=0.9962, Attack Precision=0.8039

```

