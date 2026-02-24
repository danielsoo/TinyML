# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-18 10:41:02 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8202 | 0.8383 | 0.8554 | 0.8739 | 0.8916 | 0.9079 | 0.9272 | 0.9450 | 0.9622 | 0.9794 | 0.9979 |
| QAT+Prune only | 0.4563 | 0.5063 | 0.5574 | 0.6089 | 0.6601 | 0.7115 | 0.7632 | 0.8157 | 0.8671 | 0.9171 | 0.9698 |
| QAT+PTQ | 0.4555 | 0.5056 | 0.5569 | 0.6086 | 0.6599 | 0.7114 | 0.7633 | 0.8157 | 0.8675 | 0.9176 | 0.9704 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.4555 | 0.5056 | 0.5569 | 0.6086 | 0.6599 | 0.7114 | 0.7633 | 0.8157 | 0.8675 | 0.9176 | 0.9704 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5525 | 0.7340 | 0.8260 | 0.8805 | 0.9155 | 0.9427 | 0.9621 | 0.9769 | 0.9886 | 0.9989 |
| QAT+Prune only | 0.0000 | 0.2822 | 0.4671 | 0.5981 | 0.6954 | 0.7708 | 0.8310 | 0.8805 | 0.9211 | 0.9547 | 0.9847 |
| QAT+PTQ | 0.0000 | 0.2821 | 0.4669 | 0.5980 | 0.6953 | 0.7708 | 0.8311 | 0.8806 | 0.9213 | 0.9549 | 0.9850 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2821 | 0.4669 | 0.5980 | 0.6953 | 0.7708 | 0.8311 | 0.8806 | 0.9213 | 0.9549 | 0.9850 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8202 | 0.8206 | 0.8197 | 0.8208 | 0.8208 | 0.8179 | 0.8211 | 0.8218 | 0.8195 | 0.8129 | 0.0000 |
| QAT+Prune only | 0.4563 | 0.4547 | 0.4542 | 0.4542 | 0.4536 | 0.4532 | 0.4533 | 0.4562 | 0.4562 | 0.4428 | 0.0000 |
| QAT+PTQ | 0.4555 | 0.4539 | 0.4535 | 0.4535 | 0.4529 | 0.4524 | 0.4527 | 0.4549 | 0.4558 | 0.4427 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.4555 | 0.4539 | 0.4535 | 0.4535 | 0.4529 | 0.4524 | 0.4527 | 0.4549 | 0.4558 | 0.4427 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8202 | 0.0000 | 0.0000 | 0.0000 | 0.8202 | 1.0000 |
| 90 | 10 | 299,940 | 0.8383 | 0.3820 | 0.9979 | 0.5525 | 0.8206 | 0.9997 |
| 80 | 20 | 291,350 | 0.8554 | 0.5805 | 0.9979 | 0.7340 | 0.8197 | 0.9993 |
| 70 | 30 | 194,230 | 0.8739 | 0.7046 | 0.9979 | 0.8260 | 0.8208 | 0.9989 |
| 60 | 40 | 145,675 | 0.8916 | 0.7878 | 0.9979 | 0.8805 | 0.8208 | 0.9983 |
| 50 | 50 | 116,540 | 0.9079 | 0.8457 | 0.9979 | 0.9155 | 0.8179 | 0.9974 |
| 40 | 60 | 97,115 | 0.9272 | 0.8932 | 0.9979 | 0.9427 | 0.8211 | 0.9961 |
| 30 | 70 | 83,240 | 0.9450 | 0.9289 | 0.9979 | 0.9621 | 0.8218 | 0.9939 |
| 20 | 80 | 72,835 | 0.9622 | 0.9567 | 0.9979 | 0.9769 | 0.8195 | 0.9896 |
| 10 | 90 | 64,740 | 0.9794 | 0.9796 | 0.9979 | 0.9886 | 0.8129 | 0.9768 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4563 | 0.0000 | 0.0000 | 0.0000 | 0.4563 | 1.0000 |
| 90 | 10 | 299,940 | 0.5063 | 0.1651 | 0.9705 | 0.2822 | 0.4547 | 0.9928 |
| 80 | 20 | 291,350 | 0.5574 | 0.3076 | 0.9698 | 0.4671 | 0.4542 | 0.9837 |
| 70 | 30 | 194,230 | 0.6089 | 0.4323 | 0.9698 | 0.5981 | 0.4542 | 0.9723 |
| 60 | 40 | 145,675 | 0.6601 | 0.5420 | 0.9698 | 0.6954 | 0.4536 | 0.9576 |
| 50 | 50 | 116,540 | 0.7115 | 0.6395 | 0.9698 | 0.7708 | 0.4532 | 0.9376 |
| 40 | 60 | 97,115 | 0.7632 | 0.7269 | 0.9698 | 0.8310 | 0.4533 | 0.9093 |
| 30 | 70 | 83,240 | 0.8157 | 0.8063 | 0.9698 | 0.8805 | 0.4562 | 0.8664 |
| 20 | 80 | 72,835 | 0.8671 | 0.8771 | 0.9698 | 0.9211 | 0.4562 | 0.7909 |
| 10 | 90 | 64,740 | 0.9171 | 0.9400 | 0.9698 | 0.9547 | 0.4428 | 0.6200 |
| 0 | 100 | 58,270 | 0.9698 | 1.0000 | 0.9698 | 0.9847 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4555 | 0.0000 | 0.0000 | 0.0000 | 0.4555 | 1.0000 |
| 90 | 10 | 299,940 | 0.5056 | 0.1650 | 0.9712 | 0.2821 | 0.4539 | 0.9930 |
| 80 | 20 | 291,350 | 0.5569 | 0.3074 | 0.9704 | 0.4669 | 0.4535 | 0.9839 |
| 70 | 30 | 194,230 | 0.6086 | 0.4321 | 0.9704 | 0.5980 | 0.4535 | 0.9728 |
| 60 | 40 | 145,675 | 0.6599 | 0.5418 | 0.9704 | 0.6953 | 0.4529 | 0.9582 |
| 50 | 50 | 116,540 | 0.7114 | 0.6393 | 0.9704 | 0.7708 | 0.4524 | 0.9385 |
| 40 | 60 | 97,115 | 0.7633 | 0.7268 | 0.9704 | 0.8311 | 0.4527 | 0.9106 |
| 30 | 70 | 83,240 | 0.8157 | 0.8060 | 0.9704 | 0.8806 | 0.4549 | 0.8680 |
| 20 | 80 | 72,835 | 0.8675 | 0.8770 | 0.9704 | 0.9213 | 0.4558 | 0.7936 |
| 10 | 90 | 64,740 | 0.9176 | 0.9400 | 0.9704 | 0.9549 | 0.4427 | 0.6240 |
| 0 | 100 | 58,270 | 0.9704 | 1.0000 | 0.9704 | 0.9850 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.4555 | 0.0000 | 0.0000 | 0.0000 | 0.4555 | 1.0000 |
| 90 | 10 | 299,940 | 0.5056 | 0.1650 | 0.9712 | 0.2821 | 0.4539 | 0.9930 |
| 80 | 20 | 291,350 | 0.5569 | 0.3074 | 0.9704 | 0.4669 | 0.4535 | 0.9839 |
| 70 | 30 | 194,230 | 0.6086 | 0.4321 | 0.9704 | 0.5980 | 0.4535 | 0.9728 |
| 60 | 40 | 145,675 | 0.6599 | 0.5418 | 0.9704 | 0.6953 | 0.4529 | 0.9582 |
| 50 | 50 | 116,540 | 0.7114 | 0.6393 | 0.9704 | 0.7708 | 0.4524 | 0.9385 |
| 40 | 60 | 97,115 | 0.7633 | 0.7268 | 0.9704 | 0.8311 | 0.4527 | 0.9106 |
| 30 | 70 | 83,240 | 0.8157 | 0.8060 | 0.9704 | 0.8806 | 0.4549 | 0.8680 |
| 20 | 80 | 72,835 | 0.8675 | 0.8770 | 0.9704 | 0.9213 | 0.4558 | 0.7936 |
| 10 | 90 | 64,740 | 0.9176 | 0.9400 | 0.9704 | 0.9549 | 0.4427 | 0.6240 |
| 0 | 100 | 58,270 | 0.9704 | 1.0000 | 0.9704 | 0.9850 | 0.0000 | 0.0000 |


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
0.15       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820   <--
0.20       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.25       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.30       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.35       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.40       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.45       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.50       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.55       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.60       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.65       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.70       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.75       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
0.80       0.8383   0.5525   0.8206   0.9997   0.9981   0.3820  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8383, F1=0.5525, Normal Recall=0.8206, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.3820

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
0.15       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825   <--
0.20       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.25       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.30       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.35       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.40       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.45       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.50       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.55       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.60       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.65       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.70       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.75       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
0.80       0.8565   0.7356   0.8212   0.9993   0.9979   0.5825  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8565, F1=0.7356, Normal Recall=0.8212, Normal Precision=0.9993, Attack Recall=0.9979, Attack Precision=0.5825

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
0.15       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044   <--
0.20       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.25       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.30       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.35       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.40       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.45       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.50       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.55       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.60       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.65       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.70       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.75       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
0.80       0.8737   0.8259   0.8206   0.9989   0.9979   0.7044  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8737, F1=0.8259, Normal Recall=0.8206, Normal Precision=0.9989, Attack Recall=0.9979, Attack Precision=0.7044

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
0.15       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878   <--
0.20       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.25       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.30       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.35       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.40       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.45       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.50       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.55       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.60       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.65       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.70       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.75       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
0.80       0.8916   0.8805   0.8208   0.9983   0.9979   0.7878  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8916, F1=0.8805, Normal Recall=0.8208, Normal Precision=0.9983, Attack Recall=0.9979, Attack Precision=0.7878

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
0.15       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477   <--
0.20       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.25       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.30       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.35       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.40       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.45       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.50       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.55       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.60       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.65       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.70       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.75       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
0.80       0.9093   0.9167   0.8208   0.9974   0.9979   0.8477  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9093, F1=0.9167, Normal Recall=0.8208, Normal Precision=0.9974, Attack Recall=0.9979, Attack Precision=0.8477

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
0.15       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650   <--
0.20       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.25       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.30       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.35       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.40       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.45       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.50       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.55       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.60       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.65       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.70       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.75       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
0.80       0.5062   0.2820   0.4547   0.9927   0.9698   0.1650  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5062, F1=0.2820, Normal Recall=0.4547, Normal Precision=0.9927, Attack Recall=0.9698, Attack Precision=0.1650

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
0.15       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082   <--
0.20       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.25       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.30       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.35       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.40       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.45       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.50       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.55       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.60       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.65       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.70       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.75       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
0.80       0.5585   0.4677   0.4557   0.9837   0.9698   0.3082  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5585, F1=0.4677, Normal Recall=0.4557, Normal Precision=0.9837, Attack Recall=0.9698, Attack Precision=0.3082

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
0.15       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330   <--
0.20       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.25       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.30       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.35       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.40       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.45       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.50       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.55       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.60       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.65       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.70       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.75       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
0.80       0.6099   0.5987   0.4556   0.9724   0.9698   0.4330  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6099, F1=0.5987, Normal Recall=0.4556, Normal Precision=0.9724, Attack Recall=0.9698, Attack Precision=0.4330

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
0.15       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432   <--
0.20       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.25       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.30       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.35       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.40       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.45       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.50       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.55       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.60       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.65       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.70       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.75       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
0.80       0.6617   0.6964   0.4563   0.9578   0.9698   0.5432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6617, F1=0.6964, Normal Recall=0.4563, Normal Precision=0.9578, Attack Recall=0.9698, Attack Precision=0.5432

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
0.15       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409   <--
0.20       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.25       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.30       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.35       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.40       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.45       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.50       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.55       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.60       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.65       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.70       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.75       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
0.80       0.7132   0.7718   0.4565   0.9380   0.9698   0.6409  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7132, F1=0.7718, Normal Recall=0.4565, Normal Precision=0.9380, Attack Recall=0.9698, Attack Precision=0.6409

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
0.15       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649   <--
0.20       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.25       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.30       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.35       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.40       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.45       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.50       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.55       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.60       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.65       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.70       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.75       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.80       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5056, F1=0.2818, Normal Recall=0.4539, Normal Precision=0.9927, Attack Recall=0.9702, Attack Precision=0.1649

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
0.15       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080   <--
0.20       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.25       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.30       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.35       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.40       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.45       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.50       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.55       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.60       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.65       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.70       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.75       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.80       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5580, F1=0.4676, Normal Recall=0.4549, Normal Precision=0.9840, Attack Recall=0.9704, Attack Precision=0.3080

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
0.15       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327   <--
0.20       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.25       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.30       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.35       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.40       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.45       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.50       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.55       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.60       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.65       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.70       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.75       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.80       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6095, F1=0.5986, Normal Recall=0.4549, Normal Precision=0.9728, Attack Recall=0.9704, Attack Precision=0.4327

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
0.15       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430   <--
0.20       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.25       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.30       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.35       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.40       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.45       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.50       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.55       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.60       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.65       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.70       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.75       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.80       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6615, F1=0.6963, Normal Recall=0.4555, Normal Precision=0.9584, Attack Recall=0.9704, Attack Precision=0.5430

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
0.15       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407   <--
0.20       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.25       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.30       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.35       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.40       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.45       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.50       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.55       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.60       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.65       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.70       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.75       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.80       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7130, F1=0.7718, Normal Recall=0.4557, Normal Precision=0.9389, Attack Recall=0.9704, Attack Precision=0.6407

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
0.15       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649   <--
0.20       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.25       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.30       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.35       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.40       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.45       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.50       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.55       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.60       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.65       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.70       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.75       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
0.80       0.5056   0.2818   0.4539   0.9927   0.9702   0.1649  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5056, F1=0.2818, Normal Recall=0.4539, Normal Precision=0.9927, Attack Recall=0.9702, Attack Precision=0.1649

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
0.15       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080   <--
0.20       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.25       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.30       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.35       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.40       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.45       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.50       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.55       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.60       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.65       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.70       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.75       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
0.80       0.5580   0.4676   0.4549   0.9840   0.9704   0.3080  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5580, F1=0.4676, Normal Recall=0.4549, Normal Precision=0.9840, Attack Recall=0.9704, Attack Precision=0.3080

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
0.15       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327   <--
0.20       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.25       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.30       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.35       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.40       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.45       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.50       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.55       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.60       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.65       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.70       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.75       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
0.80       0.6095   0.5986   0.4549   0.9728   0.9704   0.4327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6095, F1=0.5986, Normal Recall=0.4549, Normal Precision=0.9728, Attack Recall=0.9704, Attack Precision=0.4327

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
0.15       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430   <--
0.20       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.25       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.30       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.35       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.40       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.45       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.50       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.55       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.60       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.65       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.70       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.75       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
0.80       0.6615   0.6963   0.4555   0.9584   0.9704   0.5430  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6615, F1=0.6963, Normal Recall=0.4555, Normal Precision=0.9584, Attack Recall=0.9704, Attack Precision=0.5430

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
0.15       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407   <--
0.20       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.25       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.30       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.35       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.40       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.45       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.50       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.55       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.60       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.65       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.70       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.75       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
0.80       0.7130   0.7718   0.4557   0.9389   0.9704   0.6407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7130, F1=0.7718, Normal Recall=0.4557, Normal Precision=0.9389, Attack Recall=0.9704, Attack Precision=0.6407

```

