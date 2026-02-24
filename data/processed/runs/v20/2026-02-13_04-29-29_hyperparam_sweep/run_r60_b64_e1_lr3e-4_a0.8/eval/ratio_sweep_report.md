# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-13 07:38:17 |

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
| Original (TFLite) | 0.9369 | 0.9178 | 0.8984 | 0.8790 | 0.8586 | 0.8394 | 0.8197 | 0.8009 | 0.7805 | 0.7610 | 0.7418 |
| QAT+Prune only | 0.8632 | 0.8459 | 0.8284 | 0.8120 | 0.7949 | 0.7770 | 0.7607 | 0.7443 | 0.7273 | 0.7102 | 0.6936 |
| QAT+PTQ | 0.8627 | 0.8453 | 0.8279 | 0.8116 | 0.7945 | 0.7767 | 0.7604 | 0.7440 | 0.7271 | 0.7101 | 0.6936 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8627 | 0.8453 | 0.8279 | 0.8116 | 0.7945 | 0.7767 | 0.7604 | 0.7440 | 0.7271 | 0.7101 | 0.6936 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6434 | 0.7449 | 0.7863 | 0.8076 | 0.8220 | 0.8316 | 0.8391 | 0.8439 | 0.8482 | 0.8517 |
| QAT+Prune only | 0.0000 | 0.4735 | 0.6178 | 0.6888 | 0.7302 | 0.7567 | 0.7767 | 0.7916 | 0.8027 | 0.8116 | 0.8191 |
| QAT+PTQ | 0.0000 | 0.4726 | 0.6172 | 0.6884 | 0.7298 | 0.7564 | 0.7764 | 0.7914 | 0.8026 | 0.8116 | 0.8191 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4726 | 0.6172 | 0.6884 | 0.7298 | 0.7564 | 0.7764 | 0.7914 | 0.8026 | 0.8116 | 0.8191 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9369 | 0.9375 | 0.9375 | 0.9378 | 0.9365 | 0.9371 | 0.9367 | 0.9390 | 0.9355 | 0.9337 | 0.0000 |
| QAT+Prune only | 0.8632 | 0.8628 | 0.8621 | 0.8627 | 0.8625 | 0.8604 | 0.8613 | 0.8624 | 0.8619 | 0.8597 | 0.0000 |
| QAT+PTQ | 0.8627 | 0.8622 | 0.8615 | 0.8622 | 0.8618 | 0.8597 | 0.8605 | 0.8616 | 0.8608 | 0.8585 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8627 | 0.8622 | 0.8615 | 0.8622 | 0.8618 | 0.8597 | 0.8605 | 0.8616 | 0.8608 | 0.8585 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9369 | 0.0000 | 0.0000 | 0.0000 | 0.9369 | 1.0000 |
| 90 | 10 | 299,940 | 0.9178 | 0.5684 | 0.7412 | 0.6434 | 0.9375 | 0.9702 |
| 80 | 20 | 291,350 | 0.8984 | 0.7480 | 0.7418 | 0.7449 | 0.9375 | 0.9356 |
| 70 | 30 | 194,230 | 0.8790 | 0.8364 | 0.7418 | 0.7863 | 0.9378 | 0.8944 |
| 60 | 40 | 145,675 | 0.8586 | 0.8862 | 0.7418 | 0.8076 | 0.9365 | 0.8447 |
| 50 | 50 | 116,540 | 0.8394 | 0.9218 | 0.7418 | 0.8220 | 0.9371 | 0.7840 |
| 40 | 60 | 97,115 | 0.8197 | 0.9462 | 0.7418 | 0.8316 | 0.9367 | 0.7075 |
| 30 | 70 | 83,240 | 0.8009 | 0.9659 | 0.7418 | 0.8391 | 0.9390 | 0.6091 |
| 20 | 80 | 72,835 | 0.7805 | 0.9787 | 0.7418 | 0.8439 | 0.9355 | 0.4752 |
| 10 | 90 | 64,740 | 0.7610 | 0.9902 | 0.7418 | 0.8482 | 0.9337 | 0.2866 |
| 0 | 100 | 58,270 | 0.7418 | 1.0000 | 0.7418 | 0.8517 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8632 | 0.0000 | 0.0000 | 0.0000 | 0.8632 | 1.0000 |
| 90 | 10 | 299,940 | 0.8459 | 0.3596 | 0.6932 | 0.4735 | 0.8628 | 0.9620 |
| 80 | 20 | 291,350 | 0.8284 | 0.5570 | 0.6936 | 0.6178 | 0.8621 | 0.9184 |
| 70 | 30 | 194,230 | 0.8120 | 0.6840 | 0.6936 | 0.6888 | 0.8627 | 0.8679 |
| 60 | 40 | 145,675 | 0.7949 | 0.7708 | 0.6936 | 0.7302 | 0.8625 | 0.8085 |
| 50 | 50 | 116,540 | 0.7770 | 0.8325 | 0.6936 | 0.7567 | 0.8604 | 0.7374 |
| 40 | 60 | 97,115 | 0.7607 | 0.8824 | 0.6936 | 0.7767 | 0.8613 | 0.6521 |
| 30 | 70 | 83,240 | 0.7443 | 0.9217 | 0.6936 | 0.7916 | 0.8624 | 0.5468 |
| 20 | 80 | 72,835 | 0.7273 | 0.9526 | 0.6936 | 0.8027 | 0.8619 | 0.4129 |
| 10 | 90 | 64,740 | 0.7102 | 0.9780 | 0.6936 | 0.8116 | 0.8597 | 0.2377 |
| 0 | 100 | 58,270 | 0.6936 | 1.0000 | 0.6936 | 0.8191 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8627 | 0.0000 | 0.0000 | 0.0000 | 0.8627 | 1.0000 |
| 90 | 10 | 299,940 | 0.8453 | 0.3585 | 0.6931 | 0.4726 | 0.8622 | 0.9620 |
| 80 | 20 | 291,350 | 0.8279 | 0.5559 | 0.6936 | 0.6172 | 0.8615 | 0.9183 |
| 70 | 30 | 194,230 | 0.8116 | 0.6833 | 0.6936 | 0.6884 | 0.8622 | 0.8678 |
| 60 | 40 | 145,675 | 0.7945 | 0.7700 | 0.6936 | 0.7298 | 0.8618 | 0.8084 |
| 50 | 50 | 116,540 | 0.7767 | 0.8318 | 0.6936 | 0.7564 | 0.8597 | 0.7373 |
| 40 | 60 | 97,115 | 0.7604 | 0.8818 | 0.6936 | 0.7764 | 0.8605 | 0.6518 |
| 30 | 70 | 83,240 | 0.7440 | 0.9212 | 0.6936 | 0.7914 | 0.8616 | 0.5465 |
| 20 | 80 | 72,835 | 0.7271 | 0.9522 | 0.6936 | 0.8026 | 0.8608 | 0.4126 |
| 10 | 90 | 64,740 | 0.7101 | 0.9778 | 0.6936 | 0.8116 | 0.8585 | 0.2374 |
| 0 | 100 | 58,270 | 0.6936 | 1.0000 | 0.6936 | 0.8191 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8627 | 0.0000 | 0.0000 | 0.0000 | 0.8627 | 1.0000 |
| 90 | 10 | 299,940 | 0.8453 | 0.3585 | 0.6931 | 0.4726 | 0.8622 | 0.9620 |
| 80 | 20 | 291,350 | 0.8279 | 0.5559 | 0.6936 | 0.6172 | 0.8615 | 0.9183 |
| 70 | 30 | 194,230 | 0.8116 | 0.6833 | 0.6936 | 0.6884 | 0.8622 | 0.8678 |
| 60 | 40 | 145,675 | 0.7945 | 0.7700 | 0.6936 | 0.7298 | 0.8618 | 0.8084 |
| 50 | 50 | 116,540 | 0.7767 | 0.8318 | 0.6936 | 0.7564 | 0.8597 | 0.7373 |
| 40 | 60 | 97,115 | 0.7604 | 0.8818 | 0.6936 | 0.7764 | 0.8605 | 0.6518 |
| 30 | 70 | 83,240 | 0.7440 | 0.9212 | 0.6936 | 0.7914 | 0.8616 | 0.5465 |
| 20 | 80 | 72,835 | 0.7271 | 0.9522 | 0.6936 | 0.8026 | 0.8608 | 0.4126 |
| 10 | 90 | 64,740 | 0.7101 | 0.9778 | 0.6936 | 0.8116 | 0.8585 | 0.2374 |
| 0 | 100 | 58,270 | 0.6936 | 1.0000 | 0.6936 | 0.8191 | 0.0000 | 0.0000 |


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
0.15       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675   <--
0.20       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.25       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.30       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.35       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.40       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.45       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.50       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.55       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.60       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.65       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.70       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.75       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
0.80       0.9176   0.6418   0.9375   0.9699   0.7386   0.5675  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9176, F1=0.6418, Normal Recall=0.9375, Normal Precision=0.9699, Attack Recall=0.7386, Attack Precision=0.5675

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
0.15       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480   <--
0.20       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.25       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.30       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.35       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.40       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.45       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.50       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.55       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.60       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.65       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.70       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.75       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
0.80       0.8984   0.7449   0.9375   0.9356   0.7418   0.7480  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8984, F1=0.7449, Normal Recall=0.9375, Normal Precision=0.9356, Attack Recall=0.7418, Attack Precision=0.7480

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
0.15       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358   <--
0.20       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.25       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.30       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.35       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.40       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.45       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.50       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.55       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.60       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.65       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.70       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.75       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
0.80       0.8788   0.7860   0.9375   0.8944   0.7418   0.8358  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8788, F1=0.7860, Normal Recall=0.9375, Normal Precision=0.8944, Attack Recall=0.7418, Attack Precision=0.8358

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
0.15       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868   <--
0.20       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.25       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.30       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.35       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.40       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.45       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.50       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.55       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.60       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.65       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.70       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.75       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
0.80       0.8588   0.8078   0.9369   0.8448   0.7418   0.8868  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8588, F1=0.8078, Normal Recall=0.9369, Normal Precision=0.8448, Attack Recall=0.7418, Attack Precision=0.8868

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
0.15       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216   <--
0.20       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.25       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.30       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.35       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.40       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.45       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.50       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.55       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.60       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.65       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.70       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.75       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
0.80       0.8393   0.8220   0.9369   0.7839   0.7418   0.9216  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8393, F1=0.8220, Normal Recall=0.9369, Normal Precision=0.7839, Attack Recall=0.7418, Attack Precision=0.9216

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
0.15       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584   <--
0.20       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.25       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.30       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.35       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.40       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.45       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.50       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.55       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.60       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.65       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.70       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.75       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
0.80       0.8455   0.4717   0.8628   0.9616   0.6897   0.3584  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8455, F1=0.4717, Normal Recall=0.8628, Normal Precision=0.9616, Attack Recall=0.6897, Attack Precision=0.3584

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
0.15       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593   <--
0.20       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.25       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.30       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.35       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.40       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.45       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.50       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.55       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.60       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.65       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.70       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.75       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
0.80       0.8294   0.6193   0.8634   0.9185   0.6936   0.5593  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8294, F1=0.6193, Normal Recall=0.8634, Normal Precision=0.9185, Attack Recall=0.6936, Attack Precision=0.5593

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
0.15       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853   <--
0.20       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.25       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.30       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.35       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.40       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.45       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.50       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.55       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.60       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.65       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.70       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.75       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
0.80       0.8125   0.6894   0.8635   0.8680   0.6936   0.6853  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8125, F1=0.6894, Normal Recall=0.8635, Normal Precision=0.8680, Attack Recall=0.6936, Attack Precision=0.6853

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
0.15       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712   <--
0.20       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.25       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.30       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.35       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.40       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.45       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.50       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.55       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.60       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.65       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.70       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.75       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
0.80       0.7952   0.7304   0.8628   0.8086   0.6936   0.7712  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7952, F1=0.7304, Normal Recall=0.8628, Normal Precision=0.8086, Attack Recall=0.6936, Attack Precision=0.7712

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
0.15       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327   <--
0.20       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.25       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.30       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.35       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.40       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.45       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.50       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.55       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.60       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.65       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.70       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.75       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
0.80       0.7771   0.7568   0.8607   0.7375   0.6936   0.8327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7771, F1=0.7568, Normal Recall=0.8607, Normal Precision=0.7375, Attack Recall=0.6936, Attack Precision=0.8327

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
0.15       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574   <--
0.20       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.25       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.30       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.35       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.40       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.45       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.50       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.55       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.60       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.65       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.70       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.75       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.80       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8450, F1=0.4708, Normal Recall=0.8622, Normal Precision=0.9615, Attack Recall=0.6897, Attack Precision=0.3574

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
0.15       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583   <--
0.20       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.25       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.30       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.35       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.40       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.45       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.50       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.55       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.60       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.65       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.70       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.75       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.80       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8290, F1=0.6187, Normal Recall=0.8628, Normal Precision=0.9185, Attack Recall=0.6936, Attack Precision=0.5583

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
0.15       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844   <--
0.20       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.25       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.30       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.35       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.40       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.45       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.50       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.55       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.60       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.65       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.70       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.75       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.80       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8121, F1=0.6890, Normal Recall=0.8629, Normal Precision=0.8679, Attack Recall=0.6936, Attack Precision=0.6844

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
0.15       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706   <--
0.20       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.25       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.30       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.35       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.40       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.45       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.50       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.55       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.60       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.65       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.70       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.75       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.80       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7949, F1=0.7301, Normal Recall=0.8623, Normal Precision=0.8085, Attack Recall=0.6936, Attack Precision=0.7706

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
0.15       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321   <--
0.20       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.25       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.30       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.35       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.40       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.45       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.50       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.55       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.60       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.65       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.70       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.75       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.80       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7768, F1=0.7566, Normal Recall=0.8600, Normal Precision=0.7373, Attack Recall=0.6936, Attack Precision=0.8321

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
0.15       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574   <--
0.20       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.25       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.30       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.35       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.40       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.45       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.50       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.55       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.60       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.65       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.70       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.75       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
0.80       0.8450   0.4708   0.8622   0.9615   0.6897   0.3574  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8450, F1=0.4708, Normal Recall=0.8622, Normal Precision=0.9615, Attack Recall=0.6897, Attack Precision=0.3574

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
0.15       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583   <--
0.20       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.25       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.30       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.35       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.40       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.45       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.50       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.55       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.60       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.65       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.70       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.75       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
0.80       0.8290   0.6187   0.8628   0.9185   0.6936   0.5583  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8290, F1=0.6187, Normal Recall=0.8628, Normal Precision=0.9185, Attack Recall=0.6936, Attack Precision=0.5583

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
0.15       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844   <--
0.20       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.25       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.30       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.35       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.40       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.45       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.50       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.55       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.60       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.65       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.70       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.75       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
0.80       0.8121   0.6890   0.8629   0.8679   0.6936   0.6844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8121, F1=0.6890, Normal Recall=0.8629, Normal Precision=0.8679, Attack Recall=0.6936, Attack Precision=0.6844

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
0.15       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706   <--
0.20       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.25       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.30       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.35       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.40       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.45       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.50       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.55       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.60       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.65       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.70       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.75       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
0.80       0.7949   0.7301   0.8623   0.8085   0.6936   0.7706  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7949, F1=0.7301, Normal Recall=0.8623, Normal Precision=0.8085, Attack Recall=0.6936, Attack Precision=0.7706

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
0.15       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321   <--
0.20       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.25       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.30       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.35       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.40       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.45       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.50       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.55       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.60       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.65       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.70       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.75       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
0.80       0.7768   0.7566   0.8600   0.7373   0.6936   0.8321  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7768, F1=0.7566, Normal Recall=0.8600, Normal Precision=0.7373, Attack Recall=0.6936, Attack Precision=0.8321

```

