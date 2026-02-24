# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-15 18:34:19 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4742 | 0.5270 | 0.5793 | 0.6329 | 0.6851 | 0.7358 | 0.7891 | 0.8410 | 0.8931 | 0.9456 | 0.9980 |
| QAT+Prune only | 0.8760 | 0.8845 | 0.8929 | 0.9024 | 0.9112 | 0.9182 | 0.9282 | 0.9369 | 0.9457 | 0.9546 | 0.9635 |
| QAT+PTQ | 0.8749 | 0.8836 | 0.8921 | 0.9018 | 0.9106 | 0.9178 | 0.9278 | 0.9366 | 0.9456 | 0.9547 | 0.9637 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8749 | 0.8836 | 0.8921 | 0.9018 | 0.9106 | 0.9178 | 0.9278 | 0.9366 | 0.9456 | 0.9547 | 0.9637 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2968 | 0.4869 | 0.6199 | 0.7171 | 0.7907 | 0.8503 | 0.8978 | 0.9372 | 0.9706 | 0.9990 |
| QAT+Prune only | 0.0000 | 0.6255 | 0.7825 | 0.8556 | 0.8967 | 0.9218 | 0.9415 | 0.9553 | 0.9660 | 0.9745 | 0.9814 |
| QAT+PTQ | 0.0000 | 0.6235 | 0.7812 | 0.8548 | 0.8961 | 0.9214 | 0.9413 | 0.9551 | 0.9659 | 0.9745 | 0.9815 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6235 | 0.7812 | 0.8548 | 0.8961 | 0.9214 | 0.9413 | 0.9551 | 0.9659 | 0.9745 | 0.9815 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4742 | 0.4747 | 0.4746 | 0.4764 | 0.4764 | 0.4736 | 0.4757 | 0.4745 | 0.4735 | 0.4744 | 0.0000 |
| QAT+Prune only | 0.8760 | 0.8757 | 0.8752 | 0.8762 | 0.8763 | 0.8729 | 0.8751 | 0.8747 | 0.8744 | 0.8741 | 0.0000 |
| QAT+PTQ | 0.8749 | 0.8746 | 0.8742 | 0.8752 | 0.8752 | 0.8719 | 0.8740 | 0.8734 | 0.8733 | 0.8735 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8749 | 0.8746 | 0.8742 | 0.8752 | 0.8752 | 0.8719 | 0.8740 | 0.8734 | 0.8733 | 0.8735 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4742 | 0.0000 | 0.0000 | 0.0000 | 0.4742 | 1.0000 |
| 90 | 10 | 299,940 | 0.5270 | 0.1743 | 0.9980 | 0.2968 | 0.4747 | 0.9995 |
| 80 | 20 | 291,350 | 0.5793 | 0.3220 | 0.9980 | 0.4869 | 0.4746 | 0.9989 |
| 70 | 30 | 194,230 | 0.6329 | 0.4496 | 0.9980 | 0.6199 | 0.4764 | 0.9982 |
| 60 | 40 | 145,675 | 0.6851 | 0.5596 | 0.9980 | 0.7171 | 0.4764 | 0.9972 |
| 50 | 50 | 116,540 | 0.7358 | 0.6547 | 0.9980 | 0.7907 | 0.4736 | 0.9958 |
| 40 | 60 | 97,115 | 0.7891 | 0.7406 | 0.9980 | 0.8503 | 0.4757 | 0.9937 |
| 30 | 70 | 83,240 | 0.8410 | 0.8159 | 0.9980 | 0.8978 | 0.4745 | 0.9902 |
| 20 | 80 | 72,835 | 0.8931 | 0.8835 | 0.9980 | 0.9372 | 0.4735 | 0.9833 |
| 10 | 90 | 64,740 | 0.9456 | 0.9447 | 0.9980 | 0.9706 | 0.4744 | 0.9633 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8760 | 0.0000 | 0.0000 | 0.0000 | 0.8760 | 1.0000 |
| 90 | 10 | 299,940 | 0.8845 | 0.4629 | 0.9642 | 0.6255 | 0.8757 | 0.9955 |
| 80 | 20 | 291,350 | 0.8929 | 0.6587 | 0.9635 | 0.7825 | 0.8752 | 0.9897 |
| 70 | 30 | 194,230 | 0.9024 | 0.7694 | 0.9635 | 0.8556 | 0.8762 | 0.9825 |
| 60 | 40 | 145,675 | 0.9112 | 0.8386 | 0.9635 | 0.8967 | 0.8763 | 0.9730 |
| 50 | 50 | 116,540 | 0.9182 | 0.8834 | 0.9635 | 0.9218 | 0.8729 | 0.9599 |
| 40 | 60 | 97,115 | 0.9282 | 0.9205 | 0.9635 | 0.9415 | 0.8751 | 0.9412 |
| 30 | 70 | 83,240 | 0.9369 | 0.9472 | 0.9635 | 0.9553 | 0.8747 | 0.9114 |
| 20 | 80 | 72,835 | 0.9457 | 0.9684 | 0.9635 | 0.9660 | 0.8744 | 0.8571 |
| 10 | 90 | 64,740 | 0.9546 | 0.9857 | 0.9635 | 0.9745 | 0.8741 | 0.7271 |
| 0 | 100 | 58,270 | 0.9635 | 1.0000 | 0.9635 | 0.9814 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8749 | 0.0000 | 0.0000 | 0.0000 | 0.8749 | 1.0000 |
| 90 | 10 | 299,940 | 0.8836 | 0.4608 | 0.9642 | 0.6235 | 0.8746 | 0.9955 |
| 80 | 20 | 291,350 | 0.8921 | 0.6569 | 0.9637 | 0.7812 | 0.8742 | 0.9897 |
| 70 | 30 | 194,230 | 0.9018 | 0.7680 | 0.9637 | 0.8548 | 0.8752 | 0.9825 |
| 60 | 40 | 145,675 | 0.9106 | 0.8373 | 0.9637 | 0.8961 | 0.8752 | 0.9731 |
| 50 | 50 | 116,540 | 0.9178 | 0.8827 | 0.9637 | 0.9214 | 0.8719 | 0.9600 |
| 40 | 60 | 97,115 | 0.9278 | 0.9198 | 0.9637 | 0.9413 | 0.8740 | 0.9413 |
| 30 | 70 | 83,240 | 0.9366 | 0.9467 | 0.9637 | 0.9551 | 0.8734 | 0.9116 |
| 20 | 80 | 72,835 | 0.9456 | 0.9682 | 0.9637 | 0.9659 | 0.8733 | 0.8574 |
| 10 | 90 | 64,740 | 0.9547 | 0.9856 | 0.9637 | 0.9745 | 0.8735 | 0.7277 |
| 0 | 100 | 58,270 | 0.9637 | 1.0000 | 0.9637 | 0.9815 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8749 | 0.0000 | 0.0000 | 0.0000 | 0.8749 | 1.0000 |
| 90 | 10 | 299,940 | 0.8836 | 0.4608 | 0.9642 | 0.6235 | 0.8746 | 0.9955 |
| 80 | 20 | 291,350 | 0.8921 | 0.6569 | 0.9637 | 0.7812 | 0.8742 | 0.9897 |
| 70 | 30 | 194,230 | 0.9018 | 0.7680 | 0.9637 | 0.8548 | 0.8752 | 0.9825 |
| 60 | 40 | 145,675 | 0.9106 | 0.8373 | 0.9637 | 0.8961 | 0.8752 | 0.9731 |
| 50 | 50 | 116,540 | 0.9178 | 0.8827 | 0.9637 | 0.9214 | 0.8719 | 0.9600 |
| 40 | 60 | 97,115 | 0.9278 | 0.9198 | 0.9637 | 0.9413 | 0.8740 | 0.9413 |
| 30 | 70 | 83,240 | 0.9366 | 0.9467 | 0.9637 | 0.9551 | 0.8734 | 0.9116 |
| 20 | 80 | 72,835 | 0.9456 | 0.9682 | 0.9637 | 0.9659 | 0.8733 | 0.8574 |
| 10 | 90 | 64,740 | 0.9547 | 0.9856 | 0.9637 | 0.9745 | 0.8735 | 0.7277 |
| 0 | 100 | 58,270 | 0.9637 | 1.0000 | 0.9637 | 0.9815 | 0.0000 | 0.0000 |


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
0.15       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743   <--
0.20       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.25       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.30       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.35       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.40       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.45       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.50       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.55       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.60       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.65       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.70       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.75       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
0.80       0.5270   0.2968   0.4747   0.9996   0.9981   0.1743  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5270, F1=0.2968, Normal Recall=0.4747, Normal Precision=0.9996, Attack Recall=0.9981, Attack Precision=0.1743

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
0.15       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219   <--
0.20       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.25       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.30       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.35       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.40       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.45       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.50       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.55       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.60       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.65       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.70       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.75       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
0.80       0.5790   0.4867   0.4743   0.9989   0.9980   0.3219  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5790, F1=0.4867, Normal Recall=0.4743, Normal Precision=0.9989, Attack Recall=0.9980, Attack Precision=0.3219

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
0.15       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489   <--
0.20       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.25       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.30       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.35       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.40       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.45       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.50       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.55       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.60       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.65       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.70       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.75       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
0.80       0.6319   0.6193   0.4750   0.9982   0.9980   0.4489  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6319, F1=0.6193, Normal Recall=0.4750, Normal Precision=0.9982, Attack Recall=0.9980, Attack Precision=0.4489

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
0.15       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586   <--
0.20       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.25       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.30       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.35       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.40       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.45       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.50       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.55       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.60       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.65       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.70       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.75       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
0.80       0.6838   0.7163   0.4743   0.9972   0.9980   0.5586  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6838, F1=0.7163, Normal Recall=0.4743, Normal Precision=0.9972, Attack Recall=0.9980, Attack Precision=0.5586

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
0.15       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543   <--
0.20       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.25       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.30       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.35       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.40       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.45       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.50       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.55       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.60       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.65       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.70       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.75       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
0.80       0.7353   0.7904   0.4727   0.9958   0.9980   0.6543  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7353, F1=0.7904, Normal Recall=0.4727, Normal Precision=0.9958, Attack Recall=0.9980, Attack Precision=0.6543

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
0.15       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629   <--
0.20       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.25       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.30       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.35       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.40       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.45       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.50       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.55       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.60       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.65       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.70       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.75       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
0.80       0.8846   0.6256   0.8757   0.9955   0.9645   0.4629  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8846, F1=0.6256, Normal Recall=0.8757, Normal Precision=0.9955, Attack Recall=0.9645, Attack Precision=0.4629

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
0.15       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607   <--
0.20       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.25       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.30       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.35       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.40       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.45       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.50       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.55       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.60       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.65       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.70       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.75       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
0.80       0.8937   0.7839   0.8763   0.9897   0.9635   0.6607  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8937, F1=0.7839, Normal Recall=0.8763, Normal Precision=0.9897, Attack Recall=0.9635, Attack Precision=0.6607

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
0.15       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697   <--
0.20       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.25       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.30       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.35       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.40       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.45       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.50       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.55       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.60       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.65       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.70       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.75       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
0.80       0.9026   0.8558   0.8764   0.9825   0.9635   0.7697  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9026, F1=0.8558, Normal Recall=0.8764, Normal Precision=0.9825, Attack Recall=0.9635, Attack Precision=0.7697

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
0.15       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386   <--
0.20       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.25       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.30       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.35       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.40       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.45       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.50       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.55       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.60       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.65       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.70       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.75       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
0.80       0.9112   0.8967   0.8763   0.9730   0.9635   0.8386  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9112, F1=0.8967, Normal Recall=0.8763, Normal Precision=0.9730, Attack Recall=0.9635, Attack Precision=0.8386

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
0.15       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852   <--
0.20       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.25       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.30       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.35       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.40       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.45       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.50       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.55       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.60       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.65       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.70       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.75       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
0.80       0.9193   0.9227   0.8750   0.9600   0.9635   0.8852  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9193, F1=0.9227, Normal Recall=0.8750, Normal Precision=0.9600, Attack Recall=0.9635, Attack Precision=0.8852

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
0.15       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609   <--
0.20       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.25       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.30       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.35       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.40       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.45       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.50       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.55       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.60       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.65       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.70       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.75       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.80       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8836, F1=0.6238, Normal Recall=0.8746, Normal Precision=0.9955, Attack Recall=0.9647, Attack Precision=0.4609

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
0.15       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588   <--
0.20       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.25       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.30       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.35       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.40       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.45       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.50       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.55       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.60       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.65       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.70       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.75       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.80       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7826, Normal Recall=0.8752, Normal Precision=0.9897, Attack Recall=0.9637, Attack Precision=0.6588

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
0.15       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680   <--
0.20       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.25       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.30       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.35       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.40       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.45       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.50       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.55       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.60       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.65       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.70       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.75       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.80       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9018, F1=0.8548, Normal Recall=0.8753, Normal Precision=0.9825, Attack Recall=0.9637, Attack Precision=0.7680

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
0.15       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373   <--
0.20       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.25       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.30       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.35       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.40       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.45       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.50       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.55       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.60       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.65       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.70       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.75       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.80       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9106, F1=0.8961, Normal Recall=0.8752, Normal Precision=0.9731, Attack Recall=0.9637, Attack Precision=0.8373

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
0.15       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842   <--
0.20       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.25       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.30       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.35       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.40       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.45       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.50       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.55       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.60       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.65       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.70       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.75       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.80       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9187, F1=0.9222, Normal Recall=0.8738, Normal Precision=0.9601, Attack Recall=0.9637, Attack Precision=0.8842

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
0.15       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609   <--
0.20       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.25       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.30       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.35       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.40       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.45       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.50       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.55       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.60       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.65       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.70       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.75       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
0.80       0.8836   0.6238   0.8746   0.9955   0.9647   0.4609  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8836, F1=0.6238, Normal Recall=0.8746, Normal Precision=0.9955, Attack Recall=0.9647, Attack Precision=0.4609

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
0.15       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588   <--
0.20       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.25       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.30       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.35       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.40       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.45       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.50       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.55       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.60       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.65       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.70       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.75       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
0.80       0.8929   0.7826   0.8752   0.9897   0.9637   0.6588  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8929, F1=0.7826, Normal Recall=0.8752, Normal Precision=0.9897, Attack Recall=0.9637, Attack Precision=0.6588

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
0.15       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680   <--
0.20       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.25       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.30       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.35       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.40       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.45       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.50       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.55       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.60       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.65       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.70       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.75       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
0.80       0.9018   0.8548   0.8753   0.9825   0.9637   0.7680  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9018, F1=0.8548, Normal Recall=0.8753, Normal Precision=0.9825, Attack Recall=0.9637, Attack Precision=0.7680

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
0.15       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373   <--
0.20       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.25       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.30       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.35       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.40       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.45       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.50       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.55       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.60       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.65       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.70       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.75       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
0.80       0.9106   0.8961   0.8752   0.9731   0.9637   0.8373  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9106, F1=0.8961, Normal Recall=0.8752, Normal Precision=0.9731, Attack Recall=0.9637, Attack Precision=0.8373

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
0.15       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842   <--
0.20       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.25       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.30       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.35       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.40       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.45       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.50       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.55       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.60       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.65       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.70       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.75       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
0.80       0.9187   0.9222   0.8738   0.9601   0.9637   0.8842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9187, F1=0.9222, Normal Recall=0.8738, Normal Precision=0.9601, Attack Recall=0.9637, Attack Precision=0.8842

```

