# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-17 23:26:00 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1622 | 0.2469 | 0.3304 | 0.4145 | 0.4986 | 0.5808 | 0.6651 | 0.7494 | 0.8323 | 0.9162 | 0.9999 |
| QAT+Prune only | 0.8965 | 0.9017 | 0.9067 | 0.9123 | 0.9168 | 0.9211 | 0.9272 | 0.9313 | 0.9370 | 0.9412 | 0.9468 |
| QAT+PTQ | 0.8968 | 0.9018 | 0.9065 | 0.9117 | 0.9160 | 0.9199 | 0.9259 | 0.9297 | 0.9350 | 0.9390 | 0.9442 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8968 | 0.9018 | 0.9065 | 0.9117 | 0.9160 | 0.9199 | 0.9259 | 0.9297 | 0.9350 | 0.9390 | 0.9442 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2098 | 0.3740 | 0.5061 | 0.6147 | 0.7046 | 0.7818 | 0.8482 | 0.9051 | 0.9555 | 1.0000 |
| QAT+Prune only | 0.0000 | 0.6582 | 0.8023 | 0.8662 | 0.9010 | 0.9231 | 0.9398 | 0.9507 | 0.9601 | 0.9667 | 0.9727 |
| QAT+PTQ | 0.0000 | 0.6580 | 0.8016 | 0.8652 | 0.9000 | 0.9218 | 0.9386 | 0.9495 | 0.9587 | 0.9653 | 0.9713 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6580 | 0.8016 | 0.8652 | 0.9000 | 0.9218 | 0.9386 | 0.9495 | 0.9587 | 0.9653 | 0.9713 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1622 | 0.1632 | 0.1630 | 0.1636 | 0.1644 | 0.1616 | 0.1629 | 0.1649 | 0.1618 | 0.1627 | 0.0000 |
| QAT+Prune only | 0.8965 | 0.8967 | 0.8966 | 0.8975 | 0.8968 | 0.8954 | 0.8978 | 0.8953 | 0.8981 | 0.8914 | 0.0000 |
| QAT+PTQ | 0.8968 | 0.8971 | 0.8971 | 0.8978 | 0.8972 | 0.8955 | 0.8984 | 0.8959 | 0.8980 | 0.8919 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8968 | 0.8971 | 0.8971 | 0.8978 | 0.8972 | 0.8955 | 0.8984 | 0.8959 | 0.8980 | 0.8919 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1622 | 0.0000 | 0.0000 | 0.0000 | 0.1622 | 1.0000 |
| 90 | 10 | 299,940 | 0.2469 | 0.1172 | 0.9999 | 0.2098 | 0.1632 | 1.0000 |
| 80 | 20 | 291,350 | 0.3304 | 0.2300 | 0.9999 | 0.3740 | 0.1630 | 0.9999 |
| 70 | 30 | 194,230 | 0.4145 | 0.3388 | 0.9999 | 0.5061 | 0.1636 | 0.9998 |
| 60 | 40 | 145,675 | 0.4986 | 0.4437 | 0.9999 | 0.6147 | 0.1644 | 0.9997 |
| 50 | 50 | 116,540 | 0.5808 | 0.5439 | 0.9999 | 0.7046 | 0.1616 | 0.9996 |
| 40 | 60 | 97,115 | 0.6651 | 0.6418 | 0.9999 | 0.7818 | 0.1629 | 0.9994 |
| 30 | 70 | 83,240 | 0.7494 | 0.7364 | 0.9999 | 0.8482 | 0.1649 | 0.9990 |
| 20 | 80 | 72,835 | 0.8323 | 0.8267 | 0.9999 | 0.9051 | 0.1618 | 0.9983 |
| 10 | 90 | 64,740 | 0.9162 | 0.9149 | 0.9999 | 0.9555 | 0.1627 | 0.9962 |
| 0 | 100 | 58,270 | 0.9999 | 1.0000 | 0.9999 | 1.0000 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8965 | 0.0000 | 0.0000 | 0.0000 | 0.8965 | 1.0000 |
| 90 | 10 | 299,940 | 0.9017 | 0.5045 | 0.9466 | 0.6582 | 0.8967 | 0.9934 |
| 80 | 20 | 291,350 | 0.9067 | 0.6961 | 0.9468 | 0.8023 | 0.8966 | 0.9854 |
| 70 | 30 | 194,230 | 0.9123 | 0.7983 | 0.9468 | 0.8662 | 0.8975 | 0.9752 |
| 60 | 40 | 145,675 | 0.9168 | 0.8595 | 0.9468 | 0.9010 | 0.8968 | 0.9619 |
| 50 | 50 | 116,540 | 0.9211 | 0.9005 | 0.9468 | 0.9231 | 0.8954 | 0.9439 |
| 40 | 60 | 97,115 | 0.9272 | 0.9329 | 0.9468 | 0.9398 | 0.8978 | 0.9183 |
| 30 | 70 | 83,240 | 0.9313 | 0.9548 | 0.9468 | 0.9507 | 0.8953 | 0.8782 |
| 20 | 80 | 72,835 | 0.9370 | 0.9738 | 0.9468 | 0.9601 | 0.8981 | 0.8084 |
| 10 | 90 | 64,740 | 0.9412 | 0.9874 | 0.9468 | 0.9667 | 0.8914 | 0.6504 |
| 0 | 100 | 58,270 | 0.9468 | 1.0000 | 0.9468 | 0.9727 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8968 | 0.0000 | 0.0000 | 0.0000 | 0.8968 | 1.0000 |
| 90 | 10 | 299,940 | 0.9018 | 0.5049 | 0.9442 | 0.6580 | 0.8971 | 0.9931 |
| 80 | 20 | 291,350 | 0.9065 | 0.6964 | 0.9442 | 0.8016 | 0.8971 | 0.9847 |
| 70 | 30 | 194,230 | 0.9117 | 0.7983 | 0.9442 | 0.8652 | 0.8978 | 0.9741 |
| 60 | 40 | 145,675 | 0.9160 | 0.8597 | 0.9442 | 0.9000 | 0.8972 | 0.9602 |
| 50 | 50 | 116,540 | 0.9199 | 0.9004 | 0.9442 | 0.9218 | 0.8955 | 0.9414 |
| 40 | 60 | 97,115 | 0.9259 | 0.9330 | 0.9442 | 0.9386 | 0.8984 | 0.9148 |
| 30 | 70 | 83,240 | 0.9297 | 0.9549 | 0.9442 | 0.9495 | 0.8959 | 0.8731 |
| 20 | 80 | 72,835 | 0.9350 | 0.9737 | 0.9442 | 0.9587 | 0.8980 | 0.8010 |
| 10 | 90 | 64,740 | 0.9390 | 0.9874 | 0.9442 | 0.9653 | 0.8919 | 0.6398 |
| 0 | 100 | 58,270 | 0.9442 | 1.0000 | 0.9442 | 0.9713 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8968 | 0.0000 | 0.0000 | 0.0000 | 0.8968 | 1.0000 |
| 90 | 10 | 299,940 | 0.9018 | 0.5049 | 0.9442 | 0.6580 | 0.8971 | 0.9931 |
| 80 | 20 | 291,350 | 0.9065 | 0.6964 | 0.9442 | 0.8016 | 0.8971 | 0.9847 |
| 70 | 30 | 194,230 | 0.9117 | 0.7983 | 0.9442 | 0.8652 | 0.8978 | 0.9741 |
| 60 | 40 | 145,675 | 0.9160 | 0.8597 | 0.9442 | 0.9000 | 0.8972 | 0.9602 |
| 50 | 50 | 116,540 | 0.9199 | 0.9004 | 0.9442 | 0.9218 | 0.8955 | 0.9414 |
| 40 | 60 | 97,115 | 0.9259 | 0.9330 | 0.9442 | 0.9386 | 0.8984 | 0.9148 |
| 30 | 70 | 83,240 | 0.9297 | 0.9549 | 0.9442 | 0.9495 | 0.8959 | 0.8731 |
| 20 | 80 | 72,835 | 0.9350 | 0.9737 | 0.9442 | 0.9587 | 0.8980 | 0.8010 |
| 10 | 90 | 64,740 | 0.9390 | 0.9874 | 0.9442 | 0.9653 | 0.8919 | 0.6398 |
| 0 | 100 | 58,270 | 0.9442 | 1.0000 | 0.9442 | 0.9713 | 0.0000 | 0.0000 |


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
0.15       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172   <--
0.20       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.25       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.30       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.35       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.40       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.45       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.50       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.55       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.60       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.65       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.70       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.75       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
0.80       0.2469   0.2098   0.1632   1.0000   0.9999   0.1172  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2469, F1=0.2098, Normal Recall=0.1632, Normal Precision=1.0000, Attack Recall=0.9999, Attack Precision=0.1172

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
0.15       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300   <--
0.20       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.25       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.30       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.35       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.40       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.45       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.50       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.55       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.60       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.65       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.70       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.75       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
0.80       0.3303   0.3739   0.1629   0.9999   0.9999   0.2300  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3303, F1=0.3739, Normal Recall=0.1629, Normal Precision=0.9999, Attack Recall=0.9999, Attack Precision=0.2300

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
0.15       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387   <--
0.20       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.25       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.30       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.35       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.40       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.45       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.50       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.55       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.60       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.65       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.70       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.75       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
0.80       0.4144   0.5060   0.1634   0.9998   0.9999   0.3387  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4144, F1=0.5060, Normal Recall=0.1634, Normal Precision=0.9998, Attack Recall=0.9999, Attack Precision=0.3387

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
0.15       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431   <--
0.20       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.25       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.30       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.35       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.40       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.45       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.50       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.55       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.60       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.65       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.70       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.75       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
0.80       0.4972   0.6140   0.1621   0.9997   0.9999   0.4431  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4972, F1=0.6140, Normal Recall=0.1621, Normal Precision=0.9997, Attack Recall=0.9999, Attack Precision=0.4431

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
0.15       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439   <--
0.20       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.25       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.30       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.35       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.40       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.45       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.50       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.55       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.60       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.65       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.70       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.75       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
0.80       0.5807   0.7046   0.1615   0.9996   0.9999   0.5439  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5807, F1=0.7046, Normal Recall=0.1615, Normal Precision=0.9996, Attack Recall=0.9999, Attack Precision=0.5439

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
0.15       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045   <--
0.20       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.25       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.30       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.35       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.40       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.45       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.50       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.55       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.60       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.65       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.70       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.75       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
0.80       0.9017   0.6582   0.8967   0.9934   0.9467   0.5045  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9017, F1=0.6582, Normal Recall=0.8967, Normal Precision=0.9934, Attack Recall=0.9467, Attack Precision=0.5045

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
0.15       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966   <--
0.20       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.25       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.30       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.35       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.40       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.45       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.50       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.55       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.60       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.65       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.70       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.75       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
0.80       0.9069   0.8026   0.8969   0.9854   0.9468   0.6966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9069, F1=0.8026, Normal Recall=0.8969, Normal Precision=0.9854, Attack Recall=0.9468, Attack Precision=0.6966

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
0.15       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965   <--
0.20       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.25       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.30       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.35       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.40       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.45       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.50       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.55       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.60       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.65       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.70       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.75       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
0.80       0.9115   0.8652   0.8963   0.9752   0.9468   0.7965  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9115, F1=0.8652, Normal Recall=0.8963, Normal Precision=0.9752, Attack Recall=0.9468, Attack Precision=0.7965

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
0.15       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593   <--
0.20       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.25       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.30       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.35       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.40       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.45       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.50       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.55       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.60       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.65       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.70       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.75       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
0.80       0.9167   0.9009   0.8967   0.9619   0.9468   0.8593  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9167, F1=0.9009, Normal Recall=0.8967, Normal Precision=0.9619, Attack Recall=0.9468, Attack Precision=0.8593

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
0.15       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015   <--
0.20       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.25       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.30       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.35       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.40       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.45       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.50       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.55       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.60       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.65       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.70       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.75       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
0.80       0.9216   0.9236   0.8965   0.9439   0.9468   0.9015  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9216, F1=0.9236, Normal Recall=0.8965, Normal Precision=0.9439, Attack Recall=0.9468, Attack Precision=0.9015

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
0.15       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049   <--
0.20       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.25       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.30       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.35       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.40       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.45       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.50       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.55       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.60       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.65       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.70       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.75       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.80       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9019, F1=0.6580, Normal Recall=0.8971, Normal Precision=0.9932, Attack Recall=0.9444, Attack Precision=0.5049

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
0.15       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968   <--
0.20       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.25       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.30       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.35       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.40       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.45       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.50       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.55       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.60       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.65       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.70       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.75       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.80       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9067, F1=0.8019, Normal Recall=0.8973, Normal Precision=0.9847, Attack Recall=0.9442, Attack Precision=0.6968

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
0.15       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967   <--
0.20       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.25       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.30       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.35       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.40       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.45       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.50       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.55       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.60       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.65       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.70       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.75       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.80       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9110, F1=0.8642, Normal Recall=0.8967, Normal Precision=0.9740, Attack Recall=0.9442, Attack Precision=0.7967

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
0.15       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593   <--
0.20       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.25       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.30       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.35       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.40       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.45       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.50       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.55       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.60       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.65       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.70       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.75       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.80       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9159, F1=0.8998, Normal Recall=0.8970, Normal Precision=0.9602, Attack Recall=0.9442, Attack Precision=0.8593

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
0.15       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015   <--
0.20       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.25       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.30       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.35       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.40       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.45       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.50       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.55       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.60       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.65       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.70       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.75       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.80       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9205, F1=0.9223, Normal Recall=0.8968, Normal Precision=0.9414, Attack Recall=0.9442, Attack Precision=0.9015

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
0.15       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049   <--
0.20       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.25       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.30       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.35       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.40       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.45       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.50       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.55       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.60       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.65       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.70       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.75       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
0.80       0.9019   0.6580   0.8971   0.9932   0.9444   0.5049  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9019, F1=0.6580, Normal Recall=0.8971, Normal Precision=0.9932, Attack Recall=0.9444, Attack Precision=0.5049

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
0.15       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968   <--
0.20       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.25       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.30       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.35       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.40       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.45       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.50       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.55       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.60       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.65       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.70       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.75       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
0.80       0.9067   0.8019   0.8973   0.9847   0.9442   0.6968  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9067, F1=0.8019, Normal Recall=0.8973, Normal Precision=0.9847, Attack Recall=0.9442, Attack Precision=0.6968

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
0.15       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967   <--
0.20       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.25       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.30       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.35       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.40       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.45       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.50       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.55       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.60       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.65       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.70       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.75       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
0.80       0.9110   0.8642   0.8967   0.9740   0.9442   0.7967  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9110, F1=0.8642, Normal Recall=0.8967, Normal Precision=0.9740, Attack Recall=0.9442, Attack Precision=0.7967

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
0.15       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593   <--
0.20       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.25       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.30       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.35       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.40       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.45       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.50       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.55       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.60       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.65       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.70       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.75       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
0.80       0.9159   0.8998   0.8970   0.9602   0.9442   0.8593  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9159, F1=0.8998, Normal Recall=0.8970, Normal Precision=0.9602, Attack Recall=0.9442, Attack Precision=0.8593

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
0.15       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015   <--
0.20       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.25       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.30       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.35       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.40       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.45       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.50       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.55       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.60       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.65       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.70       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.75       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
0.80       0.9205   0.9223   0.8968   0.9414   0.9442   0.9015  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9205, F1=0.9223, Normal Recall=0.8968, Normal Precision=0.9414, Attack Recall=0.9442, Attack Precision=0.9015

```

