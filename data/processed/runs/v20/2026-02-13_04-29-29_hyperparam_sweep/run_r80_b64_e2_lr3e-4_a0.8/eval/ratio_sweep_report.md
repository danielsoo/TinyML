# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-16 17:31:29 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3649 | 0.4171 | 0.4693 | 0.5221 | 0.5774 | 0.6273 | 0.6806 | 0.7321 | 0.7837 | 0.8378 | 0.8895 |
| QAT+Prune only | 0.8154 | 0.8319 | 0.8489 | 0.8668 | 0.8824 | 0.9001 | 0.9170 | 0.9360 | 0.9522 | 0.9688 | 0.9866 |
| QAT+PTQ | 0.8108 | 0.8283 | 0.8461 | 0.8645 | 0.8807 | 0.8994 | 0.9166 | 0.9363 | 0.9532 | 0.9703 | 0.9888 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8108 | 0.8283 | 0.8461 | 0.8645 | 0.8807 | 0.8994 | 0.9166 | 0.9363 | 0.9532 | 0.9703 | 0.9888 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2339 | 0.4014 | 0.5276 | 0.6274 | 0.7047 | 0.7697 | 0.8230 | 0.8681 | 0.9080 | 0.9415 |
| QAT+Prune only | 0.0000 | 0.5399 | 0.7232 | 0.8163 | 0.8703 | 0.9081 | 0.9345 | 0.9557 | 0.9706 | 0.9828 | 0.9933 |
| QAT+PTQ | 0.0000 | 0.5352 | 0.7199 | 0.8141 | 0.8689 | 0.9076 | 0.9343 | 0.9560 | 0.9712 | 0.9836 | 0.9943 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5352 | 0.7199 | 0.8141 | 0.8689 | 0.9076 | 0.9343 | 0.9560 | 0.9712 | 0.9836 | 0.9943 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3649 | 0.3646 | 0.3643 | 0.3646 | 0.3693 | 0.3650 | 0.3672 | 0.3648 | 0.3604 | 0.3723 | 0.0000 |
| QAT+Prune only | 0.8154 | 0.8147 | 0.8145 | 0.8155 | 0.8129 | 0.8136 | 0.8126 | 0.8178 | 0.8146 | 0.8089 | 0.0000 |
| QAT+PTQ | 0.8108 | 0.8105 | 0.8104 | 0.8113 | 0.8086 | 0.8100 | 0.8083 | 0.8140 | 0.8107 | 0.8043 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8108 | 0.8105 | 0.8104 | 0.8113 | 0.8086 | 0.8100 | 0.8083 | 0.8140 | 0.8107 | 0.8043 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3649 | 0.0000 | 0.0000 | 0.0000 | 0.3649 | 1.0000 |
| 90 | 10 | 299,940 | 0.4171 | 0.1346 | 0.8896 | 0.2339 | 0.3646 | 0.9675 |
| 80 | 20 | 291,350 | 0.4693 | 0.2592 | 0.8895 | 0.4014 | 0.3643 | 0.9295 |
| 70 | 30 | 194,230 | 0.5221 | 0.3750 | 0.8895 | 0.5276 | 0.3646 | 0.8851 |
| 60 | 40 | 145,675 | 0.5774 | 0.4846 | 0.8895 | 0.6274 | 0.3693 | 0.8338 |
| 50 | 50 | 116,540 | 0.6273 | 0.5835 | 0.8895 | 0.7047 | 0.3650 | 0.7677 |
| 40 | 60 | 97,115 | 0.6806 | 0.6783 | 0.8895 | 0.7697 | 0.3672 | 0.6891 |
| 30 | 70 | 83,240 | 0.7321 | 0.7657 | 0.8895 | 0.8230 | 0.3648 | 0.5860 |
| 20 | 80 | 72,835 | 0.7837 | 0.8476 | 0.8895 | 0.8681 | 0.3604 | 0.4493 |
| 10 | 90 | 64,740 | 0.8378 | 0.9273 | 0.8896 | 0.9080 | 0.3723 | 0.2725 |
| 0 | 100 | 58,270 | 0.8895 | 1.0000 | 0.8895 | 0.9415 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8154 | 0.0000 | 0.0000 | 0.0000 | 0.8154 | 1.0000 |
| 90 | 10 | 299,940 | 0.8319 | 0.3717 | 0.9866 | 0.5399 | 0.8147 | 0.9982 |
| 80 | 20 | 291,350 | 0.8489 | 0.5708 | 0.9866 | 0.7232 | 0.8145 | 0.9959 |
| 70 | 30 | 194,230 | 0.8668 | 0.6962 | 0.9866 | 0.8163 | 0.8155 | 0.9930 |
| 60 | 40 | 145,675 | 0.8824 | 0.7785 | 0.9866 | 0.8703 | 0.8129 | 0.9891 |
| 50 | 50 | 116,540 | 0.9001 | 0.8411 | 0.9866 | 0.9081 | 0.8136 | 0.9838 |
| 40 | 60 | 97,115 | 0.9170 | 0.8876 | 0.9866 | 0.9345 | 0.8126 | 0.9759 |
| 30 | 70 | 83,240 | 0.9360 | 0.9267 | 0.9866 | 0.9557 | 0.8178 | 0.9632 |
| 20 | 80 | 72,835 | 0.9522 | 0.9551 | 0.9866 | 0.9706 | 0.8146 | 0.9383 |
| 10 | 90 | 64,740 | 0.9688 | 0.9789 | 0.9866 | 0.9828 | 0.8089 | 0.8704 |
| 0 | 100 | 58,270 | 0.9866 | 1.0000 | 0.9866 | 0.9933 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8108 | 0.0000 | 0.0000 | 0.0000 | 0.8108 | 1.0000 |
| 90 | 10 | 299,940 | 0.8283 | 0.3669 | 0.9887 | 0.5352 | 0.8105 | 0.9985 |
| 80 | 20 | 291,350 | 0.8461 | 0.5660 | 0.9888 | 0.7199 | 0.8104 | 0.9965 |
| 70 | 30 | 194,230 | 0.8645 | 0.6919 | 0.9888 | 0.8141 | 0.8113 | 0.9941 |
| 60 | 40 | 145,675 | 0.8807 | 0.7750 | 0.9888 | 0.8689 | 0.8086 | 0.9908 |
| 50 | 50 | 116,540 | 0.8994 | 0.8388 | 0.9888 | 0.9076 | 0.8100 | 0.9863 |
| 40 | 60 | 97,115 | 0.9166 | 0.8856 | 0.9888 | 0.9343 | 0.8083 | 0.9796 |
| 30 | 70 | 83,240 | 0.9363 | 0.9254 | 0.9888 | 0.9560 | 0.8140 | 0.9688 |
| 20 | 80 | 72,835 | 0.9532 | 0.9543 | 0.9888 | 0.9712 | 0.8107 | 0.9475 |
| 10 | 90 | 64,740 | 0.9703 | 0.9785 | 0.9888 | 0.9836 | 0.8043 | 0.8884 |
| 0 | 100 | 58,270 | 0.9888 | 1.0000 | 0.9888 | 0.9943 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8108 | 0.0000 | 0.0000 | 0.0000 | 0.8108 | 1.0000 |
| 90 | 10 | 299,940 | 0.8283 | 0.3669 | 0.9887 | 0.5352 | 0.8105 | 0.9985 |
| 80 | 20 | 291,350 | 0.8461 | 0.5660 | 0.9888 | 0.7199 | 0.8104 | 0.9965 |
| 70 | 30 | 194,230 | 0.8645 | 0.6919 | 0.9888 | 0.8141 | 0.8113 | 0.9941 |
| 60 | 40 | 145,675 | 0.8807 | 0.7750 | 0.9888 | 0.8689 | 0.8086 | 0.9908 |
| 50 | 50 | 116,540 | 0.8994 | 0.8388 | 0.9888 | 0.9076 | 0.8100 | 0.9863 |
| 40 | 60 | 97,115 | 0.9166 | 0.8856 | 0.9888 | 0.9343 | 0.8083 | 0.9796 |
| 30 | 70 | 83,240 | 0.9363 | 0.9254 | 0.9888 | 0.9560 | 0.8140 | 0.9688 |
| 20 | 80 | 72,835 | 0.9532 | 0.9543 | 0.9888 | 0.9712 | 0.8107 | 0.9475 |
| 10 | 90 | 64,740 | 0.9703 | 0.9785 | 0.9888 | 0.9836 | 0.8043 | 0.8884 |
| 0 | 100 | 58,270 | 0.9888 | 1.0000 | 0.9888 | 0.9943 | 0.0000 | 0.0000 |


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
0.15       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348   <--
0.20       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.25       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.30       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.35       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.40       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.45       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.50       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.55       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.60       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.65       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.70       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.75       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
0.80       0.4172   0.2341   0.3645   0.9678   0.8907   0.1348  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4172, F1=0.2341, Normal Recall=0.3645, Normal Precision=0.9678, Attack Recall=0.8907, Attack Precision=0.1348

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
0.15       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593   <--
0.20       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.25       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.30       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.35       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.40       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.45       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.50       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.55       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.60       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.65       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.70       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.75       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
0.80       0.4697   0.4016   0.3648   0.9296   0.8895   0.2593  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4697, F1=0.4016, Normal Recall=0.3648, Normal Precision=0.9296, Attack Recall=0.8895, Attack Precision=0.2593

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
0.15       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756   <--
0.20       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.25       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.30       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.35       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.40       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.45       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.50       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.55       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.60       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.65       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.70       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.75       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
0.80       0.5232   0.5282   0.3662   0.8855   0.8895   0.3756  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5232, F1=0.5282, Normal Recall=0.3662, Normal Precision=0.8855, Attack Recall=0.8895, Attack Precision=0.3756

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
0.15       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827   <--
0.20       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.25       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.30       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.35       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.40       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.45       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.50       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.55       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.60       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.65       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.70       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.75       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
0.80       0.5744   0.6258   0.3644   0.8319   0.8895   0.4827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5744, F1=0.6258, Normal Recall=0.3644, Normal Precision=0.8319, Attack Recall=0.8895, Attack Precision=0.4827

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
0.15       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827   <--
0.20       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.25       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.30       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.35       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.40       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.45       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.50       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.55       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.60       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.65       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.70       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.75       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
0.80       0.6262   0.7041   0.3629   0.7667   0.8895   0.5827  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6262, F1=0.7041, Normal Recall=0.3629, Normal Precision=0.7667, Attack Recall=0.8895, Attack Precision=0.5827

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
0.15       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716   <--
0.20       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.25       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.30       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.35       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.40       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.45       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.50       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.55       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.60       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.65       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.70       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.75       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
0.80       0.8318   0.5398   0.8147   0.9981   0.9862   0.3716  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8318, F1=0.5398, Normal Recall=0.8147, Normal Precision=0.9981, Attack Recall=0.9862, Attack Precision=0.3716

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
0.15       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717   <--
0.20       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.25       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.30       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.35       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.40       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.45       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.50       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.55       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.60       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.65       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.70       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.75       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
0.80       0.8495   0.7239   0.8152   0.9959   0.9866   0.5717  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8495, F1=0.7239, Normal Recall=0.8152, Normal Precision=0.9959, Attack Recall=0.9866, Attack Precision=0.5717

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
0.15       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966   <--
0.20       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.25       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.30       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.35       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.40       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.45       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.50       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.55       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.60       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.65       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.70       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.75       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
0.80       0.8671   0.8166   0.8158   0.9930   0.9866   0.6966  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8671, F1=0.8166, Normal Recall=0.8158, Normal Precision=0.9930, Attack Recall=0.9866, Attack Precision=0.6966

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
0.15       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814   <--
0.20       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.25       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.30       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.35       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.40       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.45       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.50       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.55       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.60       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.65       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.70       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.75       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
0.80       0.8842   0.8721   0.8160   0.9892   0.9866   0.7814  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8842, F1=0.8721, Normal Recall=0.8160, Normal Precision=0.9892, Attack Recall=0.9866, Attack Precision=0.7814

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
0.15       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426   <--
0.20       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.25       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.30       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.35       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.40       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.45       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.50       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.55       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.60       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.65       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.70       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.75       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
0.80       0.9012   0.9089   0.8157   0.9838   0.9866   0.8426  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9012, F1=0.9089, Normal Recall=0.8157, Normal Precision=0.9838, Attack Recall=0.9866, Attack Precision=0.8426

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
0.15       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669   <--
0.20       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.25       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.30       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.35       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.40       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.45       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.50       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.55       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.60       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.65       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.70       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.75       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.80       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8283, F1=0.5351, Normal Recall=0.8105, Normal Precision=0.9984, Attack Recall=0.9884, Attack Precision=0.3669

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
0.15       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667   <--
0.20       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.25       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.30       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.35       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.40       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.45       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.50       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.55       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.60       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.65       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.70       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.75       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.80       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.7205, Normal Recall=0.8110, Normal Precision=0.9965, Attack Recall=0.9888, Attack Precision=0.5667

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
0.15       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922   <--
0.20       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.25       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.30       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.35       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.40       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.45       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.50       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.55       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.60       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.65       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.70       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.75       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.80       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8647, F1=0.8143, Normal Recall=0.8116, Normal Precision=0.9941, Attack Recall=0.9888, Attack Precision=0.6922

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
0.15       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776   <--
0.20       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.25       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.30       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.35       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.40       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.45       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.50       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.55       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.60       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.65       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.70       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.75       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.80       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8824, F1=0.8706, Normal Recall=0.8115, Normal Precision=0.9909, Attack Recall=0.9888, Attack Precision=0.7776

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
0.15       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396   <--
0.20       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.25       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.30       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.35       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.40       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.45       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.50       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.55       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.60       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.65       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.70       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.75       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.80       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8999, F1=0.9081, Normal Recall=0.8111, Normal Precision=0.9863, Attack Recall=0.9888, Attack Precision=0.8396

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
0.15       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669   <--
0.20       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.25       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.30       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.35       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.40       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.45       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.50       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.55       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.60       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.65       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.70       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.75       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
0.80       0.8283   0.5351   0.8105   0.9984   0.9884   0.3669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8283, F1=0.5351, Normal Recall=0.8105, Normal Precision=0.9984, Attack Recall=0.9884, Attack Precision=0.3669

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
0.15       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667   <--
0.20       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.25       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.30       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.35       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.40       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.45       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.50       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.55       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.60       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.65       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.70       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.75       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
0.80       0.8466   0.7205   0.8110   0.9965   0.9888   0.5667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8466, F1=0.7205, Normal Recall=0.8110, Normal Precision=0.9965, Attack Recall=0.9888, Attack Precision=0.5667

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
0.15       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922   <--
0.20       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.25       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.30       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.35       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.40       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.45       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.50       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.55       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.60       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.65       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.70       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.75       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
0.80       0.8647   0.8143   0.8116   0.9941   0.9888   0.6922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8647, F1=0.8143, Normal Recall=0.8116, Normal Precision=0.9941, Attack Recall=0.9888, Attack Precision=0.6922

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
0.15       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776   <--
0.20       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.25       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.30       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.35       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.40       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.45       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.50       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.55       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.60       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.65       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.70       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.75       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
0.80       0.8824   0.8706   0.8115   0.9909   0.9888   0.7776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8824, F1=0.8706, Normal Recall=0.8115, Normal Precision=0.9909, Attack Recall=0.9888, Attack Precision=0.7776

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
0.15       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396   <--
0.20       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.25       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.30       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.35       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.40       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.45       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.50       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.55       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.60       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.65       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.70       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.75       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
0.80       0.8999   0.9081   0.8111   0.9863   0.9888   0.8396  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8999, F1=0.9081, Normal Recall=0.8111, Normal Precision=0.9863, Attack Recall=0.9888, Attack Precision=0.8396

```

