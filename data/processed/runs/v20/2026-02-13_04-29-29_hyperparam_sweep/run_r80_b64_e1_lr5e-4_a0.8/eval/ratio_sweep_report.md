# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr5e-4_a0.8.yaml` |
| **Generated** | 2026-02-16 07:54:56 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1609 | 0.2437 | 0.3269 | 0.4117 | 0.4938 | 0.5776 | 0.6605 | 0.7438 | 0.8261 | 0.9105 | 0.9930 |
| QAT+Prune only | 0.7617 | 0.7858 | 0.8088 | 0.8333 | 0.8559 | 0.8775 | 0.9025 | 0.9265 | 0.9499 | 0.9725 | 0.9969 |
| QAT+PTQ | 0.7612 | 0.7855 | 0.8086 | 0.8333 | 0.8558 | 0.8773 | 0.9026 | 0.9266 | 0.9499 | 0.9727 | 0.9971 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7612 | 0.7855 | 0.8086 | 0.8333 | 0.8558 | 0.8773 | 0.9026 | 0.9266 | 0.9499 | 0.9727 | 0.9971 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2081 | 0.3711 | 0.5032 | 0.6108 | 0.7015 | 0.7783 | 0.8444 | 0.9014 | 0.9523 | 0.9965 |
| QAT+Prune only | 0.0000 | 0.4822 | 0.6760 | 0.7820 | 0.8470 | 0.8906 | 0.9247 | 0.9500 | 0.9695 | 0.9849 | 0.9985 |
| QAT+PTQ | 0.0000 | 0.4819 | 0.6757 | 0.7821 | 0.8469 | 0.8904 | 0.9247 | 0.9500 | 0.9695 | 0.9850 | 0.9986 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4819 | 0.6757 | 0.7821 | 0.8469 | 0.8904 | 0.9247 | 0.9500 | 0.9695 | 0.9850 | 0.9986 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.1609 | 0.1604 | 0.1604 | 0.1625 | 0.1611 | 0.1621 | 0.1617 | 0.1622 | 0.1586 | 0.1684 | 0.0000 |
| QAT+Prune only | 0.7617 | 0.7623 | 0.7618 | 0.7632 | 0.7619 | 0.7581 | 0.7610 | 0.7624 | 0.7618 | 0.7532 | 0.0000 |
| QAT+PTQ | 0.7612 | 0.7620 | 0.7614 | 0.7631 | 0.7616 | 0.7575 | 0.7607 | 0.7619 | 0.7609 | 0.7532 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7612 | 0.7620 | 0.7614 | 0.7631 | 0.7616 | 0.7575 | 0.7607 | 0.7619 | 0.7609 | 0.7532 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1609 | 0.0000 | 0.0000 | 0.0000 | 0.1609 | 1.0000 |
| 90 | 10 | 299,940 | 0.2437 | 0.1162 | 0.9936 | 0.2081 | 0.1604 | 0.9956 |
| 80 | 20 | 291,350 | 0.3269 | 0.2282 | 0.9930 | 0.3711 | 0.1604 | 0.9892 |
| 70 | 30 | 194,230 | 0.4117 | 0.3369 | 0.9930 | 0.5032 | 0.1625 | 0.9819 |
| 60 | 40 | 145,675 | 0.4938 | 0.4411 | 0.9930 | 0.6108 | 0.1611 | 0.9718 |
| 50 | 50 | 116,540 | 0.5776 | 0.5424 | 0.9930 | 0.7015 | 0.1621 | 0.9586 |
| 40 | 60 | 97,115 | 0.6605 | 0.6399 | 0.9930 | 0.7783 | 0.1617 | 0.9390 |
| 30 | 70 | 83,240 | 0.7438 | 0.7344 | 0.9930 | 0.8444 | 0.1622 | 0.9085 |
| 20 | 80 | 72,835 | 0.8261 | 0.8252 | 0.9930 | 0.9014 | 0.1586 | 0.8499 |
| 10 | 90 | 64,740 | 0.9105 | 0.9149 | 0.9930 | 0.9523 | 0.1684 | 0.7276 |
| 0 | 100 | 58,270 | 0.9930 | 1.0000 | 0.9930 | 0.9965 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7617 | 0.0000 | 0.0000 | 0.0000 | 0.7617 | 1.0000 |
| 90 | 10 | 299,940 | 0.7858 | 0.3180 | 0.9972 | 0.4822 | 0.7623 | 0.9996 |
| 80 | 20 | 291,350 | 0.8088 | 0.5113 | 0.9969 | 0.6760 | 0.7618 | 0.9990 |
| 70 | 30 | 194,230 | 0.8333 | 0.6434 | 0.9969 | 0.7820 | 0.7632 | 0.9983 |
| 60 | 40 | 145,675 | 0.8559 | 0.7363 | 0.9969 | 0.8470 | 0.7619 | 0.9973 |
| 50 | 50 | 116,540 | 0.8775 | 0.8047 | 0.9969 | 0.8906 | 0.7581 | 0.9959 |
| 40 | 60 | 97,115 | 0.9025 | 0.8622 | 0.9969 | 0.9247 | 0.7610 | 0.9939 |
| 30 | 70 | 83,240 | 0.9265 | 0.9073 | 0.9969 | 0.9500 | 0.7624 | 0.9906 |
| 20 | 80 | 72,835 | 0.9499 | 0.9436 | 0.9969 | 0.9695 | 0.7618 | 0.9840 |
| 10 | 90 | 64,740 | 0.9725 | 0.9732 | 0.9969 | 0.9849 | 0.7532 | 0.9644 |
| 0 | 100 | 58,270 | 0.9969 | 1.0000 | 0.9969 | 0.9985 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7612 | 0.0000 | 0.0000 | 0.0000 | 0.7612 | 1.0000 |
| 90 | 10 | 299,940 | 0.7855 | 0.3177 | 0.9975 | 0.4819 | 0.7620 | 0.9996 |
| 80 | 20 | 291,350 | 0.8086 | 0.5110 | 0.9971 | 0.6757 | 0.7614 | 0.9991 |
| 70 | 30 | 194,230 | 0.8333 | 0.6434 | 0.9971 | 0.7821 | 0.7631 | 0.9984 |
| 60 | 40 | 145,675 | 0.8558 | 0.7360 | 0.9971 | 0.8469 | 0.7616 | 0.9975 |
| 50 | 50 | 116,540 | 0.8773 | 0.8043 | 0.9971 | 0.8904 | 0.7575 | 0.9962 |
| 40 | 60 | 97,115 | 0.9026 | 0.8621 | 0.9971 | 0.9247 | 0.7607 | 0.9944 |
| 30 | 70 | 83,240 | 0.9266 | 0.9072 | 0.9971 | 0.9500 | 0.7619 | 0.9913 |
| 20 | 80 | 72,835 | 0.9499 | 0.9434 | 0.9971 | 0.9695 | 0.7609 | 0.9852 |
| 10 | 90 | 64,740 | 0.9727 | 0.9732 | 0.9971 | 0.9850 | 0.7532 | 0.9669 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7612 | 0.0000 | 0.0000 | 0.0000 | 0.7612 | 1.0000 |
| 90 | 10 | 299,940 | 0.7855 | 0.3177 | 0.9975 | 0.4819 | 0.7620 | 0.9996 |
| 80 | 20 | 291,350 | 0.8086 | 0.5110 | 0.9971 | 0.6757 | 0.7614 | 0.9991 |
| 70 | 30 | 194,230 | 0.8333 | 0.6434 | 0.9971 | 0.7821 | 0.7631 | 0.9984 |
| 60 | 40 | 145,675 | 0.8558 | 0.7360 | 0.9971 | 0.8469 | 0.7616 | 0.9975 |
| 50 | 50 | 116,540 | 0.8773 | 0.8043 | 0.9971 | 0.8904 | 0.7575 | 0.9962 |
| 40 | 60 | 97,115 | 0.9026 | 0.8621 | 0.9971 | 0.9247 | 0.7607 | 0.9944 |
| 30 | 70 | 83,240 | 0.9266 | 0.9072 | 0.9971 | 0.9500 | 0.7619 | 0.9913 |
| 20 | 80 | 72,835 | 0.9499 | 0.9434 | 0.9971 | 0.9695 | 0.7609 | 0.9852 |
| 10 | 90 | 64,740 | 0.9727 | 0.9732 | 0.9971 | 0.9850 | 0.7532 | 0.9669 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9986 | 0.0000 | 0.0000 |


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
0.15       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162   <--
0.20       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.25       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.30       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.35       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.40       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.45       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.50       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.55       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.60       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.65       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.70       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.75       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
0.80       0.2437   0.2080   0.1604   0.9953   0.9932   0.1162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2437, F1=0.2080, Normal Recall=0.1604, Normal Precision=0.9953, Attack Recall=0.9932, Attack Precision=0.1162

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
0.15       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282   <--
0.20       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.25       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.30       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.35       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.40       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.45       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.50       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.55       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.60       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.65       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.70       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.75       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
0.80       0.3268   0.3711   0.1603   0.9892   0.9930   0.2282  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3268, F1=0.3711, Normal Recall=0.1603, Normal Precision=0.9892, Attack Recall=0.9930, Attack Precision=0.2282

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
0.15       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365   <--
0.20       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.25       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.30       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.35       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.40       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.45       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.50       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.55       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.60       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.65       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.70       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.75       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
0.80       0.4106   0.5027   0.1610   0.9817   0.9930   0.3365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4106, F1=0.5027, Normal Recall=0.1610, Normal Precision=0.9817, Attack Recall=0.9930, Attack Precision=0.3365

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
0.15       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411   <--
0.20       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.25       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.30       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.35       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.40       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.45       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.50       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.55       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.60       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.65       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.70       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.75       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
0.80       0.4938   0.6108   0.1610   0.9718   0.9930   0.4411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4938, F1=0.6108, Normal Recall=0.1610, Normal Precision=0.9718, Attack Recall=0.9930, Attack Precision=0.4411

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
0.15       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420   <--
0.20       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.25       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.30       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.35       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.40       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.45       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.50       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.55       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.60       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.65       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.70       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.75       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
0.80       0.5770   0.7013   0.1610   0.9583   0.9930   0.5420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5770, F1=0.7013, Normal Recall=0.1610, Normal Precision=0.9583, Attack Recall=0.9930, Attack Precision=0.5420

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
0.15       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179   <--
0.20       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.25       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.30       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.35       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.40       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.45       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.50       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.55       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.60       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.65       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.70       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.75       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
0.80       0.7858   0.4821   0.7623   0.9996   0.9970   0.3179  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7858, F1=0.4821, Normal Recall=0.7623, Normal Precision=0.9996, Attack Recall=0.9970, Attack Precision=0.3179

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
0.15       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128   <--
0.20       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.25       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.30       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.35       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.40       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.45       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.50       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.55       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.60       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.65       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.70       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.75       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
0.80       0.8099   0.6772   0.7632   0.9990   0.9969   0.5128  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8099, F1=0.6772, Normal Recall=0.7632, Normal Precision=0.9990, Attack Recall=0.9969, Attack Precision=0.5128

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
0.15       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422   <--
0.20       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.25       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.30       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.35       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.40       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.45       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.50       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.55       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.60       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.65       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.70       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.75       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
0.80       0.8324   0.7812   0.7619   0.9983   0.9969   0.6422  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8324, F1=0.7812, Normal Recall=0.7619, Normal Precision=0.9983, Attack Recall=0.9969, Attack Precision=0.6422

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
0.15       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360   <--
0.20       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.25       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.30       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.35       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.40       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.45       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.50       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.55       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.60       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.65       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.70       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.75       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
0.80       0.8557   0.8468   0.7616   0.9973   0.9969   0.7360  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8557, F1=0.8468, Normal Recall=0.7616, Normal Precision=0.9973, Attack Recall=0.9969, Attack Precision=0.7360

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
0.15       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062   <--
0.20       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.25       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.30       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.35       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.40       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.45       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.50       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.55       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.60       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.65       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.70       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.75       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
0.80       0.8786   0.8914   0.7603   0.9960   0.9969   0.8062  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8786, F1=0.8914, Normal Recall=0.7603, Normal Precision=0.9960, Attack Recall=0.9969, Attack Precision=0.8062

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
0.15       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176   <--
0.20       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.25       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.30       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.35       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.40       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.45       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.50       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.55       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.60       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.65       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.70       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.75       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.80       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7855, F1=0.4818, Normal Recall=0.7620, Normal Precision=0.9996, Attack Recall=0.9973, Attack Precision=0.3176

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
0.15       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125   <--
0.20       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.25       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.30       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.35       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.40       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.45       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.50       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.55       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.60       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.65       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.70       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.75       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.80       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8097, F1=0.6770, Normal Recall=0.7629, Normal Precision=0.9991, Attack Recall=0.9971, Attack Precision=0.5125

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
0.15       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418   <--
0.20       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.25       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.30       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.35       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.40       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.45       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.50       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.55       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.60       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.65       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.70       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.75       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.80       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8322, F1=0.7809, Normal Recall=0.7615, Normal Precision=0.9984, Attack Recall=0.9971, Attack Precision=0.6418

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
0.15       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356   <--
0.20       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.25       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.30       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.35       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.40       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.45       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.50       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.55       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.60       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.65       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.70       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.75       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.80       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8555, F1=0.8466, Normal Recall=0.7611, Normal Precision=0.9975, Attack Recall=0.9971, Attack Precision=0.7356

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
0.15       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058   <--
0.20       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.25       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.30       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.35       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.40       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.45       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.50       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.55       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.60       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.65       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.70       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.75       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.80       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8784, F1=0.8913, Normal Recall=0.7596, Normal Precision=0.9962, Attack Recall=0.9971, Attack Precision=0.8058

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
0.15       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176   <--
0.20       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.25       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.30       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.35       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.40       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.45       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.50       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.55       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.60       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.65       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.70       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.75       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
0.80       0.7855   0.4818   0.7620   0.9996   0.9973   0.3176  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7855, F1=0.4818, Normal Recall=0.7620, Normal Precision=0.9996, Attack Recall=0.9973, Attack Precision=0.3176

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
0.15       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125   <--
0.20       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.25       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.30       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.35       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.40       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.45       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.50       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.55       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.60       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.65       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.70       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.75       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
0.80       0.8097   0.6770   0.7629   0.9991   0.9971   0.5125  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8097, F1=0.6770, Normal Recall=0.7629, Normal Precision=0.9991, Attack Recall=0.9971, Attack Precision=0.5125

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
0.15       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418   <--
0.20       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.25       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.30       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.35       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.40       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.45       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.50       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.55       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.60       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.65       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.70       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.75       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
0.80       0.8322   0.7809   0.7615   0.9984   0.9971   0.6418  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8322, F1=0.7809, Normal Recall=0.7615, Normal Precision=0.9984, Attack Recall=0.9971, Attack Precision=0.6418

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
0.15       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356   <--
0.20       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.25       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.30       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.35       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.40       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.45       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.50       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.55       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.60       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.65       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.70       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.75       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
0.80       0.8555   0.8466   0.7611   0.9975   0.9971   0.7356  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8555, F1=0.8466, Normal Recall=0.7611, Normal Precision=0.9975, Attack Recall=0.9971, Attack Precision=0.7356

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
0.15       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058   <--
0.20       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.25       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.30       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.35       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.40       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.45       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.50       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.55       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.60       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.65       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.70       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.75       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
0.80       0.8784   0.8913   0.7596   0.9962   0.9971   0.8058  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8784, F1=0.8913, Normal Recall=0.7596, Normal Precision=0.9962, Attack Recall=0.9971, Attack Precision=0.8058

```

