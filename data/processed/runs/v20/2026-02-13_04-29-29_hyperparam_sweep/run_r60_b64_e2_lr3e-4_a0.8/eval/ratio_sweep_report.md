# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-13 19:41:19 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6613 | 0.6715 | 0.6818 | 0.6933 | 0.7015 | 0.7128 | 0.7242 | 0.7334 | 0.7430 | 0.7547 | 0.7647 |
| QAT+Prune only | 0.5739 | 0.6160 | 0.6579 | 0.6998 | 0.7434 | 0.7859 | 0.8281 | 0.8700 | 0.9135 | 0.9554 | 0.9984 |
| QAT+PTQ | 0.5707 | 0.6129 | 0.6551 | 0.6971 | 0.7413 | 0.7843 | 0.8266 | 0.8691 | 0.9129 | 0.9549 | 0.9984 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5707 | 0.6129 | 0.6551 | 0.6971 | 0.7413 | 0.7843 | 0.8266 | 0.8691 | 0.9129 | 0.9549 | 0.9984 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3179 | 0.4902 | 0.5993 | 0.6721 | 0.7270 | 0.7689 | 0.8006 | 0.8264 | 0.8487 | 0.8666 |
| QAT+Prune only | 0.0000 | 0.3421 | 0.5386 | 0.6662 | 0.7569 | 0.8234 | 0.8745 | 0.9149 | 0.9486 | 0.9758 | 0.9992 |
| QAT+PTQ | 0.0000 | 0.3403 | 0.5366 | 0.6642 | 0.7553 | 0.8223 | 0.8736 | 0.9144 | 0.9483 | 0.9755 | 0.9992 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3403 | 0.5366 | 0.6642 | 0.7553 | 0.8223 | 0.8736 | 0.9144 | 0.9483 | 0.9755 | 0.9992 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6613 | 0.6610 | 0.6611 | 0.6627 | 0.6594 | 0.6610 | 0.6636 | 0.6604 | 0.6566 | 0.6650 | 0.0000 |
| QAT+Prune only | 0.5739 | 0.5735 | 0.5727 | 0.5719 | 0.5734 | 0.5733 | 0.5725 | 0.5704 | 0.5738 | 0.5683 | 0.0000 |
| QAT+PTQ | 0.5707 | 0.5700 | 0.5692 | 0.5680 | 0.5698 | 0.5701 | 0.5690 | 0.5673 | 0.5709 | 0.5632 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5707 | 0.5700 | 0.5692 | 0.5680 | 0.5698 | 0.5701 | 0.5690 | 0.5673 | 0.5709 | 0.5632 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6613 | 0.0000 | 0.0000 | 0.0000 | 0.6613 | 1.0000 |
| 90 | 10 | 299,940 | 0.6715 | 0.2006 | 0.7656 | 0.3179 | 0.6610 | 0.9621 |
| 80 | 20 | 291,350 | 0.6818 | 0.3607 | 0.7647 | 0.4902 | 0.6611 | 0.9183 |
| 70 | 30 | 194,230 | 0.6933 | 0.4928 | 0.7647 | 0.5993 | 0.6627 | 0.8679 |
| 60 | 40 | 145,675 | 0.7015 | 0.5995 | 0.7647 | 0.6721 | 0.6594 | 0.8078 |
| 50 | 50 | 116,540 | 0.7128 | 0.6929 | 0.7647 | 0.7270 | 0.6610 | 0.7375 |
| 40 | 60 | 97,115 | 0.7242 | 0.7732 | 0.7647 | 0.7689 | 0.6636 | 0.6528 |
| 30 | 70 | 83,240 | 0.7334 | 0.8401 | 0.7647 | 0.8006 | 0.6604 | 0.5460 |
| 20 | 80 | 72,835 | 0.7430 | 0.8991 | 0.7647 | 0.8264 | 0.6566 | 0.4109 |
| 10 | 90 | 64,740 | 0.7547 | 0.9536 | 0.7646 | 0.8487 | 0.6650 | 0.2389 |
| 0 | 100 | 58,270 | 0.7647 | 1.0000 | 0.7647 | 0.8666 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5739 | 0.0000 | 0.0000 | 0.0000 | 0.5739 | 1.0000 |
| 90 | 10 | 299,940 | 0.6160 | 0.2064 | 0.9984 | 0.3421 | 0.5735 | 0.9997 |
| 80 | 20 | 291,350 | 0.6579 | 0.3688 | 0.9984 | 0.5386 | 0.5727 | 0.9993 |
| 70 | 30 | 194,230 | 0.6998 | 0.4999 | 0.9984 | 0.6662 | 0.5719 | 0.9988 |
| 60 | 40 | 145,675 | 0.7434 | 0.6094 | 0.9984 | 0.7569 | 0.5734 | 0.9981 |
| 50 | 50 | 116,540 | 0.7859 | 0.7006 | 0.9984 | 0.8234 | 0.5733 | 0.9972 |
| 40 | 60 | 97,115 | 0.8281 | 0.7780 | 0.9984 | 0.8745 | 0.5725 | 0.9958 |
| 30 | 70 | 83,240 | 0.8700 | 0.8443 | 0.9984 | 0.9149 | 0.5704 | 0.9935 |
| 20 | 80 | 72,835 | 0.9135 | 0.9036 | 0.9984 | 0.9486 | 0.5738 | 0.9890 |
| 10 | 90 | 64,740 | 0.9554 | 0.9542 | 0.9984 | 0.9758 | 0.5683 | 0.9753 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5707 | 0.0000 | 0.0000 | 0.0000 | 0.5707 | 1.0000 |
| 90 | 10 | 299,940 | 0.6129 | 0.2051 | 0.9984 | 0.3403 | 0.5700 | 0.9997 |
| 80 | 20 | 291,350 | 0.6551 | 0.3669 | 0.9984 | 0.5366 | 0.5692 | 0.9993 |
| 70 | 30 | 194,230 | 0.6971 | 0.4976 | 0.9984 | 0.6642 | 0.5680 | 0.9988 |
| 60 | 40 | 145,675 | 0.7413 | 0.6074 | 0.9984 | 0.7553 | 0.5698 | 0.9981 |
| 50 | 50 | 116,540 | 0.7843 | 0.6990 | 0.9984 | 0.8223 | 0.5701 | 0.9972 |
| 40 | 60 | 97,115 | 0.8266 | 0.7765 | 0.9984 | 0.8736 | 0.5690 | 0.9958 |
| 30 | 70 | 83,240 | 0.8691 | 0.8434 | 0.9984 | 0.9144 | 0.5673 | 0.9935 |
| 20 | 80 | 72,835 | 0.9129 | 0.9030 | 0.9984 | 0.9483 | 0.5709 | 0.9889 |
| 10 | 90 | 64,740 | 0.9549 | 0.9536 | 0.9984 | 0.9755 | 0.5632 | 0.9751 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5707 | 0.0000 | 0.0000 | 0.0000 | 0.5707 | 1.0000 |
| 90 | 10 | 299,940 | 0.6129 | 0.2051 | 0.9984 | 0.3403 | 0.5700 | 0.9997 |
| 80 | 20 | 291,350 | 0.6551 | 0.3669 | 0.9984 | 0.5366 | 0.5692 | 0.9993 |
| 70 | 30 | 194,230 | 0.6971 | 0.4976 | 0.9984 | 0.6642 | 0.5680 | 0.9988 |
| 60 | 40 | 145,675 | 0.7413 | 0.6074 | 0.9984 | 0.7553 | 0.5698 | 0.9981 |
| 50 | 50 | 116,540 | 0.7843 | 0.6990 | 0.9984 | 0.8223 | 0.5701 | 0.9972 |
| 40 | 60 | 97,115 | 0.8266 | 0.7765 | 0.9984 | 0.8736 | 0.5690 | 0.9958 |
| 30 | 70 | 83,240 | 0.8691 | 0.8434 | 0.9984 | 0.9144 | 0.5673 | 0.9935 |
| 20 | 80 | 72,835 | 0.9129 | 0.9030 | 0.9984 | 0.9483 | 0.5709 | 0.9889 |
| 10 | 90 | 64,740 | 0.9549 | 0.9536 | 0.9984 | 0.9755 | 0.5632 | 0.9751 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |


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
0.15       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005   <--
0.20       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.25       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.30       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.35       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.40       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.45       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.50       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.55       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.60       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.65       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.70       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.75       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
0.80       0.6714   0.3177   0.6610   0.9620   0.7651   0.2005  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6714, F1=0.3177, Normal Recall=0.6610, Normal Precision=0.9620, Attack Recall=0.7651, Attack Precision=0.2005

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
0.15       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603   <--
0.20       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.25       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.30       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.35       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.40       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.45       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.50       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.55       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.60       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.65       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.70       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.75       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
0.80       0.6815   0.4899   0.6607   0.9182   0.7647   0.3603  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6815, F1=0.4899, Normal Recall=0.6607, Normal Precision=0.9182, Attack Recall=0.7647, Attack Precision=0.3603

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
0.15       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914   <--
0.20       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.25       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.30       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.35       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.40       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.45       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.50       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.55       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.60       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.65       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.70       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.75       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
0.80       0.6919   0.5983   0.6608   0.8676   0.7647   0.4914  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6919, F1=0.5983, Normal Recall=0.6608, Normal Precision=0.8676, Attack Recall=0.7647, Attack Precision=0.4914

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
0.15       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014   <--
0.20       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.25       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.30       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.35       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.40       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.45       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.50       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.55       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.60       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.65       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.70       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.75       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
0.80       0.7031   0.6733   0.6621   0.8084   0.7647   0.6014  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7031, F1=0.6733, Normal Recall=0.6621, Normal Precision=0.8084, Attack Recall=0.7647, Attack Precision=0.6014

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
0.15       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946   <--
0.20       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.25       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.30       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.35       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.40       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.45       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.50       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.55       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.60       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.65       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.70       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.75       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
0.80       0.7143   0.7280   0.6638   0.7383   0.7647   0.6946  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7143, F1=0.7280, Normal Recall=0.6638, Normal Precision=0.7383, Attack Recall=0.7647, Attack Precision=0.6946

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
0.15       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064   <--
0.20       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.25       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.30       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.35       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.40       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.45       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.50       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.55       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.60       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.65       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.70       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.75       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
0.80       0.6160   0.3421   0.5735   0.9997   0.9985   0.2064  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6160, F1=0.3421, Normal Recall=0.5735, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2064

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
0.15       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692   <--
0.20       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.25       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.30       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.35       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.40       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.45       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.50       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.55       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.60       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.65       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.70       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.75       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
0.80       0.6586   0.5391   0.5736   0.9993   0.9984   0.3692  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6586, F1=0.5391, Normal Recall=0.5736, Normal Precision=0.9993, Attack Recall=0.9984, Attack Precision=0.3692

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
0.15       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007   <--
0.20       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.25       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.30       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.35       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.40       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.45       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.50       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.55       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.60       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.65       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.70       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.75       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
0.80       0.7008   0.6669   0.5733   0.9988   0.9984   0.5007  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7008, F1=0.6669, Normal Recall=0.5733, Normal Precision=0.9988, Attack Recall=0.9984, Attack Precision=0.5007

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
0.15       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091   <--
0.20       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.25       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.30       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.35       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.40       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.45       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.50       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.55       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.60       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.65       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.70       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.75       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
0.80       0.7431   0.7566   0.5729   0.9981   0.9984   0.6091  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7431, F1=0.7566, Normal Recall=0.5729, Normal Precision=0.9981, Attack Recall=0.9984, Attack Precision=0.6091

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
0.15       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993   <--
0.20       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.25       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.30       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.35       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.40       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.45       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.50       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.55       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.60       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.65       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.70       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.75       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
0.80       0.7845   0.8225   0.5707   0.9972   0.9984   0.6993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7845, F1=0.8225, Normal Recall=0.5707, Normal Precision=0.9972, Attack Recall=0.9984, Attack Precision=0.6993

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
0.15       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051   <--
0.20       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.25       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.30       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.35       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.40       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.45       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.50       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.55       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.60       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.65       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.70       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.75       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.80       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6129, F1=0.3403, Normal Recall=0.5700, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2051

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
0.15       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674   <--
0.20       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.25       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.30       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.35       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.40       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.45       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.50       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.55       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.60       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.65       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.70       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.75       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.80       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6559, F1=0.5371, Normal Recall=0.5702, Normal Precision=0.9993, Attack Recall=0.9984, Attack Precision=0.3674

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
0.15       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987   <--
0.20       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.25       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.30       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.35       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.40       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.45       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.50       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.55       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.60       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.65       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.70       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.75       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.80       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6984, F1=0.6651, Normal Recall=0.5698, Normal Precision=0.9988, Attack Recall=0.9984, Attack Precision=0.4987

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
0.15       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074   <--
0.20       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.25       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.30       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.35       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.40       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.45       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.50       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.55       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.60       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.65       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.70       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.75       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.80       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7412, F1=0.7553, Normal Recall=0.5697, Normal Precision=0.9981, Attack Recall=0.9984, Attack Precision=0.6074

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
0.15       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977   <--
0.20       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.25       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.30       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.35       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.40       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.45       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.50       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.55       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.60       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.65       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.70       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.75       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.80       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7829, F1=0.8214, Normal Recall=0.5674, Normal Precision=0.9972, Attack Recall=0.9984, Attack Precision=0.6977

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
0.15       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051   <--
0.20       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.25       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.30       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.35       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.40       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.45       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.50       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.55       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.60       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.65       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.70       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.75       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
0.80       0.6129   0.3403   0.5700   0.9997   0.9985   0.2051  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6129, F1=0.3403, Normal Recall=0.5700, Normal Precision=0.9997, Attack Recall=0.9985, Attack Precision=0.2051

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
0.15       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674   <--
0.20       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.25       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.30       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.35       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.40       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.45       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.50       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.55       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.60       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.65       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.70       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.75       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
0.80       0.6559   0.5371   0.5702   0.9993   0.9984   0.3674  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6559, F1=0.5371, Normal Recall=0.5702, Normal Precision=0.9993, Attack Recall=0.9984, Attack Precision=0.3674

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
0.15       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987   <--
0.20       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.25       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.30       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.35       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.40       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.45       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.50       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.55       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.60       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.65       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.70       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.75       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
0.80       0.6984   0.6651   0.5698   0.9988   0.9984   0.4987  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6984, F1=0.6651, Normal Recall=0.5698, Normal Precision=0.9988, Attack Recall=0.9984, Attack Precision=0.4987

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
0.15       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074   <--
0.20       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.25       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.30       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.35       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.40       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.45       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.50       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.55       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.60       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.65       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.70       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.75       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
0.80       0.7412   0.7553   0.5697   0.9981   0.9984   0.6074  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7412, F1=0.7553, Normal Recall=0.5697, Normal Precision=0.9981, Attack Recall=0.9984, Attack Precision=0.6074

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
0.15       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977   <--
0.20       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.25       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.30       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.35       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.40       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.45       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.50       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.55       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.60       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.65       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.70       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.75       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
0.80       0.7829   0.8214   0.5674   0.9972   0.9984   0.6977  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7829, F1=0.8214, Normal Recall=0.5674, Normal Precision=0.9972, Attack Recall=0.9984, Attack Precision=0.6977

```

