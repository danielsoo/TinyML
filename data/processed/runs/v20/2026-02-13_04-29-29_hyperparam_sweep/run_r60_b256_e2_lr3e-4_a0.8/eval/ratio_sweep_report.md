# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-15 16:01:23 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7633 | 0.7869 | 0.8100 | 0.8338 | 0.8578 | 0.8799 | 0.9042 | 0.9276 | 0.9518 | 0.9747 | 0.9984 |
| QAT+Prune only | 0.9546 | 0.9283 | 0.9007 | 0.8740 | 0.8464 | 0.8188 | 0.7915 | 0.7642 | 0.7366 | 0.7095 | 0.6822 |
| QAT+PTQ | 0.9600 | 0.9326 | 0.9042 | 0.8765 | 0.8482 | 0.8199 | 0.7915 | 0.7632 | 0.7349 | 0.7070 | 0.6787 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9600 | 0.9326 | 0.9042 | 0.8765 | 0.8482 | 0.8199 | 0.7915 | 0.7632 | 0.7349 | 0.7070 | 0.6787 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4838 | 0.6776 | 0.7828 | 0.8489 | 0.8926 | 0.9260 | 0.9508 | 0.9707 | 0.9861 | 0.9992 |
| QAT+Prune only | 0.0000 | 0.6559 | 0.7332 | 0.7646 | 0.7804 | 0.7901 | 0.7970 | 0.8020 | 0.8056 | 0.8087 | 0.8111 |
| QAT+PTQ | 0.0000 | 0.6685 | 0.7393 | 0.7673 | 0.7815 | 0.7903 | 0.7962 | 0.8005 | 0.8038 | 0.8065 | 0.8086 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6685 | 0.7393 | 0.7673 | 0.7815 | 0.7903 | 0.7962 | 0.8005 | 0.8038 | 0.8065 | 0.8086 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7633 | 0.7634 | 0.7629 | 0.7633 | 0.7641 | 0.7615 | 0.7631 | 0.7625 | 0.7654 | 0.7620 | 0.0000 |
| QAT+Prune only | 0.9546 | 0.9555 | 0.9553 | 0.9562 | 0.9559 | 0.9554 | 0.9556 | 0.9555 | 0.9542 | 0.9552 | 0.0000 |
| QAT+PTQ | 0.9600 | 0.9608 | 0.9606 | 0.9612 | 0.9611 | 0.9610 | 0.9607 | 0.9604 | 0.9598 | 0.9611 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9600 | 0.9608 | 0.9606 | 0.9612 | 0.9611 | 0.9610 | 0.9607 | 0.9604 | 0.9598 | 0.9611 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7633 | 0.0000 | 0.0000 | 0.0000 | 0.7633 | 1.0000 |
| 90 | 10 | 299,940 | 0.7869 | 0.3192 | 0.9984 | 0.4838 | 0.7634 | 0.9998 |
| 80 | 20 | 291,350 | 0.8100 | 0.5128 | 0.9984 | 0.6776 | 0.7629 | 0.9995 |
| 70 | 30 | 194,230 | 0.8338 | 0.6438 | 0.9984 | 0.7828 | 0.7633 | 0.9991 |
| 60 | 40 | 145,675 | 0.8578 | 0.7383 | 0.9984 | 0.8489 | 0.7641 | 0.9986 |
| 50 | 50 | 116,540 | 0.8799 | 0.8072 | 0.9984 | 0.8926 | 0.7615 | 0.9978 |
| 40 | 60 | 97,115 | 0.9042 | 0.8634 | 0.9984 | 0.9260 | 0.7631 | 0.9968 |
| 30 | 70 | 83,240 | 0.9276 | 0.9075 | 0.9984 | 0.9508 | 0.7625 | 0.9950 |
| 20 | 80 | 72,835 | 0.9518 | 0.9445 | 0.9984 | 0.9707 | 0.7654 | 0.9915 |
| 10 | 90 | 64,740 | 0.9747 | 0.9742 | 0.9984 | 0.9861 | 0.7620 | 0.9809 |
| 0 | 100 | 58,270 | 0.9984 | 1.0000 | 0.9984 | 0.9992 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9546 | 0.0000 | 0.0000 | 0.0000 | 0.9546 | 1.0000 |
| 90 | 10 | 299,940 | 0.9283 | 0.6304 | 0.6835 | 0.6559 | 0.9555 | 0.9645 |
| 80 | 20 | 291,350 | 0.9007 | 0.7924 | 0.6822 | 0.7332 | 0.9553 | 0.9232 |
| 70 | 30 | 194,230 | 0.8740 | 0.8696 | 0.6822 | 0.7646 | 0.9562 | 0.8753 |
| 60 | 40 | 145,675 | 0.8464 | 0.9116 | 0.6822 | 0.7804 | 0.9559 | 0.8186 |
| 50 | 50 | 116,540 | 0.8188 | 0.9386 | 0.6822 | 0.7901 | 0.9554 | 0.7504 |
| 40 | 60 | 97,115 | 0.7915 | 0.9584 | 0.6822 | 0.7970 | 0.9556 | 0.6672 |
| 30 | 70 | 83,240 | 0.7642 | 0.9728 | 0.6822 | 0.8020 | 0.9555 | 0.5630 |
| 20 | 80 | 72,835 | 0.7366 | 0.9835 | 0.6822 | 0.8056 | 0.9542 | 0.4288 |
| 10 | 90 | 64,740 | 0.7095 | 0.9928 | 0.6822 | 0.8087 | 0.9552 | 0.2503 |
| 0 | 100 | 58,270 | 0.6822 | 1.0000 | 0.6822 | 0.8111 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.9600 | 1.0000 |
| 90 | 10 | 299,940 | 0.9326 | 0.6581 | 0.6793 | 0.6685 | 0.9608 | 0.9642 |
| 80 | 20 | 291,350 | 0.9042 | 0.8116 | 0.6787 | 0.7393 | 0.9606 | 0.9228 |
| 70 | 30 | 194,230 | 0.8765 | 0.8824 | 0.6787 | 0.7673 | 0.9612 | 0.8747 |
| 60 | 40 | 145,675 | 0.8482 | 0.9209 | 0.6787 | 0.7815 | 0.9611 | 0.8178 |
| 50 | 50 | 116,540 | 0.8199 | 0.9457 | 0.6787 | 0.7903 | 0.9610 | 0.7495 |
| 40 | 60 | 97,115 | 0.7915 | 0.9628 | 0.6787 | 0.7962 | 0.9607 | 0.6659 |
| 30 | 70 | 83,240 | 0.7632 | 0.9756 | 0.6787 | 0.8005 | 0.9604 | 0.5616 |
| 20 | 80 | 72,835 | 0.7349 | 0.9854 | 0.6787 | 0.8038 | 0.9598 | 0.4275 |
| 10 | 90 | 64,740 | 0.7070 | 0.9937 | 0.6787 | 0.8065 | 0.9611 | 0.2495 |
| 0 | 100 | 58,270 | 0.6787 | 1.0000 | 0.6787 | 0.8086 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9600 | 0.0000 | 0.0000 | 0.0000 | 0.9600 | 1.0000 |
| 90 | 10 | 299,940 | 0.9326 | 0.6581 | 0.6793 | 0.6685 | 0.9608 | 0.9642 |
| 80 | 20 | 291,350 | 0.9042 | 0.8116 | 0.6787 | 0.7393 | 0.9606 | 0.9228 |
| 70 | 30 | 194,230 | 0.8765 | 0.8824 | 0.6787 | 0.7673 | 0.9612 | 0.8747 |
| 60 | 40 | 145,675 | 0.8482 | 0.9209 | 0.6787 | 0.7815 | 0.9611 | 0.8178 |
| 50 | 50 | 116,540 | 0.8199 | 0.9457 | 0.6787 | 0.7903 | 0.9610 | 0.7495 |
| 40 | 60 | 97,115 | 0.7915 | 0.9628 | 0.6787 | 0.7962 | 0.9607 | 0.6659 |
| 30 | 70 | 83,240 | 0.7632 | 0.9756 | 0.6787 | 0.8005 | 0.9604 | 0.5616 |
| 20 | 80 | 72,835 | 0.7349 | 0.9854 | 0.6787 | 0.8038 | 0.9598 | 0.4275 |
| 10 | 90 | 64,740 | 0.7070 | 0.9937 | 0.6787 | 0.8065 | 0.9611 | 0.2495 |
| 0 | 100 | 58,270 | 0.6787 | 1.0000 | 0.6787 | 0.8086 | 0.0000 | 0.0000 |


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
0.15       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193   <--
0.20       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.25       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.30       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.35       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.40       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.45       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.50       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.55       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.60       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.65       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.70       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.75       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
0.80       0.7869   0.4839   0.7634   0.9998   0.9987   0.3193  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7869, F1=0.4839, Normal Recall=0.7634, Normal Precision=0.9998, Attack Recall=0.9987, Attack Precision=0.3193

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
0.15       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139   <--
0.20       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.25       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.30       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.35       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.40       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.45       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.50       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.55       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.60       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.65       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.70       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.75       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
0.80       0.8108   0.6786   0.7639   0.9995   0.9984   0.5139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8108, F1=0.6786, Normal Recall=0.7639, Normal Precision=0.9995, Attack Recall=0.9984, Attack Precision=0.5139

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
0.15       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447   <--
0.20       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.25       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.30       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.35       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.40       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.45       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.50       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.55       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.60       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.65       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.70       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.75       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
0.80       0.8344   0.7835   0.7642   0.9991   0.9984   0.6447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8344, F1=0.7835, Normal Recall=0.7642, Normal Precision=0.9991, Attack Recall=0.9984, Attack Precision=0.6447

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
0.15       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377   <--
0.20       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.25       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.30       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.35       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.40       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.45       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.50       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.55       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.60       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.65       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.70       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.75       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
0.80       0.8574   0.8485   0.7634   0.9986   0.9984   0.7377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8574, F1=0.8485, Normal Recall=0.7634, Normal Precision=0.9986, Attack Recall=0.9984, Attack Precision=0.7377

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
0.15       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082   <--
0.20       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.25       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.30       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.35       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.40       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.45       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.50       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.55       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.60       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.65       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.70       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.75       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
0.80       0.8807   0.8933   0.7631   0.9978   0.9984   0.8082  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8807, F1=0.8933, Normal Recall=0.7631, Normal Precision=0.9978, Attack Recall=0.9984, Attack Precision=0.8082

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
0.15       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291   <--
0.20       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.25       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.30       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.35       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.40       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.45       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.50       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.55       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.60       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.65       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.70       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.75       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
0.80       0.9279   0.6535   0.9555   0.9641   0.6799   0.6291  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9279, F1=0.6535, Normal Recall=0.9555, Normal Precision=0.9641, Attack Recall=0.6799, Attack Precision=0.6291

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
0.15       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930   <--
0.20       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.25       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.30       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.35       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.40       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.45       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.50       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.55       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.60       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.65       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.70       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.75       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
0.80       0.9008   0.7334   0.9555   0.9232   0.6822   0.7930  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9008, F1=0.7334, Normal Recall=0.9555, Normal Precision=0.9232, Attack Recall=0.6822, Attack Precision=0.7930

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
0.15       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672   <--
0.20       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.25       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.30       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.35       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.40       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.45       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.50       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.55       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.60       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.65       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.70       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.75       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
0.80       0.8733   0.7637   0.9552   0.8752   0.6822   0.8672  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8733, F1=0.7637, Normal Recall=0.9552, Normal Precision=0.8752, Attack Recall=0.6822, Attack Precision=0.8672

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
0.15       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087   <--
0.20       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.25       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.30       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.35       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.40       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.45       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.50       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.55       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.60       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.65       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.70       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.75       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
0.80       0.8455   0.7793   0.9543   0.8183   0.6822   0.9087  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8455, F1=0.7793, Normal Recall=0.9543, Normal Precision=0.8183, Attack Recall=0.6822, Attack Precision=0.9087

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
0.15       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373   <--
0.20       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.25       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.30       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.35       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.40       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.45       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.50       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.55       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.60       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.65       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.70       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.75       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
0.80       0.8183   0.7897   0.9544   0.7502   0.6822   0.9373  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8183, F1=0.7897, Normal Recall=0.9544, Normal Precision=0.7502, Attack Recall=0.6822, Attack Precision=0.9373

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
0.15       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571   <--
0.20       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.25       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.30       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.35       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.40       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.45       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.50       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.55       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.60       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.65       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.70       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.75       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.80       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9323, F1=0.6666, Normal Recall=0.9608, Normal Precision=0.9639, Attack Recall=0.6763, Attack Precision=0.6571

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
0.15       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124   <--
0.20       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.25       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.30       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.35       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.40       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.45       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.50       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.55       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.60       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.65       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.70       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.75       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.80       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9044, F1=0.7396, Normal Recall=0.9608, Normal Precision=0.9229, Attack Recall=0.6787, Attack Precision=0.8124

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
0.15       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807   <--
0.20       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.25       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.30       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.35       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.40       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.45       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.50       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.55       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.60       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.65       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.70       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.75       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.80       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8760, F1=0.7666, Normal Recall=0.9606, Normal Precision=0.8746, Attack Recall=0.6787, Attack Precision=0.8807

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
0.15       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182   <--
0.20       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.25       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.30       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.35       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.40       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.45       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.50       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.55       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.60       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.65       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.70       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.75       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.80       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8473, F1=0.7805, Normal Recall=0.9597, Normal Precision=0.8175, Attack Recall=0.6787, Attack Precision=0.9182

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
0.15       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443   <--
0.20       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.25       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.30       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.35       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.40       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.45       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.50       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.55       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.60       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.65       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.70       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.75       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.80       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8193, F1=0.7898, Normal Recall=0.9599, Normal Precision=0.7492, Attack Recall=0.6787, Attack Precision=0.9443

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
0.15       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571   <--
0.20       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.25       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.30       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.35       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.40       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.45       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.50       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.55       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.60       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.65       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.70       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.75       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
0.80       0.9323   0.6666   0.9608   0.9639   0.6763   0.6571  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9323, F1=0.6666, Normal Recall=0.9608, Normal Precision=0.9639, Attack Recall=0.6763, Attack Precision=0.6571

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
0.15       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124   <--
0.20       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.25       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.30       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.35       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.40       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.45       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.50       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.55       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.60       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.65       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.70       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.75       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
0.80       0.9044   0.7396   0.9608   0.9229   0.6787   0.8124  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9044, F1=0.7396, Normal Recall=0.9608, Normal Precision=0.9229, Attack Recall=0.6787, Attack Precision=0.8124

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
0.15       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807   <--
0.20       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.25       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.30       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.35       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.40       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.45       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.50       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.55       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.60       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.65       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.70       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.75       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
0.80       0.8760   0.7666   0.9606   0.8746   0.6787   0.8807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8760, F1=0.7666, Normal Recall=0.9606, Normal Precision=0.8746, Attack Recall=0.6787, Attack Precision=0.8807

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
0.15       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182   <--
0.20       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.25       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.30       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.35       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.40       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.45       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.50       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.55       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.60       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.65       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.70       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.75       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
0.80       0.8473   0.7805   0.9597   0.8175   0.6787   0.9182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8473, F1=0.7805, Normal Recall=0.9597, Normal Precision=0.8175, Attack Recall=0.6787, Attack Precision=0.9182

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
0.15       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443   <--
0.20       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.25       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.30       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.35       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.40       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.45       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.50       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.55       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.60       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.65       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.70       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.75       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
0.80       0.8193   0.7898   0.9599   0.7492   0.6787   0.9443  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8193, F1=0.7898, Normal Recall=0.9599, Normal Precision=0.7492, Attack Recall=0.6787, Attack Precision=0.9443

```

