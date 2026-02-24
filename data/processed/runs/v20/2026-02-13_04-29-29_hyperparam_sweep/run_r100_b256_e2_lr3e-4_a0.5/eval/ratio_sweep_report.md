# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-22 12:12:02 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2564 | 0.3149 | 0.3734 | 0.4324 | 0.4908 | 0.5483 | 0.6076 | 0.6663 | 0.7231 | 0.7828 | 0.8406 |
| QAT+Prune only | 0.1579 | 0.2422 | 0.3260 | 0.4099 | 0.4945 | 0.5772 | 0.6616 | 0.7455 | 0.8300 | 0.9135 | 0.9974 |
| QAT+PTQ | 0.1569 | 0.2411 | 0.3250 | 0.4091 | 0.4938 | 0.5766 | 0.6611 | 0.7450 | 0.8297 | 0.9133 | 0.9974 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.1569 | 0.2411 | 0.3250 | 0.4091 | 0.4938 | 0.5766 | 0.6611 | 0.7450 | 0.8297 | 0.9133 | 0.9974 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1967 | 0.3492 | 0.4705 | 0.5691 | 0.6505 | 0.7200 | 0.7791 | 0.8293 | 0.8745 | 0.9134 |
| QAT+Prune only | 0.0000 | 0.2084 | 0.3718 | 0.5035 | 0.6122 | 0.7023 | 0.7796 | 0.8458 | 0.9037 | 0.9540 | 0.9987 |
| QAT+PTQ | 0.0000 | 0.2082 | 0.3715 | 0.5032 | 0.6119 | 0.7020 | 0.7793 | 0.8456 | 0.9036 | 0.9539 | 0.9987 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2082 | 0.3715 | 0.5032 | 0.6119 | 0.7020 | 0.7793 | 0.8456 | 0.9036 | 0.9539 | 0.9987 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2564 | 0.2566 | 0.2566 | 0.2575 | 0.2575 | 0.2560 | 0.2581 | 0.2595 | 0.2529 | 0.2624 | 0.0000 |
| QAT+Prune only | 0.1579 | 0.1583 | 0.1581 | 0.1580 | 0.1592 | 0.1569 | 0.1578 | 0.1577 | 0.1602 | 0.1576 | 0.0000 |
| QAT+PTQ | 0.1569 | 0.1571 | 0.1569 | 0.1570 | 0.1580 | 0.1557 | 0.1566 | 0.1561 | 0.1589 | 0.1555 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.1569 | 0.1571 | 0.1569 | 0.1570 | 0.1580 | 0.1557 | 0.1566 | 0.1561 | 0.1589 | 0.1555 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2564 | 0.0000 | 0.0000 | 0.0000 | 0.2564 | 1.0000 |
| 90 | 10 | 299,940 | 0.3149 | 0.1114 | 0.8391 | 0.1967 | 0.2566 | 0.9349 |
| 80 | 20 | 291,350 | 0.3734 | 0.2204 | 0.8406 | 0.3492 | 0.2566 | 0.8656 |
| 70 | 30 | 194,230 | 0.4324 | 0.3267 | 0.8406 | 0.4705 | 0.2575 | 0.7903 |
| 60 | 40 | 145,675 | 0.4908 | 0.4301 | 0.8406 | 0.5691 | 0.2575 | 0.7079 |
| 50 | 50 | 116,540 | 0.5483 | 0.5305 | 0.8406 | 0.6505 | 0.2560 | 0.6163 |
| 40 | 60 | 97,115 | 0.6076 | 0.6296 | 0.8406 | 0.7200 | 0.2581 | 0.5192 |
| 30 | 70 | 83,240 | 0.6663 | 0.7259 | 0.8406 | 0.7791 | 0.2595 | 0.4110 |
| 20 | 80 | 72,835 | 0.7231 | 0.8182 | 0.8406 | 0.8293 | 0.2529 | 0.2840 |
| 10 | 90 | 64,740 | 0.7828 | 0.9112 | 0.8406 | 0.8745 | 0.2624 | 0.1547 |
| 0 | 100 | 58,270 | 0.8406 | 1.0000 | 0.8406 | 0.9134 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1579 | 0.0000 | 0.0000 | 0.0000 | 0.1579 | 1.0000 |
| 90 | 10 | 299,940 | 0.2422 | 0.1164 | 0.9977 | 0.2084 | 0.1583 | 0.9984 |
| 80 | 20 | 291,350 | 0.3260 | 0.2285 | 0.9974 | 0.3718 | 0.1581 | 0.9960 |
| 70 | 30 | 194,230 | 0.4099 | 0.3367 | 0.9974 | 0.5035 | 0.1580 | 0.9931 |
| 60 | 40 | 145,675 | 0.4945 | 0.4416 | 0.9974 | 0.6122 | 0.1592 | 0.9894 |
| 50 | 50 | 116,540 | 0.5772 | 0.5419 | 0.9974 | 0.7023 | 0.1569 | 0.9840 |
| 40 | 60 | 97,115 | 0.6616 | 0.6398 | 0.9974 | 0.7796 | 0.1578 | 0.9763 |
| 30 | 70 | 83,240 | 0.7455 | 0.7343 | 0.9974 | 0.8458 | 0.1577 | 0.9635 |
| 20 | 80 | 72,835 | 0.8300 | 0.8261 | 0.9974 | 0.9037 | 0.1602 | 0.9400 |
| 10 | 90 | 64,740 | 0.9135 | 0.9142 | 0.9974 | 0.9540 | 0.1576 | 0.8725 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.1569 | 0.0000 | 0.0000 | 0.0000 | 0.1569 | 1.0000 |
| 90 | 10 | 299,940 | 0.2411 | 0.1162 | 0.9977 | 0.2082 | 0.1571 | 0.9984 |
| 80 | 20 | 291,350 | 0.3250 | 0.2283 | 0.9974 | 0.3715 | 0.1569 | 0.9959 |
| 70 | 30 | 194,230 | 0.4091 | 0.3365 | 0.9974 | 0.5032 | 0.1570 | 0.9931 |
| 60 | 40 | 145,675 | 0.4938 | 0.4413 | 0.9974 | 0.6119 | 0.1580 | 0.9893 |
| 50 | 50 | 116,540 | 0.5766 | 0.5416 | 0.9974 | 0.7020 | 0.1557 | 0.9838 |
| 40 | 60 | 97,115 | 0.6611 | 0.6395 | 0.9974 | 0.7793 | 0.1566 | 0.9761 |
| 30 | 70 | 83,240 | 0.7450 | 0.7339 | 0.9974 | 0.8456 | 0.1561 | 0.9632 |
| 20 | 80 | 72,835 | 0.8297 | 0.8259 | 0.9974 | 0.9036 | 0.1589 | 0.9395 |
| 10 | 90 | 64,740 | 0.9133 | 0.9140 | 0.9974 | 0.9539 | 0.1555 | 0.8711 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.1569 | 0.0000 | 0.0000 | 0.0000 | 0.1569 | 1.0000 |
| 90 | 10 | 299,940 | 0.2411 | 0.1162 | 0.9977 | 0.2082 | 0.1571 | 0.9984 |
| 80 | 20 | 291,350 | 0.3250 | 0.2283 | 0.9974 | 0.3715 | 0.1569 | 0.9959 |
| 70 | 30 | 194,230 | 0.4091 | 0.3365 | 0.9974 | 0.5032 | 0.1570 | 0.9931 |
| 60 | 40 | 145,675 | 0.4938 | 0.4413 | 0.9974 | 0.6119 | 0.1580 | 0.9893 |
| 50 | 50 | 116,540 | 0.5766 | 0.5416 | 0.9974 | 0.7020 | 0.1557 | 0.9838 |
| 40 | 60 | 97,115 | 0.6611 | 0.6395 | 0.9974 | 0.7793 | 0.1566 | 0.9761 |
| 30 | 70 | 83,240 | 0.7450 | 0.7339 | 0.9974 | 0.8456 | 0.1561 | 0.9632 |
| 20 | 80 | 72,835 | 0.8297 | 0.8259 | 0.9974 | 0.9036 | 0.1589 | 0.9395 |
| 10 | 90 | 64,740 | 0.9133 | 0.9140 | 0.9974 | 0.9539 | 0.1555 | 0.8711 |
| 0 | 100 | 58,270 | 0.9974 | 1.0000 | 0.9974 | 0.9987 | 0.0000 | 0.0000 |


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
0.15       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116   <--
0.20       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.25       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.30       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.35       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.40       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.45       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.50       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.55       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.60       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.65       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.70       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.75       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
0.80       0.3150   0.1971   0.2566   0.9354   0.8405   0.1116  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3150, F1=0.1971, Normal Recall=0.2566, Normal Precision=0.9354, Attack Recall=0.8405, Attack Precision=0.1116

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
0.15       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204   <--
0.20       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.25       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.30       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.35       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.40       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.45       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.50       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.55       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.60       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.65       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.70       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.75       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
0.80       0.3734   0.3492   0.2566   0.8656   0.8406   0.2204  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3734, F1=0.3492, Normal Recall=0.2566, Normal Precision=0.8656, Attack Recall=0.8406, Attack Precision=0.2204

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
0.15       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264   <--
0.20       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.25       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.30       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.35       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.40       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.45       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.50       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.55       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.60       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.65       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.70       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.75       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
0.80       0.4317   0.4702   0.2564   0.7896   0.8406   0.3264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4317, F1=0.4702, Normal Recall=0.2564, Normal Precision=0.7896, Attack Recall=0.8406, Attack Precision=0.3264

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
0.15       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299   <--
0.20       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.25       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.30       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.35       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.40       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.45       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.50       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.55       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.60       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.65       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.70       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.75       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
0.80       0.4903   0.5689   0.2567   0.7073   0.8406   0.4299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4903, F1=0.5689, Normal Recall=0.2567, Normal Precision=0.7073, Attack Recall=0.8406, Attack Precision=0.4299

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
0.15       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306   <--
0.20       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.25       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.30       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.35       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.40       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.45       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.50       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.55       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.60       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.65       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.70       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.75       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
0.80       0.5485   0.6506   0.2564   0.6167   0.8406   0.5306  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5485, F1=0.6506, Normal Recall=0.2564, Normal Precision=0.6167, Attack Recall=0.8406, Attack Precision=0.5306

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
0.15       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164   <--
0.20       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.25       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.30       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.35       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.40       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.45       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.50       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.55       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.60       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.65       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.70       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.75       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
0.80       0.2422   0.2084   0.1583   0.9982   0.9975   0.1164  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2422, F1=0.2084, Normal Recall=0.1583, Normal Precision=0.9982, Attack Recall=0.9975, Attack Precision=0.1164

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
0.15       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285   <--
0.20       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.25       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.30       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.35       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.40       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.45       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.50       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.55       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.60       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.65       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.70       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.75       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
0.80       0.3261   0.3719   0.1583   0.9960   0.9974   0.2285  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3261, F1=0.3719, Normal Recall=0.1583, Normal Precision=0.9960, Attack Recall=0.9974, Attack Precision=0.2285

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
0.15       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368   <--
0.20       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.25       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.30       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.35       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.40       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.45       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.50       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.55       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.60       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.65       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.70       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.75       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
0.80       0.4101   0.5036   0.1583   0.9931   0.9974   0.3368  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4101, F1=0.5036, Normal Recall=0.1583, Normal Precision=0.9931, Attack Recall=0.9974, Attack Precision=0.3368

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
0.15       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411   <--
0.20       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.25       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.30       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.35       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.40       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.45       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.50       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.55       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.60       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.65       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.70       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.75       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
0.80       0.4935   0.6117   0.1575   0.9893   0.9974   0.4411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4935, F1=0.6117, Normal Recall=0.1575, Normal Precision=0.9893, Attack Recall=0.9974, Attack Precision=0.4411

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
0.15       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418   <--
0.20       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.25       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.30       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.35       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.40       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.45       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.50       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.55       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.60       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.65       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.70       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.75       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
0.80       0.5770   0.7022   0.1565   0.9839   0.9974   0.5418  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5770, F1=0.7022, Normal Recall=0.1565, Normal Precision=0.9839, Attack Recall=0.9974, Attack Precision=0.5418

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
0.15       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162   <--
0.20       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.25       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.30       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.35       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.40       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.45       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.50       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.55       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.60       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.65       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.70       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.75       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.80       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2411, F1=0.2082, Normal Recall=0.1571, Normal Precision=0.9982, Attack Recall=0.9975, Attack Precision=0.1162

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
0.15       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283   <--
0.20       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.25       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.30       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.35       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.40       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.45       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.50       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.55       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.60       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.65       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.70       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.75       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.80       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3252, F1=0.3716, Normal Recall=0.1571, Normal Precision=0.9959, Attack Recall=0.9974, Attack Precision=0.2283

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
0.15       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365   <--
0.20       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.25       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.30       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.35       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.40       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.45       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.50       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.55       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.60       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.65       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.70       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.75       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.80       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4093, F1=0.5033, Normal Recall=0.1572, Normal Precision=0.9931, Attack Recall=0.9974, Attack Precision=0.3365

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
0.15       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408   <--
0.20       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.25       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.30       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.35       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.40       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.45       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.50       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.55       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.60       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.65       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.70       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.75       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.80       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4929, F1=0.6114, Normal Recall=0.1565, Normal Precision=0.9892, Attack Recall=0.9974, Attack Precision=0.4408

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
0.15       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415   <--
0.20       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.25       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.30       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.35       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.40       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.45       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.50       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.55       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.60       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.65       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.70       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.75       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.80       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5765, F1=0.7020, Normal Recall=0.1555, Normal Precision=0.9838, Attack Recall=0.9974, Attack Precision=0.5415

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
0.15       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162   <--
0.20       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.25       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.30       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.35       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.40       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.45       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.50       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.55       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.60       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.65       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.70       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.75       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
0.80       0.2411   0.2082   0.1571   0.9982   0.9975   0.1162  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2411, F1=0.2082, Normal Recall=0.1571, Normal Precision=0.9982, Attack Recall=0.9975, Attack Precision=0.1162

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
0.15       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283   <--
0.20       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.25       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.30       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.35       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.40       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.45       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.50       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.55       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.60       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.65       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.70       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.75       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
0.80       0.3252   0.3716   0.1571   0.9959   0.9974   0.2283  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3252, F1=0.3716, Normal Recall=0.1571, Normal Precision=0.9959, Attack Recall=0.9974, Attack Precision=0.2283

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
0.15       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365   <--
0.20       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.25       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.30       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.35       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.40       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.45       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.50       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.55       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.60       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.65       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.70       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.75       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
0.80       0.4093   0.5033   0.1572   0.9931   0.9974   0.3365  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4093, F1=0.5033, Normal Recall=0.1572, Normal Precision=0.9931, Attack Recall=0.9974, Attack Precision=0.3365

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
0.15       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408   <--
0.20       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.25       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.30       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.35       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.40       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.45       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.50       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.55       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.60       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.65       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.70       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.75       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
0.80       0.4929   0.6114   0.1565   0.9892   0.9974   0.4408  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4929, F1=0.6114, Normal Recall=0.1565, Normal Precision=0.9892, Attack Recall=0.9974, Attack Precision=0.4408

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
0.15       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415   <--
0.20       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.25       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.30       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.35       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.40       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.45       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.50       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.55       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.60       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.65       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.70       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.75       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
0.80       0.5765   0.7020   0.1555   0.9838   0.9974   0.5415  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5765, F1=0.7020, Normal Recall=0.1555, Normal Precision=0.9838, Attack Recall=0.9974, Attack Precision=0.5415

```

