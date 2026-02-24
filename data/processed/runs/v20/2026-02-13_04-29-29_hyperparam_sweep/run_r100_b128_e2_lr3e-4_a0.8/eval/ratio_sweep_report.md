# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e2_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-21 13:58:30 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4721 | 0.5224 | 0.5715 | 0.6215 | 0.6708 | 0.7192 | 0.7677 | 0.8183 | 0.8662 | 0.9170 | 0.9651 |
| QAT+Prune only | 0.2541 | 0.3267 | 0.4013 | 0.4764 | 0.5517 | 0.6267 | 0.7005 | 0.7760 | 0.8507 | 0.9243 | 1.0000 |
| QAT+PTQ | 0.2612 | 0.3331 | 0.4071 | 0.4814 | 0.5559 | 0.6303 | 0.7034 | 0.7782 | 0.8520 | 0.9251 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.2612 | 0.3331 | 0.4071 | 0.4814 | 0.5559 | 0.6303 | 0.7034 | 0.7782 | 0.8520 | 0.9251 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2881 | 0.4739 | 0.6047 | 0.7011 | 0.7746 | 0.8329 | 0.8815 | 0.9203 | 0.9544 | 0.9822 |
| QAT+Prune only | 0.0000 | 0.2290 | 0.4005 | 0.5340 | 0.6409 | 0.7281 | 0.8003 | 0.8621 | 0.9146 | 0.9597 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.2307 | 0.4029 | 0.5364 | 0.6430 | 0.7301 | 0.8018 | 0.8632 | 0.9154 | 0.9600 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2307 | 0.4029 | 0.5364 | 0.6430 | 0.7301 | 0.8018 | 0.8632 | 0.9154 | 0.9600 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4721 | 0.4731 | 0.4731 | 0.4742 | 0.4746 | 0.4733 | 0.4717 | 0.4758 | 0.4709 | 0.4844 | 0.0000 |
| QAT+Prune only | 0.2541 | 0.2519 | 0.2516 | 0.2521 | 0.2529 | 0.2533 | 0.2513 | 0.2532 | 0.2533 | 0.2434 | 0.0000 |
| QAT+PTQ | 0.2612 | 0.2591 | 0.2589 | 0.2592 | 0.2598 | 0.2607 | 0.2585 | 0.2607 | 0.2602 | 0.2507 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.2612 | 0.2591 | 0.2589 | 0.2592 | 0.2598 | 0.2607 | 0.2585 | 0.2607 | 0.2602 | 0.2507 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4721 | 0.0000 | 0.0000 | 0.0000 | 0.4721 | 1.0000 |
| 90 | 10 | 299,940 | 0.5224 | 0.1693 | 0.9662 | 0.2881 | 0.4731 | 0.9921 |
| 80 | 20 | 291,350 | 0.5715 | 0.3141 | 0.9651 | 0.4739 | 0.4731 | 0.9819 |
| 70 | 30 | 194,230 | 0.6215 | 0.4403 | 0.9651 | 0.6047 | 0.4742 | 0.9694 |
| 60 | 40 | 145,675 | 0.6708 | 0.5505 | 0.9651 | 0.7011 | 0.4746 | 0.9532 |
| 50 | 50 | 116,540 | 0.7192 | 0.6469 | 0.9651 | 0.7746 | 0.4733 | 0.9313 |
| 40 | 60 | 97,115 | 0.7677 | 0.7326 | 0.9651 | 0.8329 | 0.4717 | 0.9000 |
| 30 | 70 | 83,240 | 0.8183 | 0.8112 | 0.9651 | 0.8815 | 0.4758 | 0.8538 |
| 20 | 80 | 72,835 | 0.8662 | 0.8795 | 0.9651 | 0.9203 | 0.4709 | 0.7712 |
| 10 | 90 | 64,740 | 0.9170 | 0.9440 | 0.9651 | 0.9544 | 0.4844 | 0.6065 |
| 0 | 100 | 58,270 | 0.9651 | 1.0000 | 0.9651 | 0.9822 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2541 | 0.0000 | 0.0000 | 0.0000 | 0.2541 | 1.0000 |
| 90 | 10 | 299,940 | 0.3267 | 0.1293 | 1.0000 | 0.2290 | 0.2519 | 1.0000 |
| 80 | 20 | 291,350 | 0.4013 | 0.2504 | 1.0000 | 0.4005 | 0.2516 | 1.0000 |
| 70 | 30 | 194,230 | 0.4764 | 0.3643 | 1.0000 | 0.5340 | 0.2521 | 1.0000 |
| 60 | 40 | 145,675 | 0.5517 | 0.4716 | 1.0000 | 0.6409 | 0.2529 | 1.0000 |
| 50 | 50 | 116,540 | 0.6267 | 0.5725 | 1.0000 | 0.7281 | 0.2533 | 1.0000 |
| 40 | 60 | 97,115 | 0.7005 | 0.6671 | 1.0000 | 0.8003 | 0.2513 | 1.0000 |
| 30 | 70 | 83,240 | 0.7760 | 0.7576 | 1.0000 | 0.8621 | 0.2532 | 1.0000 |
| 20 | 80 | 72,835 | 0.8507 | 0.8427 | 1.0000 | 0.9146 | 0.2533 | 1.0000 |
| 10 | 90 | 64,740 | 0.9243 | 0.9225 | 1.0000 | 0.9597 | 0.2434 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2612 | 0.0000 | 0.0000 | 0.0000 | 0.2612 | 1.0000 |
| 90 | 10 | 299,940 | 0.3331 | 0.1304 | 1.0000 | 0.2307 | 0.2591 | 1.0000 |
| 80 | 20 | 291,350 | 0.4071 | 0.2522 | 1.0000 | 0.4029 | 0.2589 | 1.0000 |
| 70 | 30 | 194,230 | 0.4814 | 0.3665 | 1.0000 | 0.5364 | 0.2592 | 1.0000 |
| 60 | 40 | 145,675 | 0.5559 | 0.4739 | 1.0000 | 0.6430 | 0.2598 | 1.0000 |
| 50 | 50 | 116,540 | 0.6303 | 0.5749 | 1.0000 | 0.7301 | 0.2607 | 1.0000 |
| 40 | 60 | 97,115 | 0.7034 | 0.6692 | 1.0000 | 0.8018 | 0.2585 | 1.0000 |
| 30 | 70 | 83,240 | 0.7782 | 0.7594 | 1.0000 | 0.8632 | 0.2607 | 1.0000 |
| 20 | 80 | 72,835 | 0.8520 | 0.8439 | 1.0000 | 0.9154 | 0.2602 | 1.0000 |
| 10 | 90 | 64,740 | 0.9251 | 0.9231 | 1.0000 | 0.9600 | 0.2507 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.2612 | 0.0000 | 0.0000 | 0.0000 | 0.2612 | 1.0000 |
| 90 | 10 | 299,940 | 0.3331 | 0.1304 | 1.0000 | 0.2307 | 0.2591 | 1.0000 |
| 80 | 20 | 291,350 | 0.4071 | 0.2522 | 1.0000 | 0.4029 | 0.2589 | 1.0000 |
| 70 | 30 | 194,230 | 0.4814 | 0.3665 | 1.0000 | 0.5364 | 0.2592 | 1.0000 |
| 60 | 40 | 145,675 | 0.5559 | 0.4739 | 1.0000 | 0.6430 | 0.2598 | 1.0000 |
| 50 | 50 | 116,540 | 0.6303 | 0.5749 | 1.0000 | 0.7301 | 0.2607 | 1.0000 |
| 40 | 60 | 97,115 | 0.7034 | 0.6692 | 1.0000 | 0.8018 | 0.2585 | 1.0000 |
| 30 | 70 | 83,240 | 0.7782 | 0.7594 | 1.0000 | 0.8632 | 0.2607 | 1.0000 |
| 20 | 80 | 72,835 | 0.8520 | 0.8439 | 1.0000 | 0.9154 | 0.2602 | 1.0000 |
| 10 | 90 | 64,740 | 0.9251 | 0.9231 | 1.0000 | 0.9600 | 0.2507 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |


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
0.15       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692   <--
0.20       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.25       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.30       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.35       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.40       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.45       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.50       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.55       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.60       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.65       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.70       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.75       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
0.80       0.5224   0.2880   0.4731   0.9920   0.9659   0.1692  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5224, F1=0.2880, Normal Recall=0.4731, Normal Precision=0.9920, Attack Recall=0.9659, Attack Precision=0.1692

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
0.15       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141   <--
0.20       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.25       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.30       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.35       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.40       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.45       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.50       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.55       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.60       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.65       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.70       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.75       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
0.80       0.5715   0.4739   0.4731   0.9819   0.9651   0.3141  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5715, F1=0.4739, Normal Recall=0.4731, Normal Precision=0.9819, Attack Recall=0.9651, Attack Precision=0.3141

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
0.15       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395   <--
0.20       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.25       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.30       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.35       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.40       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.45       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.50       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.55       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.60       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.65       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.70       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.75       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
0.80       0.6203   0.6040   0.4726   0.9693   0.9651   0.4395  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6203, F1=0.6040, Normal Recall=0.4726, Normal Precision=0.9693, Attack Recall=0.9651, Attack Precision=0.4395

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
0.15       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494   <--
0.20       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.25       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.30       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.35       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.40       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.45       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.50       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.55       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.60       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.65       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.70       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.75       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
0.80       0.6694   0.7002   0.4723   0.9530   0.9651   0.5494  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6694, F1=0.7002, Normal Recall=0.4723, Normal Precision=0.9530, Attack Recall=0.9651, Attack Precision=0.5494

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
0.15       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458   <--
0.20       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.25       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.30       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.35       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.40       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.45       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.50       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.55       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.60       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.65       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.70       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.75       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
0.80       0.7179   0.7738   0.4708   0.9309   0.9651   0.6458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7179, F1=0.7738, Normal Recall=0.4708, Normal Precision=0.9309, Attack Recall=0.9651, Attack Precision=0.6458

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
0.15       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293   <--
0.20       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.25       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.30       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.35       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.40       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.45       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.50       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.55       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.60       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.65       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.70       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.75       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
0.80       0.3267   0.2290   0.2519   1.0000   1.0000   0.1293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3267, F1=0.2290, Normal Recall=0.2519, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1293

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
0.15       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505   <--
0.20       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.25       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.30       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.35       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.40       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.45       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.50       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.55       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.60       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.65       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.70       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.75       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
0.80       0.4017   0.4007   0.2521   1.0000   1.0000   0.2505  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4017, F1=0.4007, Normal Recall=0.2521, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2505

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
0.15       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646   <--
0.20       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.25       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.30       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.35       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.40       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.45       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.50       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.55       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.60       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.65       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.70       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.75       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
0.80       0.4772   0.5344   0.2531   1.0000   1.0000   0.3646  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4772, F1=0.5344, Normal Recall=0.2531, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3646

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
0.15       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718   <--
0.20       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.25       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.30       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.35       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.40       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.45       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.50       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.55       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.60       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.65       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.70       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.75       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
0.80       0.5521   0.6411   0.2535   1.0000   1.0000   0.4718  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5521, F1=0.6411, Normal Recall=0.2535, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4718

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
0.15       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728   <--
0.20       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.25       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.30       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.35       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.40       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.45       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.50       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.55       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.60       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.65       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.70       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.75       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
0.80       0.6271   0.7284   0.2541   1.0000   1.0000   0.5728  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6271, F1=0.7284, Normal Recall=0.2541, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5728

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
0.15       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304   <--
0.20       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.25       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.30       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.35       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.40       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.45       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.50       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.55       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.60       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.65       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.70       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.75       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.80       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3332, F1=0.2307, Normal Recall=0.2591, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1304

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
0.15       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524   <--
0.20       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.25       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.30       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.35       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.40       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.45       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.50       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.55       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.60       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.65       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.70       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.75       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.80       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4075, F1=0.4030, Normal Recall=0.2594, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2524

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
0.15       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669   <--
0.20       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.25       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.30       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.35       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.40       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.45       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.50       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.55       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.60       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.65       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.70       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.75       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.80       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4822, F1=0.5368, Normal Recall=0.2603, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3669

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
0.15       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742   <--
0.20       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.25       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.30       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.35       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.40       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.45       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.50       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.55       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.60       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.65       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.70       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.75       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.80       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5564, F1=0.6433, Normal Recall=0.2607, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4742

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
0.15       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752   <--
0.20       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.25       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.30       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.35       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.40       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.45       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.50       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.55       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.60       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.65       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.70       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.75       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.80       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6308, F1=0.7303, Normal Recall=0.2615, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5752

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
0.15       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304   <--
0.20       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.25       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.30       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.35       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.40       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.45       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.50       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.55       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.60       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.65       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.70       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.75       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
0.80       0.3332   0.2307   0.2591   1.0000   1.0000   0.1304  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3332, F1=0.2307, Normal Recall=0.2591, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1304

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
0.15       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524   <--
0.20       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.25       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.30       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.35       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.40       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.45       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.50       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.55       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.60       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.65       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.70       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.75       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
0.80       0.4075   0.4030   0.2594   1.0000   1.0000   0.2524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4075, F1=0.4030, Normal Recall=0.2594, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2524

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
0.15       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669   <--
0.20       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.25       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.30       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.35       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.40       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.45       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.50       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.55       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.60       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.65       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.70       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.75       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
0.80       0.4822   0.5368   0.2603   1.0000   1.0000   0.3669  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4822, F1=0.5368, Normal Recall=0.2603, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3669

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
0.15       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742   <--
0.20       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.25       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.30       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.35       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.40       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.45       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.50       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.55       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.60       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.65       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.70       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.75       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
0.80       0.5564   0.6433   0.2607   1.0000   1.0000   0.4742  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5564, F1=0.6433, Normal Recall=0.2607, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4742

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
0.15       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752   <--
0.20       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.25       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.30       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.35       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.40       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.45       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.50       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.55       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.60       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.65       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.70       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.75       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
0.80       0.6308   0.7303   0.2615   1.0000   1.0000   0.5752  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6308, F1=0.7303, Normal Recall=0.2615, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5752

```

