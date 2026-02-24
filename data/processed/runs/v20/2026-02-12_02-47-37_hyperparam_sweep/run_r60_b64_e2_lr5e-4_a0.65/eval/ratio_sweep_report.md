# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-12 22:48:22 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4241 | 0.4533 | 0.4844 | 0.5152 | 0.5467 | 0.5777 | 0.6073 | 0.6378 | 0.6697 | 0.7014 | 0.7311 |
| QAT+Prune only | 0.6598 | 0.6939 | 0.7274 | 0.7621 | 0.7958 | 0.8289 | 0.8640 | 0.8975 | 0.9318 | 0.9648 | 0.9996 |
| QAT+PTQ | 0.6599 | 0.6940 | 0.7275 | 0.7622 | 0.7961 | 0.8290 | 0.8640 | 0.8975 | 0.9318 | 0.9648 | 0.9996 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6599 | 0.6940 | 0.7275 | 0.7622 | 0.7961 | 0.8290 | 0.8640 | 0.8975 | 0.9318 | 0.9648 | 0.9996 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2107 | 0.3619 | 0.4750 | 0.5634 | 0.6339 | 0.6908 | 0.7386 | 0.7798 | 0.8151 | 0.8447 |
| QAT+Prune only | 0.0000 | 0.3951 | 0.5946 | 0.7160 | 0.7966 | 0.8539 | 0.8982 | 0.9318 | 0.9591 | 0.9808 | 0.9998 |
| QAT+PTQ | 0.0000 | 0.3952 | 0.5947 | 0.7160 | 0.7968 | 0.8539 | 0.8982 | 0.9318 | 0.9591 | 0.9808 | 0.9998 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3952 | 0.5947 | 0.7160 | 0.7968 | 0.8539 | 0.8982 | 0.9318 | 0.9591 | 0.9808 | 0.9998 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4241 | 0.4226 | 0.4227 | 0.4227 | 0.4238 | 0.4242 | 0.4215 | 0.4199 | 0.4241 | 0.4339 | 0.0000 |
| QAT+Prune only | 0.6598 | 0.6599 | 0.6594 | 0.6603 | 0.6600 | 0.6583 | 0.6607 | 0.6594 | 0.6609 | 0.6520 | 0.0000 |
| QAT+PTQ | 0.6599 | 0.6601 | 0.6595 | 0.6604 | 0.6605 | 0.6585 | 0.6606 | 0.6595 | 0.6608 | 0.6523 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6599 | 0.6601 | 0.6595 | 0.6604 | 0.6605 | 0.6585 | 0.6606 | 0.6595 | 0.6608 | 0.6523 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4241 | 0.0000 | 0.0000 | 0.0000 | 0.4241 | 1.0000 |
| 90 | 10 | 299,940 | 0.4533 | 0.1231 | 0.7296 | 0.2107 | 0.4226 | 0.9336 |
| 80 | 20 | 291,350 | 0.4844 | 0.2405 | 0.7311 | 0.3619 | 0.4227 | 0.8628 |
| 70 | 30 | 194,230 | 0.5152 | 0.3518 | 0.7311 | 0.4750 | 0.4227 | 0.7858 |
| 60 | 40 | 145,675 | 0.5467 | 0.4582 | 0.7311 | 0.5634 | 0.4238 | 0.7027 |
| 50 | 50 | 116,540 | 0.5777 | 0.5594 | 0.7311 | 0.6339 | 0.4242 | 0.6121 |
| 40 | 60 | 97,115 | 0.6073 | 0.6547 | 0.7311 | 0.6908 | 0.4215 | 0.5110 |
| 30 | 70 | 83,240 | 0.6378 | 0.7462 | 0.7311 | 0.7386 | 0.4199 | 0.4009 |
| 20 | 80 | 72,835 | 0.6697 | 0.8355 | 0.7311 | 0.7798 | 0.4241 | 0.2828 |
| 10 | 90 | 64,740 | 0.7014 | 0.9208 | 0.7311 | 0.8151 | 0.4339 | 0.1520 |
| 0 | 100 | 58,270 | 0.7311 | 1.0000 | 0.7311 | 0.8447 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6598 | 0.0000 | 0.0000 | 0.0000 | 0.6598 | 1.0000 |
| 90 | 10 | 299,940 | 0.6939 | 0.2462 | 0.9998 | 0.3951 | 0.6599 | 1.0000 |
| 80 | 20 | 291,350 | 0.7274 | 0.4232 | 0.9996 | 0.5946 | 0.6594 | 0.9998 |
| 70 | 30 | 194,230 | 0.7621 | 0.5577 | 0.9996 | 0.7160 | 0.6603 | 0.9997 |
| 60 | 40 | 145,675 | 0.7958 | 0.6622 | 0.9996 | 0.7966 | 0.6600 | 0.9995 |
| 50 | 50 | 116,540 | 0.8289 | 0.7452 | 0.9996 | 0.8539 | 0.6583 | 0.9993 |
| 40 | 60 | 97,115 | 0.8640 | 0.8155 | 0.9996 | 0.8982 | 0.6607 | 0.9990 |
| 30 | 70 | 83,240 | 0.8975 | 0.8726 | 0.9996 | 0.9318 | 0.6594 | 0.9984 |
| 20 | 80 | 72,835 | 0.9318 | 0.9218 | 0.9996 | 0.9591 | 0.6609 | 0.9973 |
| 10 | 90 | 64,740 | 0.9648 | 0.9628 | 0.9996 | 0.9808 | 0.6520 | 0.9939 |
| 0 | 100 | 58,270 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6599 | 0.0000 | 0.0000 | 0.0000 | 0.6599 | 1.0000 |
| 90 | 10 | 299,940 | 0.6940 | 0.2463 | 0.9998 | 0.3952 | 0.6601 | 1.0000 |
| 80 | 20 | 291,350 | 0.7275 | 0.4233 | 0.9996 | 0.5947 | 0.6595 | 0.9998 |
| 70 | 30 | 194,230 | 0.7622 | 0.5578 | 0.9996 | 0.7160 | 0.6604 | 0.9997 |
| 60 | 40 | 145,675 | 0.7961 | 0.6625 | 0.9996 | 0.7968 | 0.6605 | 0.9995 |
| 50 | 50 | 116,540 | 0.8290 | 0.7454 | 0.9996 | 0.8539 | 0.6585 | 0.9993 |
| 40 | 60 | 97,115 | 0.8640 | 0.8154 | 0.9996 | 0.8982 | 0.6606 | 0.9990 |
| 30 | 70 | 83,240 | 0.8975 | 0.8726 | 0.9996 | 0.9318 | 0.6595 | 0.9984 |
| 20 | 80 | 72,835 | 0.9318 | 0.9218 | 0.9996 | 0.9591 | 0.6608 | 0.9973 |
| 10 | 90 | 64,740 | 0.9648 | 0.9628 | 0.9996 | 0.9808 | 0.6523 | 0.9939 |
| 0 | 100 | 58,270 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6599 | 0.0000 | 0.0000 | 0.0000 | 0.6599 | 1.0000 |
| 90 | 10 | 299,940 | 0.6940 | 0.2463 | 0.9998 | 0.3952 | 0.6601 | 1.0000 |
| 80 | 20 | 291,350 | 0.7275 | 0.4233 | 0.9996 | 0.5947 | 0.6595 | 0.9998 |
| 70 | 30 | 194,230 | 0.7622 | 0.5578 | 0.9996 | 0.7160 | 0.6604 | 0.9997 |
| 60 | 40 | 145,675 | 0.7961 | 0.6625 | 0.9996 | 0.7968 | 0.6605 | 0.9995 |
| 50 | 50 | 116,540 | 0.8290 | 0.7454 | 0.9996 | 0.8539 | 0.6585 | 0.9993 |
| 40 | 60 | 97,115 | 0.8640 | 0.8154 | 0.9996 | 0.8982 | 0.6606 | 0.9990 |
| 30 | 70 | 83,240 | 0.8975 | 0.8726 | 0.9996 | 0.9318 | 0.6595 | 0.9984 |
| 20 | 80 | 72,835 | 0.9318 | 0.9218 | 0.9996 | 0.9591 | 0.6608 | 0.9973 |
| 10 | 90 | 64,740 | 0.9648 | 0.9628 | 0.9996 | 0.9808 | 0.6523 | 0.9939 |
| 0 | 100 | 58,270 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.0000 | 0.0000 |


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
0.15       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230   <--
0.20       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.25       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.30       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.35       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.40       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.45       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.50       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.55       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.60       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.65       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.70       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.75       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
0.80       0.4532   0.2105   0.4226   0.9335   0.7289   0.1230  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4532, F1=0.2105, Normal Recall=0.4226, Normal Precision=0.9335, Attack Recall=0.7289, Attack Precision=0.1230

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
0.15       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402   <--
0.20       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.25       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.30       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.35       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.40       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.45       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.50       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.55       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.60       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.65       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.70       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.75       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
0.80       0.4837   0.3616   0.4219   0.8626   0.7311   0.2402  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4837, F1=0.3616, Normal Recall=0.4219, Normal Precision=0.8626, Attack Recall=0.7311, Attack Precision=0.2402

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
0.15       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524   <--
0.20       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.25       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.30       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.35       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.40       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.45       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.50       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.55       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.60       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.65       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.70       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.75       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
0.80       0.5163   0.4756   0.4242   0.7864   0.7311   0.3524  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5163, F1=0.4756, Normal Recall=0.4242, Normal Precision=0.7864, Attack Recall=0.7311, Attack Precision=0.3524

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
0.15       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585   <--
0.20       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.25       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.30       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.35       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.40       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.45       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.50       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.55       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.60       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.65       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.70       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.75       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
0.80       0.5470   0.5636   0.4243   0.7030   0.7311   0.4585  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5470, F1=0.5636, Normal Recall=0.4243, Normal Precision=0.7030, Attack Recall=0.7311, Attack Precision=0.4585

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
0.15       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587   <--
0.20       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.25       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.30       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.35       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.40       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.45       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.50       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.55       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.60       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.65       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.70       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.75       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
0.80       0.5768   0.6334   0.4225   0.6111   0.7311   0.5587  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5768, F1=0.6334, Normal Recall=0.4225, Normal Precision=0.6111, Attack Recall=0.7311, Attack Precision=0.5587

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
0.15       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462   <--
0.20       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.25       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.30       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.35       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.40       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.45       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.50       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.55       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.60       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.65       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.70       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.75       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
0.80       0.6939   0.3950   0.6599   0.9999   0.9995   0.2462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6939, F1=0.3950, Normal Recall=0.6599, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.2462

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
0.15       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240   <--
0.20       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.25       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.30       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.35       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.40       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.45       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.50       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.55       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.60       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.65       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.70       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.75       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
0.80       0.7283   0.5954   0.6605   0.9998   0.9996   0.4240  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7283, F1=0.5954, Normal Recall=0.6605, Normal Precision=0.9998, Attack Recall=0.9996, Attack Precision=0.4240

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
0.15       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576   <--
0.20       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.25       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.30       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.35       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.40       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.45       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.50       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.55       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.60       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.65       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.70       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.75       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
0.80       0.7619   0.7159   0.6601   0.9997   0.9996   0.5576  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7619, F1=0.7159, Normal Recall=0.6601, Normal Precision=0.9997, Attack Recall=0.9996, Attack Precision=0.5576

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
0.15       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626   <--
0.20       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.25       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.30       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.35       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.40       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.45       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.50       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.55       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.60       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.65       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.70       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.75       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
0.80       0.7962   0.7969   0.6607   0.9995   0.9996   0.6626  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7962, F1=0.7969, Normal Recall=0.6607, Normal Precision=0.9995, Attack Recall=0.9996, Attack Precision=0.6626

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
0.15       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457   <--
0.20       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.25       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.30       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.35       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.40       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.45       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.50       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.55       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.60       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.65       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.70       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.75       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
0.80       0.8294   0.8542   0.6592   0.9993   0.9996   0.7457  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8294, F1=0.8542, Normal Recall=0.6592, Normal Precision=0.9993, Attack Recall=0.9996, Attack Precision=0.7457

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
0.15       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462   <--
0.20       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.25       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.30       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.35       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.40       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.45       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.50       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.55       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.60       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.65       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.70       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.75       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.80       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6940, F1=0.3951, Normal Recall=0.6601, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.2462

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
0.15       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241   <--
0.20       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.25       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.30       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.35       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.40       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.45       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.50       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.55       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.60       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.65       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.70       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.75       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.80       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7284, F1=0.5955, Normal Recall=0.6606, Normal Precision=0.9998, Attack Recall=0.9996, Attack Precision=0.4241

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
0.15       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577   <--
0.20       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.25       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.30       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.35       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.40       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.45       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.50       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.55       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.60       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.65       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.70       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.75       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.80       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7620, F1=0.7159, Normal Recall=0.6603, Normal Precision=0.9997, Attack Recall=0.9996, Attack Precision=0.5577

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
0.15       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627   <--
0.20       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.25       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.30       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.35       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.40       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.45       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.50       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.55       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.60       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.65       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.70       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.75       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.80       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7963, F1=0.7970, Normal Recall=0.6608, Normal Precision=0.9996, Attack Recall=0.9996, Attack Precision=0.6627

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
0.15       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458   <--
0.20       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.25       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.30       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.35       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.40       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.45       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.50       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.55       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.60       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.65       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.70       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.75       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.80       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8295, F1=0.8542, Normal Recall=0.6594, Normal Precision=0.9993, Attack Recall=0.9996, Attack Precision=0.7458

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
0.15       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462   <--
0.20       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.25       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.30       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.35       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.40       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.45       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.50       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.55       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.60       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.65       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.70       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.75       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
0.80       0.6940   0.3951   0.6601   0.9999   0.9995   0.2462  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6940, F1=0.3951, Normal Recall=0.6601, Normal Precision=0.9999, Attack Recall=0.9995, Attack Precision=0.2462

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
0.15       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241   <--
0.20       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.25       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.30       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.35       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.40       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.45       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.50       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.55       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.60       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.65       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.70       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.75       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
0.80       0.7284   0.5955   0.6606   0.9998   0.9996   0.4241  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7284, F1=0.5955, Normal Recall=0.6606, Normal Precision=0.9998, Attack Recall=0.9996, Attack Precision=0.4241

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
0.15       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577   <--
0.20       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.25       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.30       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.35       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.40       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.45       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.50       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.55       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.60       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.65       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.70       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.75       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
0.80       0.7620   0.7159   0.6603   0.9997   0.9996   0.5577  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7620, F1=0.7159, Normal Recall=0.6603, Normal Precision=0.9997, Attack Recall=0.9996, Attack Precision=0.5577

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
0.15       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627   <--
0.20       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.25       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.30       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.35       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.40       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.45       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.50       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.55       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.60       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.65       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.70       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.75       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
0.80       0.7963   0.7970   0.6608   0.9996   0.9996   0.6627  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7963, F1=0.7970, Normal Recall=0.6608, Normal Precision=0.9996, Attack Recall=0.9996, Attack Precision=0.6627

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
0.15       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458   <--
0.20       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.25       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.30       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.35       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.40       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.45       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.50       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.55       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.60       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.65       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.70       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.75       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
0.80       0.8295   0.8542   0.6594   0.9993   0.9996   0.7458  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8295, F1=0.8542, Normal Recall=0.6594, Normal Precision=0.9993, Attack Recall=0.9996, Attack Precision=0.7458

```

