# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-12 06:49:23 |

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
| Original (TFLite) | 0.9007 | 0.9108 | 0.9197 | 0.9296 | 0.9381 | 0.9469 | 0.9572 | 0.9665 | 0.9755 | 0.9847 | 0.9943 |
| QAT+Prune only | 0.9056 | 0.9123 | 0.9179 | 0.9245 | 0.9303 | 0.9356 | 0.9420 | 0.9481 | 0.9528 | 0.9591 | 0.9652 |
| QAT+PTQ | 0.9059 | 0.9125 | 0.9180 | 0.9244 | 0.9302 | 0.9352 | 0.9416 | 0.9477 | 0.9524 | 0.9584 | 0.9645 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9059 | 0.9125 | 0.9180 | 0.9244 | 0.9302 | 0.9352 | 0.9416 | 0.9477 | 0.9524 | 0.9584 | 0.9645 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6905 | 0.8320 | 0.8945 | 0.9278 | 0.9493 | 0.9654 | 0.9765 | 0.9848 | 0.9915 | 0.9971 |
| QAT+Prune only | 0.0000 | 0.6879 | 0.8246 | 0.8846 | 0.9172 | 0.9374 | 0.9523 | 0.9630 | 0.9703 | 0.9770 | 0.9823 |
| QAT+PTQ | 0.0000 | 0.6882 | 0.8246 | 0.8845 | 0.9170 | 0.9371 | 0.9519 | 0.9627 | 0.9701 | 0.9766 | 0.9819 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.6882 | 0.8246 | 0.8845 | 0.9170 | 0.9371 | 0.9519 | 0.9627 | 0.9701 | 0.9766 | 0.9819 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9007 | 0.9015 | 0.9010 | 0.9019 | 0.9006 | 0.8995 | 0.9016 | 0.9016 | 0.9003 | 0.8985 | 0.0000 |
| QAT+Prune only | 0.9056 | 0.9063 | 0.9060 | 0.9070 | 0.9071 | 0.9060 | 0.9074 | 0.9082 | 0.9034 | 0.9044 | 0.0000 |
| QAT+PTQ | 0.9059 | 0.9066 | 0.9063 | 0.9072 | 0.9073 | 0.9060 | 0.9072 | 0.9086 | 0.9040 | 0.9041 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9059 | 0.9066 | 0.9063 | 0.9072 | 0.9073 | 0.9060 | 0.9072 | 0.9086 | 0.9040 | 0.9041 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9007 | 0.0000 | 0.0000 | 0.0000 | 0.9007 | 1.0000 |
| 90 | 10 | 299,940 | 0.9108 | 0.5288 | 0.9947 | 0.6905 | 0.9015 | 0.9993 |
| 80 | 20 | 291,350 | 0.9197 | 0.7152 | 0.9943 | 0.8320 | 0.9010 | 0.9984 |
| 70 | 30 | 194,230 | 0.9296 | 0.8129 | 0.9943 | 0.8945 | 0.9019 | 0.9973 |
| 60 | 40 | 145,675 | 0.9381 | 0.8696 | 0.9943 | 0.9278 | 0.9006 | 0.9958 |
| 50 | 50 | 116,540 | 0.9469 | 0.9082 | 0.9943 | 0.9493 | 0.8995 | 0.9937 |
| 40 | 60 | 97,115 | 0.9572 | 0.9381 | 0.9943 | 0.9654 | 0.9016 | 0.9906 |
| 30 | 70 | 83,240 | 0.9665 | 0.9593 | 0.9943 | 0.9765 | 0.9016 | 0.9854 |
| 20 | 80 | 72,835 | 0.9755 | 0.9755 | 0.9943 | 0.9848 | 0.9003 | 0.9752 |
| 10 | 90 | 64,740 | 0.9847 | 0.9888 | 0.9943 | 0.9915 | 0.8985 | 0.9457 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9971 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9056 | 0.0000 | 0.0000 | 0.0000 | 0.9056 | 1.0000 |
| 90 | 10 | 299,940 | 0.9123 | 0.5340 | 0.9664 | 0.6879 | 0.9063 | 0.9959 |
| 80 | 20 | 291,350 | 0.9179 | 0.7198 | 0.9652 | 0.8246 | 0.9060 | 0.9905 |
| 70 | 30 | 194,230 | 0.9245 | 0.8164 | 0.9652 | 0.8846 | 0.9070 | 0.9838 |
| 60 | 40 | 145,675 | 0.9303 | 0.8738 | 0.9652 | 0.9172 | 0.9071 | 0.9750 |
| 50 | 50 | 116,540 | 0.9356 | 0.9113 | 0.9652 | 0.9374 | 0.9060 | 0.9630 |
| 40 | 60 | 97,115 | 0.9420 | 0.9399 | 0.9652 | 0.9523 | 0.9074 | 0.9455 |
| 30 | 70 | 83,240 | 0.9481 | 0.9608 | 0.9652 | 0.9630 | 0.9082 | 0.9178 |
| 20 | 80 | 72,835 | 0.9528 | 0.9756 | 0.9652 | 0.9703 | 0.9034 | 0.8664 |
| 10 | 90 | 64,740 | 0.9591 | 0.9891 | 0.9652 | 0.9770 | 0.9044 | 0.7425 |
| 0 | 100 | 58,270 | 0.9652 | 1.0000 | 0.9652 | 0.9823 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9059 | 0.0000 | 0.0000 | 0.0000 | 0.9059 | 1.0000 |
| 90 | 10 | 299,940 | 0.9125 | 0.5346 | 0.9657 | 0.6882 | 0.9066 | 0.9958 |
| 80 | 20 | 291,350 | 0.9180 | 0.7202 | 0.9645 | 0.8246 | 0.9063 | 0.9903 |
| 70 | 30 | 194,230 | 0.9244 | 0.8167 | 0.9645 | 0.8845 | 0.9072 | 0.9835 |
| 60 | 40 | 145,675 | 0.9302 | 0.8740 | 0.9645 | 0.9170 | 0.9073 | 0.9746 |
| 50 | 50 | 116,540 | 0.9352 | 0.9112 | 0.9645 | 0.9371 | 0.9060 | 0.9623 |
| 40 | 60 | 97,115 | 0.9416 | 0.9398 | 0.9645 | 0.9519 | 0.9072 | 0.9445 |
| 30 | 70 | 83,240 | 0.9477 | 0.9610 | 0.9645 | 0.9627 | 0.9086 | 0.9164 |
| 20 | 80 | 72,835 | 0.9524 | 0.9757 | 0.9645 | 0.9701 | 0.9040 | 0.8641 |
| 10 | 90 | 64,740 | 0.9584 | 0.9891 | 0.9645 | 0.9766 | 0.9041 | 0.7386 |
| 0 | 100 | 58,270 | 0.9645 | 1.0000 | 0.9645 | 0.9819 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9059 | 0.0000 | 0.0000 | 0.0000 | 0.9059 | 1.0000 |
| 90 | 10 | 299,940 | 0.9125 | 0.5346 | 0.9657 | 0.6882 | 0.9066 | 0.9958 |
| 80 | 20 | 291,350 | 0.9180 | 0.7202 | 0.9645 | 0.8246 | 0.9063 | 0.9903 |
| 70 | 30 | 194,230 | 0.9244 | 0.8167 | 0.9645 | 0.8845 | 0.9072 | 0.9835 |
| 60 | 40 | 145,675 | 0.9302 | 0.8740 | 0.9645 | 0.9170 | 0.9073 | 0.9746 |
| 50 | 50 | 116,540 | 0.9352 | 0.9112 | 0.9645 | 0.9371 | 0.9060 | 0.9623 |
| 40 | 60 | 97,115 | 0.9416 | 0.9398 | 0.9645 | 0.9519 | 0.9072 | 0.9445 |
| 30 | 70 | 83,240 | 0.9477 | 0.9610 | 0.9645 | 0.9627 | 0.9086 | 0.9164 |
| 20 | 80 | 72,835 | 0.9524 | 0.9757 | 0.9645 | 0.9701 | 0.9040 | 0.8641 |
| 10 | 90 | 64,740 | 0.9584 | 0.9891 | 0.9645 | 0.9766 | 0.9041 | 0.7386 |
| 0 | 100 | 58,270 | 0.9645 | 1.0000 | 0.9645 | 0.9819 | 0.0000 | 0.0000 |


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
0.15       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288   <--
0.20       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.25       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.30       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.35       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.40       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.45       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.50       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.55       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.60       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.65       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.70       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.75       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
0.80       0.9108   0.6905   0.9015   0.9993   0.9946   0.5288  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9108, F1=0.6905, Normal Recall=0.9015, Normal Precision=0.9993, Attack Recall=0.9946, Attack Precision=0.5288

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
0.15       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166   <--
0.20       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.25       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.30       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.35       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.40       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.45       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.50       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.55       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.60       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.65       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.70       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.75       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
0.80       0.9202   0.8329   0.9017   0.9984   0.9943   0.7166  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9202, F1=0.8329, Normal Recall=0.9017, Normal Precision=0.9984, Attack Recall=0.9943, Attack Precision=0.7166

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
0.15       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122   <--
0.20       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.25       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.30       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.35       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.40       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.45       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.50       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.55       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.60       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.65       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.70       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.75       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
0.80       0.9293   0.8941   0.9015   0.9973   0.9943   0.8122  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9293, F1=0.8941, Normal Recall=0.9015, Normal Precision=0.9973, Attack Recall=0.9943, Attack Precision=0.8122

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
0.15       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700   <--
0.20       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.25       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.30       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.35       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.40       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.45       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.50       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.55       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.60       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.65       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.70       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.75       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
0.80       0.9383   0.9280   0.9010   0.9958   0.9943   0.8700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9383, F1=0.9280, Normal Recall=0.9010, Normal Precision=0.9958, Attack Recall=0.9943, Attack Precision=0.8700

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
0.15       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097   <--
0.20       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.25       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.30       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.35       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.40       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.45       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.50       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.55       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.60       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.65       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.70       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.75       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
0.80       0.9478   0.9501   0.9013   0.9937   0.9943   0.9097  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9478, F1=0.9501, Normal Recall=0.9013, Normal Precision=0.9937, Attack Recall=0.9943, Attack Precision=0.9097

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
0.15       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340   <--
0.20       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.25       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.30       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.35       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.40       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.45       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.50       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.55       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.60       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.65       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.70       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.75       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
0.80       0.9123   0.6880   0.9063   0.9959   0.9667   0.5340  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9123, F1=0.6880, Normal Recall=0.9063, Normal Precision=0.9959, Attack Recall=0.9667, Attack Precision=0.5340

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
0.15       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206   <--
0.20       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.25       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.30       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.35       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.40       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.45       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.50       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.55       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.60       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.65       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.70       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.75       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
0.80       0.9182   0.8252   0.9065   0.9905   0.9652   0.7206  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9182, F1=0.8252, Normal Recall=0.9065, Normal Precision=0.9905, Attack Recall=0.9652, Attack Precision=0.7206

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
0.15       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145   <--
0.20       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.25       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.30       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.35       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.40       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.45       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.50       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.55       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.60       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.65       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.70       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.75       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
0.80       0.9236   0.8834   0.9058   0.9838   0.9652   0.8145  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9236, F1=0.8834, Normal Recall=0.9058, Normal Precision=0.9838, Attack Recall=0.9652, Attack Precision=0.8145

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
0.15       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722   <--
0.20       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.25       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.30       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.35       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.40       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.45       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.50       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.55       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.60       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.65       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.70       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.75       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
0.80       0.9295   0.9163   0.9057   0.9750   0.9652   0.8722  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9295, F1=0.9163, Normal Recall=0.9057, Normal Precision=0.9750, Attack Recall=0.9652, Attack Precision=0.8722

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
0.15       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113   <--
0.20       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.25       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.30       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.35       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.40       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.45       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.50       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.55       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.60       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.65       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.70       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.75       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
0.80       0.9356   0.9374   0.9060   0.9630   0.9652   0.9113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9356, F1=0.9374, Normal Recall=0.9060, Normal Precision=0.9630, Attack Recall=0.9652, Attack Precision=0.9113

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
0.15       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347   <--
0.20       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.25       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.30       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.35       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.40       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.45       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.50       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.55       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.60       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.65       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.70       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.75       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.80       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9125, F1=0.6883, Normal Recall=0.9066, Normal Precision=0.9958, Attack Recall=0.9658, Attack Precision=0.5347

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
0.15       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211   <--
0.20       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.25       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.30       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.35       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.40       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.45       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.50       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.55       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.60       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.65       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.70       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.75       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.80       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9183, F1=0.8252, Normal Recall=0.9068, Normal Precision=0.9903, Attack Recall=0.9645, Attack Precision=0.7211

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
0.15       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147   <--
0.20       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.25       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.30       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.35       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.40       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.45       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.50       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.55       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.60       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.65       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.70       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.75       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.80       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9235, F1=0.8833, Normal Recall=0.9060, Normal Precision=0.9835, Attack Recall=0.9645, Attack Precision=0.8147

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
0.15       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724   <--
0.20       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.25       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.30       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.35       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.40       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.45       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.50       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.55       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.60       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.65       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.70       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.75       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.80       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9294, F1=0.9161, Normal Recall=0.9060, Normal Precision=0.9745, Attack Recall=0.9645, Attack Precision=0.8724

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
0.15       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113   <--
0.20       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.25       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.30       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.35       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.40       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.45       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.50       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.55       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.60       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.65       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.70       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.75       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.80       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9353, F1=0.9371, Normal Recall=0.9061, Normal Precision=0.9623, Attack Recall=0.9645, Attack Precision=0.9113

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
0.15       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347   <--
0.20       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.25       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.30       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.35       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.40       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.45       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.50       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.55       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.60       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.65       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.70       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.75       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
0.80       0.9125   0.6883   0.9066   0.9958   0.9658   0.5347  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9125, F1=0.6883, Normal Recall=0.9066, Normal Precision=0.9958, Attack Recall=0.9658, Attack Precision=0.5347

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
0.15       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211   <--
0.20       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.25       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.30       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.35       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.40       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.45       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.50       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.55       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.60       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.65       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.70       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.75       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
0.80       0.9183   0.8252   0.9068   0.9903   0.9645   0.7211  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9183, F1=0.8252, Normal Recall=0.9068, Normal Precision=0.9903, Attack Recall=0.9645, Attack Precision=0.7211

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
0.15       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147   <--
0.20       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.25       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.30       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.35       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.40       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.45       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.50       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.55       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.60       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.65       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.70       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.75       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
0.80       0.9235   0.8833   0.9060   0.9835   0.9645   0.8147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9235, F1=0.8833, Normal Recall=0.9060, Normal Precision=0.9835, Attack Recall=0.9645, Attack Precision=0.8147

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
0.15       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724   <--
0.20       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.25       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.30       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.35       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.40       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.45       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.50       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.55       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.60       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.65       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.70       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.75       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
0.80       0.9294   0.9161   0.9060   0.9745   0.9645   0.8724  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9294, F1=0.9161, Normal Recall=0.9060, Normal Precision=0.9745, Attack Recall=0.9645, Attack Precision=0.8724

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
0.15       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113   <--
0.20       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.25       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.30       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.35       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.40       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.45       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.50       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.55       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.60       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.65       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.70       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.75       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
0.80       0.9353   0.9371   0.9061   0.9623   0.9645   0.9113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9353, F1=0.9371, Normal Recall=0.9061, Normal Precision=0.9623, Attack Recall=0.9645, Attack Precision=0.9113

```

