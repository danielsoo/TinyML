# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-16 13:43:16 |

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
| Original (TFLite) | 0.7891 | 0.7801 | 0.7721 | 0.7656 | 0.7569 | 0.7490 | 0.7411 | 0.7336 | 0.7254 | 0.7172 | 0.7095 |
| QAT+Prune only | 0.7303 | 0.7563 | 0.7824 | 0.8103 | 0.8372 | 0.8623 | 0.8905 | 0.9175 | 0.9448 | 0.9708 | 0.9982 |
| QAT+PTQ | 0.7299 | 0.7560 | 0.7822 | 0.8102 | 0.8368 | 0.8622 | 0.8903 | 0.9175 | 0.9446 | 0.9708 | 0.9982 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7299 | 0.7560 | 0.7822 | 0.8102 | 0.8368 | 0.8622 | 0.8903 | 0.9175 | 0.9446 | 0.9708 | 0.9982 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3915 | 0.5546 | 0.6450 | 0.7002 | 0.7387 | 0.7668 | 0.7885 | 0.8052 | 0.8187 | 0.8301 |
| QAT+Prune only | 0.0000 | 0.4503 | 0.6473 | 0.7595 | 0.8306 | 0.8788 | 0.9162 | 0.9443 | 0.9666 | 0.9840 | 0.9991 |
| QAT+PTQ | 0.0000 | 0.4500 | 0.6470 | 0.7594 | 0.8303 | 0.8787 | 0.9161 | 0.9442 | 0.9665 | 0.9840 | 0.9991 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4500 | 0.6470 | 0.7594 | 0.8303 | 0.8787 | 0.9161 | 0.9442 | 0.9665 | 0.9840 | 0.9991 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7891 | 0.7881 | 0.7877 | 0.7897 | 0.7885 | 0.7886 | 0.7885 | 0.7897 | 0.7886 | 0.7867 | 0.0000 |
| QAT+Prune only | 0.7303 | 0.7294 | 0.7285 | 0.7298 | 0.7299 | 0.7265 | 0.7289 | 0.7293 | 0.7314 | 0.7247 | 0.0000 |
| QAT+PTQ | 0.7299 | 0.7291 | 0.7282 | 0.7297 | 0.7292 | 0.7263 | 0.7285 | 0.7291 | 0.7301 | 0.7246 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7299 | 0.7291 | 0.7282 | 0.7297 | 0.7292 | 0.7263 | 0.7285 | 0.7291 | 0.7301 | 0.7246 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7891 | 0.0000 | 0.0000 | 0.0000 | 0.7891 | 1.0000 |
| 90 | 10 | 299,940 | 0.7801 | 0.2706 | 0.7076 | 0.3915 | 0.7881 | 0.9604 |
| 80 | 20 | 291,350 | 0.7721 | 0.4552 | 0.7095 | 0.5546 | 0.7877 | 0.9156 |
| 70 | 30 | 194,230 | 0.7656 | 0.5911 | 0.7096 | 0.6450 | 0.7897 | 0.8638 |
| 60 | 40 | 145,675 | 0.7569 | 0.6911 | 0.7095 | 0.7002 | 0.7885 | 0.8028 |
| 50 | 50 | 116,540 | 0.7490 | 0.7704 | 0.7095 | 0.7387 | 0.7886 | 0.7308 |
| 40 | 60 | 97,115 | 0.7411 | 0.8342 | 0.7095 | 0.7668 | 0.7885 | 0.6441 |
| 30 | 70 | 83,240 | 0.7336 | 0.8873 | 0.7095 | 0.7885 | 0.7897 | 0.5382 |
| 20 | 80 | 72,835 | 0.7254 | 0.9307 | 0.7095 | 0.8052 | 0.7886 | 0.4043 |
| 10 | 90 | 64,740 | 0.7172 | 0.9677 | 0.7095 | 0.8187 | 0.7867 | 0.2313 |
| 0 | 100 | 58,270 | 0.7095 | 1.0000 | 0.7095 | 0.8301 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7303 | 0.0000 | 0.0000 | 0.0000 | 0.7303 | 1.0000 |
| 90 | 10 | 299,940 | 0.7563 | 0.2907 | 0.9982 | 0.4503 | 0.7294 | 0.9997 |
| 80 | 20 | 291,350 | 0.7824 | 0.4789 | 0.9982 | 0.6473 | 0.7285 | 0.9994 |
| 70 | 30 | 194,230 | 0.8103 | 0.6129 | 0.9982 | 0.7595 | 0.7298 | 0.9989 |
| 60 | 40 | 145,675 | 0.8372 | 0.7113 | 0.9982 | 0.8306 | 0.7299 | 0.9983 |
| 50 | 50 | 116,540 | 0.8623 | 0.7849 | 0.9982 | 0.8788 | 0.7265 | 0.9975 |
| 40 | 60 | 97,115 | 0.8905 | 0.8467 | 0.9982 | 0.9162 | 0.7289 | 0.9963 |
| 30 | 70 | 83,240 | 0.9175 | 0.8959 | 0.9982 | 0.9443 | 0.7293 | 0.9942 |
| 20 | 80 | 72,835 | 0.9448 | 0.9370 | 0.9982 | 0.9666 | 0.7314 | 0.9901 |
| 10 | 90 | 64,740 | 0.9708 | 0.9703 | 0.9982 | 0.9840 | 0.7247 | 0.9779 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7299 | 0.0000 | 0.0000 | 0.0000 | 0.7299 | 1.0000 |
| 90 | 10 | 299,940 | 0.7560 | 0.2905 | 0.9982 | 0.4500 | 0.7291 | 0.9997 |
| 80 | 20 | 291,350 | 0.7822 | 0.4787 | 0.9982 | 0.6470 | 0.7282 | 0.9994 |
| 70 | 30 | 194,230 | 0.8102 | 0.6128 | 0.9982 | 0.7594 | 0.7297 | 0.9989 |
| 60 | 40 | 145,675 | 0.8368 | 0.7108 | 0.9982 | 0.8303 | 0.7292 | 0.9983 |
| 50 | 50 | 116,540 | 0.8622 | 0.7848 | 0.9982 | 0.8787 | 0.7263 | 0.9975 |
| 40 | 60 | 97,115 | 0.8903 | 0.8465 | 0.9982 | 0.9161 | 0.7285 | 0.9963 |
| 30 | 70 | 83,240 | 0.9175 | 0.8958 | 0.9982 | 0.9442 | 0.7291 | 0.9942 |
| 20 | 80 | 72,835 | 0.9446 | 0.9367 | 0.9982 | 0.9665 | 0.7301 | 0.9901 |
| 10 | 90 | 64,740 | 0.9708 | 0.9703 | 0.9982 | 0.9840 | 0.7246 | 0.9779 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7299 | 0.0000 | 0.0000 | 0.0000 | 0.7299 | 1.0000 |
| 90 | 10 | 299,940 | 0.7560 | 0.2905 | 0.9982 | 0.4500 | 0.7291 | 0.9997 |
| 80 | 20 | 291,350 | 0.7822 | 0.4787 | 0.9982 | 0.6470 | 0.7282 | 0.9994 |
| 70 | 30 | 194,230 | 0.8102 | 0.6128 | 0.9982 | 0.7594 | 0.7297 | 0.9989 |
| 60 | 40 | 145,675 | 0.8368 | 0.7108 | 0.9982 | 0.8303 | 0.7292 | 0.9983 |
| 50 | 50 | 116,540 | 0.8622 | 0.7848 | 0.9982 | 0.8787 | 0.7263 | 0.9975 |
| 40 | 60 | 97,115 | 0.8903 | 0.8465 | 0.9982 | 0.9161 | 0.7285 | 0.9963 |
| 30 | 70 | 83,240 | 0.9175 | 0.8958 | 0.9982 | 0.9442 | 0.7291 | 0.9942 |
| 20 | 80 | 72,835 | 0.9446 | 0.9367 | 0.9982 | 0.9665 | 0.7301 | 0.9901 |
| 10 | 90 | 64,740 | 0.9708 | 0.9703 | 0.9982 | 0.9840 | 0.7246 | 0.9779 |
| 0 | 100 | 58,270 | 0.9982 | 1.0000 | 0.9982 | 0.9991 | 0.0000 | 0.0000 |


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
0.15       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701   <--
0.20       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.25       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.30       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.35       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.40       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.45       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.50       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.55       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.60       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.65       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.70       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.75       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
0.80       0.7799   0.3907   0.7881   0.9602   0.7057   0.2701  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7799, F1=0.3907, Normal Recall=0.7881, Normal Precision=0.9602, Attack Recall=0.7057, Attack Precision=0.2701

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
0.15       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560   <--
0.20       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.25       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.30       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.35       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.40       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.45       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.50       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.55       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.60       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.65       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.70       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.75       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
0.80       0.7726   0.5552   0.7884   0.9157   0.7095   0.4560  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7726, F1=0.5552, Normal Recall=0.7884, Normal Precision=0.9157, Attack Recall=0.7095, Attack Precision=0.4560

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
0.15       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904   <--
0.20       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.25       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.30       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.35       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.40       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.45       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.50       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.55       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.60       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.65       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.70       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.75       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
0.80       0.7652   0.6445   0.7890   0.8637   0.7095   0.5904  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7652, F1=0.6445, Normal Recall=0.7890, Normal Precision=0.8637, Attack Recall=0.7095, Attack Precision=0.5904

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
0.15       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927   <--
0.20       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.25       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.30       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.35       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.40       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.45       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.50       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.55       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.60       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.65       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.70       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.75       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
0.80       0.7579   0.7010   0.7901   0.8032   0.7095   0.6927  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7579, F1=0.7010, Normal Recall=0.7901, Normal Precision=0.8032, Attack Recall=0.7095, Attack Precision=0.6927

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
0.15       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721   <--
0.20       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.25       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.30       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.35       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.40       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.45       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.50       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.55       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.60       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.65       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.70       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.75       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
0.80       0.7501   0.7395   0.7906   0.7313   0.7095   0.7721  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7501, F1=0.7395, Normal Recall=0.7906, Normal Precision=0.7313, Attack Recall=0.7095, Attack Precision=0.7721

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
0.15       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907   <--
0.20       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.25       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.30       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.35       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.40       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.45       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.50       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.55       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.60       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.65       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.70       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.75       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
0.80       0.7563   0.4503   0.7294   0.9997   0.9983   0.2907  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7563, F1=0.4503, Normal Recall=0.7294, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2907

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
0.15       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804   <--
0.20       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.25       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.30       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.35       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.40       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.45       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.50       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.55       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.60       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.65       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.70       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.75       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
0.80       0.7837   0.6487   0.7301   0.9994   0.9982   0.4804  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7837, F1=0.6487, Normal Recall=0.7301, Normal Precision=0.9994, Attack Recall=0.9982, Attack Precision=0.4804

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
0.15       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138   <--
0.20       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.25       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.30       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.35       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.40       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.45       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.50       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.55       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.60       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.65       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.70       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.75       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
0.80       0.8110   0.7602   0.7308   0.9989   0.9982   0.6138  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8110, F1=0.7602, Normal Recall=0.7308, Normal Precision=0.9989, Attack Recall=0.9982, Attack Precision=0.6138

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
0.15       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113   <--
0.20       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.25       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.30       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.35       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.40       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.45       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.50       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.55       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.60       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.65       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.70       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.75       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
0.80       0.8373   0.8307   0.7300   0.9983   0.9982   0.7113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8373, F1=0.8307, Normal Recall=0.7300, Normal Precision=0.9983, Attack Recall=0.9982, Attack Precision=0.7113

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
0.15       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863   <--
0.20       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.25       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.30       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.35       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.40       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.45       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.50       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.55       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.60       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.65       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.70       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.75       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
0.80       0.8635   0.8797   0.7288   0.9975   0.9982   0.7863  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8635, F1=0.8797, Normal Recall=0.7288, Normal Precision=0.9975, Attack Recall=0.9982, Attack Precision=0.7863

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
0.15       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905   <--
0.20       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.25       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.30       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.35       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.40       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.45       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.50       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.55       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.60       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.65       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.70       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.75       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.80       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7561, F1=0.4501, Normal Recall=0.7291, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2905

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
0.15       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802   <--
0.20       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.25       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.30       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.35       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.40       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.45       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.50       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.55       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.60       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.65       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.70       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.75       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.80       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7835, F1=0.6484, Normal Recall=0.7299, Normal Precision=0.9994, Attack Recall=0.9982, Attack Precision=0.4802

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
0.15       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136   <--
0.20       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.25       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.30       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.35       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.40       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.45       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.50       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.55       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.60       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.65       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.70       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.75       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.80       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8109, F1=0.7600, Normal Recall=0.7306, Normal Precision=0.9989, Attack Recall=0.9982, Attack Precision=0.6136

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
0.15       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112   <--
0.20       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.25       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.30       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.35       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.40       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.45       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.50       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.55       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.60       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.65       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.70       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.75       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.80       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8371, F1=0.8306, Normal Recall=0.7298, Normal Precision=0.9983, Attack Recall=0.9982, Attack Precision=0.7112

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
0.15       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863   <--
0.20       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.25       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.30       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.35       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.40       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.45       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.50       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.55       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.60       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.65       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.70       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.75       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.80       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8634, F1=0.8797, Normal Recall=0.7287, Normal Precision=0.9975, Attack Recall=0.9982, Attack Precision=0.7863

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
0.15       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905   <--
0.20       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.25       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.30       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.35       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.40       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.45       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.50       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.55       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.60       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.65       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.70       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.75       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
0.80       0.7561   0.4501   0.7291   0.9997   0.9983   0.2905  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7561, F1=0.4501, Normal Recall=0.7291, Normal Precision=0.9997, Attack Recall=0.9983, Attack Precision=0.2905

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
0.15       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802   <--
0.20       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.25       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.30       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.35       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.40       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.45       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.50       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.55       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.60       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.65       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.70       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.75       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
0.80       0.7835   0.6484   0.7299   0.9994   0.9982   0.4802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7835, F1=0.6484, Normal Recall=0.7299, Normal Precision=0.9994, Attack Recall=0.9982, Attack Precision=0.4802

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
0.15       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136   <--
0.20       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.25       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.30       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.35       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.40       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.45       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.50       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.55       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.60       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.65       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.70       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.75       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
0.80       0.8109   0.7600   0.7306   0.9989   0.9982   0.6136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8109, F1=0.7600, Normal Recall=0.7306, Normal Precision=0.9989, Attack Recall=0.9982, Attack Precision=0.6136

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
0.15       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112   <--
0.20       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.25       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.30       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.35       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.40       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.45       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.50       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.55       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.60       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.65       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.70       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.75       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
0.80       0.8371   0.8306   0.7298   0.9983   0.9982   0.7112  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8371, F1=0.8306, Normal Recall=0.7298, Normal Precision=0.9983, Attack Recall=0.9982, Attack Precision=0.7112

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
0.15       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863   <--
0.20       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.25       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.30       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.35       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.40       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.45       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.50       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.55       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.60       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.65       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.70       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.75       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
0.80       0.8634   0.8797   0.7287   0.9975   0.9982   0.7863  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8634, F1=0.8797, Normal Recall=0.7287, Normal Precision=0.9975, Attack Recall=0.9982, Attack Precision=0.7863

```

