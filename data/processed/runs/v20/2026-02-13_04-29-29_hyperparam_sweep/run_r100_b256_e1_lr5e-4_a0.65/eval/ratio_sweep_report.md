# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b256_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-22 06:18:32 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 1 |
| **Batch size** | 256 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8672 | 0.8807 | 0.8932 | 0.9071 | 0.9190 | 0.9316 | 0.9453 | 0.9586 | 0.9706 | 0.9837 | 0.9970 |
| QAT+Prune only | 0.2220 | 0.2974 | 0.3747 | 0.4537 | 0.5326 | 0.6091 | 0.6874 | 0.7663 | 0.8439 | 0.9222 | 1.0000 |
| QAT+PTQ | 0.2246 | 0.2999 | 0.3769 | 0.4558 | 0.5343 | 0.6105 | 0.6883 | 0.7668 | 0.8444 | 0.9222 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.2246 | 0.2999 | 0.3769 | 0.4558 | 0.5343 | 0.6105 | 0.6883 | 0.7668 | 0.8444 | 0.9222 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6258 | 0.7887 | 0.8655 | 0.9078 | 0.9358 | 0.9563 | 0.9712 | 0.9819 | 0.9910 | 0.9985 |
| QAT+Prune only | 0.0000 | 0.2216 | 0.3901 | 0.5234 | 0.6312 | 0.7190 | 0.7933 | 0.8570 | 0.9111 | 0.9586 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.2222 | 0.3910 | 0.5244 | 0.6321 | 0.7197 | 0.7938 | 0.8572 | 0.9114 | 0.9586 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.2222 | 0.3910 | 0.5244 | 0.6321 | 0.7197 | 0.7938 | 0.8572 | 0.9114 | 0.9586 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8672 | 0.8678 | 0.8672 | 0.8685 | 0.8670 | 0.8661 | 0.8677 | 0.8690 | 0.8650 | 0.8635 | 0.0000 |
| QAT+Prune only | 0.2220 | 0.2194 | 0.2184 | 0.2196 | 0.2209 | 0.2182 | 0.2184 | 0.2211 | 0.2193 | 0.2218 | 0.0000 |
| QAT+PTQ | 0.2246 | 0.2221 | 0.2212 | 0.2225 | 0.2238 | 0.2210 | 0.2207 | 0.2226 | 0.2221 | 0.2224 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.2246 | 0.2221 | 0.2212 | 0.2225 | 0.2238 | 0.2210 | 0.2207 | 0.2226 | 0.2221 | 0.2224 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8672 | 0.0000 | 0.0000 | 0.0000 | 0.8672 | 1.0000 |
| 90 | 10 | 299,940 | 0.8807 | 0.4559 | 0.9973 | 0.6258 | 0.8678 | 0.9997 |
| 80 | 20 | 291,350 | 0.8932 | 0.6524 | 0.9970 | 0.7887 | 0.8672 | 0.9991 |
| 70 | 30 | 194,230 | 0.9071 | 0.7647 | 0.9970 | 0.8655 | 0.8685 | 0.9985 |
| 60 | 40 | 145,675 | 0.9190 | 0.8333 | 0.9970 | 0.9078 | 0.8670 | 0.9977 |
| 50 | 50 | 116,540 | 0.9316 | 0.8816 | 0.9970 | 0.9358 | 0.8661 | 0.9966 |
| 40 | 60 | 97,115 | 0.9453 | 0.9187 | 0.9970 | 0.9563 | 0.8677 | 0.9949 |
| 30 | 70 | 83,240 | 0.9586 | 0.9467 | 0.9970 | 0.9712 | 0.8690 | 0.9921 |
| 20 | 80 | 72,835 | 0.9706 | 0.9673 | 0.9970 | 0.9819 | 0.8650 | 0.9865 |
| 10 | 90 | 64,740 | 0.9837 | 0.9850 | 0.9970 | 0.9910 | 0.8635 | 0.9701 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2220 | 0.0000 | 0.0000 | 0.0000 | 0.2220 | 1.0000 |
| 90 | 10 | 299,940 | 0.2974 | 0.1246 | 1.0000 | 0.2216 | 0.2194 | 1.0000 |
| 80 | 20 | 291,350 | 0.3747 | 0.2423 | 1.0000 | 0.3901 | 0.2184 | 1.0000 |
| 70 | 30 | 194,230 | 0.4537 | 0.3545 | 1.0000 | 0.5234 | 0.2196 | 1.0000 |
| 60 | 40 | 145,675 | 0.5326 | 0.4611 | 1.0000 | 0.6312 | 0.2209 | 1.0000 |
| 50 | 50 | 116,540 | 0.6091 | 0.5612 | 1.0000 | 0.7190 | 0.2182 | 1.0000 |
| 40 | 60 | 97,115 | 0.6874 | 0.6574 | 1.0000 | 0.7933 | 0.2184 | 1.0000 |
| 30 | 70 | 83,240 | 0.7663 | 0.7497 | 1.0000 | 0.8570 | 0.2211 | 1.0000 |
| 20 | 80 | 72,835 | 0.8439 | 0.8367 | 1.0000 | 0.9111 | 0.2193 | 1.0000 |
| 10 | 90 | 64,740 | 0.9222 | 0.9204 | 1.0000 | 0.9586 | 0.2218 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2246 | 0.0000 | 0.0000 | 0.0000 | 0.2246 | 1.0000 |
| 90 | 10 | 299,940 | 0.2999 | 0.1250 | 1.0000 | 0.2222 | 0.2221 | 1.0000 |
| 80 | 20 | 291,350 | 0.3769 | 0.2430 | 1.0000 | 0.3910 | 0.2212 | 1.0000 |
| 70 | 30 | 194,230 | 0.4558 | 0.3554 | 1.0000 | 0.5244 | 0.2225 | 1.0000 |
| 60 | 40 | 145,675 | 0.5343 | 0.4621 | 1.0000 | 0.6321 | 0.2238 | 1.0000 |
| 50 | 50 | 116,540 | 0.6105 | 0.5621 | 1.0000 | 0.7197 | 0.2210 | 1.0000 |
| 40 | 60 | 97,115 | 0.6883 | 0.6581 | 1.0000 | 0.7938 | 0.2207 | 1.0000 |
| 30 | 70 | 83,240 | 0.7668 | 0.7501 | 1.0000 | 0.8572 | 0.2226 | 1.0000 |
| 20 | 80 | 72,835 | 0.8444 | 0.8372 | 1.0000 | 0.9114 | 0.2221 | 1.0000 |
| 10 | 90 | 64,740 | 0.9222 | 0.9205 | 1.0000 | 0.9586 | 0.2224 | 1.0000 |
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
| 100 | 0 | 100,000 | 0.2246 | 0.0000 | 0.0000 | 0.0000 | 0.2246 | 1.0000 |
| 90 | 10 | 299,940 | 0.2999 | 0.1250 | 1.0000 | 0.2222 | 0.2221 | 1.0000 |
| 80 | 20 | 291,350 | 0.3769 | 0.2430 | 1.0000 | 0.3910 | 0.2212 | 1.0000 |
| 70 | 30 | 194,230 | 0.4558 | 0.3554 | 1.0000 | 0.5244 | 0.2225 | 1.0000 |
| 60 | 40 | 145,675 | 0.5343 | 0.4621 | 1.0000 | 0.6321 | 0.2238 | 1.0000 |
| 50 | 50 | 116,540 | 0.6105 | 0.5621 | 1.0000 | 0.7197 | 0.2210 | 1.0000 |
| 40 | 60 | 97,115 | 0.6883 | 0.6581 | 1.0000 | 0.7938 | 0.2207 | 1.0000 |
| 30 | 70 | 83,240 | 0.7668 | 0.7501 | 1.0000 | 0.8572 | 0.2226 | 1.0000 |
| 20 | 80 | 72,835 | 0.8444 | 0.8372 | 1.0000 | 0.9114 | 0.2221 | 1.0000 |
| 10 | 90 | 64,740 | 0.9222 | 0.9205 | 1.0000 | 0.9586 | 0.2224 | 1.0000 |
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
0.15       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560   <--
0.20       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.25       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.30       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.35       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.40       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.45       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.50       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.55       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.60       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.65       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.70       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.75       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
0.80       0.8807   0.6258   0.8678   0.9997   0.9974   0.4560  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8807, F1=0.6258, Normal Recall=0.8678, Normal Precision=0.9997, Attack Recall=0.9974, Attack Precision=0.4560

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
0.15       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540   <--
0.20       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.25       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.30       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.35       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.40       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.45       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.50       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.55       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.60       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.65       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.70       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.75       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
0.80       0.8939   0.7899   0.8681   0.9992   0.9970   0.6540  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8939, F1=0.7899, Normal Recall=0.8681, Normal Precision=0.9992, Attack Recall=0.9970, Attack Precision=0.6540

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
0.15       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643   <--
0.20       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.25       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.30       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.35       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.40       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.45       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.50       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.55       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.60       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.65       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.70       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.75       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
0.80       0.9069   0.8653   0.8682   0.9985   0.9970   0.7643  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9069, F1=0.8653, Normal Recall=0.8682, Normal Precision=0.9985, Attack Recall=0.9970, Attack Precision=0.7643

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
0.15       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335   <--
0.20       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.25       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.30       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.35       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.40       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.45       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.50       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.55       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.60       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.65       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.70       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.75       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
0.80       0.9192   0.9080   0.8673   0.9977   0.9970   0.8335  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9192, F1=0.9080, Normal Recall=0.8673, Normal Precision=0.9977, Attack Recall=0.9970, Attack Precision=0.8335

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
0.15       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819   <--
0.20       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.25       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.30       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.35       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.40       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.45       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.50       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.55       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.60       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.65       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.70       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.75       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
0.80       0.9318   0.9360   0.8665   0.9966   0.9970   0.8819  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9318, F1=0.9360, Normal Recall=0.8665, Normal Precision=0.9966, Attack Recall=0.9970, Attack Precision=0.8819

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
0.15       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246   <--
0.20       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.25       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.30       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.35       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.40       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.45       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.50       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.55       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.60       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.65       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.70       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.75       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
0.80       0.2974   0.2216   0.2194   1.0000   1.0000   0.1246  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2974, F1=0.2216, Normal Recall=0.2194, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1246

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
0.15       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426   <--
0.20       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.25       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.30       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.35       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.40       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.45       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.50       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.55       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.60       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.65       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.70       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.75       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
0.80       0.3757   0.3905   0.2196   1.0000   1.0000   0.2426  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3757, F1=0.3905, Normal Recall=0.2196, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2426

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
0.15       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549   <--
0.20       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.25       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.30       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.35       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.40       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.45       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.50       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.55       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.60       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.65       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.70       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.75       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
0.80       0.4547   0.5239   0.2210   1.0000   1.0000   0.3549  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4547, F1=0.5239, Normal Recall=0.2210, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3549

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
0.15       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613   <--
0.20       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.25       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.30       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.35       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.40       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.45       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.50       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.55       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.60       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.65       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.70       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.75       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
0.80       0.5329   0.6314   0.2215   1.0000   1.0000   0.4613  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5329, F1=0.6314, Normal Recall=0.2215, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4613

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
0.15       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624   <--
0.20       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.25       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.30       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.35       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.40       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.45       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.50       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.55       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.60       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.65       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.70       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.75       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
0.80       0.6110   0.7199   0.2220   1.0000   1.0000   0.5624  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6110, F1=0.7199, Normal Recall=0.2220, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5624

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
0.15       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250   <--
0.20       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.25       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.30       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.35       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.40       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.45       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.50       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.55       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.60       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.65       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.70       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.75       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.80       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2999, F1=0.2222, Normal Recall=0.2221, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1250

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
0.15       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433   <--
0.20       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.25       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.30       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.35       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.40       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.45       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.50       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.55       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.60       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.65       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.70       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.75       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.80       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3778, F1=0.3913, Normal Recall=0.2223, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2433

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
0.15       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557   <--
0.20       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.25       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.30       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.35       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.40       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.45       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.50       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.55       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.60       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.65       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.70       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.75       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.80       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4567, F1=0.5248, Normal Recall=0.2238, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3557

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
0.15       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621   <--
0.20       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.25       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.30       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.35       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.40       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.45       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.50       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.55       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.60       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.65       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.70       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.75       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.80       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5345, F1=0.6321, Normal Recall=0.2241, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4621

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
0.15       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633   <--
0.20       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.25       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.30       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.35       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.40       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.45       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.50       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.55       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.60       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.65       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.70       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.75       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.80       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6124, F1=0.7207, Normal Recall=0.2248, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5633

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
0.15       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250   <--
0.20       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.25       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.30       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.35       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.40       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.45       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.50       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.55       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.60       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.65       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.70       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.75       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
0.80       0.2999   0.2222   0.2221   1.0000   1.0000   0.1250  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2999, F1=0.2222, Normal Recall=0.2221, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1250

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
0.15       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433   <--
0.20       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.25       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.30       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.35       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.40       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.45       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.50       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.55       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.60       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.65       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.70       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.75       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
0.80       0.3778   0.3913   0.2223   1.0000   1.0000   0.2433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3778, F1=0.3913, Normal Recall=0.2223, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2433

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
0.15       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557   <--
0.20       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.25       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.30       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.35       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.40       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.45       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.50       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.55       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.60       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.65       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.70       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.75       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
0.80       0.4567   0.5248   0.2238   1.0000   1.0000   0.3557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4567, F1=0.5248, Normal Recall=0.2238, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3557

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
0.15       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621   <--
0.20       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.25       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.30       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.35       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.40       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.45       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.50       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.55       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.60       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.65       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.70       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.75       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
0.80       0.5345   0.6321   0.2241   1.0000   1.0000   0.4621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5345, F1=0.6321, Normal Recall=0.2241, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4621

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
0.15       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633   <--
0.20       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.25       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.30       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.35       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.40       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.45       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.50       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.55       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.60       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.65       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.70       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.75       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
0.80       0.6124   0.7207   0.2248   1.0000   1.0000   0.5633  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6124, F1=0.7207, Normal Recall=0.2248, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5633

```

