# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-16 01:55:13 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9421 | 0.9102 | 0.8772 | 0.8447 | 0.8120 | 0.7787 | 0.7459 | 0.7124 | 0.6808 | 0.6472 | 0.6147 |
| QAT+Prune only | 0.7313 | 0.7583 | 0.7848 | 0.8120 | 0.8384 | 0.8635 | 0.8913 | 0.9188 | 0.9450 | 0.9716 | 0.9986 |
| QAT+PTQ | 0.7305 | 0.7577 | 0.7843 | 0.8114 | 0.8381 | 0.8632 | 0.8911 | 0.9187 | 0.9449 | 0.9715 | 0.9986 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7305 | 0.7577 | 0.7843 | 0.8114 | 0.8381 | 0.8632 | 0.8911 | 0.9187 | 0.9449 | 0.9715 | 0.9986 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5779 | 0.6669 | 0.7037 | 0.7234 | 0.7353 | 0.7438 | 0.7496 | 0.7550 | 0.7582 | 0.7614 |
| QAT+Prune only | 0.0000 | 0.4524 | 0.6499 | 0.7612 | 0.8318 | 0.8797 | 0.9168 | 0.9451 | 0.9667 | 0.9844 | 0.9993 |
| QAT+PTQ | 0.0000 | 0.4519 | 0.6493 | 0.7606 | 0.8315 | 0.8795 | 0.9167 | 0.9451 | 0.9667 | 0.9844 | 0.9993 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4519 | 0.6493 | 0.7606 | 0.8315 | 0.8795 | 0.9167 | 0.9451 | 0.9667 | 0.9844 | 0.9993 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9421 | 0.9430 | 0.9428 | 0.9433 | 0.9435 | 0.9427 | 0.9426 | 0.9404 | 0.9447 | 0.9395 | 0.0000 |
| QAT+Prune only | 0.7313 | 0.7316 | 0.7314 | 0.7320 | 0.7317 | 0.7284 | 0.7304 | 0.7327 | 0.7306 | 0.7288 | 0.0000 |
| QAT+PTQ | 0.7305 | 0.7310 | 0.7307 | 0.7312 | 0.7311 | 0.7279 | 0.7300 | 0.7323 | 0.7301 | 0.7274 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7305 | 0.7310 | 0.7307 | 0.7312 | 0.7311 | 0.7279 | 0.7300 | 0.7323 | 0.7301 | 0.7274 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9421 | 0.0000 | 0.0000 | 0.0000 | 0.9421 | 1.0000 |
| 90 | 10 | 299,940 | 0.9102 | 0.5452 | 0.6148 | 0.5779 | 0.9430 | 0.9566 |
| 80 | 20 | 291,350 | 0.8772 | 0.7287 | 0.6147 | 0.6669 | 0.9428 | 0.9073 |
| 70 | 30 | 194,230 | 0.8447 | 0.8228 | 0.6147 | 0.7037 | 0.9433 | 0.8510 |
| 60 | 40 | 145,675 | 0.8120 | 0.8788 | 0.6147 | 0.7234 | 0.9435 | 0.7860 |
| 50 | 50 | 116,540 | 0.7787 | 0.9148 | 0.6147 | 0.7353 | 0.9427 | 0.7099 |
| 40 | 60 | 97,115 | 0.7459 | 0.9414 | 0.6147 | 0.7438 | 0.9426 | 0.6199 |
| 30 | 70 | 83,240 | 0.7124 | 0.9601 | 0.6147 | 0.7496 | 0.9404 | 0.5113 |
| 20 | 80 | 72,835 | 0.6808 | 0.9780 | 0.6148 | 0.7550 | 0.9447 | 0.3801 |
| 10 | 90 | 64,740 | 0.6472 | 0.9892 | 0.6147 | 0.7582 | 0.9395 | 0.2132 |
| 0 | 100 | 58,270 | 0.6147 | 1.0000 | 0.6147 | 0.7614 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7313 | 0.0000 | 0.0000 | 0.0000 | 0.7313 | 1.0000 |
| 90 | 10 | 299,940 | 0.7583 | 0.2925 | 0.9986 | 0.4524 | 0.7316 | 0.9998 |
| 80 | 20 | 291,350 | 0.7848 | 0.4817 | 0.9986 | 0.6499 | 0.7314 | 0.9995 |
| 70 | 30 | 194,230 | 0.8120 | 0.6149 | 0.9986 | 0.7612 | 0.7320 | 0.9992 |
| 60 | 40 | 145,675 | 0.8384 | 0.7127 | 0.9986 | 0.8318 | 0.7317 | 0.9987 |
| 50 | 50 | 116,540 | 0.8635 | 0.7862 | 0.9986 | 0.8797 | 0.7284 | 0.9980 |
| 40 | 60 | 97,115 | 0.8913 | 0.8475 | 0.9986 | 0.9168 | 0.7304 | 0.9971 |
| 30 | 70 | 83,240 | 0.9188 | 0.8971 | 0.9986 | 0.9451 | 0.7327 | 0.9955 |
| 20 | 80 | 72,835 | 0.9450 | 0.9368 | 0.9986 | 0.9667 | 0.7306 | 0.9923 |
| 10 | 90 | 64,740 | 0.9716 | 0.9707 | 0.9986 | 0.9844 | 0.7288 | 0.9827 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7305 | 0.0000 | 0.0000 | 0.0000 | 0.7305 | 1.0000 |
| 90 | 10 | 299,940 | 0.7577 | 0.2920 | 0.9986 | 0.4519 | 0.7310 | 0.9998 |
| 80 | 20 | 291,350 | 0.7843 | 0.4811 | 0.9986 | 0.6493 | 0.7307 | 0.9995 |
| 70 | 30 | 194,230 | 0.8114 | 0.6142 | 0.9986 | 0.7606 | 0.7312 | 0.9992 |
| 60 | 40 | 145,675 | 0.8381 | 0.7123 | 0.9986 | 0.8315 | 0.7311 | 0.9987 |
| 50 | 50 | 116,540 | 0.8632 | 0.7858 | 0.9986 | 0.8795 | 0.7279 | 0.9981 |
| 40 | 60 | 97,115 | 0.8911 | 0.8473 | 0.9986 | 0.9167 | 0.7300 | 0.9971 |
| 30 | 70 | 83,240 | 0.9187 | 0.8970 | 0.9986 | 0.9451 | 0.7323 | 0.9955 |
| 20 | 80 | 72,835 | 0.9449 | 0.9367 | 0.9986 | 0.9667 | 0.7301 | 0.9923 |
| 10 | 90 | 64,740 | 0.9715 | 0.9706 | 0.9986 | 0.9844 | 0.7274 | 0.9829 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7305 | 0.0000 | 0.0000 | 0.0000 | 0.7305 | 1.0000 |
| 90 | 10 | 299,940 | 0.7577 | 0.2920 | 0.9986 | 0.4519 | 0.7310 | 0.9998 |
| 80 | 20 | 291,350 | 0.7843 | 0.4811 | 0.9986 | 0.6493 | 0.7307 | 0.9995 |
| 70 | 30 | 194,230 | 0.8114 | 0.6142 | 0.9986 | 0.7606 | 0.7312 | 0.9992 |
| 60 | 40 | 145,675 | 0.8381 | 0.7123 | 0.9986 | 0.8315 | 0.7311 | 0.9987 |
| 50 | 50 | 116,540 | 0.8632 | 0.7858 | 0.9986 | 0.8795 | 0.7279 | 0.9981 |
| 40 | 60 | 97,115 | 0.8911 | 0.8473 | 0.9986 | 0.9167 | 0.7300 | 0.9971 |
| 30 | 70 | 83,240 | 0.9187 | 0.8970 | 0.9986 | 0.9451 | 0.7323 | 0.9955 |
| 20 | 80 | 72,835 | 0.9449 | 0.9367 | 0.9986 | 0.9667 | 0.7301 | 0.9923 |
| 10 | 90 | 64,740 | 0.9715 | 0.9706 | 0.9986 | 0.9844 | 0.7274 | 0.9829 |
| 0 | 100 | 58,270 | 0.9986 | 1.0000 | 0.9986 | 0.9993 | 0.0000 | 0.0000 |


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
0.15       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447   <--
0.20       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.25       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.30       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.35       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.40       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.45       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.50       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.55       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.60       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.65       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.70       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.75       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
0.80       0.9101   0.5772   0.9430   0.9565   0.6138   0.5447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9101, F1=0.5772, Normal Recall=0.9430, Normal Precision=0.9565, Attack Recall=0.6138, Attack Precision=0.5447

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
0.15       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297   <--
0.20       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.25       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.30       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.35       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.40       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.45       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.50       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.55       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.60       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.65       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.70       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.75       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
0.80       0.8774   0.6673   0.9431   0.9073   0.6147   0.7297  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8774, F1=0.6673, Normal Recall=0.9431, Normal Precision=0.9073, Attack Recall=0.6147, Attack Precision=0.7297

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
0.15       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198   <--
0.20       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.25       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.30       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.35       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.40       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.45       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.50       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.55       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.60       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.65       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.70       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.75       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
0.80       0.8439   0.7026   0.9421   0.8509   0.6147   0.8198  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8439, F1=0.7026, Normal Recall=0.9421, Normal Precision=0.8509, Attack Recall=0.6147, Attack Precision=0.8198

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
0.15       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764   <--
0.20       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.25       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.30       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.35       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.40       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.45       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.50       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.55       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.60       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.65       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.70       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.75       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
0.80       0.8112   0.7226   0.9422   0.7858   0.6147   0.8764  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8112, F1=0.7226, Normal Recall=0.9422, Normal Precision=0.7858, Attack Recall=0.6147, Attack Precision=0.8764

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
0.15       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144   <--
0.20       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.25       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.30       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.35       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.40       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.45       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.50       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.55       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.60       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.65       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.70       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.75       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
0.80       0.7786   0.7352   0.9424   0.7098   0.6147   0.9144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7786, F1=0.7352, Normal Recall=0.9424, Normal Precision=0.7098, Attack Recall=0.6147, Attack Precision=0.9144

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
0.15       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925   <--
0.20       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.25       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.30       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.35       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.40       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.45       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.50       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.55       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.60       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.65       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.70       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.75       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
0.80       0.7583   0.4524   0.7316   0.9998   0.9986   0.2925  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7583, F1=0.4524, Normal Recall=0.7316, Normal Precision=0.9998, Attack Recall=0.9986, Attack Precision=0.2925

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
0.15       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822   <--
0.20       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.25       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.30       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.35       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.40       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.45       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.50       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.55       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.60       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.65       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.70       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.75       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
0.80       0.7853   0.6504   0.7320   0.9995   0.9986   0.4822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7853, F1=0.6504, Normal Recall=0.7320, Normal Precision=0.9995, Attack Recall=0.9986, Attack Precision=0.4822

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
0.15       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144   <--
0.20       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.25       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.30       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.35       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.40       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.45       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.50       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.55       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.60       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.65       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.70       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.75       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
0.80       0.8115   0.7607   0.7314   0.9992   0.9986   0.6144  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8115, F1=0.7607, Normal Recall=0.7314, Normal Precision=0.9992, Attack Recall=0.9986, Attack Precision=0.6144

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
0.15       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129   <--
0.20       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.25       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.30       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.35       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.40       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.45       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.50       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.55       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.60       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.65       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.70       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.75       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
0.80       0.8385   0.8319   0.7318   0.9987   0.9986   0.7129  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8385, F1=0.8319, Normal Recall=0.7318, Normal Precision=0.9987, Attack Recall=0.9986, Attack Precision=0.7129

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
0.15       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881   <--
0.20       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.25       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.30       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.35       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.40       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.45       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.50       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.55       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.60       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.65       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.70       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.75       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
0.80       0.8651   0.8810   0.7315   0.9981   0.9986   0.7881  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8651, F1=0.8810, Normal Recall=0.7315, Normal Precision=0.9981, Attack Recall=0.9986, Attack Precision=0.7881

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
0.15       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920   <--
0.20       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.25       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.30       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.35       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.40       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.45       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.50       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.55       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.60       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.65       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.70       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.75       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.80       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.4518, Normal Recall=0.7310, Normal Precision=0.9998, Attack Recall=0.9986, Attack Precision=0.2920

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
0.15       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816   <--
0.20       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.25       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.30       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.35       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.40       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.45       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.50       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.55       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.60       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.65       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.70       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.75       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.80       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7848, F1=0.6498, Normal Recall=0.7313, Normal Precision=0.9995, Attack Recall=0.9986, Attack Precision=0.4816

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
0.15       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138   <--
0.20       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.25       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.30       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.35       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.40       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.45       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.50       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.55       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.60       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.65       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.70       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.75       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.80       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8111, F1=0.7603, Normal Recall=0.7307, Normal Precision=0.9992, Attack Recall=0.9986, Attack Precision=0.6138

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
0.15       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122   <--
0.20       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.25       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.30       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.35       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.40       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.45       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.50       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.55       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.60       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.65       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.70       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.75       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.80       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.8314, Normal Recall=0.7310, Normal Precision=0.9987, Attack Recall=0.9986, Attack Precision=0.7122

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
0.15       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876   <--
0.20       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.25       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.30       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.35       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.40       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.45       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.50       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.55       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.60       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.65       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.70       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.75       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.80       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8646, F1=0.8806, Normal Recall=0.7307, Normal Precision=0.9981, Attack Recall=0.9986, Attack Precision=0.7876

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
0.15       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920   <--
0.20       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.25       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.30       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.35       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.40       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.45       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.50       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.55       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.60       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.65       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.70       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.75       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
0.80       0.7577   0.4518   0.7310   0.9998   0.9986   0.2920  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7577, F1=0.4518, Normal Recall=0.7310, Normal Precision=0.9998, Attack Recall=0.9986, Attack Precision=0.2920

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
0.15       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816   <--
0.20       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.25       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.30       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.35       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.40       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.45       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.50       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.55       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.60       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.65       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.70       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.75       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
0.80       0.7848   0.6498   0.7313   0.9995   0.9986   0.4816  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7848, F1=0.6498, Normal Recall=0.7313, Normal Precision=0.9995, Attack Recall=0.9986, Attack Precision=0.4816

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
0.15       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138   <--
0.20       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.25       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.30       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.35       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.40       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.45       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.50       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.55       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.60       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.65       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.70       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.75       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
0.80       0.8111   0.7603   0.7307   0.9992   0.9986   0.6138  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8111, F1=0.7603, Normal Recall=0.7307, Normal Precision=0.9992, Attack Recall=0.9986, Attack Precision=0.6138

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
0.15       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122   <--
0.20       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.25       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.30       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.35       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.40       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.45       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.50       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.55       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.60       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.65       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.70       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.75       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
0.80       0.8380   0.8314   0.7310   0.9987   0.9986   0.7122  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8380, F1=0.8314, Normal Recall=0.7310, Normal Precision=0.9987, Attack Recall=0.9986, Attack Precision=0.7122

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
0.15       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876   <--
0.20       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.25       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.30       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.35       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.40       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.45       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.50       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.55       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.60       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.65       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.70       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.75       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
0.80       0.8646   0.8806   0.7307   0.9981   0.9986   0.7876  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8646, F1=0.8806, Normal Recall=0.7307, Normal Precision=0.9981, Attack Recall=0.9986, Attack Precision=0.7876

```

