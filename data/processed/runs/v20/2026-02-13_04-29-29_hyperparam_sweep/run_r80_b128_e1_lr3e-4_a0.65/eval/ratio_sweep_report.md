# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-17 06:40:22 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8643 | 0.8781 | 0.8905 | 0.9045 | 0.9172 | 0.9290 | 0.9425 | 0.9557 | 0.9680 | 0.9812 | 0.9943 |
| QAT+Prune only | 0.8644 | 0.8523 | 0.8389 | 0.8267 | 0.8137 | 0.7989 | 0.7875 | 0.7744 | 0.7611 | 0.7471 | 0.7348 |
| QAT+PTQ | 0.8645 | 0.8525 | 0.8390 | 0.8268 | 0.8138 | 0.7990 | 0.7876 | 0.7745 | 0.7611 | 0.7472 | 0.7348 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8645 | 0.8525 | 0.8390 | 0.8268 | 0.8138 | 0.7990 | 0.7876 | 0.7745 | 0.7611 | 0.7472 | 0.7348 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6198 | 0.7842 | 0.8620 | 0.9057 | 0.9333 | 0.9540 | 0.9692 | 0.9803 | 0.9896 | 0.9972 |
| QAT+Prune only | 0.0000 | 0.4987 | 0.6459 | 0.7178 | 0.7594 | 0.7852 | 0.8058 | 0.8202 | 0.8311 | 0.8395 | 0.8471 |
| QAT+PTQ | 0.0000 | 0.4990 | 0.6461 | 0.7180 | 0.7595 | 0.7852 | 0.8059 | 0.8202 | 0.8311 | 0.8395 | 0.8471 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4990 | 0.6461 | 0.7180 | 0.7595 | 0.7852 | 0.8059 | 0.8202 | 0.8311 | 0.8395 | 0.8471 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8643 | 0.8652 | 0.8646 | 0.8660 | 0.8657 | 0.8636 | 0.8648 | 0.8656 | 0.8628 | 0.8630 | 0.0000 |
| QAT+Prune only | 0.8644 | 0.8654 | 0.8649 | 0.8660 | 0.8663 | 0.8630 | 0.8666 | 0.8669 | 0.8663 | 0.8576 | 0.0000 |
| QAT+PTQ | 0.8645 | 0.8656 | 0.8651 | 0.8663 | 0.8666 | 0.8633 | 0.8669 | 0.8673 | 0.8665 | 0.8587 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8645 | 0.8656 | 0.8651 | 0.8663 | 0.8666 | 0.8633 | 0.8669 | 0.8673 | 0.8665 | 0.8587 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8643 | 0.0000 | 0.0000 | 0.0000 | 0.8643 | 1.0000 |
| 90 | 10 | 299,940 | 0.8781 | 0.4504 | 0.9938 | 0.6198 | 0.8652 | 0.9992 |
| 80 | 20 | 291,350 | 0.8905 | 0.6474 | 0.9943 | 0.7842 | 0.8646 | 0.9984 |
| 70 | 30 | 194,230 | 0.9045 | 0.7607 | 0.9943 | 0.8620 | 0.8660 | 0.9972 |
| 60 | 40 | 145,675 | 0.9172 | 0.8316 | 0.9943 | 0.9057 | 0.8657 | 0.9956 |
| 50 | 50 | 116,540 | 0.9290 | 0.8794 | 0.9943 | 0.9333 | 0.8636 | 0.9935 |
| 40 | 60 | 97,115 | 0.9425 | 0.9169 | 0.9943 | 0.9540 | 0.8648 | 0.9902 |
| 30 | 70 | 83,240 | 0.9557 | 0.9452 | 0.9943 | 0.9692 | 0.8656 | 0.9849 |
| 20 | 80 | 72,835 | 0.9680 | 0.9666 | 0.9943 | 0.9803 | 0.8628 | 0.9743 |
| 10 | 90 | 64,740 | 0.9812 | 0.9849 | 0.9943 | 0.9896 | 0.8630 | 0.9441 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9972 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8644 | 0.0000 | 0.0000 | 0.0000 | 0.8644 | 1.0000 |
| 90 | 10 | 299,940 | 0.8523 | 0.3775 | 0.7346 | 0.4987 | 0.8654 | 0.9671 |
| 80 | 20 | 291,350 | 0.8389 | 0.5762 | 0.7348 | 0.6459 | 0.8649 | 0.9288 |
| 70 | 30 | 194,230 | 0.8267 | 0.7016 | 0.7348 | 0.7178 | 0.8660 | 0.8840 |
| 60 | 40 | 145,675 | 0.8137 | 0.7856 | 0.7348 | 0.7594 | 0.8663 | 0.8305 |
| 50 | 50 | 116,540 | 0.7989 | 0.8429 | 0.7348 | 0.7852 | 0.8630 | 0.7650 |
| 40 | 60 | 97,115 | 0.7875 | 0.8920 | 0.7348 | 0.8058 | 0.8666 | 0.6854 |
| 30 | 70 | 83,240 | 0.7744 | 0.9280 | 0.7348 | 0.8202 | 0.8669 | 0.5835 |
| 20 | 80 | 72,835 | 0.7611 | 0.9565 | 0.7348 | 0.8311 | 0.8663 | 0.4496 |
| 10 | 90 | 64,740 | 0.7471 | 0.9789 | 0.7348 | 0.8395 | 0.8576 | 0.2644 |
| 0 | 100 | 58,270 | 0.7348 | 1.0000 | 0.7348 | 0.8471 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8645 | 0.0000 | 0.0000 | 0.0000 | 0.8645 | 1.0000 |
| 90 | 10 | 299,940 | 0.8525 | 0.3779 | 0.7345 | 0.4990 | 0.8656 | 0.9670 |
| 80 | 20 | 291,350 | 0.8390 | 0.5765 | 0.7348 | 0.6461 | 0.8651 | 0.9288 |
| 70 | 30 | 194,230 | 0.8268 | 0.7019 | 0.7348 | 0.7180 | 0.8663 | 0.8840 |
| 60 | 40 | 145,675 | 0.8138 | 0.7859 | 0.7348 | 0.7595 | 0.8666 | 0.8305 |
| 50 | 50 | 116,540 | 0.7990 | 0.8431 | 0.7348 | 0.7852 | 0.8633 | 0.7650 |
| 40 | 60 | 97,115 | 0.7876 | 0.8922 | 0.7348 | 0.8059 | 0.8669 | 0.6854 |
| 30 | 70 | 83,240 | 0.7745 | 0.9281 | 0.7348 | 0.8202 | 0.8673 | 0.5836 |
| 20 | 80 | 72,835 | 0.7611 | 0.9566 | 0.7348 | 0.8311 | 0.8665 | 0.4496 |
| 10 | 90 | 64,740 | 0.7472 | 0.9791 | 0.7348 | 0.8395 | 0.8587 | 0.2646 |
| 0 | 100 | 58,270 | 0.7348 | 1.0000 | 0.7348 | 0.8471 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8645 | 0.0000 | 0.0000 | 0.0000 | 0.8645 | 1.0000 |
| 90 | 10 | 299,940 | 0.8525 | 0.3779 | 0.7345 | 0.4990 | 0.8656 | 0.9670 |
| 80 | 20 | 291,350 | 0.8390 | 0.5765 | 0.7348 | 0.6461 | 0.8651 | 0.9288 |
| 70 | 30 | 194,230 | 0.8268 | 0.7019 | 0.7348 | 0.7180 | 0.8663 | 0.8840 |
| 60 | 40 | 145,675 | 0.8138 | 0.7859 | 0.7348 | 0.7595 | 0.8666 | 0.8305 |
| 50 | 50 | 116,540 | 0.7990 | 0.8431 | 0.7348 | 0.7852 | 0.8633 | 0.7650 |
| 40 | 60 | 97,115 | 0.7876 | 0.8922 | 0.7348 | 0.8059 | 0.8669 | 0.6854 |
| 30 | 70 | 83,240 | 0.7745 | 0.9281 | 0.7348 | 0.8202 | 0.8673 | 0.5836 |
| 20 | 80 | 72,835 | 0.7611 | 0.9566 | 0.7348 | 0.8311 | 0.8665 | 0.4496 |
| 10 | 90 | 64,740 | 0.7472 | 0.9791 | 0.7348 | 0.8395 | 0.8587 | 0.2646 |
| 0 | 100 | 58,270 | 0.7348 | 1.0000 | 0.7348 | 0.8471 | 0.0000 | 0.0000 |


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
0.15       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506   <--
0.20       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.25       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.30       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.35       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.40       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.45       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.50       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.55       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.60       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.65       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.70       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.75       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
0.80       0.8782   0.6202   0.8652   0.9993   0.9947   0.4506  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8782, F1=0.6202, Normal Recall=0.8652, Normal Precision=0.9993, Attack Recall=0.9947, Attack Precision=0.4506

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
0.15       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491   <--
0.20       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.25       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.30       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.35       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.40       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.45       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.50       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.55       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.60       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.65       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.70       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.75       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
0.80       0.8914   0.7855   0.8656   0.9984   0.9943   0.6491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8914, F1=0.7855, Normal Recall=0.8656, Normal Precision=0.9984, Attack Recall=0.9943, Attack Precision=0.6491

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
0.15       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596   <--
0.20       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.25       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.30       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.35       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.40       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.45       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.50       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.55       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.60       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.65       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.70       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.75       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
0.80       0.9039   0.8613   0.8651   0.9972   0.9943   0.7596  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9039, F1=0.8613, Normal Recall=0.8651, Normal Precision=0.9972, Attack Recall=0.9943, Attack Precision=0.7596

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
0.15       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305   <--
0.20       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.25       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.30       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.35       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.40       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.45       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.50       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.55       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.60       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.65       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.70       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.75       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
0.80       0.9166   0.9051   0.8648   0.9956   0.9943   0.8305  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9166, F1=0.9051, Normal Recall=0.8648, Normal Precision=0.9956, Attack Recall=0.9943, Attack Precision=0.8305

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
0.15       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797   <--
0.20       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.25       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.30       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.35       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.40       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.45       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.50       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.55       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.60       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.65       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.70       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.75       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
0.80       0.9292   0.9335   0.8640   0.9935   0.9943   0.8797  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9292, F1=0.9335, Normal Recall=0.8640, Normal Precision=0.9935, Attack Recall=0.9943, Attack Precision=0.8797

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
0.15       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772   <--
0.20       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.25       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.30       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.35       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.40       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.45       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.50       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.55       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.60       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.65       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.70       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.75       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
0.80       0.8522   0.4983   0.8654   0.9670   0.7338   0.3772  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8522, F1=0.4983, Normal Recall=0.8654, Normal Precision=0.9670, Attack Recall=0.7338, Attack Precision=0.3772

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
0.15       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772   <--
0.20       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.25       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.30       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.35       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.40       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.45       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.50       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.55       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.60       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.65       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.70       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.75       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
0.80       0.8393   0.6465   0.8654   0.9288   0.7348   0.5772  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8393, F1=0.6465, Normal Recall=0.8654, Normal Precision=0.9288, Attack Recall=0.7348, Attack Precision=0.5772

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
0.15       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000   <--
0.20       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.25       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.30       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.35       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.40       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.45       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.50       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.55       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.60       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.65       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.70       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.75       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
0.80       0.8260   0.7170   0.8651   0.8839   0.7348   0.7000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8260, F1=0.7170, Normal Recall=0.8651, Normal Precision=0.8839, Attack Recall=0.7348, Attack Precision=0.7000

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
0.15       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840   <--
0.20       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.25       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.30       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.35       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.40       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.45       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.50       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.55       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.60       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.65       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.70       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.75       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
0.80       0.8129   0.7586   0.8650   0.8303   0.7348   0.7840  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8129, F1=0.7586, Normal Recall=0.8650, Normal Precision=0.8303, Attack Recall=0.7348, Attack Precision=0.7840

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
0.15       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447   <--
0.20       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.25       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.30       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.35       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.40       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.45       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.50       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.55       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.60       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.65       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.70       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.75       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
0.80       0.7999   0.7859   0.8649   0.7653   0.7348   0.8447  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7999, F1=0.7859, Normal Recall=0.8649, Normal Precision=0.7653, Attack Recall=0.7348, Attack Precision=0.8447

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
0.15       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776   <--
0.20       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.25       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.30       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.35       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.40       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.45       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.50       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.55       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.60       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.65       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.70       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.75       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.80       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8524, F1=0.4986, Normal Recall=0.8656, Normal Precision=0.9669, Attack Recall=0.7337, Attack Precision=0.3776

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
0.15       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776   <--
0.20       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.25       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.30       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.35       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.40       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.45       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.50       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.55       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.60       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.65       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.70       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.75       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.80       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8395, F1=0.6468, Normal Recall=0.8657, Normal Precision=0.9289, Attack Recall=0.7348, Attack Precision=0.5776

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
0.15       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002   <--
0.20       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.25       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.30       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.35       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.40       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.45       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.50       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.55       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.60       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.65       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.70       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.75       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.80       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8260, F1=0.7171, Normal Recall=0.8652, Normal Precision=0.8839, Attack Recall=0.7348, Attack Precision=0.7002

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
0.15       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843   <--
0.20       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.25       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.30       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.35       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.40       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.45       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.50       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.55       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.60       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.65       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.70       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.75       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.80       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8131, F1=0.7587, Normal Recall=0.8653, Normal Precision=0.8303, Attack Recall=0.7348, Attack Precision=0.7843

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
0.15       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449   <--
0.20       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.25       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.30       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.35       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.40       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.45       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.50       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.55       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.60       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.65       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.70       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.75       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.80       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7999, F1=0.7860, Normal Recall=0.8651, Normal Precision=0.7654, Attack Recall=0.7348, Attack Precision=0.8449

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
0.15       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776   <--
0.20       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.25       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.30       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.35       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.40       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.45       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.50       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.55       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.60       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.65       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.70       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.75       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
0.80       0.8524   0.4986   0.8656   0.9669   0.7337   0.3776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8524, F1=0.4986, Normal Recall=0.8656, Normal Precision=0.9669, Attack Recall=0.7337, Attack Precision=0.3776

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
0.15       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776   <--
0.20       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.25       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.30       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.35       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.40       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.45       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.50       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.55       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.60       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.65       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.70       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.75       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
0.80       0.8395   0.6468   0.8657   0.9289   0.7348   0.5776  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8395, F1=0.6468, Normal Recall=0.8657, Normal Precision=0.9289, Attack Recall=0.7348, Attack Precision=0.5776

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
0.15       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002   <--
0.20       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.25       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.30       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.35       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.40       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.45       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.50       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.55       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.60       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.65       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.70       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.75       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
0.80       0.8260   0.7171   0.8652   0.8839   0.7348   0.7002  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8260, F1=0.7171, Normal Recall=0.8652, Normal Precision=0.8839, Attack Recall=0.7348, Attack Precision=0.7002

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
0.15       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843   <--
0.20       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.25       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.30       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.35       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.40       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.45       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.50       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.55       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.60       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.65       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.70       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.75       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
0.80       0.8131   0.7587   0.8653   0.8303   0.7348   0.7843  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8131, F1=0.7587, Normal Recall=0.8653, Normal Precision=0.8303, Attack Recall=0.7348, Attack Precision=0.7843

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
0.15       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449   <--
0.20       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.25       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.30       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.35       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.40       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.45       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.50       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.55       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.60       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.65       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.70       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.75       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
0.80       0.7999   0.7860   0.8651   0.7654   0.7348   0.8449  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7999, F1=0.7860, Normal Recall=0.8651, Normal Precision=0.7654, Attack Recall=0.7348, Attack Precision=0.8449

```

