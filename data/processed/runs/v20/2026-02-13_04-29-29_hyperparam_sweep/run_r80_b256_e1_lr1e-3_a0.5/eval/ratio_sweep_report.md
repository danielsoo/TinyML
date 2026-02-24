# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e1_lr1e-3_a0.5.yaml` |
| **Generated** | 2026-02-18 14:09:23 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9246 | 0.9223 | 0.9194 | 0.9179 | 0.9144 | 0.9119 | 0.9100 | 0.9079 | 0.9049 | 0.9026 | 0.9006 |
| QAT+Prune only | 0.6893 | 0.7194 | 0.7474 | 0.7758 | 0.8043 | 0.8324 | 0.8623 | 0.8905 | 0.9200 | 0.9469 | 0.9768 |
| QAT+PTQ | 0.6905 | 0.7202 | 0.7480 | 0.7762 | 0.8045 | 0.8324 | 0.8621 | 0.8903 | 0.9192 | 0.9462 | 0.9757 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6905 | 0.7202 | 0.7480 | 0.7762 | 0.8045 | 0.8324 | 0.8621 | 0.8903 | 0.9192 | 0.9462 | 0.9757 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6989 | 0.8172 | 0.8681 | 0.8938 | 0.9109 | 0.9231 | 0.9319 | 0.9381 | 0.9433 | 0.9477 |
| QAT+Prune only | 0.0000 | 0.4104 | 0.6074 | 0.7233 | 0.7997 | 0.8536 | 0.8949 | 0.9259 | 0.9513 | 0.9707 | 0.9883 |
| QAT+PTQ | 0.0000 | 0.4108 | 0.6077 | 0.7235 | 0.7997 | 0.8534 | 0.8946 | 0.9256 | 0.9508 | 0.9703 | 0.9877 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4108 | 0.6077 | 0.7235 | 0.7997 | 0.8534 | 0.8946 | 0.9256 | 0.9508 | 0.9703 | 0.9877 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9246 | 0.9246 | 0.9241 | 0.9253 | 0.9236 | 0.9232 | 0.9240 | 0.9251 | 0.9224 | 0.9208 | 0.0000 |
| QAT+Prune only | 0.6893 | 0.6908 | 0.6901 | 0.6897 | 0.6893 | 0.6880 | 0.6905 | 0.6893 | 0.6927 | 0.6776 | 0.0000 |
| QAT+PTQ | 0.6905 | 0.6919 | 0.6911 | 0.6907 | 0.6904 | 0.6891 | 0.6916 | 0.6909 | 0.6933 | 0.6803 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6905 | 0.6919 | 0.6911 | 0.6907 | 0.6904 | 0.6891 | 0.6916 | 0.6909 | 0.6933 | 0.6803 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9246 | 0.0000 | 0.0000 | 0.0000 | 0.9246 | 1.0000 |
| 90 | 10 | 299,940 | 0.9223 | 0.5705 | 0.9017 | 0.6989 | 0.9246 | 0.9883 |
| 80 | 20 | 291,350 | 0.9194 | 0.7479 | 0.9006 | 0.8172 | 0.9241 | 0.9738 |
| 70 | 30 | 194,230 | 0.9179 | 0.8379 | 0.9006 | 0.8681 | 0.9253 | 0.9560 |
| 60 | 40 | 145,675 | 0.9144 | 0.8872 | 0.9006 | 0.8938 | 0.9236 | 0.9330 |
| 50 | 50 | 116,540 | 0.9119 | 0.9214 | 0.9006 | 0.9109 | 0.9232 | 0.9028 |
| 40 | 60 | 97,115 | 0.9100 | 0.9468 | 0.9006 | 0.9231 | 0.9240 | 0.8610 |
| 30 | 70 | 83,240 | 0.9079 | 0.9656 | 0.9006 | 0.9319 | 0.9251 | 0.7995 |
| 20 | 80 | 72,835 | 0.9049 | 0.9789 | 0.9006 | 0.9381 | 0.9224 | 0.6988 |
| 10 | 90 | 64,740 | 0.9026 | 0.9903 | 0.9006 | 0.9433 | 0.9208 | 0.5071 |
| 0 | 100 | 58,270 | 0.9006 | 1.0000 | 0.9006 | 0.9477 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6893 | 0.0000 | 0.0000 | 0.0000 | 0.6893 | 1.0000 |
| 90 | 10 | 299,940 | 0.7194 | 0.2598 | 0.9765 | 0.4104 | 0.6908 | 0.9962 |
| 80 | 20 | 291,350 | 0.7474 | 0.4407 | 0.9768 | 0.6074 | 0.6901 | 0.9917 |
| 70 | 30 | 194,230 | 0.7758 | 0.5743 | 0.9768 | 0.7233 | 0.6897 | 0.9858 |
| 60 | 40 | 145,675 | 0.8043 | 0.6770 | 0.9768 | 0.7997 | 0.6893 | 0.9781 |
| 50 | 50 | 116,540 | 0.8324 | 0.7579 | 0.9768 | 0.8536 | 0.6880 | 0.9674 |
| 40 | 60 | 97,115 | 0.8623 | 0.8256 | 0.9768 | 0.8949 | 0.6905 | 0.9520 |
| 30 | 70 | 83,240 | 0.8905 | 0.8800 | 0.9768 | 0.9259 | 0.6893 | 0.9272 |
| 20 | 80 | 72,835 | 0.9200 | 0.9271 | 0.9768 | 0.9513 | 0.6927 | 0.8818 |
| 10 | 90 | 64,740 | 0.9469 | 0.9646 | 0.9768 | 0.9707 | 0.6776 | 0.7644 |
| 0 | 100 | 58,270 | 0.9768 | 1.0000 | 0.9768 | 0.9883 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6905 | 0.0000 | 0.0000 | 0.0000 | 0.6905 | 1.0000 |
| 90 | 10 | 299,940 | 0.7202 | 0.2602 | 0.9755 | 0.4108 | 0.6919 | 0.9961 |
| 80 | 20 | 291,350 | 0.7480 | 0.4413 | 0.9757 | 0.6077 | 0.6911 | 0.9913 |
| 70 | 30 | 194,230 | 0.7762 | 0.5748 | 0.9757 | 0.7235 | 0.6907 | 0.9851 |
| 60 | 40 | 145,675 | 0.8045 | 0.6775 | 0.9757 | 0.7997 | 0.6904 | 0.9771 |
| 50 | 50 | 116,540 | 0.8324 | 0.7584 | 0.9757 | 0.8534 | 0.6891 | 0.9659 |
| 40 | 60 | 97,115 | 0.8621 | 0.8259 | 0.9757 | 0.8946 | 0.6916 | 0.9499 |
| 30 | 70 | 83,240 | 0.8903 | 0.8805 | 0.9757 | 0.9256 | 0.6909 | 0.9242 |
| 20 | 80 | 72,835 | 0.9192 | 0.9271 | 0.9757 | 0.9508 | 0.6933 | 0.8770 |
| 10 | 90 | 64,740 | 0.9462 | 0.9649 | 0.9757 | 0.9703 | 0.6803 | 0.7567 |
| 0 | 100 | 58,270 | 0.9757 | 1.0000 | 0.9757 | 0.9877 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6905 | 0.0000 | 0.0000 | 0.0000 | 0.6905 | 1.0000 |
| 90 | 10 | 299,940 | 0.7202 | 0.2602 | 0.9755 | 0.4108 | 0.6919 | 0.9961 |
| 80 | 20 | 291,350 | 0.7480 | 0.4413 | 0.9757 | 0.6077 | 0.6911 | 0.9913 |
| 70 | 30 | 194,230 | 0.7762 | 0.5748 | 0.9757 | 0.7235 | 0.6907 | 0.9851 |
| 60 | 40 | 145,675 | 0.8045 | 0.6775 | 0.9757 | 0.7997 | 0.6904 | 0.9771 |
| 50 | 50 | 116,540 | 0.8324 | 0.7584 | 0.9757 | 0.8534 | 0.6891 | 0.9659 |
| 40 | 60 | 97,115 | 0.8621 | 0.8259 | 0.9757 | 0.8946 | 0.6916 | 0.9499 |
| 30 | 70 | 83,240 | 0.8903 | 0.8805 | 0.9757 | 0.9256 | 0.6909 | 0.9242 |
| 20 | 80 | 72,835 | 0.9192 | 0.9271 | 0.9757 | 0.9508 | 0.6933 | 0.8770 |
| 10 | 90 | 64,740 | 0.9462 | 0.9649 | 0.9757 | 0.9703 | 0.6803 | 0.7567 |
| 0 | 100 | 58,270 | 0.9757 | 1.0000 | 0.9757 | 0.9877 | 0.0000 | 0.0000 |


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
0.15       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703   <--
0.20       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.25       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.30       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.35       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.40       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.45       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.50       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.55       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.60       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.65       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.70       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.75       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
0.80       0.9222   0.6985   0.9246   0.9882   0.9009   0.5703  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9222, F1=0.6985, Normal Recall=0.9246, Normal Precision=0.9882, Attack Recall=0.9009, Attack Precision=0.5703

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
0.15       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495   <--
0.20       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.25       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.30       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.35       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.40       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.45       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.50       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.55       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.60       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.65       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.70       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.75       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
0.80       0.9199   0.8181   0.9248   0.9738   0.9006   0.7495  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9199, F1=0.8181, Normal Recall=0.9248, Normal Precision=0.9738, Attack Recall=0.9006, Attack Precision=0.7495

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
0.15       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377   <--
0.20       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.25       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.30       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.35       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.40       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.45       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.50       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.55       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.60       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.65       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.70       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.75       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
0.80       0.9178   0.8680   0.9252   0.9560   0.9006   0.8377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9178, F1=0.8680, Normal Recall=0.9252, Normal Precision=0.9560, Attack Recall=0.9006, Attack Precision=0.8377

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
0.15       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886   <--
0.20       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.25       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.30       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.35       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.40       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.45       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.50       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.55       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.60       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.65       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.70       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.75       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
0.80       0.9151   0.8946   0.9248   0.9331   0.9006   0.8886  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9151, F1=0.8946, Normal Recall=0.9248, Normal Precision=0.9331, Attack Recall=0.9006, Attack Precision=0.8886

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
0.15       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229   <--
0.20       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.25       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.30       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.35       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.40       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.45       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.50       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.55       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.60       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.65       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.70       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.75       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
0.80       0.9127   0.9116   0.9248   0.9029   0.9006   0.9229  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9127, F1=0.9116, Normal Recall=0.9248, Normal Precision=0.9029, Attack Recall=0.9006, Attack Precision=0.9229

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
0.15       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599   <--
0.20       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.25       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.30       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.35       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.40       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.45       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.50       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.55       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.60       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.65       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.70       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.75       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
0.80       0.7195   0.4106   0.6908   0.9963   0.9772   0.2599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7195, F1=0.4106, Normal Recall=0.6908, Normal Precision=0.9963, Attack Recall=0.9772, Attack Precision=0.2599

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
0.15       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414   <--
0.20       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.25       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.30       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.35       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.40       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.45       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.50       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.55       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.60       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.65       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.70       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.75       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
0.80       0.7481   0.6080   0.6910   0.9917   0.9768   0.4414  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7481, F1=0.6080, Normal Recall=0.6910, Normal Precision=0.9917, Attack Recall=0.9768, Attack Precision=0.4414

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
0.15       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744   <--
0.20       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.25       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.30       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.35       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.40       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.45       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.50       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.55       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.60       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.65       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.70       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.75       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
0.80       0.7759   0.7234   0.6898   0.9858   0.9768   0.5744  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7759, F1=0.7234, Normal Recall=0.6898, Normal Precision=0.9858, Attack Recall=0.9768, Attack Precision=0.5744

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
0.15       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768   <--
0.20       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.25       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.30       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.35       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.40       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.45       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.50       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.55       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.60       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.65       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.70       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.75       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
0.80       0.8041   0.7996   0.6890   0.9780   0.9768   0.6768  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8041, F1=0.7996, Normal Recall=0.6890, Normal Precision=0.9780, Attack Recall=0.9768, Attack Precision=0.6768

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
0.15       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574   <--
0.20       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.25       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.30       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.35       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.40       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.45       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.50       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.55       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.60       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.65       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.70       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.75       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
0.80       0.8320   0.8532   0.6871   0.9673   0.9768   0.7574  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8320, F1=0.8532, Normal Recall=0.6871, Normal Precision=0.9673, Attack Recall=0.9768, Attack Precision=0.7574

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
0.15       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604   <--
0.20       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.25       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.30       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.35       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.40       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.45       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.50       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.55       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.60       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.65       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.70       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.75       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.80       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7203, F1=0.4111, Normal Recall=0.6919, Normal Precision=0.9962, Attack Recall=0.9763, Attack Precision=0.2604

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
0.15       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420   <--
0.20       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.25       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.30       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.35       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.40       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.45       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.50       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.55       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.60       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.65       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.70       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.75       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.80       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7488, F1=0.6084, Normal Recall=0.6921, Normal Precision=0.9913, Attack Recall=0.9757, Attack Precision=0.4420

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
0.15       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750   <--
0.20       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.25       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.30       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.35       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.40       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.45       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.50       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.55       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.60       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.65       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.70       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.75       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.80       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7764, F1=0.7236, Normal Recall=0.6910, Normal Precision=0.9852, Attack Recall=0.9757, Attack Precision=0.5750

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
0.15       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773   <--
0.20       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.25       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.30       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.35       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.40       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.45       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.50       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.55       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.60       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.65       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.70       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.75       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.80       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8044, F1=0.7996, Normal Recall=0.6901, Normal Precision=0.9771, Attack Recall=0.9757, Attack Precision=0.6773

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
0.15       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580   <--
0.20       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.25       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.30       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.35       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.40       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.45       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.50       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.55       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.60       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.65       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.70       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.75       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.80       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8321, F1=0.8532, Normal Recall=0.6885, Normal Precision=0.9659, Attack Recall=0.9757, Attack Precision=0.7580

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
0.15       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604   <--
0.20       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.25       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.30       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.35       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.40       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.45       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.50       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.55       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.60       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.65       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.70       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.75       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
0.80       0.7203   0.4111   0.6919   0.9962   0.9763   0.2604  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7203, F1=0.4111, Normal Recall=0.6919, Normal Precision=0.9962, Attack Recall=0.9763, Attack Precision=0.2604

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
0.15       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420   <--
0.20       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.25       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.30       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.35       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.40       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.45       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.50       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.55       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.60       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.65       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.70       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.75       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
0.80       0.7488   0.6084   0.6921   0.9913   0.9757   0.4420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7488, F1=0.6084, Normal Recall=0.6921, Normal Precision=0.9913, Attack Recall=0.9757, Attack Precision=0.4420

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
0.15       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750   <--
0.20       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.25       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.30       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.35       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.40       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.45       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.50       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.55       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.60       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.65       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.70       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.75       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
0.80       0.7764   0.7236   0.6910   0.9852   0.9757   0.5750  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7764, F1=0.7236, Normal Recall=0.6910, Normal Precision=0.9852, Attack Recall=0.9757, Attack Precision=0.5750

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
0.15       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773   <--
0.20       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.25       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.30       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.35       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.40       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.45       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.50       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.55       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.60       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.65       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.70       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.75       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
0.80       0.8044   0.7996   0.6901   0.9771   0.9757   0.6773  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8044, F1=0.7996, Normal Recall=0.6901, Normal Precision=0.9771, Attack Recall=0.9757, Attack Precision=0.6773

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
0.15       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580   <--
0.20       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.25       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.30       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.35       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.40       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.45       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.50       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.55       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.60       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.65       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.70       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.75       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
0.80       0.8321   0.8532   0.6885   0.9659   0.9757   0.7580  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8321, F1=0.8532, Normal Recall=0.6885, Normal Precision=0.9659, Attack Recall=0.9757, Attack Precision=0.7580

```

