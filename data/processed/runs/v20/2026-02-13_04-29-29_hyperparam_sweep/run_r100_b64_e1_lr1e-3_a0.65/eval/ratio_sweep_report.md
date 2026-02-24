# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-19 18:18:43 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6555 | 0.6500 | 0.6438 | 0.6391 | 0.6335 | 0.6288 | 0.6226 | 0.6188 | 0.6123 | 0.6069 | 0.6015 |
| QAT+Prune only | 0.6905 | 0.7216 | 0.7518 | 0.7827 | 0.8131 | 0.8432 | 0.8742 | 0.9053 | 0.9367 | 0.9663 | 0.9979 |
| QAT+PTQ | 0.6900 | 0.7214 | 0.7516 | 0.7826 | 0.8131 | 0.8431 | 0.8742 | 0.9052 | 0.9367 | 0.9665 | 0.9981 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6900 | 0.7214 | 0.7516 | 0.7826 | 0.8131 | 0.8431 | 0.8742 | 0.9052 | 0.9367 | 0.9665 | 0.9981 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2566 | 0.4032 | 0.5001 | 0.5677 | 0.6184 | 0.6567 | 0.6884 | 0.7129 | 0.7336 | 0.7512 |
| QAT+Prune only | 0.0000 | 0.4175 | 0.6166 | 0.7338 | 0.8103 | 0.8642 | 0.9049 | 0.9365 | 0.9619 | 0.9816 | 0.9989 |
| QAT+PTQ | 0.0000 | 0.4174 | 0.6164 | 0.7337 | 0.8103 | 0.8641 | 0.9050 | 0.9365 | 0.9619 | 0.9817 | 0.9990 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4174 | 0.6164 | 0.7337 | 0.8103 | 0.8641 | 0.9050 | 0.9365 | 0.9619 | 0.9817 | 0.9990 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6555 | 0.6551 | 0.6544 | 0.6553 | 0.6548 | 0.6562 | 0.6543 | 0.6593 | 0.6554 | 0.6548 | 0.0000 |
| QAT+Prune only | 0.6905 | 0.6909 | 0.6903 | 0.6906 | 0.6899 | 0.6885 | 0.6887 | 0.6892 | 0.6920 | 0.6821 | 0.0000 |
| QAT+PTQ | 0.6900 | 0.6906 | 0.6900 | 0.6903 | 0.6898 | 0.6881 | 0.6885 | 0.6886 | 0.6914 | 0.6821 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6900 | 0.6906 | 0.6900 | 0.6903 | 0.6898 | 0.6881 | 0.6885 | 0.6886 | 0.6914 | 0.6821 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6555 | 0.0000 | 0.0000 | 0.0000 | 0.6555 | 1.0000 |
| 90 | 10 | 299,940 | 0.6500 | 0.1629 | 0.6041 | 0.2566 | 0.6551 | 0.9371 |
| 80 | 20 | 291,350 | 0.6438 | 0.3032 | 0.6015 | 0.4032 | 0.6544 | 0.8679 |
| 70 | 30 | 194,230 | 0.6391 | 0.4279 | 0.6016 | 0.5001 | 0.6553 | 0.7933 |
| 60 | 40 | 145,675 | 0.6335 | 0.5374 | 0.6015 | 0.5677 | 0.6548 | 0.7114 |
| 50 | 50 | 116,540 | 0.6288 | 0.6363 | 0.6015 | 0.6184 | 0.6562 | 0.6222 |
| 40 | 60 | 97,115 | 0.6226 | 0.7230 | 0.6015 | 0.6567 | 0.6543 | 0.5226 |
| 30 | 70 | 83,240 | 0.6188 | 0.8047 | 0.6015 | 0.6884 | 0.6593 | 0.4149 |
| 20 | 80 | 72,835 | 0.6123 | 0.8747 | 0.6015 | 0.7129 | 0.6554 | 0.2914 |
| 10 | 90 | 64,740 | 0.6069 | 0.9401 | 0.6016 | 0.7336 | 0.6548 | 0.1544 |
| 0 | 100 | 58,270 | 0.6015 | 1.0000 | 0.6015 | 0.7512 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6905 | 0.0000 | 0.0000 | 0.0000 | 0.6905 | 1.0000 |
| 90 | 10 | 299,940 | 0.7216 | 0.2640 | 0.9977 | 0.4175 | 0.6909 | 0.9996 |
| 80 | 20 | 291,350 | 0.7518 | 0.4461 | 0.9979 | 0.6166 | 0.6903 | 0.9992 |
| 70 | 30 | 194,230 | 0.7827 | 0.5802 | 0.9979 | 0.7338 | 0.6906 | 0.9987 |
| 60 | 40 | 145,675 | 0.8131 | 0.6821 | 0.9979 | 0.8103 | 0.6899 | 0.9979 |
| 50 | 50 | 116,540 | 0.8432 | 0.7621 | 0.9979 | 0.8642 | 0.6885 | 0.9969 |
| 40 | 60 | 97,115 | 0.8742 | 0.8279 | 0.9979 | 0.9049 | 0.6887 | 0.9954 |
| 30 | 70 | 83,240 | 0.9053 | 0.8822 | 0.9979 | 0.9365 | 0.6892 | 0.9928 |
| 20 | 80 | 72,835 | 0.9367 | 0.9284 | 0.9979 | 0.9619 | 0.6920 | 0.9878 |
| 10 | 90 | 64,740 | 0.9663 | 0.9658 | 0.9979 | 0.9816 | 0.6821 | 0.9727 |
| 0 | 100 | 58,270 | 0.9979 | 1.0000 | 0.9979 | 0.9989 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6900 | 0.0000 | 0.0000 | 0.0000 | 0.6900 | 1.0000 |
| 90 | 10 | 299,940 | 0.7214 | 0.2639 | 0.9980 | 0.4174 | 0.6906 | 0.9997 |
| 80 | 20 | 291,350 | 0.7516 | 0.4459 | 0.9981 | 0.6164 | 0.6900 | 0.9993 |
| 70 | 30 | 194,230 | 0.7826 | 0.5800 | 0.9981 | 0.7337 | 0.6903 | 0.9988 |
| 60 | 40 | 145,675 | 0.8131 | 0.6820 | 0.9981 | 0.8103 | 0.6898 | 0.9981 |
| 50 | 50 | 116,540 | 0.8431 | 0.7619 | 0.9981 | 0.8641 | 0.6881 | 0.9972 |
| 40 | 60 | 97,115 | 0.8742 | 0.8277 | 0.9981 | 0.9050 | 0.6885 | 0.9958 |
| 30 | 70 | 83,240 | 0.9052 | 0.8820 | 0.9981 | 0.9365 | 0.6886 | 0.9935 |
| 20 | 80 | 72,835 | 0.9367 | 0.9283 | 0.9981 | 0.9619 | 0.6914 | 0.9889 |
| 10 | 90 | 64,740 | 0.9665 | 0.9658 | 0.9981 | 0.9817 | 0.6821 | 0.9750 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6900 | 0.0000 | 0.0000 | 0.0000 | 0.6900 | 1.0000 |
| 90 | 10 | 299,940 | 0.7214 | 0.2639 | 0.9980 | 0.4174 | 0.6906 | 0.9997 |
| 80 | 20 | 291,350 | 0.7516 | 0.4459 | 0.9981 | 0.6164 | 0.6900 | 0.9993 |
| 70 | 30 | 194,230 | 0.7826 | 0.5800 | 0.9981 | 0.7337 | 0.6903 | 0.9988 |
| 60 | 40 | 145,675 | 0.8131 | 0.6820 | 0.9981 | 0.8103 | 0.6898 | 0.9981 |
| 50 | 50 | 116,540 | 0.8431 | 0.7619 | 0.9981 | 0.8641 | 0.6881 | 0.9972 |
| 40 | 60 | 97,115 | 0.8742 | 0.8277 | 0.9981 | 0.9050 | 0.6885 | 0.9958 |
| 30 | 70 | 83,240 | 0.9052 | 0.8820 | 0.9981 | 0.9365 | 0.6886 | 0.9935 |
| 20 | 80 | 72,835 | 0.9367 | 0.9283 | 0.9981 | 0.9619 | 0.6914 | 0.9889 |
| 10 | 90 | 64,740 | 0.9665 | 0.9658 | 0.9981 | 0.9817 | 0.6821 | 0.9750 |
| 0 | 100 | 58,270 | 0.9981 | 1.0000 | 0.9981 | 0.9990 | 0.0000 | 0.0000 |


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
0.15       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627   <--
0.20       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.25       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.30       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.35       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.40       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.45       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.50       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.55       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.60       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.65       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.70       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.75       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
0.80       0.6499   0.2562   0.6551   0.9369   0.6031   0.1627  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6499, F1=0.2562, Normal Recall=0.6551, Normal Precision=0.9369, Attack Recall=0.6031, Attack Precision=0.1627

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
0.15       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038   <--
0.20       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.25       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.30       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.35       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.40       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.45       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.50       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.55       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.60       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.65       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.70       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.75       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
0.80       0.6447   0.4037   0.6554   0.8681   0.6015   0.3038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6447, F1=0.4037, Normal Recall=0.6554, Normal Precision=0.8681, Attack Recall=0.6015, Attack Precision=0.3038

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
0.15       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288   <--
0.20       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.25       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.30       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.35       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.40       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.45       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.50       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.55       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.60       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.65       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.70       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.75       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
0.80       0.6400   0.5007   0.6565   0.7936   0.6015   0.4288  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6400, F1=0.5007, Normal Recall=0.6565, Normal Precision=0.7936, Attack Recall=0.6015, Attack Precision=0.4288

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
0.15       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383   <--
0.20       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.25       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.30       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.35       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.40       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.45       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.50       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.55       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.60       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.65       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.70       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.75       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
0.80       0.6342   0.5682   0.6560   0.7118   0.6015   0.5383  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6342, F1=0.5682, Normal Recall=0.6560, Normal Precision=0.7118, Attack Recall=0.6015, Attack Precision=0.5383

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
0.15       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360   <--
0.20       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.25       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.30       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.35       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.40       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.45       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.50       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.55       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.60       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.65       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.70       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.75       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
0.80       0.6287   0.6183   0.6558   0.6220   0.6015   0.6360  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6287, F1=0.6183, Normal Recall=0.6558, Normal Precision=0.6220, Attack Recall=0.6015, Attack Precision=0.6360

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
0.15       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641   <--
0.20       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.25       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.30       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.35       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.40       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.45       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.50       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.55       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.60       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.65       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.70       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.75       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
0.80       0.7216   0.4176   0.6909   0.9997   0.9981   0.2641  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7216, F1=0.4176, Normal Recall=0.6909, Normal Precision=0.9997, Attack Recall=0.9981, Attack Precision=0.2641

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
0.15       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472   <--
0.20       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.25       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.30       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.35       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.40       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.45       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.50       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.55       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.60       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.65       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.70       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.75       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
0.80       0.7528   0.6176   0.6916   0.9992   0.9979   0.4472  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7528, F1=0.6176, Normal Recall=0.6916, Normal Precision=0.9992, Attack Recall=0.9979, Attack Precision=0.4472

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
0.15       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807   <--
0.20       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.25       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.30       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.35       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.40       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.45       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.50       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.55       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.60       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.65       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.70       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.75       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
0.80       0.7832   0.7341   0.6912   0.9987   0.9979   0.5807  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7832, F1=0.7341, Normal Recall=0.6912, Normal Precision=0.9987, Attack Recall=0.9979, Attack Precision=0.5807

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
0.15       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823   <--
0.20       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.25       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.30       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.35       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.40       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.45       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.50       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.55       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.60       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.65       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.70       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.75       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
0.80       0.8133   0.8104   0.6902   0.9979   0.9979   0.6823  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8133, F1=0.8104, Normal Recall=0.6902, Normal Precision=0.9979, Attack Recall=0.9979, Attack Precision=0.6823

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
0.15       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621   <--
0.20       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.25       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.30       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.35       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.40       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.45       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.50       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.55       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.60       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.65       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.70       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.75       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
0.80       0.8432   0.8642   0.6885   0.9969   0.9979   0.7621  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8432, F1=0.8642, Normal Recall=0.6885, Normal Precision=0.9969, Attack Recall=0.9979, Attack Precision=0.7621

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
0.15       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639   <--
0.20       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.25       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.30       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.35       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.40       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.45       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.50       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.55       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.60       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.65       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.70       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.75       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.80       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7214, F1=0.4174, Normal Recall=0.6906, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2639

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
0.15       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469   <--
0.20       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.25       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.30       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.35       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.40       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.45       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.50       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.55       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.60       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.65       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.70       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.75       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.80       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7526, F1=0.6174, Normal Recall=0.6912, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4469

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
0.15       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805   <--
0.20       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.25       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.30       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.35       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.40       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.45       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.50       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.55       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.60       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.65       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.70       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.75       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.80       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7830, F1=0.7340, Normal Recall=0.6909, Normal Precision=0.9988, Attack Recall=0.9981, Attack Precision=0.5805

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
0.15       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821   <--
0.20       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.25       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.30       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.35       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.40       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.45       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.50       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.55       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.60       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.65       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.70       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.75       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.80       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8132, F1=0.8104, Normal Recall=0.6899, Normal Precision=0.9981, Attack Recall=0.9981, Attack Precision=0.6821

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
0.15       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620   <--
0.20       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.25       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.30       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.35       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.40       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.45       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.50       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.55       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.60       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.65       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.70       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.75       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.80       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8431, F1=0.8642, Normal Recall=0.6882, Normal Precision=0.9972, Attack Recall=0.9981, Attack Precision=0.7620

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
0.15       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639   <--
0.20       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.25       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.30       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.35       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.40       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.45       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.50       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.55       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.60       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.65       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.70       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.75       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
0.80       0.7214   0.4174   0.6906   0.9997   0.9982   0.2639  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7214, F1=0.4174, Normal Recall=0.6906, Normal Precision=0.9997, Attack Recall=0.9982, Attack Precision=0.2639

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
0.15       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469   <--
0.20       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.25       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.30       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.35       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.40       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.45       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.50       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.55       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.60       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.65       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.70       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.75       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
0.80       0.7526   0.6174   0.6912   0.9993   0.9981   0.4469  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7526, F1=0.6174, Normal Recall=0.6912, Normal Precision=0.9993, Attack Recall=0.9981, Attack Precision=0.4469

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
0.15       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805   <--
0.20       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.25       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.30       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.35       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.40       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.45       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.50       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.55       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.60       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.65       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.70       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.75       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
0.80       0.7830   0.7340   0.6909   0.9988   0.9981   0.5805  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7830, F1=0.7340, Normal Recall=0.6909, Normal Precision=0.9988, Attack Recall=0.9981, Attack Precision=0.5805

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
0.15       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821   <--
0.20       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.25       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.30       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.35       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.40       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.45       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.50       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.55       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.60       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.65       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.70       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.75       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
0.80       0.8132   0.8104   0.6899   0.9981   0.9981   0.6821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8132, F1=0.8104, Normal Recall=0.6899, Normal Precision=0.9981, Attack Recall=0.9981, Attack Precision=0.6821

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
0.15       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620   <--
0.20       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.25       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.30       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.35       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.40       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.45       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.50       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.55       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.60       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.65       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.70       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.75       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
0.80       0.8431   0.8642   0.6882   0.9972   0.9981   0.7620  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8431, F1=0.8642, Normal Recall=0.6882, Normal Precision=0.9972, Attack Recall=0.9981, Attack Precision=0.7620

```

