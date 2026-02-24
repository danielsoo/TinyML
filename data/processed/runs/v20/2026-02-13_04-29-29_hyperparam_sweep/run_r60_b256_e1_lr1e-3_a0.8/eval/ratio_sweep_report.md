# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-15 12:11:12 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0979 | 0.1880 | 0.2779 | 0.3678 | 0.4579 | 0.5480 | 0.6376 | 0.7273 | 0.8172 | 0.9073 | 0.9970 |
| QAT+Prune only | 0.7647 | 0.7884 | 0.8109 | 0.8344 | 0.8567 | 0.8785 | 0.9038 | 0.9252 | 0.9491 | 0.9708 | 0.9943 |
| QAT+PTQ | 0.7638 | 0.7876 | 0.8102 | 0.8337 | 0.8562 | 0.8782 | 0.9034 | 0.9251 | 0.9489 | 0.9706 | 0.9943 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7638 | 0.7876 | 0.8102 | 0.8337 | 0.8562 | 0.8782 | 0.9034 | 0.9251 | 0.9489 | 0.9706 | 0.9943 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1972 | 0.3558 | 0.4862 | 0.5954 | 0.6881 | 0.7675 | 0.8365 | 0.8972 | 0.9509 | 0.9985 |
| QAT+Prune only | 0.0000 | 0.4843 | 0.6777 | 0.7827 | 0.8473 | 0.8911 | 0.9254 | 0.9490 | 0.9690 | 0.9839 | 0.9972 |
| QAT+PTQ | 0.0000 | 0.4835 | 0.6769 | 0.7820 | 0.8469 | 0.8909 | 0.9251 | 0.9489 | 0.9689 | 0.9839 | 0.9971 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4835 | 0.6769 | 0.7820 | 0.8469 | 0.8909 | 0.9251 | 0.9489 | 0.9689 | 0.9839 | 0.9971 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0979 | 0.0982 | 0.0981 | 0.0982 | 0.0985 | 0.0991 | 0.0984 | 0.0979 | 0.0983 | 0.1001 | 0.0000 |
| QAT+Prune only | 0.7647 | 0.7655 | 0.7650 | 0.7659 | 0.7649 | 0.7626 | 0.7679 | 0.7640 | 0.7684 | 0.7587 | 0.0000 |
| QAT+PTQ | 0.7638 | 0.7647 | 0.7642 | 0.7649 | 0.7641 | 0.7621 | 0.7672 | 0.7636 | 0.7674 | 0.7576 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7638 | 0.7647 | 0.7642 | 0.7649 | 0.7641 | 0.7621 | 0.7672 | 0.7636 | 0.7674 | 0.7576 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0979 | 0.0000 | 0.0000 | 0.0000 | 0.0979 | 1.0000 |
| 90 | 10 | 299,940 | 0.1880 | 0.1094 | 0.9970 | 0.1972 | 0.0982 | 0.9966 |
| 80 | 20 | 291,350 | 0.2779 | 0.2165 | 0.9970 | 0.3558 | 0.0981 | 0.9924 |
| 70 | 30 | 194,230 | 0.3678 | 0.3215 | 0.9970 | 0.4862 | 0.0982 | 0.9870 |
| 60 | 40 | 145,675 | 0.4579 | 0.4244 | 0.9970 | 0.5954 | 0.0985 | 0.9800 |
| 50 | 50 | 116,540 | 0.5480 | 0.5253 | 0.9970 | 0.6881 | 0.0991 | 0.9704 |
| 40 | 60 | 97,115 | 0.6376 | 0.6239 | 0.9970 | 0.7675 | 0.0984 | 0.9560 |
| 30 | 70 | 83,240 | 0.7273 | 0.7206 | 0.9970 | 0.8365 | 0.0979 | 0.9329 |
| 20 | 80 | 72,835 | 0.8172 | 0.8156 | 0.9970 | 0.8972 | 0.0983 | 0.8905 |
| 10 | 90 | 64,740 | 0.9073 | 0.9088 | 0.9970 | 0.9509 | 0.1001 | 0.7864 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7647 | 0.0000 | 0.0000 | 0.0000 | 0.7647 | 1.0000 |
| 90 | 10 | 299,940 | 0.7884 | 0.3202 | 0.9939 | 0.4843 | 0.7655 | 0.9991 |
| 80 | 20 | 291,350 | 0.8109 | 0.5140 | 0.9943 | 0.6777 | 0.7650 | 0.9981 |
| 70 | 30 | 194,230 | 0.8344 | 0.6454 | 0.9943 | 0.7827 | 0.7659 | 0.9968 |
| 60 | 40 | 145,675 | 0.8567 | 0.7382 | 0.9943 | 0.8473 | 0.7649 | 0.9951 |
| 50 | 50 | 116,540 | 0.8785 | 0.8073 | 0.9943 | 0.8911 | 0.7626 | 0.9926 |
| 40 | 60 | 97,115 | 0.9038 | 0.8654 | 0.9943 | 0.9254 | 0.7679 | 0.9890 |
| 30 | 70 | 83,240 | 0.9252 | 0.9077 | 0.9943 | 0.9490 | 0.7640 | 0.9829 |
| 20 | 80 | 72,835 | 0.9491 | 0.9450 | 0.9943 | 0.9690 | 0.7684 | 0.9713 |
| 10 | 90 | 64,740 | 0.9708 | 0.9737 | 0.9943 | 0.9839 | 0.7587 | 0.9369 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9972 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7638 | 0.0000 | 0.0000 | 0.0000 | 0.7638 | 1.0000 |
| 90 | 10 | 299,940 | 0.7876 | 0.3194 | 0.9939 | 0.4835 | 0.7647 | 0.9991 |
| 80 | 20 | 291,350 | 0.8102 | 0.5131 | 0.9943 | 0.6769 | 0.7642 | 0.9981 |
| 70 | 30 | 194,230 | 0.8337 | 0.6445 | 0.9943 | 0.7820 | 0.7649 | 0.9968 |
| 60 | 40 | 145,675 | 0.8562 | 0.7375 | 0.9943 | 0.8469 | 0.7641 | 0.9951 |
| 50 | 50 | 116,540 | 0.8782 | 0.8070 | 0.9943 | 0.8909 | 0.7621 | 0.9926 |
| 40 | 60 | 97,115 | 0.9034 | 0.8650 | 0.9943 | 0.9251 | 0.7672 | 0.9890 |
| 30 | 70 | 83,240 | 0.9251 | 0.9075 | 0.9943 | 0.9489 | 0.7636 | 0.9829 |
| 20 | 80 | 72,835 | 0.9489 | 0.9447 | 0.9943 | 0.9689 | 0.7674 | 0.9712 |
| 10 | 90 | 64,740 | 0.9706 | 0.9736 | 0.9943 | 0.9839 | 0.7576 | 0.9366 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9971 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7638 | 0.0000 | 0.0000 | 0.0000 | 0.7638 | 1.0000 |
| 90 | 10 | 299,940 | 0.7876 | 0.3194 | 0.9939 | 0.4835 | 0.7647 | 0.9991 |
| 80 | 20 | 291,350 | 0.8102 | 0.5131 | 0.9943 | 0.6769 | 0.7642 | 0.9981 |
| 70 | 30 | 194,230 | 0.8337 | 0.6445 | 0.9943 | 0.7820 | 0.7649 | 0.9968 |
| 60 | 40 | 145,675 | 0.8562 | 0.7375 | 0.9943 | 0.8469 | 0.7641 | 0.9951 |
| 50 | 50 | 116,540 | 0.8782 | 0.8070 | 0.9943 | 0.8909 | 0.7621 | 0.9926 |
| 40 | 60 | 97,115 | 0.9034 | 0.8650 | 0.9943 | 0.9251 | 0.7672 | 0.9890 |
| 30 | 70 | 83,240 | 0.9251 | 0.9075 | 0.9943 | 0.9489 | 0.7636 | 0.9829 |
| 20 | 80 | 72,835 | 0.9489 | 0.9447 | 0.9943 | 0.9689 | 0.7674 | 0.9712 |
| 10 | 90 | 64,740 | 0.9706 | 0.9736 | 0.9943 | 0.9839 | 0.7576 | 0.9366 |
| 0 | 100 | 58,270 | 0.9943 | 1.0000 | 0.9943 | 0.9971 | 0.0000 | 0.0000 |


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
0.15       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093   <--
0.20       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.25       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.30       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.35       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.40       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.45       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.50       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.55       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.60       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.65       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.70       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.75       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
0.80       0.1880   0.1971   0.0982   0.9960   0.9964   0.1093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1880, F1=0.1971, Normal Recall=0.0982, Normal Precision=0.9960, Attack Recall=0.9964, Attack Precision=0.1093

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
0.15       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166   <--
0.20       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.25       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.30       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.35       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.40       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.45       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.50       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.55       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.60       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.65       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.70       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.75       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
0.80       0.2781   0.3558   0.0984   0.9924   0.9970   0.2166  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2781, F1=0.3558, Normal Recall=0.0984, Normal Precision=0.9924, Attack Recall=0.9970, Attack Precision=0.2166

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
0.15       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215   <--
0.20       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.25       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.30       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.35       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.40       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.45       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.50       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.55       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.60       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.65       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.70       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.75       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
0.80       0.3679   0.4862   0.0984   0.9870   0.9970   0.3215  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3679, F1=0.4862, Normal Recall=0.0984, Normal Precision=0.9870, Attack Recall=0.9970, Attack Precision=0.3215

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
0.15       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242   <--
0.20       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.25       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.30       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.35       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.40       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.45       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.50       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.55       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.60       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.65       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.70       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.75       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
0.80       0.4574   0.5951   0.0976   0.9798   0.9970   0.4242  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4574, F1=0.5951, Normal Recall=0.0976, Normal Precision=0.9798, Attack Recall=0.9970, Attack Precision=0.4242

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
0.15       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248   <--
0.20       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.25       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.30       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.35       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.40       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.45       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.50       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.55       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.60       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.65       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.70       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.75       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
0.80       0.5472   0.6877   0.0973   0.9699   0.9970   0.5248  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5472, F1=0.6877, Normal Recall=0.0973, Normal Precision=0.9699, Attack Recall=0.9970, Attack Precision=0.5248

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
0.15       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204   <--
0.20       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.25       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.30       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.35       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.40       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.45       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.50       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.55       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.60       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.65       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.70       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.75       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
0.80       0.7885   0.4847   0.7655   0.9993   0.9949   0.3204  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7885, F1=0.4847, Normal Recall=0.7655, Normal Precision=0.9993, Attack Recall=0.9949, Attack Precision=0.3204

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
0.15       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147   <--
0.20       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.25       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.30       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.35       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.40       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.45       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.50       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.55       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.60       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.65       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.70       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.75       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
0.80       0.8114   0.6783   0.7657   0.9981   0.9943   0.5147  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8114, F1=0.6783, Normal Recall=0.7657, Normal Precision=0.9981, Attack Recall=0.9943, Attack Precision=0.5147

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
0.15       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440   <--
0.20       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.25       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.30       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.35       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.40       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.45       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.50       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.55       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.60       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.65       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.70       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.75       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
0.80       0.8334   0.7817   0.7645   0.9968   0.9943   0.6440  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8334, F1=0.7817, Normal Recall=0.7645, Normal Precision=0.9968, Attack Recall=0.9943, Attack Precision=0.6440

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
0.15       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378   <--
0.20       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.25       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.30       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.35       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.40       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.45       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.50       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.55       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.60       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.65       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.70       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.75       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
0.80       0.8564   0.8471   0.7644   0.9951   0.9943   0.7378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8564, F1=0.8471, Normal Recall=0.7644, Normal Precision=0.9951, Attack Recall=0.9943, Attack Precision=0.7378

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
0.15       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077   <--
0.20       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.25       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.30       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.35       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.40       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.45       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.50       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.55       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.60       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.65       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.70       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.75       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
0.80       0.8788   0.8913   0.7633   0.9926   0.9943   0.8077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8788, F1=0.8913, Normal Recall=0.7633, Normal Precision=0.9926, Attack Recall=0.9943, Attack Precision=0.8077

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
0.15       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197   <--
0.20       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.25       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.30       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.35       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.40       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.45       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.50       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.55       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.60       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.65       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.70       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.75       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.80       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7877, F1=0.4839, Normal Recall=0.7647, Normal Precision=0.9993, Attack Recall=0.9949, Attack Precision=0.3197

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
0.15       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139   <--
0.20       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.25       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.30       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.35       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.40       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.45       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.50       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.55       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.60       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.65       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.70       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.75       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.80       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8108, F1=0.6776, Normal Recall=0.7649, Normal Precision=0.9981, Attack Recall=0.9943, Attack Precision=0.5139

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
0.15       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432   <--
0.20       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.25       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.30       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.35       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.40       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.45       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.50       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.55       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.60       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.65       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.70       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.75       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.80       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8328, F1=0.7811, Normal Recall=0.7636, Normal Precision=0.9968, Attack Recall=0.9943, Attack Precision=0.6432

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
0.15       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371   <--
0.20       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.25       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.30       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.35       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.40       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.45       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.50       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.55       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.60       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.65       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.70       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.75       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.80       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8559, F1=0.8466, Normal Recall=0.7636, Normal Precision=0.9951, Attack Recall=0.9943, Attack Precision=0.7371

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
0.15       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070   <--
0.20       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.25       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.30       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.35       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.40       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.45       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.50       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.55       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.60       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.65       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.70       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.75       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.80       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8783, F1=0.8909, Normal Recall=0.7623, Normal Precision=0.9926, Attack Recall=0.9943, Attack Precision=0.8070

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
0.15       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197   <--
0.20       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.25       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.30       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.35       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.40       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.45       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.50       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.55       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.60       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.65       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.70       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.75       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
0.80       0.7877   0.4839   0.7647   0.9993   0.9949   0.3197  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7877, F1=0.4839, Normal Recall=0.7647, Normal Precision=0.9993, Attack Recall=0.9949, Attack Precision=0.3197

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
0.15       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139   <--
0.20       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.25       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.30       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.35       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.40       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.45       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.50       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.55       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.60       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.65       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.70       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.75       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
0.80       0.8108   0.6776   0.7649   0.9981   0.9943   0.5139  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8108, F1=0.6776, Normal Recall=0.7649, Normal Precision=0.9981, Attack Recall=0.9943, Attack Precision=0.5139

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
0.15       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432   <--
0.20       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.25       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.30       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.35       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.40       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.45       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.50       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.55       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.60       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.65       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.70       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.75       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
0.80       0.8328   0.7811   0.7636   0.9968   0.9943   0.6432  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8328, F1=0.7811, Normal Recall=0.7636, Normal Precision=0.9968, Attack Recall=0.9943, Attack Precision=0.6432

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
0.15       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371   <--
0.20       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.25       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.30       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.35       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.40       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.45       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.50       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.55       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.60       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.65       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.70       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.75       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
0.80       0.8559   0.8466   0.7636   0.9951   0.9943   0.7371  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8559, F1=0.8466, Normal Recall=0.7636, Normal Precision=0.9951, Attack Recall=0.9943, Attack Precision=0.7371

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
0.15       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070   <--
0.20       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.25       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.30       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.35       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.40       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.45       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.50       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.55       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.60       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.65       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.70       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.75       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
0.80       0.8783   0.8909   0.7623   0.9926   0.9943   0.8070  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8783, F1=0.8909, Normal Recall=0.7623, Normal Precision=0.9926, Attack Recall=0.9943, Attack Precision=0.8070

```

