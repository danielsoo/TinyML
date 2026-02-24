# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-13 08:48:33 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2526 | 0.2934 | 0.3335 | 0.3728 | 0.4151 | 0.4532 | 0.4939 | 0.5350 | 0.5731 | 0.6140 | 0.6539 |
| QAT+Prune only | 0.6910 | 0.7232 | 0.7532 | 0.7836 | 0.8142 | 0.8436 | 0.8753 | 0.9046 | 0.9357 | 0.9654 | 0.9964 |
| QAT+PTQ | 0.6922 | 0.7246 | 0.7544 | 0.7844 | 0.8150 | 0.8444 | 0.8760 | 0.9049 | 0.9359 | 0.9655 | 0.9964 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6922 | 0.7246 | 0.7544 | 0.7844 | 0.8150 | 0.8444 | 0.8760 | 0.9049 | 0.9359 | 0.9655 | 0.9964 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.1563 | 0.2819 | 0.3849 | 0.4721 | 0.5446 | 0.6079 | 0.6632 | 0.7102 | 0.7531 | 0.7908 |
| QAT+Prune only | 0.0000 | 0.4186 | 0.6176 | 0.7342 | 0.8110 | 0.8643 | 0.9056 | 0.9360 | 0.9613 | 0.9811 | 0.9982 |
| QAT+PTQ | 0.0000 | 0.4198 | 0.6187 | 0.7350 | 0.8117 | 0.8649 | 0.9060 | 0.9361 | 0.9614 | 0.9811 | 0.9982 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4198 | 0.6187 | 0.7350 | 0.8117 | 0.8649 | 0.9060 | 0.9361 | 0.9614 | 0.9811 | 0.9982 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2526 | 0.2533 | 0.2534 | 0.2524 | 0.2559 | 0.2525 | 0.2538 | 0.2574 | 0.2498 | 0.2546 | 0.0000 |
| QAT+Prune only | 0.6910 | 0.6929 | 0.6924 | 0.6923 | 0.6928 | 0.6909 | 0.6937 | 0.6903 | 0.6931 | 0.6861 | 0.0000 |
| QAT+PTQ | 0.6922 | 0.6943 | 0.6939 | 0.6936 | 0.6941 | 0.6923 | 0.6953 | 0.6912 | 0.6939 | 0.6877 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6922 | 0.6943 | 0.6939 | 0.6936 | 0.6941 | 0.6923 | 0.6953 | 0.6912 | 0.6939 | 0.6877 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2526 | 0.0000 | 0.0000 | 0.0000 | 0.2526 | 1.0000 |
| 90 | 10 | 299,940 | 0.2934 | 0.0887 | 0.6544 | 0.1563 | 0.2533 | 0.8684 |
| 80 | 20 | 291,350 | 0.3335 | 0.1796 | 0.6539 | 0.2819 | 0.2534 | 0.7455 |
| 70 | 30 | 194,230 | 0.3728 | 0.2727 | 0.6539 | 0.3849 | 0.2524 | 0.6299 |
| 60 | 40 | 145,675 | 0.4151 | 0.3694 | 0.6539 | 0.4721 | 0.2559 | 0.5258 |
| 50 | 50 | 116,540 | 0.4532 | 0.4666 | 0.6539 | 0.5446 | 0.2525 | 0.4219 |
| 40 | 60 | 97,115 | 0.4939 | 0.5680 | 0.6539 | 0.6079 | 0.2538 | 0.3284 |
| 30 | 70 | 83,240 | 0.5350 | 0.6726 | 0.6539 | 0.6632 | 0.2574 | 0.2417 |
| 20 | 80 | 72,835 | 0.5731 | 0.7771 | 0.6539 | 0.7102 | 0.2498 | 0.1529 |
| 10 | 90 | 64,740 | 0.6140 | 0.8876 | 0.6540 | 0.7531 | 0.2546 | 0.0756 |
| 0 | 100 | 58,270 | 0.6539 | 1.0000 | 0.6539 | 0.7908 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6910 | 0.0000 | 0.0000 | 0.0000 | 0.6910 | 1.0000 |
| 90 | 10 | 299,940 | 0.7232 | 0.2650 | 0.9964 | 0.4186 | 0.6929 | 0.9994 |
| 80 | 20 | 291,350 | 0.7532 | 0.4475 | 0.9964 | 0.6176 | 0.6924 | 0.9987 |
| 70 | 30 | 194,230 | 0.7836 | 0.5812 | 0.9964 | 0.7342 | 0.6923 | 0.9978 |
| 60 | 40 | 145,675 | 0.8142 | 0.6837 | 0.9964 | 0.8110 | 0.6928 | 0.9965 |
| 50 | 50 | 116,540 | 0.8436 | 0.7632 | 0.9964 | 0.8643 | 0.6909 | 0.9948 |
| 40 | 60 | 97,115 | 0.8753 | 0.8299 | 0.9964 | 0.9056 | 0.6937 | 0.9923 |
| 30 | 70 | 83,240 | 0.9046 | 0.8824 | 0.9964 | 0.9360 | 0.6903 | 0.9880 |
| 20 | 80 | 72,835 | 0.9357 | 0.9285 | 0.9964 | 0.9613 | 0.6931 | 0.9796 |
| 10 | 90 | 64,740 | 0.9654 | 0.9662 | 0.9964 | 0.9811 | 0.6861 | 0.9549 |
| 0 | 100 | 58,270 | 0.9964 | 1.0000 | 0.9964 | 0.9982 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6922 | 0.0000 | 0.0000 | 0.0000 | 0.6922 | 1.0000 |
| 90 | 10 | 299,940 | 0.7246 | 0.2659 | 0.9964 | 0.4198 | 0.6943 | 0.9994 |
| 80 | 20 | 291,350 | 0.7544 | 0.4487 | 0.9964 | 0.6187 | 0.6939 | 0.9987 |
| 70 | 30 | 194,230 | 0.7844 | 0.5822 | 0.9964 | 0.7350 | 0.6936 | 0.9978 |
| 60 | 40 | 145,675 | 0.8150 | 0.6847 | 0.9964 | 0.8117 | 0.6941 | 0.9966 |
| 50 | 50 | 116,540 | 0.8444 | 0.7641 | 0.9964 | 0.8649 | 0.6923 | 0.9948 |
| 40 | 60 | 97,115 | 0.8760 | 0.8307 | 0.9964 | 0.9060 | 0.6953 | 0.9923 |
| 30 | 70 | 83,240 | 0.9049 | 0.8828 | 0.9964 | 0.9361 | 0.6912 | 0.9880 |
| 20 | 80 | 72,835 | 0.9359 | 0.9287 | 0.9964 | 0.9614 | 0.6939 | 0.9797 |
| 10 | 90 | 64,740 | 0.9655 | 0.9663 | 0.9964 | 0.9811 | 0.6877 | 0.9552 |
| 0 | 100 | 58,270 | 0.9964 | 1.0000 | 0.9964 | 0.9982 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6922 | 0.0000 | 0.0000 | 0.0000 | 0.6922 | 1.0000 |
| 90 | 10 | 299,940 | 0.7246 | 0.2659 | 0.9964 | 0.4198 | 0.6943 | 0.9994 |
| 80 | 20 | 291,350 | 0.7544 | 0.4487 | 0.9964 | 0.6187 | 0.6939 | 0.9987 |
| 70 | 30 | 194,230 | 0.7844 | 0.5822 | 0.9964 | 0.7350 | 0.6936 | 0.9978 |
| 60 | 40 | 145,675 | 0.8150 | 0.6847 | 0.9964 | 0.8117 | 0.6941 | 0.9966 |
| 50 | 50 | 116,540 | 0.8444 | 0.7641 | 0.9964 | 0.8649 | 0.6923 | 0.9948 |
| 40 | 60 | 97,115 | 0.8760 | 0.8307 | 0.9964 | 0.9060 | 0.6953 | 0.9923 |
| 30 | 70 | 83,240 | 0.9049 | 0.8828 | 0.9964 | 0.9361 | 0.6912 | 0.9880 |
| 20 | 80 | 72,835 | 0.9359 | 0.9287 | 0.9964 | 0.9614 | 0.6939 | 0.9797 |
| 10 | 90 | 64,740 | 0.9655 | 0.9663 | 0.9964 | 0.9811 | 0.6877 | 0.9552 |
| 0 | 100 | 58,270 | 0.9964 | 1.0000 | 0.9964 | 0.9982 | 0.0000 | 0.0000 |


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
0.15       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881   <--
0.20       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.25       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.30       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.35       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.40       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.45       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.50       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.55       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.60       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.65       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.70       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.75       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
0.80       0.2929   0.1551   0.2533   0.8666   0.6492   0.0881  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2929, F1=0.1551, Normal Recall=0.2533, Normal Precision=0.8666, Attack Recall=0.6492, Attack Precision=0.0881

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
0.15       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795   <--
0.20       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.25       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.30       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.35       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.40       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.45       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.50       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.55       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.60       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.65       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.70       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.75       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
0.80       0.3328   0.2816   0.2525   0.7448   0.6539   0.1795  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3328, F1=0.2816, Normal Recall=0.2525, Normal Precision=0.7448, Attack Recall=0.6539, Attack Precision=0.1795

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
0.15       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727   <--
0.20       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.25       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.30       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.35       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.40       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.45       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.50       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.55       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.60       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.65       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.70       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.75       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
0.80       0.3729   0.3849   0.2525   0.6299   0.6539   0.2727  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3729, F1=0.3849, Normal Recall=0.2525, Normal Precision=0.6299, Attack Recall=0.6539, Attack Precision=0.2727

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
0.15       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686   <--
0.20       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.25       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.30       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.35       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.40       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.45       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.50       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.55       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.60       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.65       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.70       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.75       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
0.80       0.4136   0.4715   0.2533   0.5234   0.6539   0.3686  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4136, F1=0.4715, Normal Recall=0.2533, Normal Precision=0.5234, Attack Recall=0.6539, Attack Precision=0.3686

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
0.15       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673   <--
0.20       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.25       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.30       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.35       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.40       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.45       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.50       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.55       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.60       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.65       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.70       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.75       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
0.80       0.4542   0.5451   0.2545   0.4238   0.6539   0.4673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4542, F1=0.5451, Normal Recall=0.2545, Normal Precision=0.4238, Attack Recall=0.6539, Attack Precision=0.4673

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
0.15       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651   <--
0.20       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.25       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.30       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.35       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.40       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.45       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.50       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.55       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.60       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.65       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.70       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.75       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
0.80       0.7233   0.4188   0.6929   0.9995   0.9970   0.2651  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7233, F1=0.4188, Normal Recall=0.6929, Normal Precision=0.9995, Attack Recall=0.9970, Attack Precision=0.2651

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
0.15       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479   <--
0.20       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.25       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.30       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.35       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.40       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.45       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.50       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.55       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.60       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.65       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.70       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.75       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
0.80       0.7537   0.6180   0.6930   0.9987   0.9964   0.4479  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7537, F1=0.6180, Normal Recall=0.6930, Normal Precision=0.9987, Attack Recall=0.9964, Attack Precision=0.4479

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
0.15       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812   <--
0.20       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.25       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.30       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.35       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.40       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.45       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.50       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.55       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.60       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.65       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.70       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.75       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
0.80       0.7835   0.7341   0.6922   0.9978   0.9964   0.5812  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7835, F1=0.7341, Normal Recall=0.6922, Normal Precision=0.9978, Attack Recall=0.9964, Attack Precision=0.5812

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
0.15       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823   <--
0.20       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.25       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.30       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.35       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.40       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.45       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.50       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.55       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.60       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.65       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.70       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.75       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
0.80       0.8129   0.8099   0.6906   0.9965   0.9964   0.6823  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8129, F1=0.8099, Normal Recall=0.6906, Normal Precision=0.9965, Attack Recall=0.9964, Attack Precision=0.6823

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
0.15       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620   <--
0.20       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.25       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.30       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.35       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.40       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.45       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.50       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.55       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.60       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.65       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.70       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.75       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
0.80       0.8426   0.8636   0.6888   0.9948   0.9964   0.7620  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8426, F1=0.8636, Normal Recall=0.6888, Normal Precision=0.9948, Attack Recall=0.9964, Attack Precision=0.7620

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
0.15       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660   <--
0.20       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.25       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.30       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.35       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.40       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.45       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.50       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.55       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.60       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.65       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.70       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.75       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.80       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7246, F1=0.4200, Normal Recall=0.6944, Normal Precision=0.9995, Attack Recall=0.9970, Attack Precision=0.2660

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
0.15       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491   <--
0.20       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.25       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.30       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.35       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.40       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.45       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.50       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.55       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.60       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.65       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.70       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.75       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.80       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7548, F1=0.6191, Normal Recall=0.6944, Normal Precision=0.9987, Attack Recall=0.9964, Attack Precision=0.4491

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
0.15       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821   <--
0.20       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.25       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.30       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.35       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.40       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.45       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.50       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.55       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.60       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.65       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.70       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.75       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.80       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7844, F1=0.7349, Normal Recall=0.6935, Normal Precision=0.9978, Attack Recall=0.9964, Attack Precision=0.5821

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
0.15       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831   <--
0.20       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.25       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.30       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.35       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.40       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.45       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.50       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.55       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.60       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.65       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.70       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.75       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.80       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8137, F1=0.8105, Normal Recall=0.6918, Normal Precision=0.9966, Attack Recall=0.9964, Attack Precision=0.6831

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
0.15       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626   <--
0.20       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.25       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.30       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.35       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.40       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.45       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.50       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.55       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.60       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.65       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.70       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.75       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.80       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8431, F1=0.8640, Normal Recall=0.6898, Normal Precision=0.9948, Attack Recall=0.9964, Attack Precision=0.7626

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
0.15       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660   <--
0.20       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.25       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.30       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.35       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.40       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.45       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.50       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.55       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.60       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.65       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.70       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.75       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
0.80       0.7246   0.4200   0.6944   0.9995   0.9970   0.2660  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7246, F1=0.4200, Normal Recall=0.6944, Normal Precision=0.9995, Attack Recall=0.9970, Attack Precision=0.2660

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
0.15       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491   <--
0.20       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.25       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.30       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.35       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.40       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.45       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.50       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.55       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.60       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.65       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.70       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.75       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
0.80       0.7548   0.6191   0.6944   0.9987   0.9964   0.4491  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7548, F1=0.6191, Normal Recall=0.6944, Normal Precision=0.9987, Attack Recall=0.9964, Attack Precision=0.4491

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
0.15       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821   <--
0.20       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.25       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.30       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.35       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.40       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.45       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.50       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.55       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.60       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.65       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.70       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.75       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
0.80       0.7844   0.7349   0.6935   0.9978   0.9964   0.5821  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7844, F1=0.7349, Normal Recall=0.6935, Normal Precision=0.9978, Attack Recall=0.9964, Attack Precision=0.5821

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
0.15       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831   <--
0.20       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.25       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.30       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.35       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.40       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.45       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.50       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.55       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.60       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.65       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.70       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.75       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
0.80       0.8137   0.8105   0.6918   0.9966   0.9964   0.6831  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8137, F1=0.8105, Normal Recall=0.6918, Normal Precision=0.9966, Attack Recall=0.9964, Attack Precision=0.6831

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
0.15       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626   <--
0.20       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.25       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.30       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.35       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.40       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.45       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.50       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.55       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.60       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.65       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.70       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.75       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
0.80       0.8431   0.8640   0.6898   0.9948   0.9964   0.7626  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8431, F1=0.8640, Normal Recall=0.6898, Normal Precision=0.9948, Attack Recall=0.9964, Attack Precision=0.7626

```

