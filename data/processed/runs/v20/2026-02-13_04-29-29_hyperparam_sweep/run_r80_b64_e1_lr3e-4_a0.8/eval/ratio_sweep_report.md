# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-16 03:27:44 |

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
| Original (TFLite) | 0.6152 | 0.6465 | 0.6786 | 0.7133 | 0.7446 | 0.7789 | 0.8092 | 0.8426 | 0.8737 | 0.9074 | 0.9401 |
| QAT+Prune only | 0.7297 | 0.7565 | 0.7825 | 0.8087 | 0.8349 | 0.8606 | 0.8880 | 0.9141 | 0.9406 | 0.9662 | 0.9931 |
| QAT+PTQ | 0.7285 | 0.7556 | 0.7817 | 0.8079 | 0.8343 | 0.8602 | 0.8876 | 0.9139 | 0.9403 | 0.9661 | 0.9931 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7285 | 0.7556 | 0.7817 | 0.8079 | 0.8343 | 0.8602 | 0.8876 | 0.9139 | 0.9403 | 0.9661 | 0.9931 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3476 | 0.5392 | 0.6630 | 0.7465 | 0.8096 | 0.8553 | 0.8932 | 0.9225 | 0.9481 | 0.9691 |
| QAT+Prune only | 0.0000 | 0.4492 | 0.6462 | 0.7570 | 0.8279 | 0.8769 | 0.9141 | 0.9418 | 0.9640 | 0.9814 | 0.9965 |
| QAT+PTQ | 0.0000 | 0.4483 | 0.6453 | 0.7562 | 0.8275 | 0.8766 | 0.9138 | 0.9417 | 0.9638 | 0.9814 | 0.9965 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4483 | 0.6453 | 0.7562 | 0.8275 | 0.8766 | 0.9138 | 0.9417 | 0.9638 | 0.9814 | 0.9965 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6152 | 0.6138 | 0.6133 | 0.6161 | 0.6142 | 0.6177 | 0.6128 | 0.6149 | 0.6079 | 0.6128 | 0.0000 |
| QAT+Prune only | 0.7297 | 0.7303 | 0.7299 | 0.7297 | 0.7294 | 0.7282 | 0.7304 | 0.7298 | 0.7305 | 0.7241 | 0.0000 |
| QAT+PTQ | 0.7285 | 0.7293 | 0.7288 | 0.7285 | 0.7285 | 0.7272 | 0.7293 | 0.7290 | 0.7291 | 0.7235 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7285 | 0.7293 | 0.7288 | 0.7285 | 0.7285 | 0.7272 | 0.7293 | 0.7290 | 0.7291 | 0.7235 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6152 | 0.0000 | 0.0000 | 0.0000 | 0.6152 | 1.0000 |
| 90 | 10 | 299,940 | 0.6465 | 0.2131 | 0.9414 | 0.3476 | 0.6138 | 0.9895 |
| 80 | 20 | 291,350 | 0.6786 | 0.3780 | 0.9401 | 0.5392 | 0.6133 | 0.9762 |
| 70 | 30 | 194,230 | 0.7133 | 0.5121 | 0.9401 | 0.6630 | 0.6161 | 0.9600 |
| 60 | 40 | 145,675 | 0.7446 | 0.6190 | 0.9401 | 0.7465 | 0.6142 | 0.9390 |
| 50 | 50 | 116,540 | 0.7789 | 0.7109 | 0.9401 | 0.8096 | 0.6177 | 0.9117 |
| 40 | 60 | 97,115 | 0.8092 | 0.7846 | 0.9401 | 0.8553 | 0.6128 | 0.8722 |
| 30 | 70 | 83,240 | 0.8426 | 0.8507 | 0.9401 | 0.8932 | 0.6149 | 0.8149 |
| 20 | 80 | 72,835 | 0.8737 | 0.9056 | 0.9401 | 0.9225 | 0.6079 | 0.7174 |
| 10 | 90 | 64,740 | 0.9074 | 0.9562 | 0.9402 | 0.9481 | 0.6128 | 0.5322 |
| 0 | 100 | 58,270 | 0.9401 | 1.0000 | 0.9401 | 0.9691 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7297 | 0.0000 | 0.0000 | 0.0000 | 0.7297 | 1.0000 |
| 90 | 10 | 299,940 | 0.7565 | 0.2903 | 0.9929 | 0.4492 | 0.7303 | 0.9989 |
| 80 | 20 | 291,350 | 0.7825 | 0.4789 | 0.9931 | 0.6462 | 0.7299 | 0.9976 |
| 70 | 30 | 194,230 | 0.8087 | 0.6116 | 0.9931 | 0.7570 | 0.7297 | 0.9960 |
| 60 | 40 | 145,675 | 0.8349 | 0.7098 | 0.9931 | 0.8279 | 0.7294 | 0.9937 |
| 50 | 50 | 116,540 | 0.8606 | 0.7851 | 0.9931 | 0.8769 | 0.7282 | 0.9906 |
| 40 | 60 | 97,115 | 0.8880 | 0.8468 | 0.9931 | 0.9141 | 0.7304 | 0.9860 |
| 30 | 70 | 83,240 | 0.9141 | 0.8956 | 0.9931 | 0.9418 | 0.7298 | 0.9784 |
| 20 | 80 | 72,835 | 0.9406 | 0.9365 | 0.9931 | 0.9640 | 0.7305 | 0.9636 |
| 10 | 90 | 64,740 | 0.9662 | 0.9701 | 0.9931 | 0.9814 | 0.7241 | 0.9210 |
| 0 | 100 | 58,270 | 0.9931 | 1.0000 | 0.9931 | 0.9965 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7285 | 0.0000 | 0.0000 | 0.0000 | 0.7285 | 1.0000 |
| 90 | 10 | 299,940 | 0.7556 | 0.2895 | 0.9929 | 0.4483 | 0.7293 | 0.9989 |
| 80 | 20 | 291,350 | 0.7817 | 0.4780 | 0.9931 | 0.6453 | 0.7288 | 0.9976 |
| 70 | 30 | 194,230 | 0.8079 | 0.6105 | 0.9931 | 0.7562 | 0.7285 | 0.9960 |
| 60 | 40 | 145,675 | 0.8343 | 0.7092 | 0.9931 | 0.8275 | 0.7285 | 0.9937 |
| 50 | 50 | 116,540 | 0.8602 | 0.7845 | 0.9931 | 0.8766 | 0.7272 | 0.9906 |
| 40 | 60 | 97,115 | 0.8876 | 0.8462 | 0.9931 | 0.9138 | 0.7293 | 0.9860 |
| 30 | 70 | 83,240 | 0.9139 | 0.8953 | 0.9931 | 0.9417 | 0.7290 | 0.9784 |
| 20 | 80 | 72,835 | 0.9403 | 0.9362 | 0.9931 | 0.9638 | 0.7291 | 0.9635 |
| 10 | 90 | 64,740 | 0.9661 | 0.9700 | 0.9931 | 0.9814 | 0.7235 | 0.9210 |
| 0 | 100 | 58,270 | 0.9931 | 1.0000 | 0.9931 | 0.9965 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7285 | 0.0000 | 0.0000 | 0.0000 | 0.7285 | 1.0000 |
| 90 | 10 | 299,940 | 0.7556 | 0.2895 | 0.9929 | 0.4483 | 0.7293 | 0.9989 |
| 80 | 20 | 291,350 | 0.7817 | 0.4780 | 0.9931 | 0.6453 | 0.7288 | 0.9976 |
| 70 | 30 | 194,230 | 0.8079 | 0.6105 | 0.9931 | 0.7562 | 0.7285 | 0.9960 |
| 60 | 40 | 145,675 | 0.8343 | 0.7092 | 0.9931 | 0.8275 | 0.7285 | 0.9937 |
| 50 | 50 | 116,540 | 0.8602 | 0.7845 | 0.9931 | 0.8766 | 0.7272 | 0.9906 |
| 40 | 60 | 97,115 | 0.8876 | 0.8462 | 0.9931 | 0.9138 | 0.7293 | 0.9860 |
| 30 | 70 | 83,240 | 0.9139 | 0.8953 | 0.9931 | 0.9417 | 0.7290 | 0.9784 |
| 20 | 80 | 72,835 | 0.9403 | 0.9362 | 0.9931 | 0.9638 | 0.7291 | 0.9635 |
| 10 | 90 | 64,740 | 0.9661 | 0.9700 | 0.9931 | 0.9814 | 0.7235 | 0.9210 |
| 0 | 100 | 58,270 | 0.9931 | 1.0000 | 0.9931 | 0.9965 | 0.0000 | 0.0000 |


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
0.15       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129   <--
0.20       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.25       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.30       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.35       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.40       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.45       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.50       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.55       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.60       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.65       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.70       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.75       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
0.80       0.6464   0.3472   0.6138   0.9893   0.9402   0.2129  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6464, F1=0.3472, Normal Recall=0.6138, Normal Precision=0.9893, Attack Recall=0.9402, Attack Precision=0.2129

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
0.15       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782   <--
0.20       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.25       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.30       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.35       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.40       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.45       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.50       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.55       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.60       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.65       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.70       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.75       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
0.80       0.6789   0.5394   0.6136   0.9762   0.9401   0.3782  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6789, F1=0.5394, Normal Recall=0.6136, Normal Precision=0.9762, Attack Recall=0.9401, Attack Precision=0.3782

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
0.15       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117   <--
0.20       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.25       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.30       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.35       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.40       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.45       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.50       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.55       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.60       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.65       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.70       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.75       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
0.80       0.7129   0.6627   0.6155   0.9600   0.9401   0.5117  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7129, F1=0.6627, Normal Recall=0.6155, Normal Precision=0.9600, Attack Recall=0.9401, Attack Precision=0.5117

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
0.15       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199   <--
0.20       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.25       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.30       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.35       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.40       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.45       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.50       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.55       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.60       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.65       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.70       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.75       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
0.80       0.7455   0.7472   0.6157   0.9391   0.9401   0.6199  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7455, F1=0.7472, Normal Recall=0.6157, Normal Precision=0.9391, Attack Recall=0.9401, Attack Precision=0.6199

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
0.15       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104   <--
0.20       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.25       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.30       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.35       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.40       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.45       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.50       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.55       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.60       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.65       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.70       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.75       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
0.80       0.7785   0.8093   0.6168   0.9115   0.9401   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7785, F1=0.8093, Normal Recall=0.6168, Normal Precision=0.9115, Attack Recall=0.9401, Attack Precision=0.7104

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
0.15       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904   <--
0.20       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.25       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.30       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.35       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.40       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.45       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.50       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.55       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.60       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.65       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.70       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.75       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
0.80       0.7566   0.4494   0.7303   0.9990   0.9935   0.2904  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7566, F1=0.4494, Normal Recall=0.7303, Normal Precision=0.9990, Attack Recall=0.9935, Attack Precision=0.2904

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
0.15       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798   <--
0.20       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.25       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.30       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.35       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.40       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.45       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.50       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.55       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.60       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.65       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.70       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.75       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
0.80       0.7833   0.6470   0.7308   0.9976   0.9931   0.4798  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7833, F1=0.6470, Normal Recall=0.7308, Normal Precision=0.9976, Attack Recall=0.9931, Attack Precision=0.4798

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
0.15       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122   <--
0.20       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.25       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.30       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.35       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.40       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.45       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.50       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.55       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.60       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.65       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.70       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.75       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
0.80       0.8092   0.7575   0.7304   0.9960   0.9931   0.6122  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8092, F1=0.7575, Normal Recall=0.7304, Normal Precision=0.9960, Attack Recall=0.9931, Attack Precision=0.6122

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
0.15       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093   <--
0.20       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.25       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.30       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.35       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.40       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.45       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.50       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.55       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.60       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.65       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.70       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.75       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
0.80       0.8345   0.8276   0.7287   0.9937   0.9931   0.7093  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8345, F1=0.8276, Normal Recall=0.7287, Normal Precision=0.9937, Attack Recall=0.9931, Attack Precision=0.7093

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
0.15       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844   <--
0.20       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.25       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.30       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.35       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.40       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.45       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.50       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.55       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.60       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.65       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.70       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.75       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
0.80       0.8601   0.8765   0.7271   0.9906   0.9931   0.7844  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8601, F1=0.8765, Normal Recall=0.7271, Normal Precision=0.9906, Attack Recall=0.9931, Attack Precision=0.7844

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
0.15       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896   <--
0.20       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.25       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.30       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.35       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.40       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.45       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.50       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.55       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.60       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.65       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.70       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.75       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.80       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7557, F1=0.4485, Normal Recall=0.7293, Normal Precision=0.9990, Attack Recall=0.9935, Attack Precision=0.2896

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
0.15       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788   <--
0.20       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.25       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.30       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.35       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.40       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.45       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.50       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.55       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.60       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.65       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.70       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.75       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.80       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7824, F1=0.6461, Normal Recall=0.7298, Normal Precision=0.9976, Attack Recall=0.9931, Attack Precision=0.4788

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
0.15       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113   <--
0.20       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.25       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.30       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.35       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.40       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.45       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.50       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.55       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.60       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.65       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.70       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.75       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.80       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8085, F1=0.7567, Normal Recall=0.7293, Normal Precision=0.9960, Attack Recall=0.9931, Attack Precision=0.6113

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
0.15       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084   <--
0.20       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.25       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.30       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.35       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.40       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.45       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.50       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.55       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.60       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.65       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.70       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.75       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.80       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8337, F1=0.8270, Normal Recall=0.7275, Normal Precision=0.9937, Attack Recall=0.9931, Attack Precision=0.7084

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
0.15       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838   <--
0.20       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.25       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.30       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.35       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.40       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.45       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.50       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.55       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.60       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.65       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.70       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.75       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.80       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8596, F1=0.8761, Normal Recall=0.7260, Normal Precision=0.9906, Attack Recall=0.9931, Attack Precision=0.7838

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
0.15       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896   <--
0.20       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.25       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.30       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.35       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.40       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.45       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.50       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.55       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.60       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.65       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.70       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.75       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
0.80       0.7557   0.4485   0.7293   0.9990   0.9935   0.2896  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7557, F1=0.4485, Normal Recall=0.7293, Normal Precision=0.9990, Attack Recall=0.9935, Attack Precision=0.2896

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
0.15       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788   <--
0.20       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.25       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.30       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.35       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.40       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.45       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.50       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.55       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.60       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.65       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.70       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.75       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
0.80       0.7824   0.6461   0.7298   0.9976   0.9931   0.4788  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7824, F1=0.6461, Normal Recall=0.7298, Normal Precision=0.9976, Attack Recall=0.9931, Attack Precision=0.4788

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
0.15       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113   <--
0.20       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.25       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.30       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.35       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.40       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.45       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.50       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.55       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.60       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.65       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.70       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.75       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
0.80       0.8085   0.7567   0.7293   0.9960   0.9931   0.6113  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8085, F1=0.7567, Normal Recall=0.7293, Normal Precision=0.9960, Attack Recall=0.9931, Attack Precision=0.6113

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
0.15       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084   <--
0.20       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.25       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.30       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.35       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.40       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.45       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.50       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.55       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.60       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.65       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.70       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.75       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
0.80       0.8337   0.8270   0.7275   0.9937   0.9931   0.7084  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8337, F1=0.8270, Normal Recall=0.7275, Normal Precision=0.9937, Attack Recall=0.9931, Attack Precision=0.7084

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
0.15       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838   <--
0.20       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.25       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.30       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.35       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.40       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.45       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.50       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.55       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.60       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.65       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.70       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.75       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
0.80       0.8596   0.8761   0.7260   0.9906   0.9931   0.7838  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8596, F1=0.8761, Normal Recall=0.7260, Normal Precision=0.9906, Attack Recall=0.9931, Attack Precision=0.7838

```

