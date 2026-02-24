# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b128_e1_lr3e-4_a0.8.yaml` |
| **Generated** | 2026-02-21 00:28:16 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7537 | 0.7740 | 0.7949 | 0.8177 | 0.8386 | 0.8581 | 0.8801 | 0.9006 | 0.9214 | 0.9434 | 0.9639 |
| QAT+Prune only | 0.7495 | 0.7728 | 0.7953 | 0.8196 | 0.8416 | 0.8638 | 0.8884 | 0.9090 | 0.9321 | 0.9561 | 0.9791 |
| QAT+PTQ | 0.7495 | 0.7728 | 0.7953 | 0.8196 | 0.8415 | 0.8639 | 0.8886 | 0.9092 | 0.9323 | 0.9562 | 0.9792 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7495 | 0.7728 | 0.7953 | 0.8196 | 0.8415 | 0.8639 | 0.8886 | 0.9092 | 0.9323 | 0.9562 | 0.9792 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.4602 | 0.6528 | 0.7603 | 0.8269 | 0.8716 | 0.9060 | 0.9314 | 0.9515 | 0.9684 | 0.9816 |
| QAT+Prune only | 0.0000 | 0.4629 | 0.6567 | 0.7651 | 0.8318 | 0.8779 | 0.9133 | 0.9377 | 0.9585 | 0.9757 | 0.9894 |
| QAT+PTQ | 0.0000 | 0.4630 | 0.6568 | 0.7651 | 0.8317 | 0.8780 | 0.9134 | 0.9379 | 0.9586 | 0.9758 | 0.9895 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4630 | 0.6568 | 0.7651 | 0.8317 | 0.8780 | 0.9134 | 0.9379 | 0.9586 | 0.9758 | 0.9895 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7537 | 0.7529 | 0.7527 | 0.7550 | 0.7551 | 0.7522 | 0.7543 | 0.7531 | 0.7516 | 0.7595 | 0.0000 |
| QAT+Prune only | 0.7495 | 0.7499 | 0.7494 | 0.7513 | 0.7500 | 0.7486 | 0.7524 | 0.7454 | 0.7444 | 0.7492 | 0.0000 |
| QAT+PTQ | 0.7495 | 0.7498 | 0.7494 | 0.7511 | 0.7497 | 0.7486 | 0.7527 | 0.7458 | 0.7447 | 0.7490 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7495 | 0.7498 | 0.7494 | 0.7511 | 0.7497 | 0.7486 | 0.7527 | 0.7458 | 0.7447 | 0.7490 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7537 | 0.0000 | 0.0000 | 0.0000 | 0.7537 | 1.0000 |
| 90 | 10 | 299,940 | 0.7740 | 0.3023 | 0.9633 | 0.4602 | 0.7529 | 0.9946 |
| 80 | 20 | 291,350 | 0.7949 | 0.4935 | 0.9639 | 0.6528 | 0.7527 | 0.9881 |
| 70 | 30 | 194,230 | 0.8177 | 0.6277 | 0.9639 | 0.7603 | 0.7550 | 0.9799 |
| 60 | 40 | 145,675 | 0.8386 | 0.7240 | 0.9639 | 0.8269 | 0.7551 | 0.9691 |
| 50 | 50 | 116,540 | 0.8581 | 0.7955 | 0.9639 | 0.8716 | 0.7522 | 0.9542 |
| 40 | 60 | 97,115 | 0.8801 | 0.8548 | 0.9639 | 0.9060 | 0.7543 | 0.9330 |
| 30 | 70 | 83,240 | 0.9006 | 0.9011 | 0.9639 | 0.9314 | 0.7531 | 0.8993 |
| 20 | 80 | 72,835 | 0.9214 | 0.9395 | 0.9639 | 0.9515 | 0.7516 | 0.8387 |
| 10 | 90 | 64,740 | 0.9434 | 0.9730 | 0.9639 | 0.9684 | 0.7595 | 0.7002 |
| 0 | 100 | 58,270 | 0.9639 | 1.0000 | 0.9639 | 0.9816 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7495 | 0.0000 | 0.0000 | 0.0000 | 0.7495 | 1.0000 |
| 90 | 10 | 299,940 | 0.7728 | 0.3031 | 0.9792 | 0.4629 | 0.7499 | 0.9969 |
| 80 | 20 | 291,350 | 0.7953 | 0.4941 | 0.9791 | 0.6567 | 0.7494 | 0.9931 |
| 70 | 30 | 194,230 | 0.8196 | 0.6279 | 0.9791 | 0.7651 | 0.7513 | 0.9882 |
| 60 | 40 | 145,675 | 0.8416 | 0.7230 | 0.9791 | 0.8318 | 0.7500 | 0.9818 |
| 50 | 50 | 116,540 | 0.8638 | 0.7957 | 0.9791 | 0.8779 | 0.7486 | 0.9728 |
| 40 | 60 | 97,115 | 0.8884 | 0.8557 | 0.9791 | 0.9133 | 0.7524 | 0.9600 |
| 30 | 70 | 83,240 | 0.9090 | 0.8997 | 0.9791 | 0.9377 | 0.7454 | 0.9386 |
| 20 | 80 | 72,835 | 0.9321 | 0.9387 | 0.9791 | 0.9585 | 0.7444 | 0.8990 |
| 10 | 90 | 64,740 | 0.9561 | 0.9723 | 0.9791 | 0.9757 | 0.7492 | 0.7993 |
| 0 | 100 | 58,270 | 0.9791 | 1.0000 | 0.9791 | 0.9894 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7495 | 0.0000 | 0.0000 | 0.0000 | 0.7495 | 1.0000 |
| 90 | 10 | 299,940 | 0.7728 | 0.3031 | 0.9793 | 0.4630 | 0.7498 | 0.9969 |
| 80 | 20 | 291,350 | 0.7953 | 0.4941 | 0.9792 | 0.6568 | 0.7494 | 0.9931 |
| 70 | 30 | 194,230 | 0.8196 | 0.6277 | 0.9792 | 0.7651 | 0.7511 | 0.9883 |
| 60 | 40 | 145,675 | 0.8415 | 0.7229 | 0.9792 | 0.8317 | 0.7497 | 0.9819 |
| 50 | 50 | 116,540 | 0.8639 | 0.7957 | 0.9792 | 0.8780 | 0.7486 | 0.9730 |
| 40 | 60 | 97,115 | 0.8886 | 0.8559 | 0.9792 | 0.9134 | 0.7527 | 0.9603 |
| 30 | 70 | 83,240 | 0.9092 | 0.8999 | 0.9792 | 0.9379 | 0.7458 | 0.9390 |
| 20 | 80 | 72,835 | 0.9323 | 0.9388 | 0.9792 | 0.9586 | 0.7447 | 0.8997 |
| 10 | 90 | 64,740 | 0.9562 | 0.9723 | 0.9792 | 0.9758 | 0.7490 | 0.8003 |
| 0 | 100 | 58,270 | 0.9792 | 1.0000 | 0.9792 | 0.9895 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7495 | 0.0000 | 0.0000 | 0.0000 | 0.7495 | 1.0000 |
| 90 | 10 | 299,940 | 0.7728 | 0.3031 | 0.9793 | 0.4630 | 0.7498 | 0.9969 |
| 80 | 20 | 291,350 | 0.7953 | 0.4941 | 0.9792 | 0.6568 | 0.7494 | 0.9931 |
| 70 | 30 | 194,230 | 0.8196 | 0.6277 | 0.9792 | 0.7651 | 0.7511 | 0.9883 |
| 60 | 40 | 145,675 | 0.8415 | 0.7229 | 0.9792 | 0.8317 | 0.7497 | 0.9819 |
| 50 | 50 | 116,540 | 0.8639 | 0.7957 | 0.9792 | 0.8780 | 0.7486 | 0.9730 |
| 40 | 60 | 97,115 | 0.8886 | 0.8559 | 0.9792 | 0.9134 | 0.7527 | 0.9603 |
| 30 | 70 | 83,240 | 0.9092 | 0.8999 | 0.9792 | 0.9379 | 0.7458 | 0.9390 |
| 20 | 80 | 72,835 | 0.9323 | 0.9388 | 0.9792 | 0.9586 | 0.7447 | 0.8997 |
| 10 | 90 | 64,740 | 0.9562 | 0.9723 | 0.9792 | 0.9758 | 0.7490 | 0.8003 |
| 0 | 100 | 58,270 | 0.9792 | 1.0000 | 0.9792 | 0.9895 | 0.0000 | 0.0000 |


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
0.15       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022   <--
0.20       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.25       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.30       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.35       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.40       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.45       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.50       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.55       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.60       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.65       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.70       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.75       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
0.80       0.7740   0.4601   0.7529   0.9946   0.9632   0.3022  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7740, F1=0.4601, Normal Recall=0.7529, Normal Precision=0.9946, Attack Recall=0.9632, Attack Precision=0.3022

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
0.15       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935   <--
0.20       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.25       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.30       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.35       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.40       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.45       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.50       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.55       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.60       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.65       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.70       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.75       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
0.80       0.7950   0.6528   0.7527   0.9881   0.9639   0.4935  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7950, F1=0.6528, Normal Recall=0.7527, Normal Precision=0.9881, Attack Recall=0.9639, Attack Precision=0.4935

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
0.15       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270   <--
0.20       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.25       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.30       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.35       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.40       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.45       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.50       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.55       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.60       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.65       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.70       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.75       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
0.80       0.8172   0.7598   0.7543   0.9799   0.9639   0.6270  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8172, F1=0.7598, Normal Recall=0.7543, Normal Precision=0.9799, Attack Recall=0.9639, Attack Precision=0.6270

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
0.15       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223   <--
0.20       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.25       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.30       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.35       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.40       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.45       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.50       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.55       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.60       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.65       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.70       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.75       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
0.80       0.8373   0.8258   0.7530   0.9690   0.9639   0.7223  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8373, F1=0.8258, Normal Recall=0.7530, Normal Precision=0.9690, Attack Recall=0.9639, Attack Precision=0.7223

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
0.15       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973   <--
0.20       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.25       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.30       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.35       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.40       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.45       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.50       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.55       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.60       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.65       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.70       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.75       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
0.80       0.8594   0.8727   0.7550   0.9543   0.9639   0.7973  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8594, F1=0.8727, Normal Recall=0.7550, Normal Precision=0.9543, Attack Recall=0.9639, Attack Precision=0.7973

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
0.15       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033   <--
0.20       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.25       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.30       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.35       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.40       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.45       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.50       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.55       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.60       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.65       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.70       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.75       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
0.80       0.7729   0.4633   0.7499   0.9971   0.9802   0.3033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7729, F1=0.4633, Normal Recall=0.7499, Normal Precision=0.9971, Attack Recall=0.9802, Attack Precision=0.3033

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
0.15       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950   <--
0.20       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.25       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.30       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.35       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.40       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.45       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.50       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.55       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.60       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.65       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.70       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.75       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
0.80       0.7960   0.6575   0.7502   0.9931   0.9791   0.4950  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7960, F1=0.6575, Normal Recall=0.7502, Normal Precision=0.9931, Attack Recall=0.9791, Attack Precision=0.4950

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
0.15       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264   <--
0.20       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.25       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.30       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.35       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.40       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.45       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.50       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.55       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.60       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.65       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.70       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.75       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
0.80       0.8185   0.7640   0.7497   0.9882   0.9791   0.6264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8185, F1=0.7640, Normal Recall=0.7497, Normal Precision=0.9882, Attack Recall=0.9791, Attack Precision=0.6264

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
0.15       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225   <--
0.20       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.25       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.30       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.35       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.40       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.45       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.50       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.55       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.60       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.65       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.70       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.75       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
0.80       0.8412   0.8315   0.7493   0.9817   0.9791   0.7225  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8412, F1=0.8315, Normal Recall=0.7493, Normal Precision=0.9817, Attack Recall=0.9791, Attack Precision=0.7225

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
0.15       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948   <--
0.20       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.25       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.30       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.35       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.40       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.45       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.50       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.55       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.60       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.65       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.70       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.75       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
0.80       0.8632   0.8774   0.7472   0.9728   0.9791   0.7948  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8632, F1=0.8774, Normal Recall=0.7472, Normal Precision=0.9728, Attack Recall=0.9791, Attack Precision=0.7948

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
0.15       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033   <--
0.20       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.25       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.30       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.35       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.40       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.45       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.50       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.55       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.60       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.65       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.70       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.75       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.80       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7729, F1=0.4633, Normal Recall=0.7498, Normal Precision=0.9971, Attack Recall=0.9803, Attack Precision=0.3033

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
0.15       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949   <--
0.20       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.25       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.30       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.35       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.40       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.45       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.50       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.55       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.60       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.65       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.70       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.75       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.80       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7960, F1=0.6575, Normal Recall=0.7502, Normal Precision=0.9931, Attack Recall=0.9792, Attack Precision=0.4949

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
0.15       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264   <--
0.20       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.25       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.30       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.35       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.40       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.45       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.50       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.55       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.60       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.65       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.70       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.75       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.80       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8186, F1=0.7641, Normal Recall=0.7497, Normal Precision=0.9883, Attack Recall=0.9792, Attack Precision=0.6264

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
0.15       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226   <--
0.20       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.25       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.30       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.35       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.40       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.45       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.50       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.55       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.60       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.65       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.70       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.75       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.80       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8413, F1=0.8315, Normal Recall=0.7494, Normal Precision=0.9819, Attack Recall=0.9792, Attack Precision=0.7226

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
0.15       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949   <--
0.20       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.25       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.30       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.35       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.40       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.45       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.50       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.55       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.60       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.65       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.70       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.75       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.80       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8633, F1=0.8775, Normal Recall=0.7473, Normal Precision=0.9730, Attack Recall=0.9792, Attack Precision=0.7949

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
0.15       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033   <--
0.20       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.25       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.30       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.35       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.40       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.45       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.50       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.55       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.60       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.65       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.70       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.75       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
0.80       0.7729   0.4633   0.7498   0.9971   0.9803   0.3033  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7729, F1=0.4633, Normal Recall=0.7498, Normal Precision=0.9971, Attack Recall=0.9803, Attack Precision=0.3033

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
0.15       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949   <--
0.20       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.25       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.30       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.35       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.40       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.45       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.50       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.55       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.60       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.65       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.70       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.75       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
0.80       0.7960   0.6575   0.7502   0.9931   0.9792   0.4949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7960, F1=0.6575, Normal Recall=0.7502, Normal Precision=0.9931, Attack Recall=0.9792, Attack Precision=0.4949

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
0.15       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264   <--
0.20       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.25       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.30       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.35       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.40       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.45       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.50       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.55       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.60       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.65       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.70       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.75       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
0.80       0.8186   0.7641   0.7497   0.9883   0.9792   0.6264  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8186, F1=0.7641, Normal Recall=0.7497, Normal Precision=0.9883, Attack Recall=0.9792, Attack Precision=0.6264

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
0.15       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226   <--
0.20       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.25       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.30       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.35       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.40       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.45       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.50       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.55       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.60       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.65       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.70       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.75       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
0.80       0.8413   0.8315   0.7494   0.9819   0.9792   0.7226  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8413, F1=0.8315, Normal Recall=0.7494, Normal Precision=0.9819, Attack Recall=0.9792, Attack Precision=0.7226

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
0.15       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949   <--
0.20       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.25       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.30       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.35       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.40       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.45       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.50       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.55       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.60       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.65       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.70       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.75       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
0.80       0.8633   0.8775   0.7473   0.9730   0.9792   0.7949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8633, F1=0.8775, Normal Recall=0.7473, Normal Precision=0.9730, Attack Recall=0.9792, Attack Precision=0.7949

```

