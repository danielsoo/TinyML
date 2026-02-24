# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-15 14:42:34 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 60 |
| **Local epochs** | 2 |
| **Batch size** | 256 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6523 | 0.6854 | 0.7172 | 0.7498 | 0.7807 | 0.8119 | 0.8448 | 0.8753 | 0.9071 | 0.9400 | 0.9713 |
| QAT+Prune only | 0.7743 | 0.7911 | 0.8073 | 0.8238 | 0.8401 | 0.8551 | 0.8727 | 0.8873 | 0.9057 | 0.9207 | 0.9376 |
| QAT+PTQ | 0.7765 | 0.7930 | 0.8087 | 0.8248 | 0.8408 | 0.8552 | 0.8722 | 0.8865 | 0.9045 | 0.9191 | 0.9354 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7765 | 0.7930 | 0.8087 | 0.8248 | 0.8408 | 0.8552 | 0.8722 | 0.8865 | 0.9045 | 0.9191 | 0.9354 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3817 | 0.5788 | 0.6996 | 0.7799 | 0.8377 | 0.8825 | 0.9160 | 0.9436 | 0.9668 | 0.9854 |
| QAT+Prune only | 0.0000 | 0.4732 | 0.6605 | 0.7615 | 0.8243 | 0.8661 | 0.8983 | 0.9209 | 0.9409 | 0.9551 | 0.9678 |
| QAT+PTQ | 0.0000 | 0.4749 | 0.6617 | 0.7621 | 0.8246 | 0.8660 | 0.8978 | 0.9203 | 0.9400 | 0.9541 | 0.9666 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4749 | 0.6617 | 0.7621 | 0.8246 | 0.8660 | 0.8978 | 0.9203 | 0.9400 | 0.9541 | 0.9666 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6523 | 0.6536 | 0.6537 | 0.6549 | 0.6536 | 0.6525 | 0.6550 | 0.6515 | 0.6507 | 0.6586 | 0.0000 |
| QAT+Prune only | 0.7743 | 0.7748 | 0.7747 | 0.7751 | 0.7751 | 0.7725 | 0.7752 | 0.7698 | 0.7780 | 0.7685 | 0.0000 |
| QAT+PTQ | 0.7765 | 0.7772 | 0.7770 | 0.7774 | 0.7778 | 0.7750 | 0.7773 | 0.7723 | 0.7805 | 0.7719 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7765 | 0.7772 | 0.7770 | 0.7774 | 0.7778 | 0.7750 | 0.7773 | 0.7723 | 0.7805 | 0.7719 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6523 | 0.0000 | 0.0000 | 0.0000 | 0.6523 | 1.0000 |
| 90 | 10 | 299,940 | 0.6854 | 0.2375 | 0.9712 | 0.3817 | 0.6536 | 0.9951 |
| 80 | 20 | 291,350 | 0.7172 | 0.4122 | 0.9713 | 0.5788 | 0.6537 | 0.9891 |
| 70 | 30 | 194,230 | 0.7498 | 0.5467 | 0.9713 | 0.6996 | 0.6549 | 0.9815 |
| 60 | 40 | 145,675 | 0.7807 | 0.6515 | 0.9713 | 0.7799 | 0.6536 | 0.9715 |
| 50 | 50 | 116,540 | 0.8119 | 0.7365 | 0.9713 | 0.8377 | 0.6525 | 0.9578 |
| 40 | 60 | 97,115 | 0.8448 | 0.8086 | 0.9713 | 0.8825 | 0.6550 | 0.9382 |
| 30 | 70 | 83,240 | 0.8753 | 0.8667 | 0.9713 | 0.9160 | 0.6515 | 0.9067 |
| 20 | 80 | 72,835 | 0.9071 | 0.9175 | 0.9713 | 0.9436 | 0.6507 | 0.8498 |
| 10 | 90 | 64,740 | 0.9400 | 0.9624 | 0.9713 | 0.9668 | 0.6586 | 0.7180 |
| 0 | 100 | 58,270 | 0.9713 | 1.0000 | 0.9713 | 0.9854 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7743 | 0.0000 | 0.0000 | 0.0000 | 0.7743 | 1.0000 |
| 90 | 10 | 299,940 | 0.7911 | 0.3164 | 0.9381 | 0.4732 | 0.7748 | 0.9912 |
| 80 | 20 | 291,350 | 0.8073 | 0.5099 | 0.9376 | 0.6605 | 0.7747 | 0.9803 |
| 70 | 30 | 194,230 | 0.8238 | 0.6411 | 0.9376 | 0.7615 | 0.7751 | 0.9667 |
| 60 | 40 | 145,675 | 0.8401 | 0.7354 | 0.9376 | 0.8243 | 0.7751 | 0.9491 |
| 50 | 50 | 116,540 | 0.8551 | 0.8048 | 0.9376 | 0.8661 | 0.7725 | 0.9253 |
| 40 | 60 | 97,115 | 0.8727 | 0.8622 | 0.9376 | 0.8983 | 0.7752 | 0.8923 |
| 30 | 70 | 83,240 | 0.8873 | 0.9048 | 0.9376 | 0.9209 | 0.7698 | 0.8410 |
| 20 | 80 | 72,835 | 0.9057 | 0.9441 | 0.9377 | 0.9409 | 0.7780 | 0.7572 |
| 10 | 90 | 64,740 | 0.9207 | 0.9733 | 0.9376 | 0.9551 | 0.7685 | 0.5779 |
| 0 | 100 | 58,270 | 0.9376 | 1.0000 | 0.9376 | 0.9678 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7765 | 0.0000 | 0.0000 | 0.0000 | 0.7765 | 1.0000 |
| 90 | 10 | 299,940 | 0.7930 | 0.3182 | 0.9359 | 0.4749 | 0.7772 | 0.9909 |
| 80 | 20 | 291,350 | 0.8087 | 0.5119 | 0.9354 | 0.6617 | 0.7770 | 0.9796 |
| 70 | 30 | 194,230 | 0.8248 | 0.6430 | 0.9354 | 0.7621 | 0.7774 | 0.9656 |
| 60 | 40 | 145,675 | 0.8408 | 0.7373 | 0.9354 | 0.8246 | 0.7778 | 0.9476 |
| 50 | 50 | 116,540 | 0.8552 | 0.8061 | 0.9354 | 0.8660 | 0.7750 | 0.9231 |
| 40 | 60 | 97,115 | 0.8722 | 0.8630 | 0.9354 | 0.8978 | 0.7773 | 0.8892 |
| 30 | 70 | 83,240 | 0.8865 | 0.9056 | 0.9354 | 0.9203 | 0.7723 | 0.8368 |
| 20 | 80 | 72,835 | 0.9045 | 0.9446 | 0.9355 | 0.9400 | 0.7805 | 0.7514 |
| 10 | 90 | 64,740 | 0.9191 | 0.9736 | 0.9354 | 0.9541 | 0.7719 | 0.5705 |
| 0 | 100 | 58,270 | 0.9354 | 1.0000 | 0.9354 | 0.9666 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7765 | 0.0000 | 0.0000 | 0.0000 | 0.7765 | 1.0000 |
| 90 | 10 | 299,940 | 0.7930 | 0.3182 | 0.9359 | 0.4749 | 0.7772 | 0.9909 |
| 80 | 20 | 291,350 | 0.8087 | 0.5119 | 0.9354 | 0.6617 | 0.7770 | 0.9796 |
| 70 | 30 | 194,230 | 0.8248 | 0.6430 | 0.9354 | 0.7621 | 0.7774 | 0.9656 |
| 60 | 40 | 145,675 | 0.8408 | 0.7373 | 0.9354 | 0.8246 | 0.7778 | 0.9476 |
| 50 | 50 | 116,540 | 0.8552 | 0.8061 | 0.9354 | 0.8660 | 0.7750 | 0.9231 |
| 40 | 60 | 97,115 | 0.8722 | 0.8630 | 0.9354 | 0.8978 | 0.7773 | 0.8892 |
| 30 | 70 | 83,240 | 0.8865 | 0.9056 | 0.9354 | 0.9203 | 0.7723 | 0.8368 |
| 20 | 80 | 72,835 | 0.9045 | 0.9446 | 0.9355 | 0.9400 | 0.7805 | 0.7514 |
| 10 | 90 | 64,740 | 0.9191 | 0.9736 | 0.9354 | 0.9541 | 0.7719 | 0.5705 |
| 0 | 100 | 58,270 | 0.9354 | 1.0000 | 0.9354 | 0.9666 | 0.0000 | 0.0000 |


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
0.15       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377   <--
0.20       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.25       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.30       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.35       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.40       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.45       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.50       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.55       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.60       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.65       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.70       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.75       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
0.80       0.6855   0.3821   0.6536   0.9953   0.9723   0.2377  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6855, F1=0.3821, Normal Recall=0.6536, Normal Precision=0.9953, Attack Recall=0.9723, Attack Precision=0.2377

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
0.15       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116   <--
0.20       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.25       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.30       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.35       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.40       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.45       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.50       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.55       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.60       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.65       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.70       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.75       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
0.80       0.7166   0.5782   0.6529   0.9891   0.9713   0.4116  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7166, F1=0.5782, Normal Recall=0.6529, Normal Precision=0.9891, Attack Recall=0.9713, Attack Precision=0.4116

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
0.15       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452   <--
0.20       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.25       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.30       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.35       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.40       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.45       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.50       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.55       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.60       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.65       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.70       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.75       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
0.80       0.7483   0.6983   0.6527   0.9815   0.9713   0.5452  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7483, F1=0.6983, Normal Recall=0.6527, Normal Precision=0.9815, Attack Recall=0.9713, Attack Precision=0.5452

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
0.15       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514   <--
0.20       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.25       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.30       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.35       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.40       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.45       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.50       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.55       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.60       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.65       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.70       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.75       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
0.80       0.7806   0.7798   0.6535   0.9715   0.9713   0.6514  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7806, F1=0.7798, Normal Recall=0.6535, Normal Precision=0.9715, Attack Recall=0.9713, Attack Precision=0.6514

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
0.15       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368   <--
0.20       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.25       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.30       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.35       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.40       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.45       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.50       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.55       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.60       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.65       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.70       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.75       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
0.80       0.8122   0.8380   0.6531   0.9578   0.9713   0.7368  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8122, F1=0.8380, Normal Recall=0.6531, Normal Precision=0.9578, Attack Recall=0.9713, Attack Precision=0.7368

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
0.15       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165   <--
0.20       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.25       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.30       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.35       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.40       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.45       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.50       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.55       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.60       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.65       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.70       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.75       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
0.80       0.7912   0.4733   0.7748   0.9912   0.9384   0.3165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7912, F1=0.4733, Normal Recall=0.7748, Normal Precision=0.9912, Attack Recall=0.9384, Attack Precision=0.3165

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
0.15       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105   <--
0.20       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.25       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.30       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.35       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.40       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.45       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.50       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.55       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.60       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.65       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.70       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.75       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
0.80       0.8077   0.6611   0.7753   0.9803   0.9376   0.5105  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8077, F1=0.6611, Normal Recall=0.7753, Normal Precision=0.9803, Attack Recall=0.9376, Attack Precision=0.5105

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
0.15       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403   <--
0.20       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.25       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.30       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.35       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.40       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.45       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.50       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.55       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.60       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.65       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.70       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.75       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
0.80       0.8232   0.7609   0.7742   0.9666   0.9376   0.6403  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8232, F1=0.7609, Normal Recall=0.7742, Normal Precision=0.9666, Attack Recall=0.9376, Attack Precision=0.6403

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
0.15       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344   <--
0.20       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.25       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.30       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.35       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.40       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.45       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.50       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.55       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.60       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.65       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.70       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.75       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
0.80       0.8394   0.8236   0.7739   0.9490   0.9376   0.7344  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8394, F1=0.8236, Normal Recall=0.7739, Normal Precision=0.9490, Attack Recall=0.9376, Attack Precision=0.7344

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
0.15       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052   <--
0.20       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.25       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.30       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.35       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.40       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.45       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.50       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.55       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.60       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.65       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.70       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.75       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
0.80       0.8554   0.8664   0.7731   0.9254   0.9376   0.8052  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8554, F1=0.8664, Normal Recall=0.7731, Normal Precision=0.9254, Attack Recall=0.9376, Attack Precision=0.8052

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
0.15       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182   <--
0.20       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.25       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.30       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.35       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.40       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.45       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.50       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.55       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.60       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.65       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.70       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.75       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.80       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7931, F1=0.4750, Normal Recall=0.7772, Normal Precision=0.9910, Attack Recall=0.9362, Attack Precision=0.3182

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
0.15       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125   <--
0.20       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.25       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.30       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.35       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.40       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.45       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.50       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.55       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.60       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.65       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.70       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.75       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.80       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8091, F1=0.6622, Normal Recall=0.7776, Normal Precision=0.9797, Attack Recall=0.9354, Attack Precision=0.5125

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
0.15       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420   <--
0.20       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.25       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.30       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.35       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.40       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.45       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.50       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.55       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.60       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.65       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.70       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.75       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.80       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8241, F1=0.7614, Normal Recall=0.7764, Normal Precision=0.9656, Attack Recall=0.9354, Attack Precision=0.6420

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
0.15       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357   <--
0.20       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.25       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.30       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.35       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.40       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.45       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.50       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.55       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.60       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.65       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.70       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.75       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.80       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8397, F1=0.8236, Normal Recall=0.7760, Normal Precision=0.9474, Attack Recall=0.9354, Attack Precision=0.7357

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
0.15       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062   <--
0.20       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.25       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.30       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.35       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.40       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.45       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.50       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.55       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.60       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.65       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.70       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.75       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.80       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8553, F1=0.8660, Normal Recall=0.7751, Normal Precision=0.9231, Attack Recall=0.9354, Attack Precision=0.8062

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
0.15       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182   <--
0.20       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.25       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.30       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.35       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.40       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.45       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.50       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.55       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.60       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.65       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.70       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.75       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
0.80       0.7931   0.4750   0.7772   0.9910   0.9362   0.3182  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7931, F1=0.4750, Normal Recall=0.7772, Normal Precision=0.9910, Attack Recall=0.9362, Attack Precision=0.3182

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
0.15       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125   <--
0.20       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.25       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.30       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.35       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.40       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.45       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.50       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.55       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.60       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.65       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.70       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.75       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
0.80       0.8091   0.6622   0.7776   0.9797   0.9354   0.5125  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8091, F1=0.6622, Normal Recall=0.7776, Normal Precision=0.9797, Attack Recall=0.9354, Attack Precision=0.5125

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
0.15       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420   <--
0.20       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.25       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.30       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.35       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.40       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.45       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.50       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.55       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.60       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.65       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.70       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.75       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
0.80       0.8241   0.7614   0.7764   0.9656   0.9354   0.6420  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8241, F1=0.7614, Normal Recall=0.7764, Normal Precision=0.9656, Attack Recall=0.9354, Attack Precision=0.6420

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
0.15       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357   <--
0.20       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.25       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.30       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.35       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.40       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.45       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.50       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.55       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.60       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.65       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.70       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.75       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
0.80       0.8397   0.8236   0.7760   0.9474   0.9354   0.7357  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8397, F1=0.8236, Normal Recall=0.7760, Normal Precision=0.9474, Attack Recall=0.9354, Attack Precision=0.7357

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
0.15       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062   <--
0.20       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.25       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.30       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.35       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.40       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.45       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.50       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.55       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.60       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.65       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.70       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.75       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
0.80       0.8553   0.8660   0.7751   0.9231   0.9354   0.8062  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8553, F1=0.8660, Normal Recall=0.7751, Normal Precision=0.9231, Attack Recall=0.9354, Attack Precision=0.8062

```

