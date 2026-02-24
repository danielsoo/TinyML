# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e2_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-17 18:28:44 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 80 |
| **Local epochs** | 2 |
| **Batch size** | 128 |
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2888 | 0.3588 | 0.4306 | 0.5020 | 0.5723 | 0.6430 | 0.7149 | 0.7859 | 0.8567 | 0.9289 | 0.9996 |
| QAT+Prune only | 0.9677 | 0.9495 | 0.9311 | 0.9129 | 0.8945 | 0.8763 | 0.8584 | 0.8395 | 0.8212 | 0.8033 | 0.7850 |
| QAT+PTQ | 0.9669 | 0.9487 | 0.9303 | 0.9123 | 0.8939 | 0.8757 | 0.8578 | 0.8390 | 0.8208 | 0.8029 | 0.7846 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9669 | 0.9487 | 0.9303 | 0.9123 | 0.8939 | 0.8757 | 0.8578 | 0.8390 | 0.8208 | 0.8029 | 0.7846 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2377 | 0.4125 | 0.5463 | 0.6515 | 0.7368 | 0.8080 | 0.8673 | 0.9178 | 0.9620 | 0.9998 |
| QAT+Prune only | 0.0000 | 0.7568 | 0.8200 | 0.8440 | 0.8561 | 0.8639 | 0.8693 | 0.8726 | 0.8754 | 0.8778 | 0.8795 |
| QAT+PTQ | 0.0000 | 0.7540 | 0.8184 | 0.8430 | 0.8554 | 0.8633 | 0.8688 | 0.8722 | 0.8751 | 0.8775 | 0.8793 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7540 | 0.8184 | 0.8430 | 0.8554 | 0.8633 | 0.8688 | 0.8722 | 0.8751 | 0.8775 | 0.8793 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.2888 | 0.2876 | 0.2883 | 0.2887 | 0.2874 | 0.2864 | 0.2879 | 0.2874 | 0.2855 | 0.2927 | 0.0000 |
| QAT+Prune only | 0.9677 | 0.9677 | 0.9676 | 0.9678 | 0.9675 | 0.9677 | 0.9686 | 0.9668 | 0.9660 | 0.9683 | 0.0000 |
| QAT+PTQ | 0.9669 | 0.9669 | 0.9668 | 0.9670 | 0.9668 | 0.9668 | 0.9676 | 0.9660 | 0.9655 | 0.9674 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9669 | 0.9669 | 0.9668 | 0.9670 | 0.9668 | 0.9668 | 0.9676 | 0.9660 | 0.9655 | 0.9674 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.2888 | 0.0000 | 0.0000 | 0.0000 | 0.2888 | 1.0000 |
| 90 | 10 | 299,940 | 0.3588 | 0.1349 | 0.9996 | 0.2377 | 0.2876 | 0.9999 |
| 80 | 20 | 291,350 | 0.4306 | 0.2599 | 0.9996 | 0.4125 | 0.2883 | 0.9996 |
| 70 | 30 | 194,230 | 0.5020 | 0.3759 | 0.9996 | 0.5463 | 0.2887 | 0.9993 |
| 60 | 40 | 145,675 | 0.5723 | 0.4833 | 0.9996 | 0.6515 | 0.2874 | 0.9990 |
| 50 | 50 | 116,540 | 0.6430 | 0.5835 | 0.9996 | 0.7368 | 0.2864 | 0.9984 |
| 40 | 60 | 97,115 | 0.7149 | 0.6780 | 0.9996 | 0.8080 | 0.2879 | 0.9977 |
| 30 | 70 | 83,240 | 0.7859 | 0.7660 | 0.9996 | 0.8673 | 0.2874 | 0.9964 |
| 20 | 80 | 72,835 | 0.8567 | 0.8484 | 0.9996 | 0.9178 | 0.2855 | 0.9938 |
| 10 | 90 | 64,740 | 0.9289 | 0.9271 | 0.9996 | 0.9620 | 0.2927 | 0.9865 |
| 0 | 100 | 58,270 | 0.9996 | 1.0000 | 0.9996 | 0.9998 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9677 | 0.0000 | 0.0000 | 0.0000 | 0.9677 | 1.0000 |
| 90 | 10 | 299,940 | 0.9495 | 0.7299 | 0.7859 | 0.7568 | 0.9677 | 0.9760 |
| 80 | 20 | 291,350 | 0.9311 | 0.8582 | 0.7850 | 0.8200 | 0.9676 | 0.9474 |
| 70 | 30 | 194,230 | 0.9129 | 0.9126 | 0.7850 | 0.8440 | 0.9678 | 0.9131 |
| 60 | 40 | 145,675 | 0.8945 | 0.9414 | 0.7850 | 0.8561 | 0.9675 | 0.8709 |
| 50 | 50 | 116,540 | 0.8763 | 0.9605 | 0.7850 | 0.8639 | 0.9677 | 0.8182 |
| 40 | 60 | 97,115 | 0.8584 | 0.9740 | 0.7850 | 0.8693 | 0.9686 | 0.7502 |
| 30 | 70 | 83,240 | 0.8395 | 0.9822 | 0.7850 | 0.8726 | 0.9668 | 0.6583 |
| 20 | 80 | 72,835 | 0.8212 | 0.9893 | 0.7850 | 0.8754 | 0.9660 | 0.5290 |
| 10 | 90 | 64,740 | 0.8033 | 0.9955 | 0.7850 | 0.8778 | 0.9683 | 0.3335 |
| 0 | 100 | 58,270 | 0.7850 | 1.0000 | 0.7850 | 0.8795 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9669 | 0.0000 | 0.0000 | 0.0000 | 0.9669 | 1.0000 |
| 90 | 10 | 299,940 | 0.9487 | 0.7249 | 0.7855 | 0.7540 | 0.9669 | 0.9759 |
| 80 | 20 | 291,350 | 0.9303 | 0.8552 | 0.7846 | 0.8184 | 0.9668 | 0.9472 |
| 70 | 30 | 194,230 | 0.9123 | 0.9107 | 0.7846 | 0.8430 | 0.9670 | 0.9129 |
| 60 | 40 | 145,675 | 0.8939 | 0.9403 | 0.7846 | 0.8554 | 0.9668 | 0.8707 |
| 50 | 50 | 116,540 | 0.8757 | 0.9594 | 0.7846 | 0.8633 | 0.9668 | 0.8178 |
| 40 | 60 | 97,115 | 0.8578 | 0.9732 | 0.7846 | 0.8688 | 0.9676 | 0.7497 |
| 30 | 70 | 83,240 | 0.8390 | 0.9818 | 0.7846 | 0.8722 | 0.9660 | 0.6578 |
| 20 | 80 | 72,835 | 0.8208 | 0.9891 | 0.7846 | 0.8751 | 0.9655 | 0.5285 |
| 10 | 90 | 64,740 | 0.8029 | 0.9954 | 0.7846 | 0.8775 | 0.9674 | 0.3329 |
| 0 | 100 | 58,270 | 0.7846 | 1.0000 | 0.7846 | 0.8793 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9669 | 0.0000 | 0.0000 | 0.0000 | 0.9669 | 1.0000 |
| 90 | 10 | 299,940 | 0.9487 | 0.7249 | 0.7855 | 0.7540 | 0.9669 | 0.9759 |
| 80 | 20 | 291,350 | 0.9303 | 0.8552 | 0.7846 | 0.8184 | 0.9668 | 0.9472 |
| 70 | 30 | 194,230 | 0.9123 | 0.9107 | 0.7846 | 0.8430 | 0.9670 | 0.9129 |
| 60 | 40 | 145,675 | 0.8939 | 0.9403 | 0.7846 | 0.8554 | 0.9668 | 0.8707 |
| 50 | 50 | 116,540 | 0.8757 | 0.9594 | 0.7846 | 0.8633 | 0.9668 | 0.8178 |
| 40 | 60 | 97,115 | 0.8578 | 0.9732 | 0.7846 | 0.8688 | 0.9676 | 0.7497 |
| 30 | 70 | 83,240 | 0.8390 | 0.9818 | 0.7846 | 0.8722 | 0.9660 | 0.6578 |
| 20 | 80 | 72,835 | 0.8208 | 0.9891 | 0.7846 | 0.8751 | 0.9655 | 0.5285 |
| 10 | 90 | 64,740 | 0.8029 | 0.9954 | 0.7846 | 0.8775 | 0.9674 | 0.3329 |
| 0 | 100 | 58,270 | 0.7846 | 1.0000 | 0.7846 | 0.8793 | 0.0000 | 0.0000 |


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
0.15       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349   <--
0.20       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.25       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.30       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.35       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.40       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.45       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.50       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.55       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.60       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.65       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.70       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.75       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
0.80       0.3588   0.2377   0.2876   0.9998   0.9995   0.1349  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3588, F1=0.2377, Normal Recall=0.2876, Normal Precision=0.9998, Attack Recall=0.9995, Attack Precision=0.1349

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
0.15       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597   <--
0.20       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.25       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.30       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.35       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.40       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.45       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.50       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.55       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.60       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.65       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.70       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.75       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
0.80       0.4301   0.4123   0.2877   0.9996   0.9996   0.2597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4301, F1=0.4123, Normal Recall=0.2877, Normal Precision=0.9996, Attack Recall=0.9996, Attack Precision=0.2597

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
0.15       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761   <--
0.20       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.25       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.30       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.35       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.40       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.45       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.50       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.55       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.60       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.65       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.70       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.75       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
0.80       0.5024   0.5465   0.2894   0.9993   0.9996   0.3761  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5024, F1=0.5465, Normal Recall=0.2894, Normal Precision=0.9993, Attack Recall=0.9996, Attack Precision=0.3761

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
0.15       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837   <--
0.20       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.25       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.30       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.35       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.40       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.45       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.50       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.55       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.60       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.65       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.70       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.75       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
0.80       0.5731   0.6519   0.2888   0.9990   0.9996   0.4837  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5731, F1=0.6519, Normal Recall=0.2888, Normal Precision=0.9990, Attack Recall=0.9996, Attack Precision=0.4837

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
0.15       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847   <--
0.20       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.25       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.30       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.35       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.40       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.45       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.50       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.55       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.60       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.65       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.70       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.75       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
0.80       0.6449   0.7378   0.2902   0.9985   0.9996   0.5847  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6449, F1=0.7378, Normal Recall=0.2902, Normal Precision=0.9985, Attack Recall=0.9996, Attack Precision=0.5847

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
0.15       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296   <--
0.20       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.25       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.30       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.35       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.40       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.45       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.50       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.55       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.60       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.65       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.70       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.75       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
0.80       0.9494   0.7563   0.9677   0.9759   0.7850   0.7296  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9494, F1=0.7563, Normal Recall=0.9677, Normal Precision=0.9759, Attack Recall=0.7850, Attack Precision=0.7296

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
0.15       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587   <--
0.20       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.25       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.30       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.35       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.40       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.45       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.50       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.55       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.60       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.65       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.70       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.75       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
0.80       0.9312   0.8202   0.9677   0.9474   0.7850   0.8587  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9312, F1=0.8202, Normal Recall=0.9677, Normal Precision=0.9474, Attack Recall=0.7850, Attack Precision=0.8587

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
0.15       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126   <--
0.20       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.25       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.30       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.35       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.40       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.45       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.50       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.55       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.60       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.65       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.70       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.75       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
0.80       0.9129   0.8440   0.9678   0.9131   0.7850   0.9126  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9129, F1=0.8440, Normal Recall=0.9678, Normal Precision=0.9131, Attack Recall=0.7850, Attack Precision=0.9126

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
0.15       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413   <--
0.20       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.25       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.30       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.35       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.40       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.45       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.50       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.55       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.60       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.65       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.70       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.75       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
0.80       0.8944   0.8560   0.9673   0.8709   0.7850   0.9413  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8944, F1=0.8560, Normal Recall=0.9673, Normal Precision=0.8709, Attack Recall=0.7850, Attack Precision=0.9413

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
0.15       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589   <--
0.20       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.25       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.30       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.35       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.40       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.45       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.50       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.55       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.60       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.65       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.70       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.75       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
0.80       0.8757   0.8633   0.9664   0.8180   0.7850   0.9589  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8757, F1=0.8633, Normal Recall=0.9664, Normal Precision=0.8180, Attack Recall=0.7850, Attack Precision=0.9589

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
0.15       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247   <--
0.20       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.25       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.30       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.35       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.40       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.45       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.50       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.55       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.60       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.65       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.70       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.75       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.80       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9486, F1=0.7534, Normal Recall=0.9669, Normal Precision=0.9758, Attack Recall=0.7845, Attack Precision=0.7247

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
0.15       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557   <--
0.20       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.25       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.30       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.35       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.40       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.45       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.50       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.55       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.60       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.65       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.70       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.75       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.80       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9305, F1=0.8186, Normal Recall=0.9669, Normal Precision=0.9473, Attack Recall=0.7846, Attack Precision=0.8557

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
0.15       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106   <--
0.20       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.25       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.30       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.35       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.40       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.45       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.50       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.55       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.60       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.65       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.70       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.75       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.80       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9123, F1=0.8429, Normal Recall=0.9670, Normal Precision=0.9129, Attack Recall=0.7846, Attack Precision=0.9106

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
0.15       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398   <--
0.20       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.25       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.30       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.35       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.40       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.45       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.50       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.55       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.60       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.65       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.70       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.75       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.80       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8937, F1=0.8552, Normal Recall=0.9665, Normal Precision=0.8707, Attack Recall=0.7846, Attack Precision=0.9398

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
0.15       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578   <--
0.20       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.25       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.30       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.35       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.40       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.45       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.50       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.55       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.60       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.65       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.70       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.75       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.80       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8750, F1=0.8626, Normal Recall=0.9655, Normal Precision=0.8176, Attack Recall=0.7846, Attack Precision=0.9578

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
0.15       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247   <--
0.20       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.25       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.30       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.35       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.40       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.45       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.50       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.55       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.60       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.65       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.70       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.75       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
0.80       0.9486   0.7534   0.9669   0.9758   0.7845   0.7247  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9486, F1=0.7534, Normal Recall=0.9669, Normal Precision=0.9758, Attack Recall=0.7845, Attack Precision=0.7247

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
0.15       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557   <--
0.20       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.25       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.30       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.35       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.40       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.45       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.50       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.55       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.60       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.65       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.70       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.75       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
0.80       0.9305   0.8186   0.9669   0.9473   0.7846   0.8557  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9305, F1=0.8186, Normal Recall=0.9669, Normal Precision=0.9473, Attack Recall=0.7846, Attack Precision=0.8557

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
0.15       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106   <--
0.20       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.25       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.30       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.35       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.40       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.45       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.50       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.55       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.60       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.65       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.70       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.75       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
0.80       0.9123   0.8429   0.9670   0.9129   0.7846   0.9106  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9123, F1=0.8429, Normal Recall=0.9670, Normal Precision=0.9129, Attack Recall=0.7846, Attack Precision=0.9106

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
0.15       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398   <--
0.20       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.25       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.30       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.35       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.40       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.45       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.50       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.55       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.60       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.65       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.70       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.75       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
0.80       0.8937   0.8552   0.9665   0.8707   0.7846   0.9398  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8937, F1=0.8552, Normal Recall=0.9665, Normal Precision=0.8707, Attack Recall=0.7846, Attack Precision=0.9398

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
0.15       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578   <--
0.20       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.25       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.30       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.35       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.40       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.45       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.50       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.55       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.60       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.65       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.70       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.75       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
0.80       0.8750   0.8626   0.9655   0.8176   0.7846   0.9578  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8750, F1=0.8626, Normal Recall=0.9655, Normal Precision=0.8176, Attack Recall=0.7846, Attack Precision=0.9578

```

