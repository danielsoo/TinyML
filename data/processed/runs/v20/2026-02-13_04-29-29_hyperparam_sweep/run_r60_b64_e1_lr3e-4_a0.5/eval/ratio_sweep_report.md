# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-13 05:16:12 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8763 | 0.8894 | 0.9010 | 0.9138 | 0.9251 | 0.9367 | 0.9490 | 0.9615 | 0.9732 | 0.9849 | 0.9973 |
| QAT+Prune only | 0.7999 | 0.8162 | 0.8308 | 0.8449 | 0.8596 | 0.8734 | 0.8892 | 0.9038 | 0.9199 | 0.9326 | 0.9486 |
| QAT+PTQ | 0.7988 | 0.8154 | 0.8302 | 0.8444 | 0.8593 | 0.8731 | 0.8893 | 0.9040 | 0.9202 | 0.9331 | 0.9493 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7988 | 0.8154 | 0.8302 | 0.8444 | 0.8593 | 0.8731 | 0.8893 | 0.9040 | 0.9202 | 0.9331 | 0.9493 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6433 | 0.8012 | 0.8741 | 0.9141 | 0.9403 | 0.9591 | 0.9732 | 0.9835 | 0.9916 | 0.9987 |
| QAT+Prune only | 0.0000 | 0.5080 | 0.6916 | 0.7858 | 0.8439 | 0.8822 | 0.9113 | 0.9324 | 0.9498 | 0.9620 | 0.9736 |
| QAT+PTQ | 0.0000 | 0.5071 | 0.6909 | 0.7854 | 0.8437 | 0.8821 | 0.9114 | 0.9326 | 0.9501 | 0.9623 | 0.9740 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5071 | 0.6909 | 0.7854 | 0.8437 | 0.8821 | 0.9114 | 0.9326 | 0.9501 | 0.9623 | 0.9740 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8763 | 0.8774 | 0.8770 | 0.8780 | 0.8769 | 0.8760 | 0.8765 | 0.8781 | 0.8765 | 0.8727 | 0.0000 |
| QAT+Prune only | 0.7999 | 0.8014 | 0.8014 | 0.8004 | 0.8003 | 0.7982 | 0.8001 | 0.7992 | 0.8048 | 0.7890 | 0.0000 |
| QAT+PTQ | 0.7988 | 0.8004 | 0.8004 | 0.7995 | 0.7993 | 0.7969 | 0.7993 | 0.7983 | 0.8039 | 0.7878 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7988 | 0.8004 | 0.8004 | 0.7995 | 0.7993 | 0.7969 | 0.7993 | 0.7983 | 0.8039 | 0.7878 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8763 | 0.0000 | 0.0000 | 0.0000 | 0.8763 | 1.0000 |
| 90 | 10 | 299,940 | 0.8894 | 0.4747 | 0.9974 | 0.6433 | 0.8774 | 0.9997 |
| 80 | 20 | 291,350 | 0.9010 | 0.6696 | 0.9973 | 0.8012 | 0.8770 | 0.9992 |
| 70 | 30 | 194,230 | 0.9138 | 0.7780 | 0.9973 | 0.8741 | 0.8780 | 0.9987 |
| 60 | 40 | 145,675 | 0.9251 | 0.8438 | 0.9973 | 0.9141 | 0.8769 | 0.9980 |
| 50 | 50 | 116,540 | 0.9367 | 0.8895 | 0.9973 | 0.9403 | 0.8760 | 0.9970 |
| 40 | 60 | 97,115 | 0.9490 | 0.9237 | 0.9973 | 0.9591 | 0.8765 | 0.9954 |
| 30 | 70 | 83,240 | 0.9615 | 0.9502 | 0.9973 | 0.9732 | 0.8781 | 0.9929 |
| 20 | 80 | 72,835 | 0.9732 | 0.9700 | 0.9973 | 0.9835 | 0.8765 | 0.9879 |
| 10 | 90 | 64,740 | 0.9849 | 0.9860 | 0.9973 | 0.9916 | 0.8727 | 0.9731 |
| 0 | 100 | 58,270 | 0.9973 | 1.0000 | 0.9973 | 0.9987 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7999 | 0.0000 | 0.0000 | 0.0000 | 0.7999 | 1.0000 |
| 90 | 10 | 299,940 | 0.8162 | 0.3468 | 0.9490 | 0.5080 | 0.8014 | 0.9930 |
| 80 | 20 | 291,350 | 0.8308 | 0.5442 | 0.9486 | 0.6916 | 0.8014 | 0.9842 |
| 70 | 30 | 194,230 | 0.8449 | 0.6707 | 0.9486 | 0.7858 | 0.8004 | 0.9732 |
| 60 | 40 | 145,675 | 0.8596 | 0.7600 | 0.9486 | 0.8439 | 0.8003 | 0.9589 |
| 50 | 50 | 116,540 | 0.8734 | 0.8246 | 0.9486 | 0.8822 | 0.7982 | 0.9395 |
| 40 | 60 | 97,115 | 0.8892 | 0.8768 | 0.9486 | 0.9113 | 0.8001 | 0.9121 |
| 30 | 70 | 83,240 | 0.9038 | 0.9168 | 0.9486 | 0.9324 | 0.7992 | 0.8695 |
| 20 | 80 | 72,835 | 0.9199 | 0.9511 | 0.9486 | 0.9498 | 0.8048 | 0.7966 |
| 10 | 90 | 64,740 | 0.9326 | 0.9759 | 0.9486 | 0.9620 | 0.7890 | 0.6304 |
| 0 | 100 | 58,270 | 0.9486 | 1.0000 | 0.9486 | 0.9736 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7988 | 0.0000 | 0.0000 | 0.0000 | 0.7988 | 1.0000 |
| 90 | 10 | 299,940 | 0.8154 | 0.3459 | 0.9498 | 0.5071 | 0.8004 | 0.9931 |
| 80 | 20 | 291,350 | 0.8302 | 0.5431 | 0.9493 | 0.6909 | 0.8004 | 0.9844 |
| 70 | 30 | 194,230 | 0.8444 | 0.6698 | 0.9493 | 0.7854 | 0.7995 | 0.9735 |
| 60 | 40 | 145,675 | 0.8593 | 0.7592 | 0.9493 | 0.8437 | 0.7993 | 0.9594 |
| 50 | 50 | 116,540 | 0.8731 | 0.8237 | 0.9493 | 0.8821 | 0.7969 | 0.9402 |
| 40 | 60 | 97,115 | 0.8893 | 0.8764 | 0.9493 | 0.9114 | 0.7993 | 0.9131 |
| 30 | 70 | 83,240 | 0.9040 | 0.9165 | 0.9493 | 0.9326 | 0.7983 | 0.8709 |
| 20 | 80 | 72,835 | 0.9202 | 0.9509 | 0.9493 | 0.9501 | 0.8039 | 0.7986 |
| 10 | 90 | 64,740 | 0.9331 | 0.9758 | 0.9493 | 0.9623 | 0.7878 | 0.6332 |
| 0 | 100 | 58,270 | 0.9493 | 1.0000 | 0.9493 | 0.9740 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7988 | 0.0000 | 0.0000 | 0.0000 | 0.7988 | 1.0000 |
| 90 | 10 | 299,940 | 0.8154 | 0.3459 | 0.9498 | 0.5071 | 0.8004 | 0.9931 |
| 80 | 20 | 291,350 | 0.8302 | 0.5431 | 0.9493 | 0.6909 | 0.8004 | 0.9844 |
| 70 | 30 | 194,230 | 0.8444 | 0.6698 | 0.9493 | 0.7854 | 0.7995 | 0.9735 |
| 60 | 40 | 145,675 | 0.8593 | 0.7592 | 0.9493 | 0.8437 | 0.7993 | 0.9594 |
| 50 | 50 | 116,540 | 0.8731 | 0.8237 | 0.9493 | 0.8821 | 0.7969 | 0.9402 |
| 40 | 60 | 97,115 | 0.8893 | 0.8764 | 0.9493 | 0.9114 | 0.7993 | 0.9131 |
| 30 | 70 | 83,240 | 0.9040 | 0.9165 | 0.9493 | 0.9326 | 0.7983 | 0.8709 |
| 20 | 80 | 72,835 | 0.9202 | 0.9509 | 0.9493 | 0.9501 | 0.8039 | 0.7986 |
| 10 | 90 | 64,740 | 0.9331 | 0.9758 | 0.9493 | 0.9623 | 0.7878 | 0.6332 |
| 0 | 100 | 58,270 | 0.9493 | 1.0000 | 0.9493 | 0.9740 | 0.0000 | 0.0000 |


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
0.15       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748   <--
0.20       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.25       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.30       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.35       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.40       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.45       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.50       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.55       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.60       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.65       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.70       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.75       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
0.80       0.8894   0.6434   0.8774   0.9997   0.9976   0.4748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8894, F1=0.6434, Normal Recall=0.8774, Normal Precision=0.9997, Attack Recall=0.9976, Attack Precision=0.4748

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
0.15       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709   <--
0.20       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.25       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.30       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.35       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.40       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.45       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.50       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.55       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.60       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.65       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.70       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.75       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
0.80       0.9016   0.8022   0.8777   0.9992   0.9973   0.6709  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9016, F1=0.8022, Normal Recall=0.8777, Normal Precision=0.9992, Attack Recall=0.9973, Attack Precision=0.6709

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
0.15       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770   <--
0.20       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.25       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.30       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.35       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.40       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.45       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.50       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.55       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.60       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.65       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.70       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.75       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
0.80       0.9133   0.8735   0.8773   0.9987   0.9973   0.7770  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9133, F1=0.8735, Normal Recall=0.8773, Normal Precision=0.9987, Attack Recall=0.9973, Attack Precision=0.7770

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
0.15       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433   <--
0.20       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.25       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.30       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.35       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.40       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.45       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.50       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.55       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.60       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.65       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.70       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.75       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
0.80       0.9248   0.9139   0.8765   0.9980   0.9973   0.8433  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9248, F1=0.9139, Normal Recall=0.8765, Normal Precision=0.9980, Attack Recall=0.9973, Attack Precision=0.8433

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
0.15       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902   <--
0.20       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.25       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.30       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.35       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.40       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.45       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.50       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.55       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.60       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.65       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.70       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.75       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
0.80       0.9371   0.9407   0.8769   0.9970   0.9973   0.8902  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9371, F1=0.9407, Normal Recall=0.8769, Normal Precision=0.9970, Attack Recall=0.9973, Attack Precision=0.8902

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
0.15       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470   <--
0.20       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.25       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.30       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.35       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.40       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.45       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.50       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.55       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.60       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.65       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.70       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.75       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
0.80       0.8163   0.5083   0.8014   0.9931   0.9496   0.3470  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8163, F1=0.5083, Normal Recall=0.8014, Normal Precision=0.9931, Attack Recall=0.9496, Attack Precision=0.3470

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
0.15       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445   <--
0.20       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.25       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.30       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.35       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.40       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.45       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.50       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.55       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.60       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.65       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.70       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.75       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
0.80       0.8310   0.6918   0.8016   0.9842   0.9486   0.5445  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8310, F1=0.6918, Normal Recall=0.8016, Normal Precision=0.9842, Attack Recall=0.9486, Attack Precision=0.5445

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
0.15       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710   <--
0.20       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.25       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.30       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.35       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.40       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.45       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.50       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.55       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.60       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.65       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.70       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.75       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
0.80       0.8450   0.7860   0.8006   0.9732   0.9486   0.6710  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8450, F1=0.7860, Normal Recall=0.8006, Normal Precision=0.9732, Attack Recall=0.9486, Attack Precision=0.6710

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
0.15       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599   <--
0.20       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.25       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.30       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.35       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.40       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.45       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.50       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.55       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.60       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.65       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.70       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.75       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
0.80       0.8595   0.8438   0.8001   0.9589   0.9486   0.7599  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8595, F1=0.8438, Normal Recall=0.8001, Normal Precision=0.9589, Attack Recall=0.9486, Attack Precision=0.7599

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
0.15       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263   <--
0.20       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.25       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.30       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.35       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.40       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.45       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.50       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.55       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.60       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.65       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.70       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.75       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
0.80       0.8746   0.8833   0.8006   0.9397   0.9486   0.8263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8746, F1=0.8833, Normal Recall=0.8006, Normal Precision=0.9397, Attack Recall=0.9486, Attack Precision=0.8263

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
0.15       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460   <--
0.20       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.25       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.30       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.35       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.40       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.45       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.50       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.55       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.60       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.65       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.70       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.75       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.80       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8154, F1=0.5073, Normal Recall=0.8004, Normal Precision=0.9932, Attack Recall=0.9503, Attack Precision=0.3460

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
0.15       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434   <--
0.20       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.25       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.30       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.35       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.40       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.45       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.50       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.55       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.60       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.65       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.70       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.75       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.80       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8303, F1=0.6912, Normal Recall=0.8006, Normal Precision=0.9844, Attack Recall=0.9493, Attack Precision=0.5434

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
0.15       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700   <--
0.20       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.25       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.30       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.35       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.40       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.45       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.50       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.55       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.60       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.65       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.70       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.75       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.80       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8445, F1=0.7856, Normal Recall=0.7996, Normal Precision=0.9735, Attack Recall=0.9493, Attack Precision=0.6700

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
0.15       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590   <--
0.20       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.25       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.30       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.35       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.40       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.45       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.50       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.55       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.60       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.65       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.70       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.75       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.80       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8591, F1=0.8435, Normal Recall=0.7990, Normal Precision=0.9594, Attack Recall=0.9493, Attack Precision=0.7590

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
0.15       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256   <--
0.20       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.25       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.30       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.35       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.40       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.45       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.50       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.55       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.60       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.65       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.70       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.75       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.80       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8744, F1=0.8832, Normal Recall=0.7995, Normal Precision=0.9404, Attack Recall=0.9493, Attack Precision=0.8256

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
0.15       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460   <--
0.20       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.25       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.30       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.35       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.40       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.45       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.50       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.55       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.60       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.65       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.70       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.75       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
0.80       0.8154   0.5073   0.8004   0.9932   0.9503   0.3460  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8154, F1=0.5073, Normal Recall=0.8004, Normal Precision=0.9932, Attack Recall=0.9503, Attack Precision=0.3460

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
0.15       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434   <--
0.20       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.25       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.30       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.35       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.40       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.45       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.50       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.55       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.60       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.65       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.70       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.75       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
0.80       0.8303   0.6912   0.8006   0.9844   0.9493   0.5434  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8303, F1=0.6912, Normal Recall=0.8006, Normal Precision=0.9844, Attack Recall=0.9493, Attack Precision=0.5434

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
0.15       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700   <--
0.20       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.25       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.30       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.35       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.40       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.45       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.50       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.55       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.60       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.65       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.70       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.75       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
0.80       0.8445   0.7856   0.7996   0.9735   0.9493   0.6700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8445, F1=0.7856, Normal Recall=0.7996, Normal Precision=0.9735, Attack Recall=0.9493, Attack Precision=0.6700

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
0.15       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590   <--
0.20       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.25       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.30       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.35       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.40       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.45       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.50       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.55       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.60       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.65       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.70       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.75       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
0.80       0.8591   0.8435   0.7990   0.9594   0.9493   0.7590  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8591, F1=0.8435, Normal Recall=0.7990, Normal Precision=0.9594, Attack Recall=0.9493, Attack Precision=0.7590

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
0.15       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256   <--
0.20       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.25       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.30       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.35       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.40       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.45       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.50       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.55       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.60       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.65       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.70       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.75       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
0.80       0.8744   0.8832   0.7995   0.9404   0.9493   0.8256  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8744, F1=0.8832, Normal Recall=0.7995, Normal Precision=0.9404, Attack Recall=0.9493, Attack Precision=0.8256

```

