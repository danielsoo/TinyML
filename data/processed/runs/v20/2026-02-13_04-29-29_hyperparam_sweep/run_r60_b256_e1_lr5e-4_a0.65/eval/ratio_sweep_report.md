# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-15 08:30:05 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8754 | 0.8865 | 0.8983 | 0.9115 | 0.9227 | 0.9348 | 0.9469 | 0.9593 | 0.9715 | 0.9836 | 0.9956 |
| QAT+Prune only | 0.5657 | 0.6072 | 0.6495 | 0.6940 | 0.7367 | 0.7771 | 0.8213 | 0.8640 | 0.9067 | 0.9498 | 0.9928 |
| QAT+PTQ | 0.5695 | 0.6108 | 0.6526 | 0.6965 | 0.7389 | 0.7788 | 0.8226 | 0.8645 | 0.9068 | 0.9494 | 0.9920 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.5695 | 0.6108 | 0.6526 | 0.6965 | 0.7389 | 0.7788 | 0.8226 | 0.8645 | 0.9068 | 0.9494 | 0.9920 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.6370 | 0.7966 | 0.8709 | 0.9115 | 0.9385 | 0.9574 | 0.9716 | 0.9824 | 0.9909 | 0.9978 |
| QAT+Prune only | 0.0000 | 0.3358 | 0.5312 | 0.6607 | 0.7511 | 0.8166 | 0.8696 | 0.9109 | 0.9445 | 0.9727 | 0.9964 |
| QAT+PTQ | 0.0000 | 0.3377 | 0.5332 | 0.6623 | 0.7524 | 0.8177 | 0.8703 | 0.9111 | 0.9445 | 0.9724 | 0.9960 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3377 | 0.5332 | 0.6623 | 0.7524 | 0.8177 | 0.8703 | 0.9111 | 0.9445 | 0.9724 | 0.9960 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8754 | 0.8744 | 0.8740 | 0.8755 | 0.8741 | 0.8740 | 0.8739 | 0.8747 | 0.8752 | 0.8758 | 0.0000 |
| QAT+Prune only | 0.5657 | 0.5643 | 0.5637 | 0.5660 | 0.5660 | 0.5613 | 0.5640 | 0.5633 | 0.5622 | 0.5629 | 0.0000 |
| QAT+PTQ | 0.5695 | 0.5684 | 0.5678 | 0.5699 | 0.5701 | 0.5657 | 0.5684 | 0.5671 | 0.5660 | 0.5653 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.5695 | 0.5684 | 0.5678 | 0.5699 | 0.5701 | 0.5657 | 0.5684 | 0.5671 | 0.5660 | 0.5653 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8754 | 0.0000 | 0.0000 | 0.0000 | 0.8754 | 1.0000 |
| 90 | 10 | 299,940 | 0.8865 | 0.4683 | 0.9956 | 0.6370 | 0.8744 | 0.9994 |
| 80 | 20 | 291,350 | 0.8983 | 0.6639 | 0.9956 | 0.7966 | 0.8740 | 0.9987 |
| 70 | 30 | 194,230 | 0.9115 | 0.7741 | 0.9956 | 0.8709 | 0.8755 | 0.9978 |
| 60 | 40 | 145,675 | 0.9227 | 0.8406 | 0.9956 | 0.9115 | 0.8741 | 0.9966 |
| 50 | 50 | 116,540 | 0.9348 | 0.8876 | 0.9956 | 0.9385 | 0.8740 | 0.9949 |
| 40 | 60 | 97,115 | 0.9469 | 0.9221 | 0.9956 | 0.9574 | 0.8739 | 0.9924 |
| 30 | 70 | 83,240 | 0.9593 | 0.9488 | 0.9956 | 0.9716 | 0.8747 | 0.9883 |
| 20 | 80 | 72,835 | 0.9715 | 0.9696 | 0.9956 | 0.9824 | 0.8752 | 0.9801 |
| 10 | 90 | 64,740 | 0.9836 | 0.9863 | 0.9956 | 0.9909 | 0.8758 | 0.9563 |
| 0 | 100 | 58,270 | 0.9956 | 1.0000 | 0.9956 | 0.9978 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5657 | 0.0000 | 0.0000 | 0.0000 | 0.5657 | 1.0000 |
| 90 | 10 | 299,940 | 0.6072 | 0.2021 | 0.9929 | 0.3358 | 0.5643 | 0.9986 |
| 80 | 20 | 291,350 | 0.6495 | 0.3626 | 0.9928 | 0.5312 | 0.5637 | 0.9968 |
| 70 | 30 | 194,230 | 0.6940 | 0.4950 | 0.9928 | 0.6607 | 0.5660 | 0.9946 |
| 60 | 40 | 145,675 | 0.7367 | 0.6040 | 0.9928 | 0.7511 | 0.5660 | 0.9916 |
| 50 | 50 | 116,540 | 0.7771 | 0.6936 | 0.9928 | 0.8166 | 0.5613 | 0.9874 |
| 40 | 60 | 97,115 | 0.8213 | 0.7735 | 0.9928 | 0.8696 | 0.5640 | 0.9812 |
| 30 | 70 | 83,240 | 0.8640 | 0.8414 | 0.9928 | 0.9109 | 0.5633 | 0.9711 |
| 20 | 80 | 72,835 | 0.9067 | 0.9007 | 0.9928 | 0.9445 | 0.5622 | 0.9513 |
| 10 | 90 | 64,740 | 0.9498 | 0.9534 | 0.9928 | 0.9727 | 0.5629 | 0.8969 |
| 0 | 100 | 58,270 | 0.9928 | 1.0000 | 0.9928 | 0.9964 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5695 | 0.0000 | 0.0000 | 0.0000 | 0.5695 | 1.0000 |
| 90 | 10 | 299,940 | 0.6108 | 0.2035 | 0.9922 | 0.3377 | 0.5684 | 0.9985 |
| 80 | 20 | 291,350 | 0.6526 | 0.3646 | 0.9920 | 0.5332 | 0.5678 | 0.9965 |
| 70 | 30 | 194,230 | 0.6965 | 0.4971 | 0.9920 | 0.6623 | 0.5699 | 0.9940 |
| 60 | 40 | 145,675 | 0.7389 | 0.6061 | 0.9920 | 0.7524 | 0.5701 | 0.9908 |
| 50 | 50 | 116,540 | 0.7788 | 0.6955 | 0.9920 | 0.8177 | 0.5657 | 0.9861 |
| 40 | 60 | 97,115 | 0.8226 | 0.7752 | 0.9920 | 0.8703 | 0.5684 | 0.9794 |
| 30 | 70 | 83,240 | 0.8645 | 0.8424 | 0.9920 | 0.9111 | 0.5671 | 0.9682 |
| 20 | 80 | 72,835 | 0.9068 | 0.9014 | 0.9920 | 0.9445 | 0.5660 | 0.9466 |
| 10 | 90 | 64,740 | 0.9494 | 0.9536 | 0.9920 | 0.9724 | 0.5653 | 0.8873 |
| 0 | 100 | 58,270 | 0.9920 | 1.0000 | 0.9920 | 0.9960 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.5695 | 0.0000 | 0.0000 | 0.0000 | 0.5695 | 1.0000 |
| 90 | 10 | 299,940 | 0.6108 | 0.2035 | 0.9922 | 0.3377 | 0.5684 | 0.9985 |
| 80 | 20 | 291,350 | 0.6526 | 0.3646 | 0.9920 | 0.5332 | 0.5678 | 0.9965 |
| 70 | 30 | 194,230 | 0.6965 | 0.4971 | 0.9920 | 0.6623 | 0.5699 | 0.9940 |
| 60 | 40 | 145,675 | 0.7389 | 0.6061 | 0.9920 | 0.7524 | 0.5701 | 0.9908 |
| 50 | 50 | 116,540 | 0.7788 | 0.6955 | 0.9920 | 0.8177 | 0.5657 | 0.9861 |
| 40 | 60 | 97,115 | 0.8226 | 0.7752 | 0.9920 | 0.8703 | 0.5684 | 0.9794 |
| 30 | 70 | 83,240 | 0.8645 | 0.8424 | 0.9920 | 0.9111 | 0.5671 | 0.9682 |
| 20 | 80 | 72,835 | 0.9068 | 0.9014 | 0.9920 | 0.9445 | 0.5660 | 0.9466 |
| 10 | 90 | 64,740 | 0.9494 | 0.9536 | 0.9920 | 0.9724 | 0.5653 | 0.8873 |
| 0 | 100 | 58,270 | 0.9920 | 1.0000 | 0.9920 | 0.9960 | 0.0000 | 0.0000 |


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
0.15       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684   <--
0.20       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.25       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.30       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.35       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.40       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.45       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.50       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.55       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.60       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.65       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.70       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.75       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
0.80       0.8866   0.6371   0.8744   0.9995   0.9958   0.4684  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8866, F1=0.6371, Normal Recall=0.8744, Normal Precision=0.9995, Attack Recall=0.9958, Attack Precision=0.4684

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
0.15       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656   <--
0.20       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.25       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.30       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.35       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.40       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.45       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.50       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.55       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.60       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.65       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.70       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.75       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
0.80       0.8991   0.7978   0.8749   0.9987   0.9956   0.6656  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8991, F1=0.7978, Normal Recall=0.8749, Normal Precision=0.9987, Attack Recall=0.9956, Attack Precision=0.6656

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
0.15       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743   <--
0.20       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.25       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.30       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.35       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.40       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.45       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.50       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.55       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.60       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.65       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.70       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.75       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
0.80       0.9116   0.8711   0.8756   0.9978   0.9956   0.7743  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9116, F1=0.8711, Normal Recall=0.8756, Normal Precision=0.9978, Attack Recall=0.9956, Attack Precision=0.7743

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
0.15       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422   <--
0.20       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.25       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.30       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.35       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.40       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.45       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.50       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.55       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.60       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.65       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.70       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.75       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
0.80       0.9236   0.9125   0.8756   0.9966   0.9956   0.8422  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9236, F1=0.9125, Normal Recall=0.8756, Normal Precision=0.9966, Attack Recall=0.9956, Attack Precision=0.8422

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
0.15       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886   <--
0.20       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.25       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.30       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.35       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.40       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.45       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.50       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.55       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.60       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.65       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.70       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.75       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
0.80       0.9354   0.9391   0.8752   0.9949   0.9956   0.8886  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9354, F1=0.9391, Normal Recall=0.8752, Normal Precision=0.9949, Attack Recall=0.9956, Attack Precision=0.8886

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
0.15       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020   <--
0.20       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.25       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.30       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.35       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.40       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.45       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.50       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.55       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.60       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.65       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.70       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.75       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
0.80       0.6071   0.3357   0.5643   0.9985   0.9925   0.2020  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6071, F1=0.3357, Normal Recall=0.5643, Normal Precision=0.9985, Attack Recall=0.9925, Attack Precision=0.2020

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
0.15       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632   <--
0.20       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.25       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.30       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.35       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.40       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.45       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.50       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.55       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.60       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.65       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.70       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.75       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
0.80       0.6504   0.5318   0.5647   0.9968   0.9928   0.3632  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6504, F1=0.5318, Normal Recall=0.5647, Normal Precision=0.9968, Attack Recall=0.9928, Attack Precision=0.3632

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
0.15       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949   <--
0.20       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.25       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.30       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.35       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.40       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.45       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.50       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.55       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.60       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.65       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.70       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.75       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
0.80       0.6938   0.6605   0.5657   0.9946   0.9928   0.4949  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6938, F1=0.6605, Normal Recall=0.5657, Normal Precision=0.9946, Attack Recall=0.9928, Attack Precision=0.4949

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
0.15       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040   <--
0.20       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.25       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.30       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.35       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.40       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.45       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.50       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.55       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.60       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.65       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.70       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.75       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
0.80       0.7367   0.7510   0.5660   0.9916   0.9928   0.6040  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7367, F1=0.7510, Normal Recall=0.5660, Normal Precision=0.9916, Attack Recall=0.9928, Attack Precision=0.6040

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
0.15       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957   <--
0.20       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.25       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.30       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.35       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.40       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.45       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.50       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.55       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.60       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.65       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.70       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.75       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
0.80       0.7793   0.8181   0.5657   0.9874   0.9928   0.6957  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7793, F1=0.8181, Normal Recall=0.5657, Normal Precision=0.9874, Attack Recall=0.9928, Attack Precision=0.6957

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
0.15       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034   <--
0.20       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.25       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.30       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.35       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.40       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.45       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.50       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.55       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.60       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.65       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.70       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.75       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.80       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6108, F1=0.3375, Normal Recall=0.5685, Normal Precision=0.9983, Attack Recall=0.9915, Attack Precision=0.2034

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
0.15       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652   <--
0.20       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.25       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.30       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.35       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.40       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.45       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.50       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.55       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.60       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.65       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.70       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.75       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.80       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6535, F1=0.5338, Normal Recall=0.5689, Normal Precision=0.9965, Attack Recall=0.9920, Attack Precision=0.3652

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
0.15       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969   <--
0.20       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.25       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.30       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.35       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.40       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.45       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.50       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.55       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.60       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.65       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.70       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.75       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.80       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6962, F1=0.6621, Normal Recall=0.5695, Normal Precision=0.9940, Attack Recall=0.9920, Attack Precision=0.4969

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
0.15       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060   <--
0.20       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.25       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.30       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.35       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.40       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.45       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.50       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.55       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.60       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.65       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.70       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.75       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.80       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7388, F1=0.7524, Normal Recall=0.5699, Normal Precision=0.9908, Attack Recall=0.9920, Attack Precision=0.6060

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
0.15       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975   <--
0.20       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.25       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.30       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.35       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.40       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.45       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.50       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.55       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.60       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.65       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.70       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.75       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.80       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7809, F1=0.8191, Normal Recall=0.5698, Normal Precision=0.9862, Attack Recall=0.9920, Attack Precision=0.6975

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
0.15       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034   <--
0.20       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.25       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.30       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.35       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.40       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.45       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.50       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.55       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.60       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.65       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.70       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.75       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
0.80       0.6108   0.3375   0.5685   0.9983   0.9915   0.2034  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6108, F1=0.3375, Normal Recall=0.5685, Normal Precision=0.9983, Attack Recall=0.9915, Attack Precision=0.2034

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
0.15       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652   <--
0.20       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.25       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.30       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.35       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.40       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.45       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.50       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.55       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.60       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.65       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.70       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.75       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
0.80       0.6535   0.5338   0.5689   0.9965   0.9920   0.3652  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6535, F1=0.5338, Normal Recall=0.5689, Normal Precision=0.9965, Attack Recall=0.9920, Attack Precision=0.3652

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
0.15       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969   <--
0.20       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.25       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.30       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.35       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.40       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.45       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.50       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.55       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.60       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.65       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.70       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.75       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
0.80       0.6962   0.6621   0.5695   0.9940   0.9920   0.4969  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6962, F1=0.6621, Normal Recall=0.5695, Normal Precision=0.9940, Attack Recall=0.9920, Attack Precision=0.4969

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
0.15       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060   <--
0.20       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.25       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.30       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.35       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.40       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.45       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.50       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.55       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.60       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.65       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.70       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.75       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
0.80       0.7388   0.7524   0.5699   0.9908   0.9920   0.6060  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7388, F1=0.7524, Normal Recall=0.5699, Normal Precision=0.9908, Attack Recall=0.9920, Attack Precision=0.6060

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
0.15       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975   <--
0.20       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.25       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.30       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.35       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.40       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.45       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.50       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.55       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.60       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.65       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.70       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.75       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
0.80       0.7809   0.8191   0.5698   0.9862   0.9920   0.6975  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7809, F1=0.8191, Normal Recall=0.5698, Normal Precision=0.9862, Attack Recall=0.9920, Attack Precision=0.6975

```

