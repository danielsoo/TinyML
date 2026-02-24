# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr1e-3_a0.65.yaml` |
| **Generated** | 2026-02-14 15:06:35 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4990 | 0.5491 | 0.5984 | 0.6489 | 0.6978 | 0.7480 | 0.7978 | 0.8471 | 0.8976 | 0.9480 | 0.9976 |
| QAT+Prune only | 0.7720 | 0.7949 | 0.8166 | 0.8401 | 0.8622 | 0.8826 | 0.9067 | 0.9294 | 0.9521 | 0.9741 | 0.9970 |
| QAT+PTQ | 0.7727 | 0.7954 | 0.8171 | 0.8406 | 0.8626 | 0.8829 | 0.9069 | 0.9295 | 0.9522 | 0.9741 | 0.9970 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7727 | 0.7954 | 0.8171 | 0.8406 | 0.8626 | 0.8829 | 0.9069 | 0.9295 | 0.9522 | 0.9741 | 0.9970 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3068 | 0.4984 | 0.6303 | 0.7254 | 0.7983 | 0.8555 | 0.9013 | 0.9397 | 0.9718 | 0.9988 |
| QAT+Prune only | 0.0000 | 0.4930 | 0.6850 | 0.7890 | 0.8527 | 0.8946 | 0.9276 | 0.9518 | 0.9708 | 0.9858 | 0.9985 |
| QAT+PTQ | 0.0000 | 0.4937 | 0.6856 | 0.7896 | 0.8530 | 0.8949 | 0.9278 | 0.9519 | 0.9709 | 0.9858 | 0.9985 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4937 | 0.6856 | 0.7896 | 0.8530 | 0.8949 | 0.9278 | 0.9519 | 0.9709 | 0.9858 | 0.9985 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4990 | 0.4992 | 0.4986 | 0.4994 | 0.4980 | 0.4985 | 0.4982 | 0.4958 | 0.4976 | 0.5015 | 0.0000 |
| QAT+Prune only | 0.7720 | 0.7724 | 0.7715 | 0.7728 | 0.7724 | 0.7681 | 0.7712 | 0.7717 | 0.7725 | 0.7680 | 0.0000 |
| QAT+PTQ | 0.7727 | 0.7730 | 0.7722 | 0.7735 | 0.7730 | 0.7688 | 0.7718 | 0.7721 | 0.7733 | 0.7680 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7727 | 0.7730 | 0.7722 | 0.7735 | 0.7730 | 0.7688 | 0.7718 | 0.7721 | 0.7733 | 0.7680 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4990 | 0.0000 | 0.0000 | 0.0000 | 0.4990 | 1.0000 |
| 90 | 10 | 299,940 | 0.5491 | 0.1813 | 0.9978 | 0.3068 | 0.4992 | 0.9995 |
| 80 | 20 | 291,350 | 0.5984 | 0.3322 | 0.9976 | 0.4984 | 0.4986 | 0.9988 |
| 70 | 30 | 194,230 | 0.6489 | 0.4606 | 0.9976 | 0.6303 | 0.4994 | 0.9979 |
| 60 | 40 | 145,675 | 0.6978 | 0.5699 | 0.9976 | 0.7254 | 0.4980 | 0.9968 |
| 50 | 50 | 116,540 | 0.7480 | 0.6654 | 0.9976 | 0.7983 | 0.4985 | 0.9952 |
| 40 | 60 | 97,115 | 0.7978 | 0.7489 | 0.9976 | 0.8555 | 0.4982 | 0.9928 |
| 30 | 70 | 83,240 | 0.8471 | 0.8220 | 0.9976 | 0.9013 | 0.4958 | 0.9887 |
| 20 | 80 | 72,835 | 0.8976 | 0.8882 | 0.9976 | 0.9397 | 0.4976 | 0.9809 |
| 10 | 90 | 64,740 | 0.9480 | 0.9474 | 0.9976 | 0.9718 | 0.5015 | 0.9584 |
| 0 | 100 | 58,270 | 0.9976 | 1.0000 | 0.9976 | 0.9988 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7720 | 0.0000 | 0.0000 | 0.0000 | 0.7720 | 1.0000 |
| 90 | 10 | 299,940 | 0.7949 | 0.3274 | 0.9973 | 0.4930 | 0.7724 | 0.9996 |
| 80 | 20 | 291,350 | 0.8166 | 0.5218 | 0.9970 | 0.6850 | 0.7715 | 0.9990 |
| 70 | 30 | 194,230 | 0.8401 | 0.6529 | 0.9970 | 0.7890 | 0.7728 | 0.9983 |
| 60 | 40 | 145,675 | 0.8622 | 0.7449 | 0.9970 | 0.8527 | 0.7724 | 0.9974 |
| 50 | 50 | 116,540 | 0.8826 | 0.8113 | 0.9970 | 0.8946 | 0.7681 | 0.9961 |
| 40 | 60 | 97,115 | 0.9067 | 0.8673 | 0.9970 | 0.9276 | 0.7712 | 0.9942 |
| 30 | 70 | 83,240 | 0.9294 | 0.9106 | 0.9970 | 0.9518 | 0.7717 | 0.9909 |
| 20 | 80 | 72,835 | 0.9521 | 0.9460 | 0.9970 | 0.9708 | 0.7725 | 0.9846 |
| 10 | 90 | 64,740 | 0.9741 | 0.9748 | 0.9970 | 0.9858 | 0.7680 | 0.9658 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7727 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 1.0000 |
| 90 | 10 | 299,940 | 0.7954 | 0.3280 | 0.9973 | 0.4937 | 0.7730 | 0.9996 |
| 80 | 20 | 291,350 | 0.8171 | 0.5224 | 0.9970 | 0.6856 | 0.7722 | 0.9990 |
| 70 | 30 | 194,230 | 0.8406 | 0.6536 | 0.9970 | 0.7896 | 0.7735 | 0.9983 |
| 60 | 40 | 145,675 | 0.8626 | 0.7454 | 0.9970 | 0.8530 | 0.7730 | 0.9974 |
| 50 | 50 | 116,540 | 0.8829 | 0.8118 | 0.9970 | 0.8949 | 0.7688 | 0.9961 |
| 40 | 60 | 97,115 | 0.9069 | 0.8676 | 0.9970 | 0.9278 | 0.7718 | 0.9942 |
| 30 | 70 | 83,240 | 0.9295 | 0.9108 | 0.9970 | 0.9519 | 0.7721 | 0.9910 |
| 20 | 80 | 72,835 | 0.9522 | 0.9462 | 0.9970 | 0.9709 | 0.7733 | 0.9847 |
| 10 | 90 | 64,740 | 0.9741 | 0.9748 | 0.9970 | 0.9858 | 0.7680 | 0.9660 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7727 | 0.0000 | 0.0000 | 0.0000 | 0.7727 | 1.0000 |
| 90 | 10 | 299,940 | 0.7954 | 0.3280 | 0.9973 | 0.4937 | 0.7730 | 0.9996 |
| 80 | 20 | 291,350 | 0.8171 | 0.5224 | 0.9970 | 0.6856 | 0.7722 | 0.9990 |
| 70 | 30 | 194,230 | 0.8406 | 0.6536 | 0.9970 | 0.7896 | 0.7735 | 0.9983 |
| 60 | 40 | 145,675 | 0.8626 | 0.7454 | 0.9970 | 0.8530 | 0.7730 | 0.9974 |
| 50 | 50 | 116,540 | 0.8829 | 0.8118 | 0.9970 | 0.8949 | 0.7688 | 0.9961 |
| 40 | 60 | 97,115 | 0.9069 | 0.8676 | 0.9970 | 0.9278 | 0.7718 | 0.9942 |
| 30 | 70 | 83,240 | 0.9295 | 0.9108 | 0.9970 | 0.9519 | 0.7721 | 0.9910 |
| 20 | 80 | 72,835 | 0.9522 | 0.9462 | 0.9970 | 0.9709 | 0.7733 | 0.9847 |
| 10 | 90 | 64,740 | 0.9741 | 0.9748 | 0.9970 | 0.9858 | 0.7680 | 0.9660 |
| 0 | 100 | 58,270 | 0.9970 | 1.0000 | 0.9970 | 0.9985 | 0.0000 | 0.0000 |


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
0.15       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813   <--
0.20       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.25       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.30       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.35       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.40       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.45       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.50       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.55       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.60       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.65       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.70       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.75       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
0.80       0.5491   0.3068   0.4992   0.9995   0.9978   0.1813  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5491, F1=0.3068, Normal Recall=0.4992, Normal Precision=0.9995, Attack Recall=0.9978, Attack Precision=0.1813

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
0.15       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327   <--
0.20       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.25       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.30       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.35       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.40       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.45       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.50       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.55       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.60       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.65       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.70       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.75       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
0.80       0.5993   0.4990   0.4997   0.9988   0.9976   0.3327  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5993, F1=0.4990, Normal Recall=0.4997, Normal Precision=0.9988, Attack Recall=0.9976, Attack Precision=0.3327

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
0.15       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606   <--
0.20       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.25       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.30       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.35       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.40       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.45       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.50       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.55       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.60       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.65       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.70       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.75       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
0.80       0.6487   0.6302   0.4992   0.9979   0.9976   0.4606  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6487, F1=0.6302, Normal Recall=0.4992, Normal Precision=0.9979, Attack Recall=0.9976, Attack Precision=0.4606

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
0.15       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702   <--
0.20       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.25       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.30       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.35       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.40       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.45       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.50       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.55       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.60       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.65       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.70       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.75       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
0.80       0.6983   0.7257   0.4988   0.9968   0.9976   0.5702  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6983, F1=0.7257, Normal Recall=0.4988, Normal Precision=0.9968, Attack Recall=0.9976, Attack Precision=0.5702

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
0.15       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653   <--
0.20       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.25       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.30       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.35       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.40       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.45       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.50       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.55       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.60       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.65       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.70       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.75       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
0.80       0.7478   0.7982   0.4980   0.9952   0.9976   0.6653  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7478, F1=0.7982, Normal Recall=0.4980, Normal Precision=0.9952, Attack Recall=0.9976, Attack Precision=0.6653

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
0.15       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274   <--
0.20       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.25       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.30       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.35       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.40       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.45       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.50       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.55       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.60       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.65       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.70       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.75       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
0.80       0.7949   0.4930   0.7724   0.9996   0.9972   0.3274  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7949, F1=0.4930, Normal Recall=0.7724, Normal Precision=0.9996, Attack Recall=0.9972, Attack Precision=0.3274

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
0.15       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235   <--
0.20       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.25       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.30       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.35       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.40       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.45       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.50       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.55       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.60       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.65       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.70       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.75       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
0.80       0.8179   0.6866   0.7732   0.9990   0.9970   0.5235  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8179, F1=0.6866, Normal Recall=0.7732, Normal Precision=0.9990, Attack Recall=0.9970, Attack Precision=0.5235

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
0.15       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525   <--
0.20       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.25       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.30       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.35       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.40       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.45       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.50       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.55       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.60       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.65       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.70       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.75       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
0.80       0.8398   0.7887   0.7724   0.9983   0.9970   0.6525  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8398, F1=0.7887, Normal Recall=0.7724, Normal Precision=0.9983, Attack Recall=0.9970, Attack Precision=0.6525

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
0.15       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444   <--
0.20       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.25       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.30       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.35       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.40       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.45       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.50       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.55       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.60       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.65       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.70       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.75       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
0.80       0.8619   0.8524   0.7718   0.9974   0.9970   0.7444  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8619, F1=0.8524, Normal Recall=0.7718, Normal Precision=0.9974, Attack Recall=0.9970, Attack Precision=0.7444

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
0.15       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123   <--
0.20       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.25       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.30       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.35       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.40       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.45       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.50       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.55       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.60       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.65       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.70       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.75       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
0.80       0.8833   0.8952   0.7696   0.9961   0.9970   0.8123  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8833, F1=0.8952, Normal Recall=0.7696, Normal Precision=0.9961, Attack Recall=0.9970, Attack Precision=0.8123

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
0.15       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280   <--
0.20       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.25       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.30       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.35       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.40       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.45       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.50       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.55       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.60       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.65       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.70       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.75       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.80       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7954, F1=0.4937, Normal Recall=0.7730, Normal Precision=0.9996, Attack Recall=0.9972, Attack Precision=0.3280

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
0.15       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243   <--
0.20       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.25       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.30       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.35       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.40       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.45       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.50       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.55       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.60       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.65       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.70       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.75       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.80       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8185, F1=0.6872, Normal Recall=0.7738, Normal Precision=0.9990, Attack Recall=0.9970, Attack Precision=0.5243

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
0.15       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532   <--
0.20       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.25       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.30       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.35       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.40       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.45       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.50       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.55       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.60       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.65       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.70       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.75       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.80       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8403, F1=0.7893, Normal Recall=0.7731, Normal Precision=0.9983, Attack Recall=0.9970, Attack Precision=0.6532

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
0.15       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450   <--
0.20       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.25       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.30       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.35       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.40       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.45       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.50       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.55       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.60       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.65       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.70       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.75       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.80       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8623, F1=0.8528, Normal Recall=0.7725, Normal Precision=0.9974, Attack Recall=0.9970, Attack Precision=0.7450

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
0.15       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126   <--
0.20       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.25       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.30       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.35       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.40       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.45       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.50       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.55       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.60       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.65       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.70       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.75       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.80       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8836, F1=0.8954, Normal Recall=0.7701, Normal Precision=0.9961, Attack Recall=0.9970, Attack Precision=0.8126

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
0.15       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280   <--
0.20       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.25       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.30       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.35       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.40       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.45       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.50       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.55       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.60       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.65       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.70       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.75       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
0.80       0.7954   0.4937   0.7730   0.9996   0.9972   0.3280  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7954, F1=0.4937, Normal Recall=0.7730, Normal Precision=0.9996, Attack Recall=0.9972, Attack Precision=0.3280

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
0.15       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243   <--
0.20       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.25       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.30       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.35       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.40       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.45       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.50       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.55       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.60       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.65       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.70       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.75       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
0.80       0.8185   0.6872   0.7738   0.9990   0.9970   0.5243  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8185, F1=0.6872, Normal Recall=0.7738, Normal Precision=0.9990, Attack Recall=0.9970, Attack Precision=0.5243

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
0.15       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532   <--
0.20       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.25       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.30       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.35       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.40       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.45       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.50       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.55       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.60       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.65       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.70       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.75       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
0.80       0.8403   0.7893   0.7731   0.9983   0.9970   0.6532  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8403, F1=0.7893, Normal Recall=0.7731, Normal Precision=0.9983, Attack Recall=0.9970, Attack Precision=0.6532

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
0.15       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450   <--
0.20       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.25       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.30       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.35       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.40       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.45       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.50       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.55       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.60       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.65       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.70       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.75       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
0.80       0.8623   0.8528   0.7725   0.9974   0.9970   0.7450  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8623, F1=0.8528, Normal Recall=0.7725, Normal Precision=0.9974, Attack Recall=0.9970, Attack Precision=0.7450

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
0.15       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126   <--
0.20       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.25       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.30       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.35       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.40       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.45       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.50       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.55       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.60       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.65       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.70       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.75       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
0.80       0.8836   0.8954   0.7701   0.9961   0.9970   0.8126  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8836, F1=0.8954, Normal Recall=0.7701, Normal Precision=0.9961, Attack Recall=0.9970, Attack Precision=0.8126

```

