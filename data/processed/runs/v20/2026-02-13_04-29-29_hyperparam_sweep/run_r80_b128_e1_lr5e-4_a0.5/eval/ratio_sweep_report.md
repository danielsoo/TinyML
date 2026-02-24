# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b128_e1_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-17 08:51:01 |

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
| **Batch size** | 128 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5331 | 0.5810 | 0.6268 | 0.6739 | 0.7205 | 0.7657 | 0.8136 | 0.8578 | 0.9049 | 0.9509 | 0.9976 |
| QAT+Prune only | 0.7502 | 0.7744 | 0.7987 | 0.8234 | 0.8483 | 0.8711 | 0.8976 | 0.9216 | 0.9480 | 0.9709 | 0.9964 |
| QAT+PTQ | 0.7499 | 0.7742 | 0.7985 | 0.8234 | 0.8482 | 0.8711 | 0.8976 | 0.9217 | 0.9479 | 0.9710 | 0.9965 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7499 | 0.7742 | 0.7985 | 0.8234 | 0.8482 | 0.8711 | 0.8976 | 0.9217 | 0.9479 | 0.9710 | 0.9965 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3226 | 0.5167 | 0.6473 | 0.7406 | 0.8098 | 0.8653 | 0.9076 | 0.9438 | 0.9734 | 0.9988 |
| QAT+Prune only | 0.0000 | 0.4691 | 0.6644 | 0.7719 | 0.8401 | 0.8854 | 0.9211 | 0.9468 | 0.9684 | 0.9840 | 0.9982 |
| QAT+PTQ | 0.0000 | 0.4689 | 0.6642 | 0.7720 | 0.8400 | 0.8855 | 0.9211 | 0.9468 | 0.9684 | 0.9841 | 0.9983 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4689 | 0.6642 | 0.7720 | 0.8400 | 0.8855 | 0.9211 | 0.9468 | 0.9684 | 0.9841 | 0.9983 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5331 | 0.5347 | 0.5341 | 0.5352 | 0.5357 | 0.5339 | 0.5375 | 0.5316 | 0.5344 | 0.5310 | 0.0000 |
| QAT+Prune only | 0.7502 | 0.7497 | 0.7492 | 0.7492 | 0.7496 | 0.7458 | 0.7493 | 0.7469 | 0.7542 | 0.7411 | 0.0000 |
| QAT+PTQ | 0.7499 | 0.7495 | 0.7490 | 0.7492 | 0.7492 | 0.7457 | 0.7491 | 0.7470 | 0.7535 | 0.7411 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7499 | 0.7495 | 0.7490 | 0.7492 | 0.7492 | 0.7457 | 0.7491 | 0.7470 | 0.7535 | 0.7411 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5331 | 0.0000 | 0.0000 | 0.0000 | 0.5331 | 1.0000 |
| 90 | 10 | 299,940 | 0.5810 | 0.1924 | 0.9977 | 0.3226 | 0.5347 | 0.9995 |
| 80 | 20 | 291,350 | 0.6268 | 0.3486 | 0.9976 | 0.5167 | 0.5341 | 0.9989 |
| 70 | 30 | 194,230 | 0.6739 | 0.4791 | 0.9976 | 0.6473 | 0.5352 | 0.9981 |
| 60 | 40 | 145,675 | 0.7205 | 0.5889 | 0.9976 | 0.7406 | 0.5357 | 0.9970 |
| 50 | 50 | 116,540 | 0.7657 | 0.6815 | 0.9976 | 0.8098 | 0.5339 | 0.9955 |
| 40 | 60 | 97,115 | 0.8136 | 0.7639 | 0.9976 | 0.8653 | 0.5375 | 0.9933 |
| 30 | 70 | 83,240 | 0.8578 | 0.8325 | 0.9976 | 0.9076 | 0.5316 | 0.9896 |
| 20 | 80 | 72,835 | 0.9049 | 0.8955 | 0.9976 | 0.9438 | 0.5344 | 0.9823 |
| 10 | 90 | 64,740 | 0.9509 | 0.9504 | 0.9976 | 0.9734 | 0.5310 | 0.9609 |
| 0 | 100 | 58,270 | 0.9976 | 1.0000 | 0.9976 | 0.9988 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7502 | 0.0000 | 0.0000 | 0.0000 | 0.7502 | 1.0000 |
| 90 | 10 | 299,940 | 0.7744 | 0.3067 | 0.9966 | 0.4691 | 0.7497 | 0.9995 |
| 80 | 20 | 291,350 | 0.7987 | 0.4983 | 0.9964 | 0.6644 | 0.7492 | 0.9988 |
| 70 | 30 | 194,230 | 0.8234 | 0.6300 | 0.9964 | 0.7719 | 0.7492 | 0.9980 |
| 60 | 40 | 145,675 | 0.8483 | 0.7263 | 0.9964 | 0.8401 | 0.7496 | 0.9968 |
| 50 | 50 | 116,540 | 0.8711 | 0.7967 | 0.9964 | 0.8854 | 0.7458 | 0.9952 |
| 40 | 60 | 97,115 | 0.8976 | 0.8564 | 0.9964 | 0.9211 | 0.7493 | 0.9929 |
| 30 | 70 | 83,240 | 0.9216 | 0.9018 | 0.9964 | 0.9468 | 0.7469 | 0.9889 |
| 20 | 80 | 72,835 | 0.9480 | 0.9419 | 0.9964 | 0.9684 | 0.7542 | 0.9813 |
| 10 | 90 | 64,740 | 0.9709 | 0.9719 | 0.9964 | 0.9840 | 0.7411 | 0.9583 |
| 0 | 100 | 58,270 | 0.9964 | 1.0000 | 0.9964 | 0.9982 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7499 | 0.0000 | 0.0000 | 0.0000 | 0.7499 | 1.0000 |
| 90 | 10 | 299,940 | 0.7742 | 0.3066 | 0.9967 | 0.4689 | 0.7495 | 0.9995 |
| 80 | 20 | 291,350 | 0.7985 | 0.4981 | 0.9965 | 0.6642 | 0.7490 | 0.9988 |
| 70 | 30 | 194,230 | 0.8234 | 0.6300 | 0.9965 | 0.7720 | 0.7492 | 0.9980 |
| 60 | 40 | 145,675 | 0.8482 | 0.7260 | 0.9965 | 0.8400 | 0.7492 | 0.9969 |
| 50 | 50 | 116,540 | 0.8711 | 0.7967 | 0.9965 | 0.8855 | 0.7457 | 0.9954 |
| 40 | 60 | 97,115 | 0.8976 | 0.8563 | 0.9965 | 0.9211 | 0.7491 | 0.9931 |
| 30 | 70 | 83,240 | 0.9217 | 0.9019 | 0.9965 | 0.9468 | 0.7470 | 0.9893 |
| 20 | 80 | 72,835 | 0.9479 | 0.9418 | 0.9965 | 0.9684 | 0.7535 | 0.9819 |
| 10 | 90 | 64,740 | 0.9710 | 0.9719 | 0.9965 | 0.9841 | 0.7411 | 0.9596 |
| 0 | 100 | 58,270 | 0.9965 | 1.0000 | 0.9965 | 0.9983 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7499 | 0.0000 | 0.0000 | 0.0000 | 0.7499 | 1.0000 |
| 90 | 10 | 299,940 | 0.7742 | 0.3066 | 0.9967 | 0.4689 | 0.7495 | 0.9995 |
| 80 | 20 | 291,350 | 0.7985 | 0.4981 | 0.9965 | 0.6642 | 0.7490 | 0.9988 |
| 70 | 30 | 194,230 | 0.8234 | 0.6300 | 0.9965 | 0.7720 | 0.7492 | 0.9980 |
| 60 | 40 | 145,675 | 0.8482 | 0.7260 | 0.9965 | 0.8400 | 0.7492 | 0.9969 |
| 50 | 50 | 116,540 | 0.8711 | 0.7967 | 0.9965 | 0.8855 | 0.7457 | 0.9954 |
| 40 | 60 | 97,115 | 0.8976 | 0.8563 | 0.9965 | 0.9211 | 0.7491 | 0.9931 |
| 30 | 70 | 83,240 | 0.9217 | 0.9019 | 0.9965 | 0.9468 | 0.7470 | 0.9893 |
| 20 | 80 | 72,835 | 0.9479 | 0.9418 | 0.9965 | 0.9684 | 0.7535 | 0.9819 |
| 10 | 90 | 64,740 | 0.9710 | 0.9719 | 0.9965 | 0.9841 | 0.7411 | 0.9596 |
| 0 | 100 | 58,270 | 0.9965 | 1.0000 | 0.9965 | 0.9983 | 0.0000 | 0.0000 |


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
0.15       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924   <--
0.20       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.25       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.30       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.35       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.40       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.45       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.50       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.55       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.60       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.65       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.70       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.75       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
0.80       0.5810   0.3226   0.5347   0.9995   0.9978   0.1924  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5810, F1=0.3226, Normal Recall=0.5347, Normal Precision=0.9995, Attack Recall=0.9978, Attack Precision=0.1924

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
0.15       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490   <--
0.20       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.25       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.30       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.35       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.40       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.45       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.50       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.55       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.60       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.65       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.70       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.75       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
0.80       0.6273   0.5171   0.5348   0.9989   0.9976   0.3490  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6273, F1=0.5171, Normal Recall=0.5348, Normal Precision=0.9989, Attack Recall=0.9976, Attack Precision=0.3490

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
0.15       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787   <--
0.20       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.25       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.30       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.35       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.40       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.45       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.50       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.55       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.60       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.65       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.70       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.75       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
0.80       0.6734   0.6470   0.5345   0.9981   0.9976   0.4787  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6734, F1=0.6470, Normal Recall=0.5345, Normal Precision=0.9981, Attack Recall=0.9976, Attack Precision=0.4787

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
0.15       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877   <--
0.20       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.25       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.30       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.35       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.40       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.45       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.50       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.55       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.60       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.65       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.70       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.75       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
0.80       0.7190   0.7396   0.5333   0.9970   0.9976   0.5877  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7190, F1=0.7396, Normal Recall=0.5333, Normal Precision=0.9970, Attack Recall=0.9976, Attack Precision=0.5877

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
0.15       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811   <--
0.20       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.25       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.30       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.35       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.40       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.45       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.50       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.55       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.60       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.65       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.70       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.75       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
0.80       0.7653   0.8095   0.5330   0.9955   0.9976   0.6811  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7653, F1=0.8095, Normal Recall=0.5330, Normal Precision=0.9955, Attack Recall=0.9976, Attack Precision=0.6811

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
0.15       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067   <--
0.20       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.25       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.30       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.35       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.40       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.45       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.50       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.55       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.60       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.65       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.70       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.75       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
0.80       0.7744   0.4691   0.7497   0.9995   0.9967   0.3067  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7744, F1=0.4691, Normal Recall=0.7497, Normal Precision=0.9995, Attack Recall=0.9967, Attack Precision=0.3067

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
0.15       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994   <--
0.20       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.25       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.30       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.35       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.40       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.45       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.50       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.55       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.60       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.65       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.70       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.75       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
0.80       0.7995   0.6654   0.7503   0.9988   0.9964   0.4994  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7995, F1=0.6654, Normal Recall=0.7503, Normal Precision=0.9988, Attack Recall=0.9964, Attack Precision=0.4994

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
0.15       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309   <--
0.20       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.25       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.30       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.35       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.40       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.45       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.50       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.55       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.60       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.65       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.70       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.75       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
0.80       0.8241   0.7726   0.7502   0.9980   0.9964   0.6309  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8241, F1=0.7726, Normal Recall=0.7502, Normal Precision=0.9980, Attack Recall=0.9964, Attack Precision=0.6309

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
0.15       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267   <--
0.20       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.25       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.30       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.35       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.40       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.45       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.50       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.55       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.60       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.65       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.70       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.75       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
0.80       0.8487   0.8405   0.7502   0.9968   0.9964   0.7267  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8487, F1=0.8405, Normal Recall=0.7502, Normal Precision=0.9968, Attack Recall=0.9964, Attack Precision=0.7267

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
0.15       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983   <--
0.20       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.25       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.30       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.35       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.40       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.45       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.50       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.55       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.60       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.65       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.70       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.75       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
0.80       0.8723   0.8864   0.7482   0.9952   0.9964   0.7983  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8723, F1=0.8864, Normal Recall=0.7482, Normal Precision=0.9952, Attack Recall=0.9964, Attack Precision=0.7983

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
0.15       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066   <--
0.20       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.25       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.30       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.35       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.40       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.45       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.50       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.55       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.60       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.65       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.70       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.75       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.80       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7742, F1=0.4689, Normal Recall=0.7495, Normal Precision=0.9995, Attack Recall=0.9968, Attack Precision=0.3066

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
0.15       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993   <--
0.20       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.25       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.30       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.35       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.40       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.45       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.50       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.55       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.60       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.65       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.70       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.75       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.80       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7994, F1=0.6653, Normal Recall=0.7501, Normal Precision=0.9988, Attack Recall=0.9965, Attack Precision=0.4993

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
0.15       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308   <--
0.20       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.25       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.30       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.35       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.40       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.45       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.50       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.55       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.60       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.65       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.70       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.75       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.80       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8240, F1=0.7725, Normal Recall=0.7500, Normal Precision=0.9980, Attack Recall=0.9965, Attack Precision=0.6308

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
0.15       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265   <--
0.20       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.25       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.30       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.35       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.40       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.45       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.50       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.55       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.60       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.65       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.70       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.75       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.80       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8485, F1=0.8403, Normal Recall=0.7499, Normal Precision=0.9969, Attack Recall=0.9965, Attack Precision=0.7265

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
0.15       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982   <--
0.20       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.25       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.30       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.35       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.40       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.45       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.50       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.55       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.60       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.65       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.70       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.75       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.80       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8723, F1=0.8864, Normal Recall=0.7480, Normal Precision=0.9954, Attack Recall=0.9965, Attack Precision=0.7982

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
0.15       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066   <--
0.20       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.25       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.30       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.35       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.40       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.45       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.50       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.55       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.60       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.65       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.70       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.75       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
0.80       0.7742   0.4689   0.7495   0.9995   0.9968   0.3066  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7742, F1=0.4689, Normal Recall=0.7495, Normal Precision=0.9995, Attack Recall=0.9968, Attack Precision=0.3066

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
0.15       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993   <--
0.20       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.25       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.30       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.35       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.40       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.45       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.50       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.55       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.60       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.65       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.70       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.75       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
0.80       0.7994   0.6653   0.7501   0.9988   0.9965   0.4993  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7994, F1=0.6653, Normal Recall=0.7501, Normal Precision=0.9988, Attack Recall=0.9965, Attack Precision=0.4993

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
0.15       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308   <--
0.20       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.25       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.30       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.35       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.40       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.45       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.50       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.55       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.60       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.65       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.70       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.75       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
0.80       0.8240   0.7725   0.7500   0.9980   0.9965   0.6308  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8240, F1=0.7725, Normal Recall=0.7500, Normal Precision=0.9980, Attack Recall=0.9965, Attack Precision=0.6308

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
0.15       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265   <--
0.20       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.25       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.30       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.35       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.40       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.45       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.50       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.55       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.60       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.65       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.70       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.75       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
0.80       0.8485   0.8403   0.7499   0.9969   0.9965   0.7265  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8485, F1=0.8403, Normal Recall=0.7499, Normal Precision=0.9969, Attack Recall=0.9965, Attack Precision=0.7265

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
0.15       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982   <--
0.20       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.25       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.30       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.35       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.40       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.45       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.50       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.55       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.60       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.65       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.70       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.75       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
0.80       0.8723   0.8864   0.7480   0.9954   0.9965   0.7982  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8723, F1=0.8864, Normal Recall=0.7480, Normal Precision=0.9954, Attack Recall=0.9965, Attack Precision=0.7982

```

