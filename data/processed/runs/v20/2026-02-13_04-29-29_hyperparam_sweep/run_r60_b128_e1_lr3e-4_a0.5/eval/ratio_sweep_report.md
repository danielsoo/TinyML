# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b128_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-14 07:31:59 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4030 | 0.4643 | 0.5233 | 0.5829 | 0.6430 | 0.6998 | 0.7602 | 0.8193 | 0.8787 | 0.9379 | 0.9971 |
| QAT+Prune only | 0.0001 | 0.1001 | 0.2001 | 0.3001 | 0.4001 | 0.5001 | 0.6000 | 0.7000 | 0.8001 | 0.9000 | 1.0000 |
| QAT+PTQ | 0.0001 | 0.1001 | 0.2001 | 0.3001 | 0.4001 | 0.5001 | 0.6000 | 0.7000 | 0.8001 | 0.9000 | 1.0000 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.0001 | 0.1001 | 0.2001 | 0.3001 | 0.4001 | 0.5001 | 0.6000 | 0.7000 | 0.8001 | 0.9000 | 1.0000 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2713 | 0.4555 | 0.5892 | 0.6908 | 0.7686 | 0.8331 | 0.8854 | 0.9294 | 0.9665 | 0.9985 |
| QAT+Prune only | 0.0000 | 0.1818 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| QAT+PTQ | 0.0000 | 0.1818 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.1818 | 0.3334 | 0.4616 | 0.5715 | 0.6667 | 0.7500 | 0.8235 | 0.8889 | 0.9474 | 1.0000 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.4030 | 0.4051 | 0.4048 | 0.4054 | 0.4070 | 0.4024 | 0.4050 | 0.4044 | 0.4054 | 0.4052 | 0.0000 |
| QAT+Prune only | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0000 | 0.0003 | 0.0002 | 0.0000 |
| QAT+PTQ | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0000 | 0.0003 | 0.0002 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0001 | 0.0002 | 0.0001 | 0.0000 | 0.0003 | 0.0002 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.4030 | 0.0000 | 0.0000 | 0.0000 | 0.4030 | 1.0000 |
| 90 | 10 | 299,940 | 0.4643 | 0.1570 | 0.9974 | 0.2713 | 0.4051 | 0.9993 |
| 80 | 20 | 291,350 | 0.5233 | 0.2952 | 0.9971 | 0.4555 | 0.4048 | 0.9982 |
| 70 | 30 | 194,230 | 0.5829 | 0.4182 | 0.9971 | 0.5892 | 0.4054 | 0.9969 |
| 60 | 40 | 145,675 | 0.6430 | 0.5285 | 0.9971 | 0.6908 | 0.4070 | 0.9952 |
| 50 | 50 | 116,540 | 0.6998 | 0.6253 | 0.9971 | 0.7686 | 0.4024 | 0.9928 |
| 40 | 60 | 97,115 | 0.7602 | 0.7154 | 0.9971 | 0.8331 | 0.4050 | 0.9892 |
| 30 | 70 | 83,240 | 0.8193 | 0.7962 | 0.9971 | 0.8854 | 0.4044 | 0.9833 |
| 20 | 80 | 72,835 | 0.8787 | 0.8703 | 0.9971 | 0.9294 | 0.4054 | 0.9719 |
| 10 | 90 | 64,740 | 0.9379 | 0.9378 | 0.9971 | 0.9665 | 0.4052 | 0.9388 |
| 0 | 100 | 58,270 | 0.9971 | 1.0000 | 0.9971 | 0.9985 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 299,940 | 0.1001 | 0.1000 | 1.0000 | 0.1818 | 0.0001 | 1.0000 |
| 80 | 20 | 291,350 | 0.2001 | 0.2000 | 1.0000 | 0.3334 | 0.0001 | 1.0000 |
| 70 | 30 | 194,230 | 0.3001 | 0.3000 | 1.0000 | 0.4616 | 0.0001 | 1.0000 |
| 60 | 40 | 145,675 | 0.4001 | 0.4000 | 1.0000 | 0.5715 | 0.0001 | 1.0000 |
| 50 | 50 | 116,540 | 0.5001 | 0.5000 | 1.0000 | 0.6667 | 0.0002 | 1.0000 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0001 | 1.0000 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0000 | 1.0000 |
| 20 | 80 | 72,835 | 0.8001 | 0.8000 | 1.0000 | 0.8889 | 0.0003 | 1.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 299,940 | 0.1001 | 0.1000 | 1.0000 | 0.1818 | 0.0001 | 1.0000 |
| 80 | 20 | 291,350 | 0.2001 | 0.2000 | 1.0000 | 0.3334 | 0.0001 | 1.0000 |
| 70 | 30 | 194,230 | 0.3001 | 0.3000 | 1.0000 | 0.4616 | 0.0001 | 1.0000 |
| 60 | 40 | 145,675 | 0.4001 | 0.4000 | 1.0000 | 0.5715 | 0.0001 | 1.0000 |
| 50 | 50 | 116,540 | 0.5001 | 0.5000 | 1.0000 | 0.6667 | 0.0002 | 1.0000 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0001 | 1.0000 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0000 | 1.0000 |
| 20 | 80 | 72,835 | 0.8001 | 0.8000 | 1.0000 | 0.8889 | 0.0003 | 1.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.0001 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | 1.0000 |
| 90 | 10 | 299,940 | 0.1001 | 0.1000 | 1.0000 | 0.1818 | 0.0001 | 1.0000 |
| 80 | 20 | 291,350 | 0.2001 | 0.2000 | 1.0000 | 0.3334 | 0.0001 | 1.0000 |
| 70 | 30 | 194,230 | 0.3001 | 0.3000 | 1.0000 | 0.4616 | 0.0001 | 1.0000 |
| 60 | 40 | 145,675 | 0.4001 | 0.4000 | 1.0000 | 0.5715 | 0.0001 | 1.0000 |
| 50 | 50 | 116,540 | 0.5001 | 0.5000 | 1.0000 | 0.6667 | 0.0002 | 1.0000 |
| 40 | 60 | 97,115 | 0.6000 | 0.6000 | 1.0000 | 0.7500 | 0.0001 | 1.0000 |
| 30 | 70 | 83,240 | 0.7000 | 0.7000 | 1.0000 | 0.8235 | 0.0000 | 1.0000 |
| 20 | 80 | 72,835 | 0.8001 | 0.8000 | 1.0000 | 0.8889 | 0.0003 | 1.0000 |
| 10 | 90 | 64,740 | 0.9000 | 0.9000 | 1.0000 | 0.9474 | 0.0002 | 1.0000 |
| 0 | 100 | 58,270 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |


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
0.15       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570   <--
0.20       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.25       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.30       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.35       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.40       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.45       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.50       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.55       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.60       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.65       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.70       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.75       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
0.80       0.4643   0.2713   0.4051   0.9992   0.9971   0.1570  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4643, F1=0.2713, Normal Recall=0.4051, Normal Precision=0.9992, Attack Recall=0.9971, Attack Precision=0.1570

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
0.15       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950   <--
0.20       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.25       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.30       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.35       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.40       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.45       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.50       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.55       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.60       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.65       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.70       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.75       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
0.80       0.5229   0.4553   0.4044   0.9982   0.9971   0.2950  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5229, F1=0.4553, Normal Recall=0.4044, Normal Precision=0.9982, Attack Recall=0.9971, Attack Precision=0.2950

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
0.15       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176   <--
0.20       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.25       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.30       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.35       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.40       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.45       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.50       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.55       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.60       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.65       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.70       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.75       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
0.80       0.5820   0.5887   0.4040   0.9969   0.9971   0.4176  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5820, F1=0.5887, Normal Recall=0.4040, Normal Precision=0.9969, Attack Recall=0.9971, Attack Precision=0.4176

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
0.15       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270   <--
0.20       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.25       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.30       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.35       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.40       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.45       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.50       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.55       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.60       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.65       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.70       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.75       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
0.80       0.6409   0.6895   0.4034   0.9952   0.9971   0.5270  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6409, F1=0.6895, Normal Recall=0.4034, Normal Precision=0.9952, Attack Recall=0.9971, Attack Precision=0.5270

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
0.15       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252   <--
0.20       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.25       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.30       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.35       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.40       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.45       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.50       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.55       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.60       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.65       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.70       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.75       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
0.80       0.6997   0.7685   0.4023   0.9928   0.9971   0.6252  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6997, F1=0.7685, Normal Recall=0.4023, Normal Precision=0.9928, Attack Recall=0.9971, Attack Precision=0.6252

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
0.15       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000   <--
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.30       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.35       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.40       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.45       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.50       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.55       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.60       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.65       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.70       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.75       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.80       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1001, F1=0.1818, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1000

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
0.15       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000   <--
0.20       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.25       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.30       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.35       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.40       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.45       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.50       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.55       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.60       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.65       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.70       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.75       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.80       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2001, F1=0.3334, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2000

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
0.15       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000   <--
0.20       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.25       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.30       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.35       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.40       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.45       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.50       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.55       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.60       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.65       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.70       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.75       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.80       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3001, F1=0.4616, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3000

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
0.15       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000   <--
0.20       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.25       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.30       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.35       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.40       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.45       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.50       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.55       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.60       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.65       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.70       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.75       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.80       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4001, F1=0.5715, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4000

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
0.15       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000   <--
0.20       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.25       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.30       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.35       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.40       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.45       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.50       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.55       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.60       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.65       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.70       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.75       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.80       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5001, F1=0.6667, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5000

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
0.15       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000   <--
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.30       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.35       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.40       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.45       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.50       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.55       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.60       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.65       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.70       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.75       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.80       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1001, F1=0.1818, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1000

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
0.15       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000   <--
0.20       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.25       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.30       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.35       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.40       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.45       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.50       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.55       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.60       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.65       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.70       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.75       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.80       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2001, F1=0.3334, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2000

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
0.15       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000   <--
0.20       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.25       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.30       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.35       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.40       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.45       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.50       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.55       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.60       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.65       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.70       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.75       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.80       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3001, F1=0.4616, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3000

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
0.15       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000   <--
0.20       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.25       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.30       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.35       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.40       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.45       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.50       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.55       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.60       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.65       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.70       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.75       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.80       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4001, F1=0.5715, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4000

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
0.15       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000   <--
0.20       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.25       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.30       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.35       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.40       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.45       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.50       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.55       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.60       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.65       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.70       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.75       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.80       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5001, F1=0.6667, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5000

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
0.15       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000   <--
0.20       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.25       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.30       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.35       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.40       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.45       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.50       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.55       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.60       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.65       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.70       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.75       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
0.80       0.1001   0.1818   0.0001   1.0000   1.0000   0.1000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.1001, F1=0.1818, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.1000

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
0.15       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000   <--
0.20       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.25       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.30       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.35       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.40       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.45       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.50       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.55       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.60       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.65       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.70       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.75       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
0.80       0.2001   0.3334   0.0001   1.0000   1.0000   0.2000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.2001, F1=0.3334, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.2000

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
0.15       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000   <--
0.20       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.25       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.30       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.35       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.40       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.45       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.50       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.55       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.60       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.65       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.70       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.75       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
0.80       0.3001   0.4616   0.0002   1.0000   1.0000   0.3000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3001, F1=0.4616, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.3000

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
0.15       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000   <--
0.20       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.25       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.30       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.35       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.40       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.45       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.50       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.55       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.60       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.65       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.70       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.75       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
0.80       0.4001   0.5715   0.0001   1.0000   1.0000   0.4000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4001, F1=0.5715, Normal Recall=0.0001, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.4000

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
0.15       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000   <--
0.20       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.25       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.30       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.35       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.40       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.45       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.50       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.55       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.60       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.65       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.70       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.75       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
0.80       0.5001   0.6667   0.0002   1.0000   1.0000   0.5000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5001, F1=0.6667, Normal Recall=0.0002, Normal Precision=1.0000, Attack Recall=1.0000, Attack Precision=0.5000

```

