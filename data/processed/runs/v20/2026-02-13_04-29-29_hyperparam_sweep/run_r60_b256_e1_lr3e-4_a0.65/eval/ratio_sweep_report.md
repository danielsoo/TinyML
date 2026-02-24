# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b256_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-15 06:02:02 |

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
| **Learning rate** | 0.0003 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8504 | 0.8658 | 0.8799 | 0.8947 | 0.9093 | 0.9222 | 0.9384 | 0.9527 | 0.9665 | 0.9804 | 0.9954 |
| QAT+Prune only | 0.7774 | 0.7797 | 0.7824 | 0.7855 | 0.7877 | 0.7892 | 0.7931 | 0.7944 | 0.7987 | 0.8004 | 0.8036 |
| QAT+PTQ | 0.7776 | 0.7800 | 0.7827 | 0.7859 | 0.7882 | 0.7899 | 0.7939 | 0.7952 | 0.7996 | 0.8014 | 0.8049 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7776 | 0.7800 | 0.7827 | 0.7859 | 0.7882 | 0.7899 | 0.7939 | 0.7952 | 0.7996 | 0.8014 | 0.8049 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5973 | 0.7682 | 0.8501 | 0.8977 | 0.9275 | 0.9509 | 0.9672 | 0.9794 | 0.9892 | 0.9977 |
| QAT+Prune only | 0.0000 | 0.4214 | 0.5963 | 0.6921 | 0.7517 | 0.7922 | 0.8234 | 0.8455 | 0.8646 | 0.8787 | 0.8911 |
| QAT+PTQ | 0.0000 | 0.4220 | 0.5970 | 0.6929 | 0.7525 | 0.7930 | 0.8241 | 0.8462 | 0.8654 | 0.8795 | 0.8919 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4220 | 0.5970 | 0.6929 | 0.7525 | 0.7930 | 0.8241 | 0.8462 | 0.8654 | 0.8795 | 0.8919 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.8504 | 0.8514 | 0.8510 | 0.8515 | 0.8519 | 0.8490 | 0.8528 | 0.8530 | 0.8506 | 0.8449 | 0.0000 |
| QAT+Prune only | 0.7774 | 0.7773 | 0.7771 | 0.7778 | 0.7771 | 0.7748 | 0.7774 | 0.7728 | 0.7787 | 0.7708 | 0.0000 |
| QAT+PTQ | 0.7776 | 0.7774 | 0.7772 | 0.7778 | 0.7771 | 0.7749 | 0.7774 | 0.7727 | 0.7787 | 0.7705 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7776 | 0.7774 | 0.7772 | 0.7778 | 0.7771 | 0.7749 | 0.7774 | 0.7727 | 0.7787 | 0.7705 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8504 | 0.0000 | 0.0000 | 0.0000 | 0.8504 | 1.0000 |
| 90 | 10 | 299,940 | 0.8658 | 0.4267 | 0.9954 | 0.5973 | 0.8514 | 0.9994 |
| 80 | 20 | 291,350 | 0.8799 | 0.6255 | 0.9954 | 0.7682 | 0.8510 | 0.9987 |
| 70 | 30 | 194,230 | 0.8947 | 0.7418 | 0.9954 | 0.8501 | 0.8515 | 0.9977 |
| 60 | 40 | 145,675 | 0.9093 | 0.8175 | 0.9954 | 0.8977 | 0.8519 | 0.9964 |
| 50 | 50 | 116,540 | 0.9222 | 0.8683 | 0.9954 | 0.9275 | 0.8490 | 0.9946 |
| 40 | 60 | 97,115 | 0.9384 | 0.9103 | 0.9954 | 0.9509 | 0.8528 | 0.9920 |
| 30 | 70 | 83,240 | 0.9527 | 0.9405 | 0.9954 | 0.9672 | 0.8530 | 0.9876 |
| 20 | 80 | 72,835 | 0.9665 | 0.9638 | 0.9954 | 0.9794 | 0.8506 | 0.9789 |
| 10 | 90 | 64,740 | 0.9804 | 0.9830 | 0.9954 | 0.9892 | 0.8449 | 0.9535 |
| 0 | 100 | 58,270 | 0.9954 | 1.0000 | 0.9954 | 0.9977 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7774 | 0.0000 | 0.0000 | 0.0000 | 0.7774 | 1.0000 |
| 90 | 10 | 299,940 | 0.7797 | 0.2857 | 0.8019 | 0.4214 | 0.7773 | 0.9725 |
| 80 | 20 | 291,350 | 0.7824 | 0.4740 | 0.8036 | 0.5963 | 0.7771 | 0.9406 |
| 70 | 30 | 194,230 | 0.7855 | 0.6078 | 0.8036 | 0.6921 | 0.7778 | 0.9024 |
| 60 | 40 | 145,675 | 0.7877 | 0.7062 | 0.8036 | 0.7517 | 0.7771 | 0.8558 |
| 50 | 50 | 116,540 | 0.7892 | 0.7811 | 0.8036 | 0.7922 | 0.7748 | 0.7978 |
| 40 | 60 | 97,115 | 0.7931 | 0.8441 | 0.8037 | 0.8234 | 0.7774 | 0.7252 |
| 30 | 70 | 83,240 | 0.7944 | 0.8919 | 0.8036 | 0.8455 | 0.7728 | 0.6278 |
| 20 | 80 | 72,835 | 0.7987 | 0.9356 | 0.8036 | 0.8646 | 0.7787 | 0.4979 |
| 10 | 90 | 64,740 | 0.8004 | 0.9693 | 0.8036 | 0.8787 | 0.7708 | 0.3037 |
| 0 | 100 | 58,270 | 0.8036 | 1.0000 | 0.8036 | 0.8911 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7776 | 0.0000 | 0.0000 | 0.0000 | 0.7776 | 1.0000 |
| 90 | 10 | 299,940 | 0.7800 | 0.2862 | 0.8031 | 0.4220 | 0.7774 | 0.9726 |
| 80 | 20 | 291,350 | 0.7827 | 0.4745 | 0.8049 | 0.5970 | 0.7772 | 0.9409 |
| 70 | 30 | 194,230 | 0.7859 | 0.6082 | 0.8049 | 0.6929 | 0.7778 | 0.9029 |
| 60 | 40 | 145,675 | 0.7882 | 0.7065 | 0.8049 | 0.7525 | 0.7771 | 0.8566 |
| 50 | 50 | 116,540 | 0.7899 | 0.7814 | 0.8049 | 0.7930 | 0.7749 | 0.7988 |
| 40 | 60 | 97,115 | 0.7939 | 0.8443 | 0.8049 | 0.8241 | 0.7774 | 0.7265 |
| 30 | 70 | 83,240 | 0.7952 | 0.8920 | 0.8049 | 0.8462 | 0.7727 | 0.6292 |
| 20 | 80 | 72,835 | 0.7996 | 0.9357 | 0.8049 | 0.8654 | 0.7787 | 0.4994 |
| 10 | 90 | 64,740 | 0.8014 | 0.9693 | 0.8049 | 0.8795 | 0.7705 | 0.3049 |
| 0 | 100 | 58,270 | 0.8049 | 1.0000 | 0.8049 | 0.8919 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7776 | 0.0000 | 0.0000 | 0.0000 | 0.7776 | 1.0000 |
| 90 | 10 | 299,940 | 0.7800 | 0.2862 | 0.8031 | 0.4220 | 0.7774 | 0.9726 |
| 80 | 20 | 291,350 | 0.7827 | 0.4745 | 0.8049 | 0.5970 | 0.7772 | 0.9409 |
| 70 | 30 | 194,230 | 0.7859 | 0.6082 | 0.8049 | 0.6929 | 0.7778 | 0.9029 |
| 60 | 40 | 145,675 | 0.7882 | 0.7065 | 0.8049 | 0.7525 | 0.7771 | 0.8566 |
| 50 | 50 | 116,540 | 0.7899 | 0.7814 | 0.8049 | 0.7930 | 0.7749 | 0.7988 |
| 40 | 60 | 97,115 | 0.7939 | 0.8443 | 0.8049 | 0.8241 | 0.7774 | 0.7265 |
| 30 | 70 | 83,240 | 0.7952 | 0.8920 | 0.8049 | 0.8462 | 0.7727 | 0.6292 |
| 20 | 80 | 72,835 | 0.7996 | 0.9357 | 0.8049 | 0.8654 | 0.7787 | 0.4994 |
| 10 | 90 | 64,740 | 0.8014 | 0.9693 | 0.8049 | 0.8795 | 0.7705 | 0.3049 |
| 0 | 100 | 58,270 | 0.8049 | 1.0000 | 0.8049 | 0.8919 | 0.0000 | 0.0000 |


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
0.15       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268   <--
0.20       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.25       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.30       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.35       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.40       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.45       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.50       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.55       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.60       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.65       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.70       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.75       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
0.80       0.8658   0.5975   0.8514   0.9995   0.9959   0.4268  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8658, F1=0.5975, Normal Recall=0.8514, Normal Precision=0.9995, Attack Recall=0.9959, Attack Precision=0.4268

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
0.15       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263   <--
0.20       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.25       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.30       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.35       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.40       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.45       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.50       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.55       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.60       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.65       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.70       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.75       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
0.80       0.8803   0.7689   0.8515   0.9987   0.9954   0.6263  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8803, F1=0.7689, Normal Recall=0.8515, Normal Precision=0.9987, Attack Recall=0.9954, Attack Precision=0.6263

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
0.15       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407   <--
0.20       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.25       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.30       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.35       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.40       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.45       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.50       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.55       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.60       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.65       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.70       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.75       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
0.80       0.8941   0.8494   0.8506   0.9977   0.9954   0.7407  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8941, F1=0.8494, Normal Recall=0.8506, Normal Precision=0.9977, Attack Recall=0.9954, Attack Precision=0.7407

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
0.15       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168   <--
0.20       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.25       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.30       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.35       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.40       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.45       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.50       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.55       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.60       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.65       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.70       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.75       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
0.80       0.9089   0.8973   0.8512   0.9964   0.9954   0.8168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9089, F1=0.8973, Normal Recall=0.8512, Normal Precision=0.9964, Attack Recall=0.9954, Attack Precision=0.8168

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
0.15       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700   <--
0.20       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.25       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.30       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.35       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.40       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.45       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.50       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.55       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.60       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.65       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.70       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.75       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
0.80       0.9233   0.9285   0.8512   0.9946   0.9954   0.8700  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9233, F1=0.9285, Normal Recall=0.8512, Normal Precision=0.9946, Attack Recall=0.9954, Attack Precision=0.8700

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
0.15       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869   <--
0.20       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.25       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.30       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.35       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.40       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.45       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.50       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.55       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.60       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.65       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.70       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.75       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
0.80       0.7802   0.4232   0.7773   0.9731   0.8065   0.2869  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7802, F1=0.4232, Normal Recall=0.7773, Normal Precision=0.9731, Attack Recall=0.8065, Attack Precision=0.2869

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
0.15       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748   <--
0.20       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.25       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.30       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.35       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.40       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.45       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.50       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.55       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.60       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.65       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.70       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.75       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
0.80       0.7829   0.5969   0.7777   0.9406   0.8036   0.4748  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7829, F1=0.5969, Normal Recall=0.7777, Normal Precision=0.9406, Attack Recall=0.8036, Attack Precision=0.4748

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
0.15       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071   <--
0.20       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.25       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.30       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.35       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.40       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.45       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.50       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.55       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.60       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.65       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.70       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.75       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
0.80       0.7851   0.6917   0.7771   0.9023   0.8037   0.6071  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7851, F1=0.6917, Normal Recall=0.7771, Normal Precision=0.9023, Attack Recall=0.8037, Attack Precision=0.6071

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
0.15       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061   <--
0.20       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.25       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.30       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.35       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.40       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.45       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.50       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.55       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.60       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.65       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.70       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.75       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
0.80       0.7876   0.7517   0.7770   0.8558   0.8036   0.7061  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7876, F1=0.7517, Normal Recall=0.7770, Normal Precision=0.8558, Attack Recall=0.8036, Attack Precision=0.7061

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
0.15       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818   <--
0.20       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.25       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.30       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.35       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.40       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.45       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.50       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.55       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.60       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.65       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.70       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.75       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
0.80       0.7897   0.7926   0.7757   0.7980   0.8036   0.7818  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7897, F1=0.7926, Normal Recall=0.7757, Normal Precision=0.7980, Attack Recall=0.8036, Attack Precision=0.7818

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
0.15       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873   <--
0.20       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.25       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.30       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.35       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.40       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.45       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.50       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.55       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.60       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.65       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.70       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.75       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.80       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7804, F1=0.4238, Normal Recall=0.7774, Normal Precision=0.9732, Attack Recall=0.8075, Attack Precision=0.2873

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
0.15       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753   <--
0.20       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.25       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.30       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.35       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.40       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.45       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.50       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.55       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.60       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.65       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.70       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.75       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.80       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7833, F1=0.5977, Normal Recall=0.7779, Normal Precision=0.9410, Attack Recall=0.8049, Attack Precision=0.4753

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
0.15       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077   <--
0.20       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.25       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.30       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.35       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.40       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.45       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.50       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.55       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.60       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.65       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.70       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.75       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.80       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7856, F1=0.6925, Normal Recall=0.7773, Normal Precision=0.9029, Attack Recall=0.8049, Attack Precision=0.6077

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
0.15       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065   <--
0.20       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.25       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.30       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.35       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.40       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.45       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.50       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.55       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.60       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.65       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.70       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.75       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.80       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7882, F1=0.7525, Normal Recall=0.7771, Normal Precision=0.8566, Attack Recall=0.8049, Attack Precision=0.7065

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
0.15       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822   <--
0.20       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.25       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.30       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.35       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.40       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.45       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.50       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.55       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.60       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.65       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.70       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.75       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.80       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7904, F1=0.7934, Normal Recall=0.7758, Normal Precision=0.7990, Attack Recall=0.8049, Attack Precision=0.7822

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
0.15       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873   <--
0.20       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.25       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.30       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.35       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.40       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.45       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.50       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.55       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.60       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.65       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.70       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.75       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
0.80       0.7804   0.4238   0.7774   0.9732   0.8075   0.2873  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7804, F1=0.4238, Normal Recall=0.7774, Normal Precision=0.9732, Attack Recall=0.8075, Attack Precision=0.2873

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
0.15       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753   <--
0.20       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.25       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.30       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.35       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.40       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.45       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.50       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.55       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.60       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.65       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.70       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.75       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
0.80       0.7833   0.5977   0.7779   0.9410   0.8049   0.4753  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7833, F1=0.5977, Normal Recall=0.7779, Normal Precision=0.9410, Attack Recall=0.8049, Attack Precision=0.4753

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
0.15       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077   <--
0.20       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.25       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.30       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.35       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.40       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.45       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.50       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.55       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.60       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.65       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.70       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.75       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
0.80       0.7856   0.6925   0.7773   0.9029   0.8049   0.6077  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7856, F1=0.6925, Normal Recall=0.7773, Normal Precision=0.9029, Attack Recall=0.8049, Attack Precision=0.6077

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
0.15       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065   <--
0.20       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.25       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.30       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.35       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.40       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.45       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.50       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.55       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.60       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.65       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.70       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.75       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
0.80       0.7882   0.7525   0.7771   0.8566   0.8049   0.7065  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7882, F1=0.7525, Normal Recall=0.7771, Normal Precision=0.8566, Attack Recall=0.8049, Attack Precision=0.7065

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
0.15       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822   <--
0.20       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.25       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.30       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.35       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.40       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.45       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.50       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.55       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.60       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.65       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.70       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.75       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
0.80       0.7904   0.7934   0.7758   0.7990   0.8049   0.7822  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7904, F1=0.7934, Normal Recall=0.7758, Normal Precision=0.7990, Attack Recall=0.8049, Attack Precision=0.7822

```

