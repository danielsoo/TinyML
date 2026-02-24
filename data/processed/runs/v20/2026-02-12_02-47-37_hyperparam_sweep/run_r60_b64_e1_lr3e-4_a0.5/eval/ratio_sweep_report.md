# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.5.yaml` |
| **Generated** | 2026-02-12 04:16:30 |

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
| Original (TFLite) | 0.6555 | 0.6605 | 0.6661 | 0.6716 | 0.6764 | 0.6816 | 0.6860 | 0.6922 | 0.6955 | 0.7010 | 0.7063 |
| QAT+Prune only | 0.5996 | 0.6388 | 0.6782 | 0.7186 | 0.7580 | 0.7968 | 0.8374 | 0.8772 | 0.9169 | 0.9568 | 0.9968 |
| QAT+PTQ | 0.6006 | 0.6396 | 0.6789 | 0.7193 | 0.7587 | 0.7972 | 0.8377 | 0.8774 | 0.9173 | 0.9568 | 0.9968 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.6006 | 0.6396 | 0.6789 | 0.7193 | 0.7587 | 0.7972 | 0.8377 | 0.8774 | 0.9173 | 0.9568 | 0.9968 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2931 | 0.4584 | 0.5634 | 0.6359 | 0.6893 | 0.7297 | 0.7626 | 0.7877 | 0.8096 | 0.8279 |
| QAT+Prune only | 0.0000 | 0.3556 | 0.5534 | 0.6801 | 0.7672 | 0.8307 | 0.8803 | 0.9191 | 0.9505 | 0.9765 | 0.9984 |
| QAT+PTQ | 0.0000 | 0.3561 | 0.5539 | 0.6806 | 0.7677 | 0.8310 | 0.8805 | 0.9192 | 0.9507 | 0.9765 | 0.9984 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3561 | 0.5539 | 0.6806 | 0.7677 | 0.8310 | 0.8805 | 0.9192 | 0.9507 | 0.9765 | 0.9984 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.6555 | 0.6556 | 0.6561 | 0.6567 | 0.6564 | 0.6568 | 0.6555 | 0.6591 | 0.6521 | 0.6532 | 0.0000 |
| QAT+Prune only | 0.5996 | 0.5990 | 0.5985 | 0.5994 | 0.5988 | 0.5968 | 0.5982 | 0.5981 | 0.5972 | 0.5962 | 0.0000 |
| QAT+PTQ | 0.6006 | 0.6000 | 0.5995 | 0.6004 | 0.6000 | 0.5977 | 0.5991 | 0.5988 | 0.5992 | 0.5968 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.6006 | 0.6000 | 0.5995 | 0.6004 | 0.6000 | 0.5977 | 0.5991 | 0.5988 | 0.5992 | 0.5968 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6555 | 0.0000 | 0.0000 | 0.0000 | 0.6555 | 1.0000 |
| 90 | 10 | 299,940 | 0.6605 | 0.1851 | 0.7040 | 0.2931 | 0.6556 | 0.9522 |
| 80 | 20 | 291,350 | 0.6661 | 0.3393 | 0.7063 | 0.4584 | 0.6561 | 0.8994 |
| 70 | 30 | 194,230 | 0.6716 | 0.4686 | 0.7063 | 0.5634 | 0.6567 | 0.8392 |
| 60 | 40 | 145,675 | 0.6764 | 0.5782 | 0.7063 | 0.6359 | 0.6564 | 0.7703 |
| 50 | 50 | 116,540 | 0.6816 | 0.6730 | 0.7063 | 0.6893 | 0.6568 | 0.6910 |
| 40 | 60 | 97,115 | 0.6860 | 0.7546 | 0.7063 | 0.7297 | 0.6555 | 0.5981 |
| 30 | 70 | 83,240 | 0.6922 | 0.8286 | 0.7064 | 0.7626 | 0.6591 | 0.4903 |
| 20 | 80 | 72,835 | 0.6955 | 0.8904 | 0.7063 | 0.7877 | 0.6521 | 0.3570 |
| 10 | 90 | 64,740 | 0.7010 | 0.9483 | 0.7063 | 0.8096 | 0.6532 | 0.1982 |
| 0 | 100 | 58,270 | 0.7063 | 1.0000 | 0.7063 | 0.8279 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5996 | 0.0000 | 0.0000 | 0.0000 | 0.5996 | 1.0000 |
| 90 | 10 | 299,940 | 0.6388 | 0.2164 | 0.9968 | 0.3556 | 0.5990 | 0.9994 |
| 80 | 20 | 291,350 | 0.6782 | 0.3830 | 0.9968 | 0.5534 | 0.5985 | 0.9987 |
| 70 | 30 | 194,230 | 0.7186 | 0.5161 | 0.9968 | 0.6801 | 0.5994 | 0.9977 |
| 60 | 40 | 145,675 | 0.7580 | 0.6235 | 0.9968 | 0.7672 | 0.5988 | 0.9965 |
| 50 | 50 | 116,540 | 0.7968 | 0.7120 | 0.9968 | 0.8307 | 0.5968 | 0.9947 |
| 40 | 60 | 97,115 | 0.8374 | 0.7882 | 0.9968 | 0.8803 | 0.5982 | 0.9921 |
| 30 | 70 | 83,240 | 0.8772 | 0.8527 | 0.9968 | 0.9191 | 0.5981 | 0.9878 |
| 20 | 80 | 72,835 | 0.9169 | 0.9083 | 0.9968 | 0.9505 | 0.5972 | 0.9792 |
| 10 | 90 | 64,740 | 0.9568 | 0.9569 | 0.9968 | 0.9765 | 0.5962 | 0.9543 |
| 0 | 100 | 58,270 | 0.9968 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.6006 | 0.0000 | 0.0000 | 0.0000 | 0.6006 | 1.0000 |
| 90 | 10 | 299,940 | 0.6396 | 0.2168 | 0.9967 | 0.3561 | 0.6000 | 0.9994 |
| 80 | 20 | 291,350 | 0.6789 | 0.3835 | 0.9968 | 0.5539 | 0.5995 | 0.9987 |
| 70 | 30 | 194,230 | 0.7193 | 0.5167 | 0.9968 | 0.6806 | 0.6004 | 0.9977 |
| 60 | 40 | 145,675 | 0.7587 | 0.6242 | 0.9968 | 0.7677 | 0.6000 | 0.9964 |
| 50 | 50 | 116,540 | 0.7972 | 0.7124 | 0.9968 | 0.8310 | 0.5977 | 0.9947 |
| 40 | 60 | 97,115 | 0.8377 | 0.7886 | 0.9968 | 0.8805 | 0.5991 | 0.9920 |
| 30 | 70 | 83,240 | 0.8774 | 0.8529 | 0.9968 | 0.9192 | 0.5988 | 0.9876 |
| 20 | 80 | 72,835 | 0.9173 | 0.9087 | 0.9968 | 0.9507 | 0.5992 | 0.9790 |
| 10 | 90 | 64,740 | 0.9568 | 0.9570 | 0.9968 | 0.9765 | 0.5968 | 0.9538 |
| 0 | 100 | 58,270 | 0.9968 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.6006 | 0.0000 | 0.0000 | 0.0000 | 0.6006 | 1.0000 |
| 90 | 10 | 299,940 | 0.6396 | 0.2168 | 0.9967 | 0.3561 | 0.6000 | 0.9994 |
| 80 | 20 | 291,350 | 0.6789 | 0.3835 | 0.9968 | 0.5539 | 0.5995 | 0.9987 |
| 70 | 30 | 194,230 | 0.7193 | 0.5167 | 0.9968 | 0.6806 | 0.6004 | 0.9977 |
| 60 | 40 | 145,675 | 0.7587 | 0.6242 | 0.9968 | 0.7677 | 0.6000 | 0.9964 |
| 50 | 50 | 116,540 | 0.7972 | 0.7124 | 0.9968 | 0.8310 | 0.5977 | 0.9947 |
| 40 | 60 | 97,115 | 0.8377 | 0.7886 | 0.9968 | 0.8805 | 0.5991 | 0.9920 |
| 30 | 70 | 83,240 | 0.8774 | 0.8529 | 0.9968 | 0.9192 | 0.5988 | 0.9876 |
| 20 | 80 | 72,835 | 0.9173 | 0.9087 | 0.9968 | 0.9507 | 0.5992 | 0.9790 |
| 10 | 90 | 64,740 | 0.9568 | 0.9570 | 0.9968 | 0.9765 | 0.5968 | 0.9538 |
| 0 | 100 | 58,270 | 0.9968 | 1.0000 | 0.9968 | 0.9984 | 0.0000 | 0.0000 |


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
0.15       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852   <--
0.20       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.25       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.30       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.35       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.40       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.45       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.50       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.55       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.60       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.65       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.70       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.75       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
0.80       0.6605   0.2932   0.6556   0.9523   0.7043   0.1852  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6605, F1=0.2932, Normal Recall=0.6556, Normal Precision=0.9523, Attack Recall=0.7043, Attack Precision=0.1852

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
0.15       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388   <--
0.20       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.25       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.30       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.35       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.40       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.45       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.50       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.55       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.60       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.65       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.70       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.75       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
0.80       0.6656   0.4580   0.6554   0.8993   0.7063   0.3388  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6656, F1=0.4580, Normal Recall=0.6554, Normal Precision=0.8993, Attack Recall=0.7063, Attack Precision=0.3388

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
0.15       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683   <--
0.20       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.25       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.30       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.35       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.40       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.45       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.50       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.55       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.60       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.65       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.70       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.75       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
0.80       0.6713   0.5632   0.6563   0.8391   0.7063   0.4683  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6713, F1=0.5632, Normal Recall=0.6563, Normal Precision=0.8391, Attack Recall=0.7063, Attack Precision=0.4683

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
0.15       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780   <--
0.20       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.25       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.30       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.35       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.40       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.45       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.50       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.55       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.60       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.65       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.70       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.75       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
0.80       0.6763   0.6358   0.6562   0.7702   0.7063   0.5780  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6763, F1=0.6358, Normal Recall=0.6562, Normal Precision=0.7702, Attack Recall=0.7063, Attack Precision=0.5780

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
0.15       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716   <--
0.20       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.25       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.30       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.35       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.40       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.45       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.50       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.55       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.60       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.65       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.70       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.75       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
0.80       0.6805   0.6885   0.6546   0.6903   0.7063   0.6716  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6805, F1=0.6885, Normal Recall=0.6546, Normal Precision=0.6903, Attack Recall=0.7063, Attack Precision=0.6716

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
0.15       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164   <--
0.20       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.25       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.30       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.35       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.40       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.45       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.50       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.55       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.60       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.65       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.70       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.75       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
0.80       0.6388   0.3556   0.5990   0.9994   0.9966   0.2164  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6388, F1=0.3556, Normal Recall=0.5990, Normal Precision=0.9994, Attack Recall=0.9966, Attack Precision=0.2164

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
0.15       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837   <--
0.20       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.25       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.30       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.35       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.40       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.45       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.50       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.55       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.60       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.65       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.70       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.75       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
0.80       0.6791   0.5541   0.5996   0.9987   0.9968   0.3837  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6791, F1=0.5541, Normal Recall=0.5996, Normal Precision=0.9987, Attack Recall=0.9968, Attack Precision=0.3837

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
0.15       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159   <--
0.20       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.25       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.30       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.35       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.40       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.45       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.50       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.55       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.60       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.65       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.70       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.75       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
0.80       0.7184   0.6799   0.5990   0.9977   0.9968   0.5159  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7184, F1=0.6799, Normal Recall=0.5990, Normal Precision=0.9977, Attack Recall=0.9968, Attack Precision=0.5159

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
0.15       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243   <--
0.20       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.25       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.30       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.35       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.40       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.45       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.50       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.55       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.60       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.65       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.70       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.75       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
0.80       0.7587   0.7677   0.6000   0.9965   0.9968   0.6243  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7587, F1=0.7677, Normal Recall=0.6000, Normal Precision=0.9965, Attack Recall=0.9968, Attack Precision=0.6243

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
0.15       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132   <--
0.20       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.25       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.30       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.35       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.40       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.45       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.50       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.55       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.60       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.65       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.70       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.75       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
0.80       0.7980   0.8315   0.5992   0.9947   0.9968   0.7132  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7980, F1=0.8315, Normal Recall=0.5992, Normal Precision=0.9947, Attack Recall=0.9968, Attack Precision=0.7132

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
0.15       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168   <--
0.20       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.25       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.30       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.35       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.40       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.45       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.50       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.55       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.60       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.65       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.70       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.75       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.80       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6396, F1=0.3561, Normal Recall=0.6000, Normal Precision=0.9994, Attack Recall=0.9965, Attack Precision=0.2168

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
0.15       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842   <--
0.20       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.25       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.30       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.35       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.40       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.45       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.50       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.55       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.60       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.65       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.70       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.75       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.80       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6798, F1=0.5546, Normal Recall=0.6005, Normal Precision=0.9987, Attack Recall=0.9968, Attack Precision=0.3842

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
0.15       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165   <--
0.20       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.25       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.30       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.35       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.40       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.45       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.50       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.55       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.60       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.65       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.70       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.75       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.80       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7191, F1=0.6804, Normal Recall=0.6000, Normal Precision=0.9977, Attack Recall=0.9968, Attack Precision=0.5165

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
0.15       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248   <--
0.20       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.25       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.30       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.35       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.40       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.45       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.50       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.55       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.60       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.65       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.70       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.75       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.80       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7593, F1=0.7681, Normal Recall=0.6009, Normal Precision=0.9965, Attack Recall=0.9968, Attack Precision=0.6248

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
0.15       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136   <--
0.20       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.25       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.30       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.35       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.40       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.45       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.50       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.55       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.60       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.65       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.70       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.75       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.80       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7984, F1=0.8318, Normal Recall=0.6000, Normal Precision=0.9947, Attack Recall=0.9968, Attack Precision=0.7136

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
0.15       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168   <--
0.20       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.25       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.30       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.35       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.40       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.45       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.50       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.55       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.60       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.65       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.70       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.75       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
0.80       0.6396   0.3561   0.6000   0.9994   0.9965   0.2168  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6396, F1=0.3561, Normal Recall=0.6000, Normal Precision=0.9994, Attack Recall=0.9965, Attack Precision=0.2168

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
0.15       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842   <--
0.20       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.25       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.30       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.35       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.40       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.45       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.50       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.55       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.60       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.65       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.70       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.75       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
0.80       0.6798   0.5546   0.6005   0.9987   0.9968   0.3842  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6798, F1=0.5546, Normal Recall=0.6005, Normal Precision=0.9987, Attack Recall=0.9968, Attack Precision=0.3842

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
0.15       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165   <--
0.20       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.25       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.30       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.35       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.40       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.45       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.50       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.55       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.60       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.65       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.70       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.75       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
0.80       0.7191   0.6804   0.6000   0.9977   0.9968   0.5165  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7191, F1=0.6804, Normal Recall=0.6000, Normal Precision=0.9977, Attack Recall=0.9968, Attack Precision=0.5165

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
0.15       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248   <--
0.20       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.25       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.30       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.35       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.40       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.45       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.50       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.55       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.60       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.65       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.70       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.75       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
0.80       0.7593   0.7681   0.6009   0.9965   0.9968   0.6248  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7593, F1=0.7681, Normal Recall=0.6009, Normal Precision=0.9965, Attack Recall=0.9968, Attack Precision=0.6248

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
0.15       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136   <--
0.20       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.25       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.30       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.35       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.40       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.45       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.50       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.55       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.60       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.65       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.70       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.75       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
0.80       0.7984   0.8318   0.6000   0.9947   0.9968   0.7136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7984, F1=0.8318, Normal Recall=0.6000, Normal Precision=0.9947, Attack Recall=0.9968, Attack Precision=0.7136

```

