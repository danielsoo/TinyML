# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r60_b64_e1_lr3e-4_a0.65.yaml` |
| **Generated** | 2026-02-23 02:39:03 |

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
| Original (TFLite) | 0.5595 | 0.6005 | 0.6443 | 0.6902 | 0.7345 | 0.7779 | 0.8207 | 0.8661 | 0.9092 | 0.9541 | 0.9980 |
| QAT+Prune only | 0.7224 | 0.7486 | 0.7749 | 0.8027 | 0.8293 | 0.8559 | 0.8820 | 0.9103 | 0.9371 | 0.9637 | 0.9911 |
| QAT+PTQ | 0.7223 | 0.7484 | 0.7747 | 0.8026 | 0.8291 | 0.8557 | 0.8819 | 0.9103 | 0.9372 | 0.9638 | 0.9911 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7223 | 0.7484 | 0.7747 | 0.8026 | 0.8291 | 0.8557 | 0.8819 | 0.9103 | 0.9372 | 0.9638 | 0.9911 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3332 | 0.5288 | 0.6590 | 0.7505 | 0.8179 | 0.8698 | 0.9125 | 0.9462 | 0.9751 | 0.9990 |
| QAT+Prune only | 0.0000 | 0.4408 | 0.6378 | 0.7509 | 0.8228 | 0.8730 | 0.9097 | 0.9393 | 0.9618 | 0.9801 | 0.9955 |
| QAT+PTQ | 0.0000 | 0.4406 | 0.6377 | 0.7508 | 0.8227 | 0.8729 | 0.9097 | 0.9393 | 0.9619 | 0.9801 | 0.9955 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.4406 | 0.6377 | 0.7508 | 0.8227 | 0.8729 | 0.9097 | 0.9393 | 0.9619 | 0.9801 | 0.9955 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5595 | 0.5564 | 0.5558 | 0.5582 | 0.5589 | 0.5577 | 0.5547 | 0.5582 | 0.5536 | 0.5585 | 0.0000 |
| QAT+Prune only | 0.7224 | 0.7217 | 0.7209 | 0.7220 | 0.7214 | 0.7206 | 0.7182 | 0.7218 | 0.7211 | 0.7170 | 0.0000 |
| QAT+PTQ | 0.7223 | 0.7215 | 0.7206 | 0.7218 | 0.7211 | 0.7204 | 0.7180 | 0.7216 | 0.7213 | 0.7176 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7223 | 0.7215 | 0.7206 | 0.7218 | 0.7211 | 0.7204 | 0.7180 | 0.7216 | 0.7213 | 0.7176 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5595 | 0.0000 | 0.0000 | 0.0000 | 0.5595 | 1.0000 |
| 90 | 10 | 299,940 | 0.6005 | 0.2000 | 0.9980 | 0.3332 | 0.5564 | 0.9996 |
| 80 | 20 | 291,350 | 0.6443 | 0.3597 | 0.9980 | 0.5288 | 0.5558 | 0.9991 |
| 70 | 30 | 194,230 | 0.6902 | 0.4919 | 0.9980 | 0.6590 | 0.5582 | 0.9985 |
| 60 | 40 | 145,675 | 0.7345 | 0.6013 | 0.9980 | 0.7505 | 0.5589 | 0.9977 |
| 50 | 50 | 116,540 | 0.7779 | 0.6929 | 0.9980 | 0.8179 | 0.5577 | 0.9965 |
| 40 | 60 | 97,115 | 0.8207 | 0.7707 | 0.9980 | 0.8698 | 0.5547 | 0.9947 |
| 30 | 70 | 83,240 | 0.8661 | 0.8405 | 0.9980 | 0.9125 | 0.5582 | 0.9918 |
| 20 | 80 | 72,835 | 0.9092 | 0.8994 | 0.9980 | 0.9462 | 0.5536 | 0.9859 |
| 10 | 90 | 64,740 | 0.9541 | 0.9532 | 0.9980 | 0.9751 | 0.5585 | 0.9692 |
| 0 | 100 | 58,270 | 0.9980 | 1.0000 | 0.9980 | 0.9990 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7224 | 0.0000 | 0.0000 | 0.0000 | 0.7224 | 1.0000 |
| 90 | 10 | 299,940 | 0.7486 | 0.2835 | 0.9908 | 0.4408 | 0.7217 | 0.9986 |
| 80 | 20 | 291,350 | 0.7749 | 0.4702 | 0.9911 | 0.6378 | 0.7209 | 0.9969 |
| 70 | 30 | 194,230 | 0.8027 | 0.6044 | 0.9911 | 0.7509 | 0.7220 | 0.9947 |
| 60 | 40 | 145,675 | 0.8293 | 0.7034 | 0.9911 | 0.8228 | 0.7214 | 0.9918 |
| 50 | 50 | 116,540 | 0.8559 | 0.7801 | 0.9911 | 0.8730 | 0.7206 | 0.9878 |
| 40 | 60 | 97,115 | 0.8820 | 0.8407 | 0.9911 | 0.9097 | 0.7182 | 0.9817 |
| 30 | 70 | 83,240 | 0.9103 | 0.8926 | 0.9911 | 0.9393 | 0.7218 | 0.9720 |
| 20 | 80 | 72,835 | 0.9371 | 0.9343 | 0.9911 | 0.9618 | 0.7211 | 0.9529 |
| 10 | 90 | 64,740 | 0.9637 | 0.9693 | 0.9911 | 0.9801 | 0.7170 | 0.8994 |
| 0 | 100 | 58,270 | 0.9911 | 1.0000 | 0.9911 | 0.9955 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7223 | 0.0000 | 0.0000 | 0.0000 | 0.7223 | 1.0000 |
| 90 | 10 | 299,940 | 0.7484 | 0.2833 | 0.9908 | 0.4406 | 0.7215 | 0.9986 |
| 80 | 20 | 291,350 | 0.7747 | 0.4700 | 0.9911 | 0.6377 | 0.7206 | 0.9969 |
| 70 | 30 | 194,230 | 0.8026 | 0.6042 | 0.9911 | 0.7508 | 0.7218 | 0.9948 |
| 60 | 40 | 145,675 | 0.8291 | 0.7032 | 0.9911 | 0.8227 | 0.7211 | 0.9919 |
| 50 | 50 | 116,540 | 0.8557 | 0.7800 | 0.9911 | 0.8729 | 0.7204 | 0.9878 |
| 40 | 60 | 97,115 | 0.8819 | 0.8406 | 0.9911 | 0.9097 | 0.7180 | 0.9818 |
| 30 | 70 | 83,240 | 0.9103 | 0.8925 | 0.9911 | 0.9393 | 0.7216 | 0.9721 |
| 20 | 80 | 72,835 | 0.9372 | 0.9343 | 0.9911 | 0.9619 | 0.7213 | 0.9531 |
| 10 | 90 | 64,740 | 0.9638 | 0.9693 | 0.9911 | 0.9801 | 0.7176 | 0.8999 |
| 0 | 100 | 58,270 | 0.9911 | 1.0000 | 0.9911 | 0.9955 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7223 | 0.0000 | 0.0000 | 0.0000 | 0.7223 | 1.0000 |
| 90 | 10 | 299,940 | 0.7484 | 0.2833 | 0.9908 | 0.4406 | 0.7215 | 0.9986 |
| 80 | 20 | 291,350 | 0.7747 | 0.4700 | 0.9911 | 0.6377 | 0.7206 | 0.9969 |
| 70 | 30 | 194,230 | 0.8026 | 0.6042 | 0.9911 | 0.7508 | 0.7218 | 0.9948 |
| 60 | 40 | 145,675 | 0.8291 | 0.7032 | 0.9911 | 0.8227 | 0.7211 | 0.9919 |
| 50 | 50 | 116,540 | 0.8557 | 0.7800 | 0.9911 | 0.8729 | 0.7204 | 0.9878 |
| 40 | 60 | 97,115 | 0.8819 | 0.8406 | 0.9911 | 0.9097 | 0.7180 | 0.9818 |
| 30 | 70 | 83,240 | 0.9103 | 0.8925 | 0.9911 | 0.9393 | 0.7216 | 0.9721 |
| 20 | 80 | 72,835 | 0.9372 | 0.9343 | 0.9911 | 0.9619 | 0.7213 | 0.9531 |
| 10 | 90 | 64,740 | 0.9638 | 0.9693 | 0.9911 | 0.9801 | 0.7176 | 0.8999 |
| 0 | 100 | 58,270 | 0.9911 | 1.0000 | 0.9911 | 0.9955 | 0.0000 | 0.0000 |


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
0.15       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000   <--
0.20       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.25       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.30       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.35       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.40       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.45       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.50       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.55       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.60       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.65       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.70       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.75       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
0.80       0.6005   0.3332   0.5563   0.9996   0.9982   0.2000  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6005, F1=0.3332, Normal Recall=0.5563, Normal Precision=0.9996, Attack Recall=0.9982, Attack Precision=0.2000

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
0.15       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600   <--
0.20       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.25       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.30       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.35       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.40       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.45       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.50       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.55       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.60       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.65       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.70       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.75       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
0.80       0.6448   0.5292   0.5565   0.9991   0.9980   0.3600  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6448, F1=0.5292, Normal Recall=0.5565, Normal Precision=0.9991, Attack Recall=0.9980, Attack Precision=0.3600

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
0.15       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921   <--
0.20       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.25       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.30       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.35       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.40       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.45       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.50       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.55       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.60       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.65       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.70       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.75       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
0.80       0.6904   0.6592   0.5585   0.9985   0.9980   0.4921  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6904, F1=0.6592, Normal Recall=0.5585, Normal Precision=0.9985, Attack Recall=0.9980, Attack Precision=0.4921

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
0.15       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012   <--
0.20       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.25       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.30       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.35       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.40       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.45       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.50       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.55       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.60       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.65       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.70       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.75       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
0.80       0.7344   0.7503   0.5586   0.9977   0.9980   0.6012  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7344, F1=0.7503, Normal Recall=0.5586, Normal Precision=0.9977, Attack Recall=0.9980, Attack Precision=0.6012

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
0.15       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932   <--
0.20       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.25       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.30       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.35       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.40       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.45       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.50       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.55       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.60       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.65       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.70       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.75       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
0.80       0.7781   0.8181   0.5583   0.9965   0.9980   0.6932  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7781, F1=0.8181, Normal Recall=0.5583, Normal Precision=0.9965, Attack Recall=0.9980, Attack Precision=0.6932

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
0.15       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835   <--
0.20       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.25       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.30       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.35       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.40       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.45       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.50       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.55       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.60       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.65       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.70       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.75       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
0.80       0.7486   0.4408   0.7217   0.9986   0.9909   0.2835  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7486, F1=0.4408, Normal Recall=0.7217, Normal Precision=0.9986, Attack Recall=0.9909, Attack Precision=0.2835

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
0.15       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714   <--
0.20       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.25       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.30       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.35       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.40       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.45       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.50       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.55       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.60       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.65       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.70       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.75       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
0.80       0.7759   0.6389   0.7221   0.9969   0.9911   0.4714  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7759, F1=0.6389, Normal Recall=0.7221, Normal Precision=0.9969, Attack Recall=0.9911, Attack Precision=0.4714

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
0.15       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048   <--
0.20       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.25       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.30       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.35       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.40       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.45       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.50       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.55       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.60       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.65       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.70       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.75       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
0.80       0.8030   0.7512   0.7224   0.9947   0.9911   0.6048  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8030, F1=0.7512, Normal Recall=0.7224, Normal Precision=0.9947, Attack Recall=0.9911, Attack Precision=0.6048

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
0.15       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039   <--
0.20       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.25       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.30       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.35       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.40       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.45       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.50       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.55       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.60       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.65       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.70       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.75       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
0.80       0.8297   0.8232   0.7221   0.9918   0.9911   0.7039  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8297, F1=0.8232, Normal Recall=0.7221, Normal Precision=0.9918, Attack Recall=0.9911, Attack Precision=0.7039

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
0.15       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802   <--
0.20       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.25       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.30       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.35       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.40       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.45       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.50       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.55       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.60       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.65       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.70       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.75       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
0.80       0.8559   0.8731   0.7208   0.9878   0.9911   0.7802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8559, F1=0.8731, Normal Recall=0.7208, Normal Precision=0.9878, Attack Recall=0.9911, Attack Precision=0.7802

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
0.15       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833   <--
0.20       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.25       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.30       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.35       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.40       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.45       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.50       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.55       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.60       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.65       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.70       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.75       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.80       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7484, F1=0.4407, Normal Recall=0.7215, Normal Precision=0.9986, Attack Recall=0.9910, Attack Precision=0.2833

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
0.15       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712   <--
0.20       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.25       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.30       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.35       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.40       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.45       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.50       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.55       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.60       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.65       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.70       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.75       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.80       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7757, F1=0.6387, Normal Recall=0.7219, Normal Precision=0.9969, Attack Recall=0.9911, Attack Precision=0.4712

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
0.15       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046   <--
0.20       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.25       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.30       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.35       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.40       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.45       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.50       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.55       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.60       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.65       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.70       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.75       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.80       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8029, F1=0.7511, Normal Recall=0.7223, Normal Precision=0.9948, Attack Recall=0.9911, Attack Precision=0.6046

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
0.15       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038   <--
0.20       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.25       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.30       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.35       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.40       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.45       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.50       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.55       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.60       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.65       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.70       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.75       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.80       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8296, F1=0.8231, Normal Recall=0.7219, Normal Precision=0.9919, Attack Recall=0.9911, Attack Precision=0.7038

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
0.15       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802   <--
0.20       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.25       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.30       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.35       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.40       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.45       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.50       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.55       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.60       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.65       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.70       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.75       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.80       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8559, F1=0.8731, Normal Recall=0.7207, Normal Precision=0.9878, Attack Recall=0.9911, Attack Precision=0.7802

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
0.15       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833   <--
0.20       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.25       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.30       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.35       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.40       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.45       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.50       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.55       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.60       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.65       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.70       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.75       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
0.80       0.7484   0.4407   0.7215   0.9986   0.9910   0.2833  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7484, F1=0.4407, Normal Recall=0.7215, Normal Precision=0.9986, Attack Recall=0.9910, Attack Precision=0.2833

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
0.15       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712   <--
0.20       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.25       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.30       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.35       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.40       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.45       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.50       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.55       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.60       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.65       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.70       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.75       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
0.80       0.7757   0.6387   0.7219   0.9969   0.9911   0.4712  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7757, F1=0.6387, Normal Recall=0.7219, Normal Precision=0.9969, Attack Recall=0.9911, Attack Precision=0.4712

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
0.15       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046   <--
0.20       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.25       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.30       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.35       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.40       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.45       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.50       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.55       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.60       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.65       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.70       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.75       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
0.80       0.8029   0.7511   0.7223   0.9948   0.9911   0.6046  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8029, F1=0.7511, Normal Recall=0.7223, Normal Precision=0.9948, Attack Recall=0.9911, Attack Precision=0.6046

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
0.15       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038   <--
0.20       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.25       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.30       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.35       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.40       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.45       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.50       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.55       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.60       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.65       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.70       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.75       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
0.80       0.8296   0.8231   0.7219   0.9919   0.9911   0.7038  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8296, F1=0.8231, Normal Recall=0.7219, Normal Precision=0.9919, Attack Recall=0.9911, Attack Precision=0.7038

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
0.15       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802   <--
0.20       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.25       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.30       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.35       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.40       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.45       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.50       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.55       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.60       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.65       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.70       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.75       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
0.80       0.8559   0.8731   0.7207   0.9878   0.9911   0.7802  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8559, F1=0.8731, Normal Recall=0.7207, Normal Precision=0.9878, Attack Recall=0.9911, Attack Precision=0.7802

```

