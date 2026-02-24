# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr1e-3_a0.8.yaml` |
| **Generated** | 2026-02-19 06:10:16 |

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
| **Batch size** | 256 |
| **Learning rate** | 0.001 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7597 | 0.7605 | 0.7608 | 0.7617 | 0.7625 | 0.7640 | 0.7640 | 0.7667 | 0.7655 | 0.7670 | 0.7682 |
| QAT+Prune only | 0.8287 | 0.8442 | 0.8594 | 0.8752 | 0.8913 | 0.9045 | 0.9223 | 0.9376 | 0.9534 | 0.9680 | 0.9845 |
| QAT+PTQ | 0.8278 | 0.8436 | 0.8589 | 0.8747 | 0.8909 | 0.9041 | 0.9221 | 0.9375 | 0.9533 | 0.9680 | 0.9845 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.8278 | 0.8436 | 0.8589 | 0.8747 | 0.8909 | 0.9041 | 0.9221 | 0.9375 | 0.9533 | 0.9680 | 0.9845 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.3909 | 0.5623 | 0.6592 | 0.7213 | 0.7650 | 0.7962 | 0.8218 | 0.8398 | 0.8558 | 0.8689 |
| QAT+Prune only | 0.0000 | 0.5583 | 0.7369 | 0.8256 | 0.8787 | 0.9116 | 0.9383 | 0.9567 | 0.9713 | 0.9823 | 0.9922 |
| QAT+PTQ | 0.0000 | 0.5574 | 0.7362 | 0.8250 | 0.8783 | 0.9112 | 0.9381 | 0.9566 | 0.9712 | 0.9823 | 0.9922 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.5574 | 0.7362 | 0.8250 | 0.8783 | 0.9112 | 0.9381 | 0.9566 | 0.9712 | 0.9823 | 0.9922 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.7597 | 0.7597 | 0.7589 | 0.7590 | 0.7587 | 0.7599 | 0.7579 | 0.7634 | 0.7547 | 0.7564 | 0.0000 |
| QAT+Prune only | 0.8287 | 0.8286 | 0.8282 | 0.8284 | 0.8291 | 0.8246 | 0.8290 | 0.8280 | 0.8289 | 0.8196 | 0.0000 |
| QAT+PTQ | 0.8278 | 0.8280 | 0.8275 | 0.8276 | 0.8285 | 0.8237 | 0.8284 | 0.8278 | 0.8285 | 0.8196 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.8278 | 0.8280 | 0.8275 | 0.8276 | 0.8285 | 0.8237 | 0.8284 | 0.8278 | 0.8285 | 0.8196 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7597 | 0.0000 | 0.0000 | 0.0000 | 0.7597 | 1.0000 |
| 90 | 10 | 299,940 | 0.7605 | 0.2621 | 0.7684 | 0.3909 | 0.7597 | 0.9672 |
| 80 | 20 | 291,350 | 0.7608 | 0.4434 | 0.7682 | 0.5623 | 0.7589 | 0.9291 |
| 70 | 30 | 194,230 | 0.7617 | 0.5773 | 0.7682 | 0.6592 | 0.7590 | 0.8842 |
| 60 | 40 | 145,675 | 0.7625 | 0.6797 | 0.7682 | 0.7213 | 0.7587 | 0.8308 |
| 50 | 50 | 116,540 | 0.7640 | 0.7619 | 0.7682 | 0.7650 | 0.7599 | 0.7662 |
| 40 | 60 | 97,115 | 0.7640 | 0.8263 | 0.7682 | 0.7962 | 0.7579 | 0.6855 |
| 30 | 70 | 83,240 | 0.7667 | 0.8834 | 0.7682 | 0.8218 | 0.7634 | 0.5853 |
| 20 | 80 | 72,835 | 0.7655 | 0.9261 | 0.7682 | 0.8398 | 0.7547 | 0.4487 |
| 10 | 90 | 64,740 | 0.7670 | 0.9660 | 0.7681 | 0.8558 | 0.7564 | 0.2661 |
| 0 | 100 | 58,270 | 0.7682 | 1.0000 | 0.7682 | 0.8689 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8287 | 0.0000 | 0.0000 | 0.0000 | 0.8287 | 1.0000 |
| 90 | 10 | 299,940 | 0.8442 | 0.3896 | 0.9845 | 0.5583 | 0.8286 | 0.9979 |
| 80 | 20 | 291,350 | 0.8594 | 0.5889 | 0.9845 | 0.7369 | 0.8282 | 0.9953 |
| 70 | 30 | 194,230 | 0.8752 | 0.7109 | 0.9845 | 0.8256 | 0.8284 | 0.9921 |
| 60 | 40 | 145,675 | 0.8913 | 0.7934 | 0.9845 | 0.8787 | 0.8291 | 0.9877 |
| 50 | 50 | 116,540 | 0.9045 | 0.8488 | 0.9845 | 0.9116 | 0.8246 | 0.9816 |
| 40 | 60 | 97,115 | 0.9223 | 0.8962 | 0.9845 | 0.9383 | 0.8290 | 0.9728 |
| 30 | 70 | 83,240 | 0.9376 | 0.9303 | 0.9845 | 0.9567 | 0.8280 | 0.9582 |
| 20 | 80 | 72,835 | 0.9534 | 0.9584 | 0.9845 | 0.9713 | 0.8289 | 0.9305 |
| 10 | 90 | 64,740 | 0.9680 | 0.9800 | 0.9845 | 0.9823 | 0.8196 | 0.8547 |
| 0 | 100 | 58,270 | 0.9845 | 1.0000 | 0.9845 | 0.9922 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.8278 | 0.0000 | 0.0000 | 0.0000 | 0.8278 | 1.0000 |
| 90 | 10 | 299,940 | 0.8436 | 0.3887 | 0.9846 | 0.5574 | 0.8280 | 0.9979 |
| 80 | 20 | 291,350 | 0.8589 | 0.5879 | 0.9845 | 0.7362 | 0.8275 | 0.9953 |
| 70 | 30 | 194,230 | 0.8747 | 0.7099 | 0.9845 | 0.8250 | 0.8276 | 0.9920 |
| 60 | 40 | 145,675 | 0.8909 | 0.7928 | 0.9845 | 0.8783 | 0.8285 | 0.9877 |
| 50 | 50 | 116,540 | 0.9041 | 0.8481 | 0.9845 | 0.9112 | 0.8237 | 0.9816 |
| 40 | 60 | 97,115 | 0.9221 | 0.8959 | 0.9845 | 0.9381 | 0.8284 | 0.9727 |
| 30 | 70 | 83,240 | 0.9375 | 0.9303 | 0.9845 | 0.9566 | 0.8278 | 0.9582 |
| 20 | 80 | 72,835 | 0.9533 | 0.9583 | 0.9845 | 0.9712 | 0.8285 | 0.9305 |
| 10 | 90 | 64,740 | 0.9680 | 0.9800 | 0.9845 | 0.9823 | 0.8196 | 0.8547 |
| 0 | 100 | 58,270 | 0.9845 | 1.0000 | 0.9845 | 0.9922 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.8278 | 0.0000 | 0.0000 | 0.0000 | 0.8278 | 1.0000 |
| 90 | 10 | 299,940 | 0.8436 | 0.3887 | 0.9846 | 0.5574 | 0.8280 | 0.9979 |
| 80 | 20 | 291,350 | 0.8589 | 0.5879 | 0.9845 | 0.7362 | 0.8275 | 0.9953 |
| 70 | 30 | 194,230 | 0.8747 | 0.7099 | 0.9845 | 0.8250 | 0.8276 | 0.9920 |
| 60 | 40 | 145,675 | 0.8909 | 0.7928 | 0.9845 | 0.8783 | 0.8285 | 0.9877 |
| 50 | 50 | 116,540 | 0.9041 | 0.8481 | 0.9845 | 0.9112 | 0.8237 | 0.9816 |
| 40 | 60 | 97,115 | 0.9221 | 0.8959 | 0.9845 | 0.9381 | 0.8284 | 0.9727 |
| 30 | 70 | 83,240 | 0.9375 | 0.9303 | 0.9845 | 0.9566 | 0.8278 | 0.9582 |
| 20 | 80 | 72,835 | 0.9533 | 0.9583 | 0.9845 | 0.9712 | 0.8285 | 0.9305 |
| 10 | 90 | 64,740 | 0.9680 | 0.9800 | 0.9845 | 0.9823 | 0.8196 | 0.8547 |
| 0 | 100 | 58,270 | 0.9845 | 1.0000 | 0.9845 | 0.9922 | 0.0000 | 0.0000 |


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
0.15       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622   <--
0.20       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.25       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.30       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.35       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.40       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.45       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.50       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.55       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.60       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.65       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.70       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.75       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
0.80       0.7606   0.3910   0.7597   0.9673   0.7687   0.2622  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7606, F1=0.3910, Normal Recall=0.7597, Normal Precision=0.9673, Attack Recall=0.7687, Attack Precision=0.2622

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
0.15       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442   <--
0.20       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.25       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.30       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.35       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.40       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.45       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.50       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.55       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.60       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.65       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.70       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.75       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
0.80       0.7614   0.5629   0.7597   0.9291   0.7682   0.4442  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7614, F1=0.5629, Normal Recall=0.7597, Normal Precision=0.9291, Attack Recall=0.7682, Attack Precision=0.4442

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
0.15       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782   <--
0.20       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.25       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.30       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.35       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.40       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.45       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.50       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.55       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.60       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.65       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.70       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.75       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
0.80       0.7623   0.6598   0.7598   0.8844   0.7682   0.5782  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7623, F1=0.6598, Normal Recall=0.7598, Normal Precision=0.8844, Attack Recall=0.7682, Attack Precision=0.5782

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
0.15       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813   <--
0.20       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.25       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.30       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.35       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.40       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.45       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.50       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.55       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.60       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.65       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.70       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.75       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
0.80       0.7636   0.7222   0.7605   0.8311   0.7682   0.6813  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7636, F1=0.7222, Normal Recall=0.7605, Normal Precision=0.8311, Attack Recall=0.7682, Attack Precision=0.6813

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
0.15       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631   <--
0.20       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.25       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.30       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.35       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.40       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.45       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.50       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.55       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.60       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.65       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.70       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.75       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
0.80       0.7649   0.7656   0.7616   0.7666   0.7682   0.7631  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7649, F1=0.7656, Normal Recall=0.7616, Normal Precision=0.7666, Attack Recall=0.7682, Attack Precision=0.7631

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
0.15       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898   <--
0.20       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.25       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.30       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.35       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.40       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.45       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.50       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.55       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.60       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.65       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.70       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.75       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
0.80       0.8443   0.5586   0.8286   0.9980   0.9853   0.3898  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8443, F1=0.5586, Normal Recall=0.8286, Normal Precision=0.9980, Attack Recall=0.9853, Attack Precision=0.3898

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
0.15       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899   <--
0.20       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.25       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.30       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.35       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.40       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.45       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.50       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.55       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.60       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.65       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.70       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.75       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
0.80       0.8600   0.7377   0.8289   0.9954   0.9845   0.5899  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8600, F1=0.7377, Normal Recall=0.8289, Normal Precision=0.9954, Attack Recall=0.9845, Attack Precision=0.5899

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
0.15       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114   <--
0.20       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.25       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.30       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.35       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.40       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.45       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.50       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.55       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.60       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.65       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.70       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.75       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
0.80       0.8756   0.8260   0.8289   0.9921   0.9845   0.7114  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8756, F1=0.8260, Normal Recall=0.8289, Normal Precision=0.9921, Attack Recall=0.9845, Attack Precision=0.7114

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
0.15       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929   <--
0.20       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.25       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.30       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.35       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.40       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.45       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.50       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.55       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.60       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.65       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.70       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.75       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
0.80       0.8910   0.8784   0.8286   0.9877   0.9845   0.7929  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8910, F1=0.8784, Normal Recall=0.8286, Normal Precision=0.9877, Attack Recall=0.9845, Attack Precision=0.7929

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
0.15       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509   <--
0.20       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.25       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.30       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.35       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.40       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.45       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.50       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.55       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.60       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.65       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.70       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.75       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
0.80       0.9060   0.9129   0.8275   0.9816   0.9845   0.8509  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9060, F1=0.9129, Normal Recall=0.8275, Normal Precision=0.9816, Attack Recall=0.9845, Attack Precision=0.8509

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
0.15       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889   <--
0.20       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.25       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.30       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.35       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.40       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.45       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.50       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.55       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.60       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.65       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.70       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.75       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.80       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8437, F1=0.5577, Normal Recall=0.8280, Normal Precision=0.9980, Attack Recall=0.9854, Attack Precision=0.3889

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
0.15       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889   <--
0.20       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.25       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.30       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.35       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.40       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.45       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.50       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.55       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.60       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.65       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.70       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.75       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.80       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8595, F1=0.7370, Normal Recall=0.8282, Normal Precision=0.9953, Attack Recall=0.9845, Attack Precision=0.5889

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
0.15       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104   <--
0.20       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.25       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.30       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.35       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.40       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.45       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.50       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.55       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.60       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.65       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.70       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.75       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.80       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8750, F1=0.8253, Normal Recall=0.8280, Normal Precision=0.9921, Attack Recall=0.9845, Attack Precision=0.7104

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
0.15       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922   <--
0.20       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.25       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.30       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.35       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.40       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.45       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.50       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.55       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.60       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.65       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.70       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.75       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.80       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8905, F1=0.8780, Normal Recall=0.8279, Normal Precision=0.9877, Attack Recall=0.9845, Attack Precision=0.7922

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
0.15       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504   <--
0.20       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.25       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.30       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.35       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.40       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.45       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.50       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.55       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.60       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.65       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.70       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.75       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.80       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9057, F1=0.9126, Normal Recall=0.8268, Normal Precision=0.9816, Attack Recall=0.9845, Attack Precision=0.8504

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
0.15       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889   <--
0.20       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.25       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.30       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.35       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.40       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.45       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.50       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.55       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.60       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.65       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.70       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.75       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
0.80       0.8437   0.5577   0.8280   0.9980   0.9854   0.3889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8437, F1=0.5577, Normal Recall=0.8280, Normal Precision=0.9980, Attack Recall=0.9854, Attack Precision=0.3889

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
0.15       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889   <--
0.20       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.25       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.30       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.35       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.40       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.45       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.50       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.55       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.60       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.65       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.70       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.75       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
0.80       0.8595   0.7370   0.8282   0.9953   0.9845   0.5889  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8595, F1=0.7370, Normal Recall=0.8282, Normal Precision=0.9953, Attack Recall=0.9845, Attack Precision=0.5889

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
0.15       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104   <--
0.20       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.25       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.30       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.35       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.40       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.45       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.50       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.55       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.60       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.65       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.70       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.75       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
0.80       0.8750   0.8253   0.8280   0.9921   0.9845   0.7104  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8750, F1=0.8253, Normal Recall=0.8280, Normal Precision=0.9921, Attack Recall=0.9845, Attack Precision=0.7104

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
0.15       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922   <--
0.20       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.25       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.30       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.35       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.40       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.45       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.50       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.55       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.60       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.65       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.70       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.75       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
0.80       0.8905   0.8780   0.8279   0.9877   0.9845   0.7922  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8905, F1=0.8780, Normal Recall=0.8279, Normal Precision=0.9877, Attack Recall=0.9845, Attack Precision=0.7922

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
0.15       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504   <--
0.20       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.25       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.30       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.35       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.40       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.45       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.50       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.55       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.60       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.65       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.70       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.75       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
0.80       0.9057   0.9126   0.8268   0.9816   0.9845   0.8504  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9057, F1=0.9126, Normal Recall=0.8268, Normal Precision=0.9816, Attack Recall=0.9845, Attack Precision=0.8504

```

