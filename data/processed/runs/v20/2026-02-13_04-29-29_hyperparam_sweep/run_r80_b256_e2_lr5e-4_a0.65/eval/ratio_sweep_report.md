# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b256_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-19 00:05:36 |

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
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9071 | 0.8795 | 0.8515 | 0.8234 | 0.7955 | 0.7678 | 0.7395 | 0.7121 | 0.6840 | 0.6555 | 0.6279 |
| QAT+Prune only | 0.7712 | 0.7642 | 0.7565 | 0.7498 | 0.7425 | 0.7335 | 0.7286 | 0.7213 | 0.7149 | 0.7061 | 0.6998 |
| QAT+PTQ | 0.7745 | 0.7672 | 0.7592 | 0.7523 | 0.7445 | 0.7354 | 0.7301 | 0.7223 | 0.7155 | 0.7066 | 0.7001 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.7745 | 0.7672 | 0.7592 | 0.7523 | 0.7445 | 0.7354 | 0.7301 | 0.7223 | 0.7155 | 0.7066 | 0.7001 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.5093 | 0.6284 | 0.6808 | 0.7106 | 0.7300 | 0.7431 | 0.7533 | 0.7607 | 0.7664 | 0.7714 |
| QAT+Prune only | 0.0000 | 0.3721 | 0.5348 | 0.6266 | 0.6850 | 0.7242 | 0.7558 | 0.7785 | 0.7970 | 0.8108 | 0.8234 |
| QAT+PTQ | 0.0000 | 0.3753 | 0.5377 | 0.6290 | 0.6867 | 0.7257 | 0.7568 | 0.7792 | 0.7974 | 0.8112 | 0.8236 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.3753 | 0.5377 | 0.6290 | 0.6867 | 0.7257 | 0.7568 | 0.7792 | 0.7974 | 0.8112 | 0.8236 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.9071 | 0.9077 | 0.9074 | 0.9072 | 0.9072 | 0.9077 | 0.9069 | 0.9085 | 0.9084 | 0.9039 | 0.0000 |
| QAT+Prune only | 0.7712 | 0.7714 | 0.7707 | 0.7712 | 0.7710 | 0.7672 | 0.7718 | 0.7713 | 0.7750 | 0.7626 | 0.0000 |
| QAT+PTQ | 0.7745 | 0.7747 | 0.7740 | 0.7746 | 0.7741 | 0.7708 | 0.7750 | 0.7741 | 0.7770 | 0.7657 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.7745 | 0.7747 | 0.7740 | 0.7746 | 0.7741 | 0.7708 | 0.7750 | 0.7741 | 0.7770 | 0.7657 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9071 | 0.0000 | 0.0000 | 0.0000 | 0.9071 | 1.0000 |
| 90 | 10 | 299,940 | 0.8795 | 0.4296 | 0.6254 | 0.5093 | 0.9077 | 0.9562 |
| 80 | 20 | 291,350 | 0.8515 | 0.6289 | 0.6279 | 0.6284 | 0.9074 | 0.9070 |
| 70 | 30 | 194,230 | 0.8234 | 0.7435 | 0.6279 | 0.6808 | 0.9072 | 0.8505 |
| 60 | 40 | 145,675 | 0.7955 | 0.8186 | 0.6279 | 0.7106 | 0.9072 | 0.7853 |
| 50 | 50 | 116,540 | 0.7678 | 0.8719 | 0.6279 | 0.7300 | 0.9077 | 0.7092 |
| 40 | 60 | 97,115 | 0.7395 | 0.9100 | 0.6279 | 0.7431 | 0.9069 | 0.6190 |
| 30 | 70 | 83,240 | 0.7121 | 0.9412 | 0.6279 | 0.7533 | 0.9085 | 0.5113 |
| 20 | 80 | 72,835 | 0.6840 | 0.9648 | 0.6279 | 0.7607 | 0.9084 | 0.3790 |
| 10 | 90 | 64,740 | 0.6555 | 0.9833 | 0.6279 | 0.7664 | 0.9039 | 0.2125 |
| 0 | 100 | 58,270 | 0.6279 | 1.0000 | 0.6279 | 0.7714 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7712 | 0.0000 | 0.0000 | 0.0000 | 0.7712 | 1.0000 |
| 90 | 10 | 299,940 | 0.7642 | 0.2536 | 0.6989 | 0.3721 | 0.7714 | 0.9584 |
| 80 | 20 | 291,350 | 0.7565 | 0.4328 | 0.6998 | 0.5348 | 0.7707 | 0.9113 |
| 70 | 30 | 194,230 | 0.7498 | 0.5672 | 0.6998 | 0.6266 | 0.7712 | 0.8570 |
| 60 | 40 | 145,675 | 0.7425 | 0.6708 | 0.6998 | 0.6850 | 0.7710 | 0.7939 |
| 50 | 50 | 116,540 | 0.7335 | 0.7504 | 0.6998 | 0.7242 | 0.7672 | 0.7188 |
| 40 | 60 | 97,115 | 0.7286 | 0.8215 | 0.6998 | 0.7558 | 0.7718 | 0.6316 |
| 30 | 70 | 83,240 | 0.7213 | 0.8772 | 0.6998 | 0.7785 | 0.7713 | 0.5241 |
| 20 | 80 | 72,835 | 0.7149 | 0.9256 | 0.6998 | 0.7970 | 0.7750 | 0.3923 |
| 10 | 90 | 64,740 | 0.7061 | 0.9637 | 0.6998 | 0.8108 | 0.7626 | 0.2201 |
| 0 | 100 | 58,270 | 0.6998 | 1.0000 | 0.6998 | 0.8234 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.7745 | 0.0000 | 0.0000 | 0.0000 | 0.7745 | 1.0000 |
| 90 | 10 | 299,940 | 0.7672 | 0.2565 | 0.6993 | 0.3753 | 0.7747 | 0.9587 |
| 80 | 20 | 291,350 | 0.7592 | 0.4364 | 0.7001 | 0.5377 | 0.7740 | 0.9117 |
| 70 | 30 | 194,230 | 0.7523 | 0.5711 | 0.7001 | 0.6290 | 0.7746 | 0.8577 |
| 60 | 40 | 145,675 | 0.7445 | 0.6738 | 0.7001 | 0.6867 | 0.7741 | 0.7947 |
| 50 | 50 | 116,540 | 0.7354 | 0.7533 | 0.7001 | 0.7257 | 0.7708 | 0.7199 |
| 40 | 60 | 97,115 | 0.7301 | 0.8236 | 0.7001 | 0.7568 | 0.7750 | 0.6327 |
| 30 | 70 | 83,240 | 0.7223 | 0.8785 | 0.7001 | 0.7792 | 0.7741 | 0.5252 |
| 20 | 80 | 72,835 | 0.7155 | 0.9262 | 0.7001 | 0.7974 | 0.7770 | 0.3931 |
| 10 | 90 | 64,740 | 0.7066 | 0.9641 | 0.7001 | 0.8112 | 0.7657 | 0.2210 |
| 0 | 100 | 58,270 | 0.7001 | 1.0000 | 0.7001 | 0.8236 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.7745 | 0.0000 | 0.0000 | 0.0000 | 0.7745 | 1.0000 |
| 90 | 10 | 299,940 | 0.7672 | 0.2565 | 0.6993 | 0.3753 | 0.7747 | 0.9587 |
| 80 | 20 | 291,350 | 0.7592 | 0.4364 | 0.7001 | 0.5377 | 0.7740 | 0.9117 |
| 70 | 30 | 194,230 | 0.7523 | 0.5711 | 0.7001 | 0.6290 | 0.7746 | 0.8577 |
| 60 | 40 | 145,675 | 0.7445 | 0.6738 | 0.7001 | 0.6867 | 0.7741 | 0.7947 |
| 50 | 50 | 116,540 | 0.7354 | 0.7533 | 0.7001 | 0.7257 | 0.7708 | 0.7199 |
| 40 | 60 | 97,115 | 0.7301 | 0.8236 | 0.7001 | 0.7568 | 0.7750 | 0.6327 |
| 30 | 70 | 83,240 | 0.7223 | 0.8785 | 0.7001 | 0.7792 | 0.7741 | 0.5252 |
| 20 | 80 | 72,835 | 0.7155 | 0.9262 | 0.7001 | 0.7974 | 0.7770 | 0.3931 |
| 10 | 90 | 64,740 | 0.7066 | 0.9641 | 0.7001 | 0.8112 | 0.7657 | 0.2210 |
| 0 | 100 | 58,270 | 0.7001 | 1.0000 | 0.7001 | 0.8236 | 0.0000 | 0.0000 |


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
0.15       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290   <--
0.20       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.25       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.30       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.35       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.40       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.45       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.50       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.55       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.60       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.65       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.70       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.75       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
0.80       0.8794   0.5085   0.9077   0.9560   0.6240   0.4290  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8794, F1=0.5085, Normal Recall=0.9077, Normal Precision=0.9560, Attack Recall=0.6240, Attack Precision=0.4290

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
0.15       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299   <--
0.20       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.25       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.30       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.35       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.40       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.45       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.50       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.55       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.60       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.65       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.70       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.75       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
0.80       0.8518   0.6289   0.9078   0.9070   0.6279   0.6299  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8518, F1=0.6289, Normal Recall=0.9078, Normal Precision=0.9070, Attack Recall=0.6279, Attack Precision=0.6299

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
0.15       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436   <--
0.20       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.25       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.30       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.35       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.40       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.45       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.50       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.55       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.60       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.65       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.70       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.75       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
0.80       0.8234   0.6808   0.9072   0.8505   0.6279   0.7436  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.8234, F1=0.6808, Normal Recall=0.9072, Normal Precision=0.8505, Attack Recall=0.6279, Attack Precision=0.7436

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
0.15       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173   <--
0.20       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.25       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.30       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.35       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.40       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.45       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.50       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.55       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.60       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.65       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.70       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.75       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
0.80       0.7950   0.7102   0.9064   0.7851   0.6279   0.8173  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7950, F1=0.7102, Normal Recall=0.9064, Normal Precision=0.7851, Attack Recall=0.6279, Attack Precision=0.8173

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
0.15       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704   <--
0.20       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.25       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.30       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.35       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.40       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.45       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.50       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.55       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.60       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.65       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.70       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.75       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
0.80       0.7672   0.7295   0.9065   0.7090   0.6279   0.8704  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7672, F1=0.7295, Normal Recall=0.9065, Normal Precision=0.7090, Attack Recall=0.6279, Attack Precision=0.8704

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
0.15       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530   <--
0.20       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.25       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.30       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.35       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.40       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.45       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.50       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.55       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.60       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.65       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.70       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.75       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
0.80       0.7640   0.3713   0.7714   0.9582   0.6969   0.2530  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7640, F1=0.3713, Normal Recall=0.7714, Normal Precision=0.9582, Attack Recall=0.6969, Attack Precision=0.2530

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
0.15       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341   <--
0.20       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.25       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.30       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.35       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.40       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.45       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.50       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.55       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.60       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.65       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.70       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.75       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
0.80       0.7575   0.5359   0.7720   0.9114   0.6998   0.4341  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7575, F1=0.5359, Normal Recall=0.7720, Normal Precision=0.9114, Attack Recall=0.6998, Attack Precision=0.4341

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
0.15       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667   <--
0.20       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.25       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.30       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.35       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.40       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.45       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.50       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.55       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.60       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.65       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.70       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.75       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
0.80       0.7494   0.6263   0.7707   0.8570   0.6998   0.5667  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7494, F1=0.6263, Normal Recall=0.7707, Normal Precision=0.8570, Attack Recall=0.6998, Attack Precision=0.5667

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
0.15       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707   <--
0.20       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.25       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.30       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.35       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.40       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.45       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.50       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.55       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.60       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.65       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.70       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.75       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
0.80       0.7425   0.6850   0.7709   0.7939   0.6998   0.6707  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7425, F1=0.6850, Normal Recall=0.7709, Normal Precision=0.7939, Attack Recall=0.6998, Attack Precision=0.6707

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
0.15       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522   <--
0.20       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.25       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.30       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.35       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.40       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.45       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.50       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.55       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.60       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.65       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.70       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.75       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
0.80       0.7346   0.7251   0.7694   0.7194   0.6998   0.7522  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7346, F1=0.7251, Normal Recall=0.7694, Normal Precision=0.7194, Attack Recall=0.6998, Attack Precision=0.7522

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
0.15       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559   <--
0.20       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.25       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.30       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.35       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.40       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.45       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.50       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.55       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.60       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.65       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.70       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.75       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.80       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7670, F1=0.3744, Normal Recall=0.7747, Normal Precision=0.9584, Attack Recall=0.6972, Attack Precision=0.2559

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
0.15       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378   <--
0.20       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.25       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.30       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.35       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.40       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.45       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.50       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.55       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.60       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.65       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.70       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.75       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.80       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7602, F1=0.5387, Normal Recall=0.7753, Normal Precision=0.9118, Attack Recall=0.7001, Attack Precision=0.4378

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
0.15       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704   <--
0.20       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.25       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.30       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.35       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.40       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.45       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.50       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.55       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.60       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.65       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.70       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.75       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.80       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.6286, Normal Recall=0.7740, Normal Precision=0.8576, Attack Recall=0.7001, Attack Precision=0.5704

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
0.15       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741   <--
0.20       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.25       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.30       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.35       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.40       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.45       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.50       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.55       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.60       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.65       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.70       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.75       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.80       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7446, F1=0.6868, Normal Recall=0.7743, Normal Precision=0.7948, Attack Recall=0.7001, Attack Precision=0.6741

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
0.15       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550   <--
0.20       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.25       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.30       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.35       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.40       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.45       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.50       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.55       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.60       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.65       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.70       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.75       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.80       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7364, F1=0.7265, Normal Recall=0.7728, Normal Precision=0.7204, Attack Recall=0.7001, Attack Precision=0.7550

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
0.15       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559   <--
0.20       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.25       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.30       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.35       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.40       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.45       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.50       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.55       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.60       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.65       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.70       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.75       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
0.80       0.7670   0.3744   0.7747   0.9584   0.6972   0.2559  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7670, F1=0.3744, Normal Recall=0.7747, Normal Precision=0.9584, Attack Recall=0.6972, Attack Precision=0.2559

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
0.15       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378   <--
0.20       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.25       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.30       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.35       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.40       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.45       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.50       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.55       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.60       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.65       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.70       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.75       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
0.80       0.7602   0.5387   0.7753   0.9118   0.7001   0.4378  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7602, F1=0.5387, Normal Recall=0.7753, Normal Precision=0.9118, Attack Recall=0.7001, Attack Precision=0.4378

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
0.15       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704   <--
0.20       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.25       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.30       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.35       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.40       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.45       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.50       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.55       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.60       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.65       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.70       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.75       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
0.80       0.7518   0.6286   0.7740   0.8576   0.7001   0.5704  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7518, F1=0.6286, Normal Recall=0.7740, Normal Precision=0.8576, Attack Recall=0.7001, Attack Precision=0.5704

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
0.15       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741   <--
0.20       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.25       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.30       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.35       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.40       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.45       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.50       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.55       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.60       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.65       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.70       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.75       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
0.80       0.7446   0.6868   0.7743   0.7948   0.7001   0.6741  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7446, F1=0.6868, Normal Recall=0.7743, Normal Precision=0.7948, Attack Recall=0.7001, Attack Precision=0.6741

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
0.15       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550   <--
0.20       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.25       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.30       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.35       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.40       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.45       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.50       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.55       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.60       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.65       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.70       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.75       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
0.80       0.7364   0.7265   0.7728   0.7204   0.7001   0.7550  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.7364, F1=0.7265, Normal Recall=0.7728, Normal Precision=0.7204, Attack Recall=0.7001, Attack Precision=0.7550

```

