# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r80_b64_e2_lr5e-4_a0.5.yaml` |
| **Generated** | 2026-02-16 19:20:26 |

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
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5982 | 0.5976 | 0.5975 | 0.5990 | 0.5986 | 0.6005 | 0.6014 | 0.6025 | 0.6019 | 0.6018 | 0.6034 |
| QAT+Prune only | 0.9476 | 0.9439 | 0.9398 | 0.9365 | 0.9316 | 0.9280 | 0.9242 | 0.9203 | 0.9157 | 0.9118 | 0.9082 |
| QAT+PTQ | 0.9477 | 0.9440 | 0.9400 | 0.9367 | 0.9319 | 0.9282 | 0.9246 | 0.9205 | 0.9161 | 0.9123 | 0.9087 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9477 | 0.9440 | 0.9400 | 0.9367 | 0.9319 | 0.9282 | 0.9246 | 0.9205 | 0.9161 | 0.9123 | 0.9087 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2307 | 0.3749 | 0.4744 | 0.5460 | 0.6016 | 0.6449 | 0.6800 | 0.7080 | 0.7317 | 0.7526 |
| QAT+Prune only | 0.0000 | 0.7640 | 0.8579 | 0.8956 | 0.9140 | 0.9266 | 0.9350 | 0.9410 | 0.9452 | 0.9488 | 0.9519 |
| QAT+PTQ | 0.0000 | 0.7644 | 0.8582 | 0.8959 | 0.9143 | 0.9268 | 0.9354 | 0.9412 | 0.9454 | 0.9491 | 0.9521 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7644 | 0.8582 | 0.8959 | 0.9143 | 0.9268 | 0.9354 | 0.9412 | 0.9454 | 0.9491 | 0.9521 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.5982 | 0.5970 | 0.5961 | 0.5971 | 0.5954 | 0.5976 | 0.5984 | 0.6005 | 0.5963 | 0.5880 | 0.0000 |
| QAT+Prune only | 0.9476 | 0.9479 | 0.9478 | 0.9487 | 0.9472 | 0.9479 | 0.9483 | 0.9485 | 0.9459 | 0.9447 | 0.0000 |
| QAT+PTQ | 0.9477 | 0.9479 | 0.9478 | 0.9487 | 0.9473 | 0.9477 | 0.9486 | 0.9481 | 0.9458 | 0.9450 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9477 | 0.9479 | 0.9478 | 0.9487 | 0.9473 | 0.9477 | 0.9486 | 0.9481 | 0.9458 | 0.9450 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.5982 | 0.0000 | 0.0000 | 0.0000 | 0.5982 | 1.0000 |
| 90 | 10 | 299,940 | 0.5976 | 0.1426 | 0.6033 | 0.2307 | 0.5970 | 0.9312 |
| 80 | 20 | 291,350 | 0.5975 | 0.2719 | 0.6034 | 0.3749 | 0.5961 | 0.8574 |
| 70 | 30 | 194,230 | 0.5990 | 0.3909 | 0.6034 | 0.4744 | 0.5971 | 0.7784 |
| 60 | 40 | 145,675 | 0.5986 | 0.4985 | 0.6034 | 0.5460 | 0.5954 | 0.6925 |
| 50 | 50 | 116,540 | 0.6005 | 0.5999 | 0.6034 | 0.6016 | 0.5976 | 0.6010 |
| 40 | 60 | 97,115 | 0.6014 | 0.6926 | 0.6034 | 0.6449 | 0.5984 | 0.5014 |
| 30 | 70 | 83,240 | 0.6025 | 0.7789 | 0.6034 | 0.6800 | 0.6005 | 0.3935 |
| 20 | 80 | 72,835 | 0.6019 | 0.8567 | 0.6034 | 0.7080 | 0.5963 | 0.2732 |
| 10 | 90 | 64,740 | 0.6018 | 0.9295 | 0.6034 | 0.7317 | 0.5880 | 0.1414 |
| 0 | 100 | 58,270 | 0.6034 | 1.0000 | 0.6034 | 0.7526 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9476 | 0.0000 | 0.0000 | 0.0000 | 0.9476 | 1.0000 |
| 90 | 10 | 299,940 | 0.9439 | 0.6594 | 0.9080 | 0.7640 | 0.9479 | 0.9893 |
| 80 | 20 | 291,350 | 0.9398 | 0.8130 | 0.9082 | 0.8579 | 0.9478 | 0.9763 |
| 70 | 30 | 194,230 | 0.9365 | 0.8835 | 0.9082 | 0.8956 | 0.9487 | 0.9602 |
| 60 | 40 | 145,675 | 0.9316 | 0.9198 | 0.9082 | 0.9140 | 0.9472 | 0.9393 |
| 50 | 50 | 116,540 | 0.9280 | 0.9457 | 0.9082 | 0.9266 | 0.9479 | 0.9117 |
| 40 | 60 | 97,115 | 0.9242 | 0.9634 | 0.9082 | 0.9350 | 0.9483 | 0.8731 |
| 30 | 70 | 83,240 | 0.9203 | 0.9763 | 0.9081 | 0.9410 | 0.9485 | 0.8157 |
| 20 | 80 | 72,835 | 0.9157 | 0.9853 | 0.9082 | 0.9452 | 0.9459 | 0.7203 |
| 10 | 90 | 64,740 | 0.9118 | 0.9933 | 0.9082 | 0.9488 | 0.9447 | 0.5334 |
| 0 | 100 | 58,270 | 0.9082 | 1.0000 | 0.9082 | 0.9519 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9477 | 0.0000 | 0.0000 | 0.0000 | 0.9477 | 1.0000 |
| 90 | 10 | 299,940 | 0.9440 | 0.6597 | 0.9085 | 0.7644 | 0.9479 | 0.9894 |
| 80 | 20 | 291,350 | 0.9400 | 0.8131 | 0.9087 | 0.8582 | 0.9478 | 0.9765 |
| 70 | 30 | 194,230 | 0.9367 | 0.8835 | 0.9087 | 0.8959 | 0.9487 | 0.9604 |
| 60 | 40 | 145,675 | 0.9319 | 0.9200 | 0.9087 | 0.9143 | 0.9473 | 0.9396 |
| 50 | 50 | 116,540 | 0.9282 | 0.9456 | 0.9087 | 0.9268 | 0.9477 | 0.9121 |
| 40 | 60 | 97,115 | 0.9246 | 0.9637 | 0.9087 | 0.9354 | 0.9486 | 0.8738 |
| 30 | 70 | 83,240 | 0.9205 | 0.9761 | 0.9087 | 0.9412 | 0.9481 | 0.8165 |
| 20 | 80 | 72,835 | 0.9161 | 0.9853 | 0.9087 | 0.9454 | 0.9458 | 0.7214 |
| 10 | 90 | 64,740 | 0.9123 | 0.9933 | 0.9087 | 0.9491 | 0.9450 | 0.5348 |
| 0 | 100 | 58,270 | 0.9087 | 1.0000 | 0.9087 | 0.9521 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9477 | 0.0000 | 0.0000 | 0.0000 | 0.9477 | 1.0000 |
| 90 | 10 | 299,940 | 0.9440 | 0.6597 | 0.9085 | 0.7644 | 0.9479 | 0.9894 |
| 80 | 20 | 291,350 | 0.9400 | 0.8131 | 0.9087 | 0.8582 | 0.9478 | 0.9765 |
| 70 | 30 | 194,230 | 0.9367 | 0.8835 | 0.9087 | 0.8959 | 0.9487 | 0.9604 |
| 60 | 40 | 145,675 | 0.9319 | 0.9200 | 0.9087 | 0.9143 | 0.9473 | 0.9396 |
| 50 | 50 | 116,540 | 0.9282 | 0.9456 | 0.9087 | 0.9268 | 0.9477 | 0.9121 |
| 40 | 60 | 97,115 | 0.9246 | 0.9637 | 0.9087 | 0.9354 | 0.9486 | 0.8738 |
| 30 | 70 | 83,240 | 0.9205 | 0.9761 | 0.9087 | 0.9412 | 0.9481 | 0.8165 |
| 20 | 80 | 72,835 | 0.9161 | 0.9853 | 0.9087 | 0.9454 | 0.9458 | 0.7214 |
| 10 | 90 | 64,740 | 0.9123 | 0.9933 | 0.9087 | 0.9491 | 0.9450 | 0.5348 |
| 0 | 100 | 58,270 | 0.9087 | 1.0000 | 0.9087 | 0.9521 | 0.0000 | 0.0000 |


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
0.15       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424   <--
0.20       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.25       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.30       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.35       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.40       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.45       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.50       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.55       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.60       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.65       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.70       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.75       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
0.80       0.5975   0.2303   0.5970   0.9311   0.6023   0.1424  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5975, F1=0.2303, Normal Recall=0.5970, Normal Precision=0.9311, Attack Recall=0.6023, Attack Precision=0.1424

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
0.15       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723   <--
0.20       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.25       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.30       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.35       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.40       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.45       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.50       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.55       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.60       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.65       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.70       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.75       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
0.80       0.5982   0.3753   0.5969   0.8575   0.6034   0.2723  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5982, F1=0.3753, Normal Recall=0.5969, Normal Precision=0.8575, Attack Recall=0.6034, Attack Precision=0.2723

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
0.15       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917   <--
0.20       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.25       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.30       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.35       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.40       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.45       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.50       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.55       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.60       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.65       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.70       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.75       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
0.80       0.5999   0.4750   0.5985   0.7788   0.6034   0.3917  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5999, F1=0.4750, Normal Recall=0.5985, Normal Precision=0.7788, Attack Recall=0.6034, Attack Precision=0.3917

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
0.15       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002   <--
0.20       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.25       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.30       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.35       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.40       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.45       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.50       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.55       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.60       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.65       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.70       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.75       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
0.80       0.6002   0.5470   0.5981   0.6934   0.6034   0.5002  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6002, F1=0.5470, Normal Recall=0.5981, Normal Precision=0.6934, Attack Recall=0.6034, Attack Precision=0.5002

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
0.15       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014   <--
0.20       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.25       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.30       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.35       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.40       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.45       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.50       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.55       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.60       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.65       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.70       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.75       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
0.80       0.6017   0.6024   0.6001   0.6020   0.6034   0.6014  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.6017, F1=0.6024, Normal Recall=0.6001, Normal Precision=0.6020, Attack Recall=0.6034, Attack Precision=0.6014

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
0.15       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595   <--
0.20       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.25       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.30       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.35       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.40       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.45       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.50       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.55       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.60       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.65       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.70       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.75       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
0.80       0.9439   0.7641   0.9479   0.9894   0.9082   0.6595  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9439, F1=0.7641, Normal Recall=0.9479, Normal Precision=0.9894, Attack Recall=0.9082, Attack Precision=0.6595

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
0.15       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134   <--
0.20       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.25       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.30       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.35       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.40       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.45       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.50       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.55       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.60       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.65       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.70       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.75       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
0.80       0.9400   0.8582   0.9479   0.9763   0.9082   0.8134  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9400, F1=0.8582, Normal Recall=0.9479, Normal Precision=0.9763, Attack Recall=0.9082, Attack Precision=0.8134

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
0.15       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818   <--
0.20       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.25       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.30       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.35       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.40       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.45       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.50       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.55       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.60       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.65       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.70       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.75       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
0.80       0.9359   0.8948   0.9478   0.9601   0.9082   0.8818  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9359, F1=0.8948, Normal Recall=0.9478, Normal Precision=0.9601, Attack Recall=0.9082, Attack Precision=0.8818

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
0.15       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208   <--
0.20       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.25       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.30       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.35       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.40       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.45       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.50       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.55       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.60       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.65       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.70       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.75       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
0.80       0.9320   0.9144   0.9479   0.9393   0.9082   0.9208  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9320, F1=0.9144, Normal Recall=0.9479, Normal Precision=0.9393, Attack Recall=0.9082, Attack Precision=0.9208

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
0.15       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455   <--
0.20       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.25       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.30       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.35       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.40       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.45       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.50       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.55       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.60       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.65       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.70       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.75       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
0.80       0.9279   0.9264   0.9476   0.9116   0.9082   0.9455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9279, F1=0.9264, Normal Recall=0.9476, Normal Precision=0.9116, Attack Recall=0.9082, Attack Precision=0.9455

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
0.15       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597   <--
0.20       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.25       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.30       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.35       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.40       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.45       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.50       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.55       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.60       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.65       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.70       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.75       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.80       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9440, F1=0.7644, Normal Recall=0.9479, Normal Precision=0.9894, Attack Recall=0.9086, Attack Precision=0.6597

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
0.15       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136   <--
0.20       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.25       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.30       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.35       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.40       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.45       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.50       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.55       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.60       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.65       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.70       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.75       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.80       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9401, F1=0.8585, Normal Recall=0.9479, Normal Precision=0.9765, Attack Recall=0.9087, Attack Precision=0.8136

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
0.15       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820   <--
0.20       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.25       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.30       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.35       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.40       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.45       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.50       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.55       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.60       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.65       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.70       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.75       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.80       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9361, F1=0.8951, Normal Recall=0.9479, Normal Precision=0.9603, Attack Recall=0.9087, Attack Precision=0.8820

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
0.15       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210   <--
0.20       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.25       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.30       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.35       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.40       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.45       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.50       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.55       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.60       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.65       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.70       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.75       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.80       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9323, F1=0.9148, Normal Recall=0.9480, Normal Precision=0.9396, Attack Recall=0.9087, Attack Precision=0.9210

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
0.15       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455   <--
0.20       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.25       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.30       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.35       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.40       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.45       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.50       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.55       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.60       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.65       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.70       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.75       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.80       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9282, F1=0.9267, Normal Recall=0.9476, Normal Precision=0.9121, Attack Recall=0.9087, Attack Precision=0.9455

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
0.15       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597   <--
0.20       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.25       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.30       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.35       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.40       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.45       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.50       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.55       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.60       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.65       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.70       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.75       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
0.80       0.9440   0.7644   0.9479   0.9894   0.9086   0.6597  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9440, F1=0.7644, Normal Recall=0.9479, Normal Precision=0.9894, Attack Recall=0.9086, Attack Precision=0.6597

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
0.15       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136   <--
0.20       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.25       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.30       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.35       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.40       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.45       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.50       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.55       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.60       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.65       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.70       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.75       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
0.80       0.9401   0.8585   0.9479   0.9765   0.9087   0.8136  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9401, F1=0.8585, Normal Recall=0.9479, Normal Precision=0.9765, Attack Recall=0.9087, Attack Precision=0.8136

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
0.15       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820   <--
0.20       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.25       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.30       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.35       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.40       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.45       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.50       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.55       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.60       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.65       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.70       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.75       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
0.80       0.9361   0.8951   0.9479   0.9603   0.9087   0.8820  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9361, F1=0.8951, Normal Recall=0.9479, Normal Precision=0.9603, Attack Recall=0.9087, Attack Precision=0.8820

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
0.15       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210   <--
0.20       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.25       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.30       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.35       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.40       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.45       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.50       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.55       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.60       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.65       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.70       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.75       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
0.80       0.9323   0.9148   0.9480   0.9396   0.9087   0.9210  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9323, F1=0.9148, Normal Recall=0.9480, Normal Precision=0.9396, Attack Recall=0.9087, Attack Precision=0.9210

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
0.15       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455   <--
0.20       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.25       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.30       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.35       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.40       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.45       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.50       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.55       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.60       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.65       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.70       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.75       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
0.80       0.9282   0.9267   0.9476   0.9121   0.9087   0.9455  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9282, F1=0.9267, Normal Recall=0.9476, Normal Precision=0.9121, Attack Recall=0.9087, Attack Precision=0.9455

```

