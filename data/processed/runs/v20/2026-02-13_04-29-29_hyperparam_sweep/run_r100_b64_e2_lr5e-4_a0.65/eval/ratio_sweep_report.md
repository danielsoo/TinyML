# Ratio Sweep Report (All Models)

| Item | Value |
|------|-------|
| **Models** | 5 models (same as compression_analysis) |
| **Config** | `config/hyperparam_sweep/federated_r100_b64_e2_lr5e-4_a0.65.yaml` |
| **Generated** | 2026-02-20 09:39:02 |

## Run / Training Configuration

| Item | Value |
|------|-------|
| **Data** | cicids2017 |
| **Max samples** | 2000000 |
| **Balance ratio** | 4.0 (normal:attack 8:2) |
| **Num clients** | 4 |
| **Model** | mlp |
| **FL rounds** | 100 |
| **Local epochs** | 2 |
| **Batch size** | 64 |
| **Learning rate** | 0.0005 |
| **Use QAT** | True |

## Summary

Total models: 5, Total ratios: 11

## Comparison — Accuracy (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3486 | 0.3913 | 0.4355 | 0.4801 | 0.5226 | 0.5662 | 0.6115 | 0.6541 | 0.6990 | 0.7436 | 0.7866 |
| QAT+Prune only | 0.9267 | 0.9304 | 0.9336 | 0.9382 | 0.9411 | 0.9444 | 0.9488 | 0.9519 | 0.9550 | 0.9592 | 0.9625 |
| QAT+PTQ | 0.9271 | 0.9306 | 0.9337 | 0.9382 | 0.9412 | 0.9446 | 0.9488 | 0.9517 | 0.9549 | 0.9591 | 0.9623 |
| noQAT+PTQ | 1.0000 | 0.9000 | 0.8000 | 0.7000 | 0.6000 | 0.5000 | 0.4000 | 0.3000 | 0.2000 | 0.1000 | 0.0000 |
| Compressed (PTQ) | 0.9271 | 0.9306 | 0.9337 | 0.9382 | 0.9412 | 0.9446 | 0.9488 | 0.9517 | 0.9549 | 0.9591 | 0.9623 |

## Comparison — F1-Score (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.0000 | 0.2054 | 0.3579 | 0.4758 | 0.5686 | 0.6445 | 0.7084 | 0.7610 | 0.8070 | 0.8467 | 0.8806 |
| QAT+Prune only | 0.0000 | 0.7345 | 0.8529 | 0.9033 | 0.9290 | 0.9454 | 0.9576 | 0.9656 | 0.9716 | 0.9770 | 0.9809 |
| QAT+PTQ | 0.0000 | 0.7350 | 0.8531 | 0.9033 | 0.9290 | 0.9456 | 0.9576 | 0.9654 | 0.9715 | 0.9769 | 0.9808 |
| noQAT+PTQ | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Compressed (PTQ) | 0.0000 | 0.7350 | 0.8531 | 0.9033 | 0.9290 | 0.9456 | 0.9576 | 0.9654 | 0.9715 | 0.9769 | 0.9808 |

## Comparison — Normal Recall (Model × Normal%)

| Model | 100 | 90 | 80 | 70 | 60 | 50 | 40 | 30 | 20 | 10 | 0 |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| Original (TFLite) | 0.3486 | 0.3474 | 0.3477 | 0.3487 | 0.3465 | 0.3457 | 0.3489 | 0.3448 | 0.3485 | 0.3567 | 0.0000 |
| QAT+Prune only | 0.9267 | 0.9269 | 0.9264 | 0.9277 | 0.9269 | 0.9263 | 0.9283 | 0.9272 | 0.9250 | 0.9291 | 0.0000 |
| QAT+PTQ | 0.9271 | 0.9271 | 0.9266 | 0.9279 | 0.9271 | 0.9269 | 0.9286 | 0.9270 | 0.9250 | 0.9297 | 0.0000 |
| noQAT+PTQ | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| Compressed (PTQ) | 0.9271 | 0.9271 | 0.9266 | 0.9279 | 0.9271 | 0.9269 | 0.9286 | 0.9270 | 0.9250 | 0.9297 | 0.0000 |

## Detailed (per model)

### Original (TFLite)

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.3486 | 0.0000 | 0.0000 | 0.0000 | 0.3486 | 1.0000 |
| 90 | 10 | 299,940 | 0.3913 | 0.1181 | 0.7867 | 0.2054 | 0.3474 | 0.9361 |
| 80 | 20 | 291,350 | 0.4355 | 0.2316 | 0.7866 | 0.3579 | 0.3477 | 0.8670 |
| 70 | 30 | 194,230 | 0.4801 | 0.3411 | 0.7866 | 0.4758 | 0.3487 | 0.7923 |
| 60 | 40 | 145,675 | 0.5226 | 0.4452 | 0.7866 | 0.5686 | 0.3465 | 0.7090 |
| 50 | 50 | 116,540 | 0.5662 | 0.5459 | 0.7866 | 0.6445 | 0.3457 | 0.6184 |
| 40 | 60 | 97,115 | 0.6115 | 0.6444 | 0.7866 | 0.7084 | 0.3489 | 0.5216 |
| 30 | 70 | 83,240 | 0.6541 | 0.7369 | 0.7866 | 0.7610 | 0.3448 | 0.4092 |
| 20 | 80 | 72,835 | 0.6990 | 0.8285 | 0.7866 | 0.8070 | 0.3485 | 0.2899 |
| 10 | 90 | 64,740 | 0.7436 | 0.9167 | 0.7866 | 0.8467 | 0.3567 | 0.1566 |
| 0 | 100 | 58,270 | 0.7866 | 1.0000 | 0.7866 | 0.8806 | 0.0000 | 0.0000 |

### QAT+Prune only

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9267 | 0.0000 | 0.0000 | 0.0000 | 0.9267 | 1.0000 |
| 90 | 10 | 299,940 | 0.9304 | 0.5939 | 0.9624 | 0.7345 | 0.9269 | 0.9955 |
| 80 | 20 | 291,350 | 0.9336 | 0.7658 | 0.9625 | 0.8529 | 0.9264 | 0.9900 |
| 70 | 30 | 194,230 | 0.9382 | 0.8509 | 0.9625 | 0.9033 | 0.9277 | 0.9830 |
| 60 | 40 | 145,675 | 0.9411 | 0.8977 | 0.9625 | 0.9290 | 0.9269 | 0.9738 |
| 50 | 50 | 116,540 | 0.9444 | 0.9289 | 0.9625 | 0.9454 | 0.9263 | 0.9611 |
| 40 | 60 | 97,115 | 0.9488 | 0.9527 | 0.9625 | 0.9576 | 0.9283 | 0.9429 |
| 30 | 70 | 83,240 | 0.9519 | 0.9686 | 0.9625 | 0.9656 | 0.9272 | 0.9138 |
| 20 | 80 | 72,835 | 0.9550 | 0.9809 | 0.9625 | 0.9716 | 0.9250 | 0.8606 |
| 10 | 90 | 64,740 | 0.9592 | 0.9919 | 0.9626 | 0.9770 | 0.9291 | 0.7338 |
| 0 | 100 | 58,270 | 0.9625 | 1.0000 | 0.9625 | 0.9809 | 0.0000 | 0.0000 |

### QAT+PTQ

| Normal% | Attack% | n_total | Accuracy | Precision | Recall | F1-Score | Normal Recall | Normal Precision |
|---------|----------|---------|----------|------------|--------|----------|---------------|-------------------|
| 100 | 0 | 100,000 | 0.9271 | 0.0000 | 0.0000 | 0.0000 | 0.9271 | 1.0000 |
| 90 | 10 | 299,940 | 0.9306 | 0.5946 | 0.9622 | 0.7350 | 0.9271 | 0.9955 |
| 80 | 20 | 291,350 | 0.9337 | 0.7662 | 0.9623 | 0.8531 | 0.9266 | 0.9899 |
| 70 | 30 | 194,230 | 0.9382 | 0.8512 | 0.9623 | 0.9033 | 0.9279 | 0.9829 |
| 60 | 40 | 145,675 | 0.9412 | 0.8979 | 0.9623 | 0.9290 | 0.9271 | 0.9736 |
| 50 | 50 | 116,540 | 0.9446 | 0.9294 | 0.9623 | 0.9456 | 0.9269 | 0.9609 |
| 40 | 60 | 97,115 | 0.9488 | 0.9529 | 0.9623 | 0.9576 | 0.9286 | 0.9426 |
| 30 | 70 | 83,240 | 0.9517 | 0.9685 | 0.9623 | 0.9654 | 0.9270 | 0.9134 |
| 20 | 80 | 72,835 | 0.9549 | 0.9809 | 0.9623 | 0.9715 | 0.9250 | 0.8599 |
| 10 | 90 | 64,740 | 0.9591 | 0.9920 | 0.9623 | 0.9769 | 0.9297 | 0.7328 |
| 0 | 100 | 58,270 | 0.9623 | 1.0000 | 0.9623 | 0.9808 | 0.0000 | 0.0000 |

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
| 100 | 0 | 100,000 | 0.9271 | 0.0000 | 0.0000 | 0.0000 | 0.9271 | 1.0000 |
| 90 | 10 | 299,940 | 0.9306 | 0.5946 | 0.9622 | 0.7350 | 0.9271 | 0.9955 |
| 80 | 20 | 291,350 | 0.9337 | 0.7662 | 0.9623 | 0.8531 | 0.9266 | 0.9899 |
| 70 | 30 | 194,230 | 0.9382 | 0.8512 | 0.9623 | 0.9033 | 0.9279 | 0.9829 |
| 60 | 40 | 145,675 | 0.9412 | 0.8979 | 0.9623 | 0.9290 | 0.9271 | 0.9736 |
| 50 | 50 | 116,540 | 0.9446 | 0.9294 | 0.9623 | 0.9456 | 0.9269 | 0.9609 |
| 40 | 60 | 97,115 | 0.9488 | 0.9529 | 0.9623 | 0.9576 | 0.9286 | 0.9426 |
| 30 | 70 | 83,240 | 0.9517 | 0.9685 | 0.9623 | 0.9654 | 0.9270 | 0.9134 |
| 20 | 80 | 72,835 | 0.9549 | 0.9809 | 0.9623 | 0.9715 | 0.9250 | 0.8599 |
| 10 | 90 | 64,740 | 0.9591 | 0.9920 | 0.9623 | 0.9769 | 0.9297 | 0.7328 |
| 0 | 100 | 58,270 | 0.9623 | 1.0000 | 0.9623 | 0.9808 | 0.0000 | 0.0000 |


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
0.15       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181   <--
0.20       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.25       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.30       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.35       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.40       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.45       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.50       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.55       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.60       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.65       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.70       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.75       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
0.80       0.3913   0.2053   0.3474   0.9361   0.7864   0.1181  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.3913, F1=0.2053, Normal Recall=0.3474, Normal Precision=0.9361, Attack Recall=0.7864, Attack Precision=0.1181

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
0.15       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316   <--
0.20       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.25       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.30       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.35       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.40       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.45       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.50       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.55       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.60       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.65       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.70       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.75       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
0.80       0.4353   0.3578   0.3474   0.8669   0.7866   0.2316  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4353, F1=0.3578, Normal Recall=0.3474, Normal Precision=0.8669, Attack Recall=0.7866, Attack Precision=0.2316

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
0.15       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411   <--
0.20       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.25       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.30       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.35       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.40       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.45       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.50       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.55       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.60       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.65       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.70       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.75       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
0.80       0.4801   0.4758   0.3487   0.7922   0.7866   0.3411  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.4801, F1=0.4758, Normal Recall=0.3487, Normal Precision=0.7922, Attack Recall=0.7866, Attack Precision=0.3411

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
0.15       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461   <--
0.20       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.25       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.30       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.35       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.40       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.45       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.50       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.55       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.60       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.65       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.70       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.75       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
0.80       0.5240   0.5694   0.3489   0.7104   0.7866   0.4461  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5240, F1=0.5694, Normal Recall=0.3489, Normal Precision=0.7104, Attack Recall=0.7866, Attack Precision=0.4461

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
0.15       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477   <--
0.20       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.25       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.30       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.35       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.40       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.45       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.50       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.55       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.60       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.65       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.70       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.75       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
0.80       0.5685   0.6458   0.3505   0.6216   0.7866   0.5477  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.5685, F1=0.6458, Normal Recall=0.3505, Normal Precision=0.6216, Attack Recall=0.7866, Attack Precision=0.5477

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
0.15       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940   <--
0.20       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.25       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.30       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.35       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.40       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.45       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.50       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.55       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.60       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.65       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.70       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.75       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
0.80       0.9305   0.7347   0.9269   0.9956   0.9628   0.5940  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9305, F1=0.7347, Normal Recall=0.9269, Normal Precision=0.9956, Attack Recall=0.9628, Attack Precision=0.5940

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
0.15       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673   <--
0.20       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.25       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.30       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.35       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.40       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.45       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.50       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.55       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.60       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.65       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.70       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.75       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
0.80       0.9341   0.8539   0.9270   0.9900   0.9625   0.7673  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9341, F1=0.8539, Normal Recall=0.9270, Normal Precision=0.9900, Attack Recall=0.9625, Attack Precision=0.7673

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
0.15       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499   <--
0.20       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.25       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.30       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.35       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.40       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.45       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.50       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.55       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.60       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.65       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.70       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.75       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
0.80       0.9378   0.9027   0.9272   0.9830   0.9625   0.8499  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9378, F1=0.9027, Normal Recall=0.9272, Normal Precision=0.9830, Attack Recall=0.9625, Attack Precision=0.8499

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
0.15       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978   <--
0.20       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.25       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.30       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.35       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.40       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.45       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.50       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.55       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.60       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.65       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.70       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.75       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
0.80       0.9412   0.9290   0.9269   0.9738   0.9625   0.8978  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9412, F1=0.9290, Normal Recall=0.9269, Normal Precision=0.9738, Attack Recall=0.9625, Attack Precision=0.8978

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
0.15       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293   <--
0.20       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.25       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.30       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.35       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.40       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.45       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.50       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.55       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.60       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.65       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.70       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.75       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
0.80       0.9447   0.9456   0.9268   0.9611   0.9625   0.9293  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9447, F1=0.9456, Normal Recall=0.9268, Normal Precision=0.9611, Attack Recall=0.9625, Attack Precision=0.9293

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
0.15       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947   <--
0.20       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.25       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.30       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.35       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.40       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.45       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.50       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.55       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.60       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.65       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.70       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.75       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.80       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9307, F1=0.7352, Normal Recall=0.9271, Normal Precision=0.9955, Attack Recall=0.9626, Attack Precision=0.5947

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
0.15       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679   <--
0.20       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.25       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.30       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.35       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.40       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.45       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.50       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.55       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.60       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.65       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.70       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.75       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.80       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9343, F1=0.8542, Normal Recall=0.9273, Normal Precision=0.9899, Attack Recall=0.9623, Attack Precision=0.7679

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
0.15       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503   <--
0.20       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.25       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.30       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.35       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.40       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.45       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.50       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.55       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.60       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.65       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.70       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.75       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.80       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9379, F1=0.9028, Normal Recall=0.9274, Normal Precision=0.9829, Attack Recall=0.9623, Attack Precision=0.8503

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
0.15       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981   <--
0.20       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.25       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.30       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.35       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.40       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.45       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.50       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.55       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.60       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.65       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.70       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.75       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.80       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9413, F1=0.9291, Normal Recall=0.9272, Normal Precision=0.9736, Attack Recall=0.9623, Attack Precision=0.8981

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
0.15       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294   <--
0.20       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.25       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.30       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.35       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.40       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.45       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.50       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.55       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.60       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.65       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.70       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.75       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.80       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9446, F1=0.9456, Normal Recall=0.9269, Normal Precision=0.9609, Attack Recall=0.9623, Attack Precision=0.9294

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
0.15       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947   <--
0.20       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.25       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.30       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.35       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.40       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.45       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.50       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.55       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.60       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.65       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.70       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.75       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
0.80       0.9307   0.7352   0.9271   0.9955   0.9626   0.5947  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9307, F1=0.7352, Normal Recall=0.9271, Normal Precision=0.9955, Attack Recall=0.9626, Attack Precision=0.5947

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
0.15       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679   <--
0.20       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.25       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.30       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.35       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.40       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.45       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.50       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.55       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.60       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.65       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.70       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.75       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
0.80       0.9343   0.8542   0.9273   0.9899   0.9623   0.7679  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9343, F1=0.8542, Normal Recall=0.9273, Normal Precision=0.9899, Attack Recall=0.9623, Attack Precision=0.7679

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
0.15       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503   <--
0.20       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.25       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.30       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.35       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.40       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.45       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.50       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.55       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.60       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.65       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.70       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.75       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
0.80       0.9379   0.9028   0.9274   0.9829   0.9623   0.8503  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9379, F1=0.9028, Normal Recall=0.9274, Normal Precision=0.9829, Attack Recall=0.9623, Attack Precision=0.8503

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
0.15       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981   <--
0.20       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.25       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.30       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.35       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.40       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.45       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.50       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.55       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.60       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.65       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.70       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.75       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
0.80       0.9413   0.9291   0.9272   0.9736   0.9623   0.8981  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9413, F1=0.9291, Normal Recall=0.9272, Normal Precision=0.9736, Attack Recall=0.9623, Attack Precision=0.8981

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
0.15       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294   <--
0.20       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.25       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.30       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.35       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.40       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.45       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.50       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.55       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.60       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.65       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.70       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.75       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
0.80       0.9446   0.9456   0.9269   0.9609   0.9623   0.9294  
--------------------------------------------------------------------------

Recommended threshold (by f1): 0.15
  At threshold 0.15: Accuracy=0.9446, F1=0.9456, Normal Recall=0.9269, Normal Precision=0.9609, Attack Recall=0.9623, Attack Precision=0.9294

```

